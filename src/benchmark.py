"""Measurement core: runs every configured detector on the dataset, records
quality metrics (F1, AUROC, FNR), guardrail metrics (FPR on humans, latency, RAM)
and per-document scores."""
import os
import time
import numpy as np
import pandas as pd
import psutil
from sklearn.metrics import f1_score, roc_auc_score

from detectors import build


def _timed_score(det, texts, timeout):
    t0 = time.perf_counter()
    s = det.score(texts)
    dt = time.perf_counter() - t0
    if dt > timeout:
        raise TimeoutError(f"{det.name} exceeded timeout {timeout}s ({dt:.1f}s)")
    return s, dt


def run(cfg):
    d, ex = cfg["data"], cfg["experiment"]
    df = pd.read_csv(d["path"])
    for col in (d["text_col"], d["label_col"]):
        if col not in df.columns:
            raise KeyError(f"Column '{col}' missing in {d['path']}")
    texts = df[d["text_col"]].astype(str).tolist()
    y = df[d["label_col"]].astype(int).to_numpy()
    cats = df[d["category_col"]].tolist() if d["category_col"] in df.columns else ["na"] * len(df)
    thr = cfg["decision"]["threshold"]
    g = cfg["guardrails"]
    proc = psutil.Process(os.getpid())

    runs, doc_rows = [], []
    for name, det in build(cfg["detectors"], seed=ex["seed"]).items():
        errors = 0
        try:
            for _ in range(ex["warmup_runs"]):
                det.score(texts[:1])
            for r in range(ex["n_runs"]):
                s, dt = _timed_score(det, texts, ex["timeout_sec"])
                s = np.asarray(s, dtype=float)
                if np.isnan(s).any():
                    errors += 1
                pred = (s >= thr).astype(int)
                neg, pos = (y == 0), (y == 1)
                runs.append({
                    "detector": name, "run": r,
                    "f1": f1_score(y, pred, zero_division=0),                      # PRIMARY
                    "auroc": roc_auc_score(y, s) if len(set(y)) > 1 else np.nan,
                    "fnr": float(((pred == 0) & pos).sum() / max(pos.sum(), 1)),
                    "fpr_human": float(((pred == 1) & neg).sum() / max(neg.sum(), 1)),
                    "sec_per_doc": dt / len(texts),
                    "ram_mb": proc.memory_info().rss / 1e6,
                    "errors": errors,
                })
                if r == 0:
                    for i, (sc, p) in enumerate(zip(s, pred)):
                        row = df.iloc[i]
                        doc_rows.append({
                            "id": row.get("id", i), "detector": name, "category": cats[i],
                            "ai_fraction": row.get("ai_fraction", np.nan),
                            "ai_segment_tokens": row.get("ai_segment_tokens", np.nan),
                            "label": int(y[i]), "score": float(sc), "pred": int(p)})
        except Exception as e:  # recorded, not hidden: guardrail "no errors"
            errors += 1
            runs.append({"detector": name, "run": -1, "errors": errors, "error_msg": repr(e)})
    return pd.DataFrame(runs), pd.DataFrame(doc_rows)


def summarize(runs, cfg):
    """Mean over runs + guardrail verdicts."""
    g = cfg["guardrails"]
    ok = runs[runs["run"] >= 0]
    summ = ok.groupby("detector").mean(numeric_only=True).drop(columns=["run"])
    summ["guard_fpr"] = summ["fpr_human"] <= g["max_fpr_human"]
    summ["guard_latency"] = summ["sec_per_doc"] <= g["max_sec_per_doc"]
    summ["guard_ram"] = summ["ram_mb"] <= g["max_ram_mb"]
    summ["guard_errors"] = summ["errors"] == 0 if g["require_no_errors"] else True
    failed = runs[runs["run"] < 0]["detector"].tolist()
    return summ, failed
