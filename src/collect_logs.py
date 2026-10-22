"""Week 7-8 data collection: run every detector on the corpus and write RAW logs.

Outputs (JSON Lines, one record per line):
  data/raw/test_run.jsonl   latency/resource runs, phase = "warmup" | "eval"
  data/raw/doc_scores.jsonl per-document detector scores (untimed, one pass)

Latency "load" = batch size (documents per request): 1, 4, 16, 64, 256.
Warm-up repetitions are logged but discarded later by src/process_data.py.

Usage: python src/collect_logs.py --corpus data/raw/corpus.csv --out data/raw
"""
import argparse
import json
import math
import os
import platform
import time

import numpy as np
import pandas as pd
import psutil

from detectors import build

SEED = 42
DETECTORS = ["random_baseline", "stylometric_heuristic", "lexical_formality", "ensemble_mean"]
BATCH_SIZES = [1, 4, 16, 64, 256]
WARMUP_REPS = 3
EVAL_REPS = 30


def collect(corpus_path, out_dir, seed=SEED):
    df = pd.read_csv(corpus_path)
    texts = df["text"].astype(str).tolist()
    rng = np.random.default_rng(seed)
    proc = psutil.Process(os.getpid())
    os.makedirs(out_dir, exist_ok=True)
    meta = {"record": "meta", "seed": seed, "python": platform.python_version(),
            "platform": platform.platform(), "n_docs": len(df), "batch_sizes": BATCH_SIZES,
            "warmup_reps": WARMUP_REPS, "eval_reps": EVAL_REPS, "unit_latency": "ms",
            "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}

    with open(os.path.join(out_dir, "test_run.jsonl"), "w", encoding="utf-8") as f_run, \
         open(os.path.join(out_dir, "doc_scores.jsonl"), "w", encoding="utf-8") as f_doc:
        f_run.write(json.dumps(meta) + "\n")
        for name, det in build(DETECTORS, seed=seed).items():
            # ---- per-document scores (untimed pass over the whole corpus)
            err = 0
            try:
                scores = np.asarray(det.score(texts), dtype=float)
                if np.isnan(scores).any():
                    err = 1
            except Exception:  # logged, never hidden (guardrail: errors)
                scores, err = np.full(len(df), np.nan), 1
            for i, row in df.iterrows():
                f_doc.write(json.dumps({
                    "record": "score", "detector": name, "id": int(row["id"]), "score": None if math.isnan(scores[i]) else float(scores[i]),
                    "label": int(row["label"]), "category": row["category"], "series": row["series"],
                    "ai_fraction": float(row["ai_fraction"]), "ai_segment_tokens": int(row["ai_segment_tokens"]),
                    "ai_fraction_target": None if pd.isna(row["ai_fraction_target"]) else float(row["ai_fraction_target"]),
                    "segment_target": int(row["segment_target"]), "split": row["split"], "error": err}) + "\n")
            # ---- latency vs load: warm-up repetitions, then evaluation repetitions
            for bs in BATCH_SIZES:
                for phase, reps in (("warmup", WARMUP_REPS), ("eval", EVAL_REPS)):
                    for rep in range(reps):
                        batch = [texts[j] for j in rng.choice(len(texts), size=bs, replace=False)]
                        error = 0
                        t0 = time.perf_counter()
                        try:
                            s = np.asarray(det.score(batch), dtype=float)
                            error = int(np.isnan(s).any())
                        except Exception:
                            error = 1
                        dt_ms = (time.perf_counter() - t0) * 1000
                        f_run.write(json.dumps({
                            "record": "latency", "detector": name, "phase": phase, "rep": rep,
                            "batch_size": bs, "latency_ms": dt_ms, "ms_per_doc": dt_ms / bs,
                            "ram_mb": proc.memory_info().rss / 1e6, "error": error}) + "\n")
    return len(df)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default="data/raw/corpus.csv")
    ap.add_argument("--out", default="data/raw")
    a = ap.parse_args()
    n = collect(a.corpus, a.out)
    print(f"collected logs for {n} documents -> {a.out}/test_run.jsonl, doc_scores.jsonl")