"""Data processing and statistical aggregation (Task 3, criterion 1).

Pipeline
  1. load raw JSONL logs                         -> load_jsonl
  2. drop warm-up ("cold") runs                  -> split_warmup
  3. latency statistics (median, p25/p75, p95, p99, bootstrap 95 % CI) -> latency_summary
  4. freeze one threshold per detector on the VALIDATION split          -> tune_thresholds
  5. quality metrics on the TEST split with bootstrap CI (F1, AUROC, FNR, FPR on humans)
  6. guardrails: FPR on pure-human, errors, latency, RAM
  7. hypothesis tests H2a, H2b, H3 (Holm-corrected over the primary detector)

Every random procedure takes an explicit seed (default 42).
Usage:  python src/process_data.py --raw data/raw --out data/processed
"""
import argparse
import json
import os

import numpy as np
import pandas as pd
from scipy import stats as sps
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, roc_auc_score

from stats import paired_bootstrap_f1

SEED = 42
N_BOOT = 2000
GUARDRAILS = {"max_fpr_human": 0.05, "max_ms_per_doc": 10_000.0, "max_ram_mb": 8192.0, "max_errors": 0}
PRIMARY_DETECTOR = "ensemble_mean"      # the "proposed method"
BASELINE_DETECTOR = "stylometric_heuristic"  # reference single detector (see README)


# --------------------------------------------------------------------------- 1-2. ingestion
def load_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return pd.DataFrame([json.loads(line) for line in f if line.strip()])


def split_warmup(runs):
    """Return (eval_runs, warmup_runs). Warm-up (cold) repetitions never enter statistics."""
    lat = runs[runs["record"] == "latency"]
    return lat[lat["phase"] == "eval"].copy(), lat[lat["phase"] == "warmup"].copy()


# --------------------------------------------------------------------------- bootstrap helpers
def bootstrap_ci(x, stat=np.median, n_boot=N_BOOT, alpha=0.05, seed=SEED):
    """Percentile bootstrap CI for `stat` of a 1-D sample."""
    x = np.asarray(x, dtype=float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(x), size=(n_boot, len(x)))
    vals = np.array([stat(x[i]) for i in idx])
    return float(np.percentile(vals, 100 * alpha / 2)), float(np.percentile(vals, 100 * (1 - alpha / 2)))


def wilson_ci(k, n, z=1.96):
    """Wilson 95 % interval for a proportion (well behaved near 0 and 1)."""
    if n == 0:
        return np.nan, np.nan
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return float(max(c - h, 0.0)), float(min(c + h, 1.0))


# --------------------------------------------------------------------------- 3. latency
def latency_summary(ev, seed=SEED):
    rows = []
    for (det, bs), g in ev.groupby(["detector", "batch_size"]):
        x = g["latency_ms"].to_numpy()
        lo, hi = bootstrap_ci(x, np.median, seed=seed)
        rows.append({
            "detector": det, "batch_size": int(bs), "n_runs": len(x),
            "median_ms": np.median(x), "p25_ms": np.percentile(x, 25), "p75_ms": np.percentile(x, 75),
            "p95_ms": np.percentile(x, 95), "p99_ms": np.percentile(x, 99),
            "mean_ms": x.mean(), "sd_ms": x.std(ddof=1),
            "ci95_lo_ms": lo, "ci95_hi_ms": hi,
            "median_ms_per_doc": np.median(g["ms_per_doc"]), "p99_ms_per_doc": np.percentile(g["ms_per_doc"], 99),
            "max_ram_mb": g["ram_mb"].max(), "errors": int(g["error"].sum()),
        })
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- 4. thresholds
def tune_thresholds(scores, max_fpr=GUARDRAILS["max_fpr_human"]):
    """One threshold per detector, chosen ONCE on the validation split, then frozen.

    Rule: maximise F1 subject to the guardrail FPR(pure-human) <= max_fpr (5 %), because a false
    accusation is worse than a miss (docs/methodology_card.md). If no threshold meets the guardrail,
    take the one with the lowest FPR (the detector will then FAIL the guardrail in the report)."""
    thr = {}
    for det, g in scores[scores["split"] == "val"].groupby("detector"):
        s, y = g["score"].to_numpy(), g["label"].to_numpy()
        u = np.unique(s)
        cands = np.concatenate([[u[0] - 1e-9], (u[:-1] + u[1:]) / 2, [u[-1] + 1e-9]])
        f1 = np.array([f1_score(y, (s >= c).astype(int), zero_division=0) for c in cands])
        fpr = np.array([((s >= c) & (y == 0)).sum() / max((y == 0).sum(), 1) for c in cands])
        ok = fpr <= max_fpr
        if ok.any():
            best = f1 == f1[ok].max()
            thr[det] = float(np.median(cands[best & ok]))
        else:
            thr[det] = float(cands[np.argmin(fpr)])
    return thr


def add_predictions(scores, thr):
    out = scores.copy()
    out["pred"] = (out["score"] >= out["detector"].map(thr)).astype(int)
    return out


# --------------------------------------------------------------------------- 5. quality metrics
def _boot_metrics(y, s, pred, n_boot, seed):
    rng = np.random.default_rng(seed)
    n = len(y)
    idx = rng.integers(0, n, size=(n_boot, n))
    yb, pb = y[idx], pred[idx]
    tp = (yb * pb).sum(1); fp = ((1 - yb) * pb).sum(1); fn = (yb * (1 - pb)).sum(1)
    f1 = np.where(2 * tp + fp + fn > 0, 2 * tp / np.maximum(2 * tp + fp + fn, 1), 0.0)
    fnr = fn / np.maximum(yb.sum(1), 1)
    neg = (1 - yb).sum(1)
    fpr = fp / np.maximum(neg, 1)
    auc = []
    for i in range(n_boot):
        auc.append(roc_auc_score(yb[i], s[idx[i]]) if 0 < yb[i].sum() < n else np.nan)
    return {"f1": f1, "fnr": fnr, "fpr_human": fpr, "auroc": np.array(auc)}


def quality_table(scored, n_boot=N_BOOT, seed=SEED):
    """Test-split metrics per detector, with bootstrap mean, SD and 95 % CI."""
    rows = []
    for det, g in scored[scored["split"] == "test"].groupby("detector"):
        y, s, p = g["label"].to_numpy(), g["score"].to_numpy(), g["pred"].to_numpy()
        point = {"f1": f1_score(y, p, zero_division=0), "auroc": roc_auc_score(y, s),
                 "fnr": float(((p == 0) & (y == 1)).sum() / max((y == 1).sum(), 1)),
                 "fpr_human": float(((p == 1) & (y == 0)).sum() / max((y == 0).sum(), 1))}
        boot = _boot_metrics(y, s, p, n_boot, seed)
        row = {"detector": det, "n_test_docs": len(g)}
        for m, v in point.items():
            b = boot[m][~np.isnan(boot[m])]
            row.update({m: v, f"{m}_boot_mean": b.mean(), f"{m}_boot_sd": b.std(ddof=1),
                        f"{m}_ci95_lo": np.percentile(b, 2.5), f"{m}_ci95_hi": np.percentile(b, 97.5)})
        rows.append(row)
    return pd.DataFrame(rows)


def scores_by_category(scored):
    return scored[scored["split"] == "test"][["detector", "category", "score"]]


# --------------------------------------------------------------------------- RQ2 / RQ3 curves
def flag_rate_by_proportion(scored):
    """Share of documents flagged as 'contains AI' vs AI-content proportion (test split).
    0 % = FPR on pure-human, 100 % = TPR on pure-AI, 25/50/75 % = sentence-interleaved hybrids."""
    t = scored[scored["split"] == "test"]
    keep = t[(t["series"] == "pure_human") | (t["series"] == "pure_ai") | (t["series"] == "proportion")].copy()
    keep["proportion_pct"] = np.where(keep["series"] == "pure_human", 0,
                                      np.where(keep["series"] == "pure_ai", 100, keep["ai_fraction_target"] * 100)).astype(int)
    rows = []
    for (det, pr), g in keep.groupby(["detector", "proportion_pct"]):
        k, n = int(g["pred"].sum()), len(g)
        lo, hi = wilson_ci(k, n)
        rows.append({"detector": det, "proportion_pct": pr, "n_docs": n, "flag_rate": k / n, "ci95_lo": lo, "ci95_hi": hi})
    return pd.DataFrame(rows)


def fnr_by_segment(scored):
    """False-negative rate vs length of the single embedded AI block (test split)."""
    t = scored[(scored["split"] == "test") & (scored["series"] == "segment")]
    rows = []
    for (det, L), g in t.groupby(["detector", "segment_target"]):
        k, n = int((g["pred"] == 0).sum()), len(g)
        lo, hi = wilson_ci(k, n)
        rows.append({"detector": det, "segment_tokens": int(L), "n_docs": n, "fnr": k / n, "ci95_lo": lo, "ci95_hi": hi})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- 7. hypothesis tests
def holm(pvals):
    """Holm-Bonferroni adjusted p-values (same order as input)."""
    p = np.asarray(pvals, dtype=float)
    order = np.argsort(p)
    adj = np.empty_like(p)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (len(p) - rank) * p[i])
        adj[i] = min(running, 1.0)
    return adj


def logistic_slope_h2a(scored, det, n_boot=N_BOOT, seed=SEED):
    """H2a: slope of P(detected) on AI proportion among AI-containing documents (test split)."""
    g = scored[(scored["split"] == "test") & (scored["detector"] == det) & (scored["label"] == 1) &
               (scored["series"].isin(["pure_ai", "proportion"]))]
    x = g["ai_fraction"].to_numpy().reshape(-1, 1)
    y = g["pred"].to_numpy()

    def slope(xx, yy):
        if yy.min() == yy.max():  # degenerate resample/all-same outcome
            return np.nan
        return LogisticRegression(C=1e6, max_iter=1000).fit(xx, yy).coef_[0, 0]

    obs = slope(x, y)
    if np.isnan(obs):
        return {"test": "H2a logistic slope (AI proportion)", "detector": det, "estimate": np.nan, "ci95_lo": np.nan,
                "ci95_hi": np.nan, "p_one_sided": np.nan, "n": len(y),
                "note": f"not estimable: {'all' if y.min() == 1 else 'no'} AI-containing documents detected (ceiling/floor)"}
    rng = np.random.default_rng(seed)
    b = []
    for _ in range(n_boot // 4):
        i = rng.integers(0, len(y), len(y))
        b.append(slope(x[i], y[i]))
    b = np.array(b, dtype=float); b = b[~np.isnan(b)]
    p = float((b <= 0).mean()) if len(b) else np.nan  # one-sided: H1 slope > 0
    return {"test": "H2a logistic slope (AI proportion)", "detector": det, "estimate": obs,
            "ci95_lo": np.percentile(b, 2.5) if len(b) else np.nan,
            "ci95_hi": np.percentile(b, 97.5) if len(b) else np.nan, "p_one_sided": p, "n": len(y)}


def cochran_armitage(k, n, w):
    """Cochran-Armitage trend z (positive z = proportion rises with weight)."""
    k, n, w = map(lambda a: np.asarray(a, float), (k, n, w))
    N, pbar = n.sum(), k.sum() / n.sum()
    num = (w * (k - n * pbar)).sum()
    var = pbar * (1 - pbar) * ((n * w ** 2).sum() - ((n * w).sum() ** 2) / N)
    return float(num / np.sqrt(var)) if var > 0 else np.nan


def segment_length_h3(scored, det):
    """H3: FNR(short <= 50 tokens) > FNR(long >= 200 tokens), one-sided Fisher exact + trend test."""
    g = scored[(scored["split"] == "test") & (scored["detector"] == det) & (scored["series"] == "segment")]
    short, long_ = g[g["segment_target"] <= 50], g[g["segment_target"] >= 200]
    fn_s, fn_l = int((short["pred"] == 0).sum()), int((long_["pred"] == 0).sum())
    table = [[fn_s, len(short) - fn_s], [fn_l, len(long_) - fn_l]]
    _, p = sps.fisher_exact(table, alternative="greater")
    cells = g.groupby("segment_target")["pred"].agg(n="size", hit="sum")
    z = cochran_armitage((cells["n"] - cells["hit"]).to_numpy(), cells["n"].to_numpy(), np.log(cells.index.to_numpy()))
    return {"test": "H3 FNR short(<=50) vs long(>=200)", "detector": det,
            "estimate": fn_s / len(short) - fn_l / len(long_), "ci95_lo": np.nan, "ci95_hi": np.nan,
            "p_one_sided": float(p), "n": len(g), "trend_z_fnr_vs_log_length": z}


def hypothesis_tests(scored, primary=PRIMARY_DETECTOR, baseline=BASELINE_DETECTOR, n_boot=N_BOOT, seed=SEED):
    """H2a (proportion), H2b (ensemble vs best single), H3 (segment length) + Holm over the primary three."""
    t = scored[scored["split"] == "test"]
    v = scored[scored["split"] == "val"]
    singles = [d for d in t["detector"].unique() if d not in ("random_baseline", primary)]
    f1s = {d: f1_score(v[v.detector == d]["label"], v[v.detector == d]["pred"], zero_division=0) for d in singles}
    best_single = max(f1s, key=f1s.get)  # reference chosen on VALIDATION, compared on TEST
    b = paired_bootstrap_f1(t, primary, best_single, n_boot=n_boot, seed=seed)
    rows = [logistic_slope_h2a(scored, primary, seed=seed),
            {"test": f"H2b F1({primary}) - F1({best_single})", "detector": primary, "estimate": b["delta_f1"],
             "ci95_lo": b["ci95"][0], "ci95_hi": b["ci95"][1], "p_one_sided": b["p_one_sided"], "n": len(t[t.detector == primary])},
            segment_length_h3(scored, primary)]
    df = pd.DataFrame(rows)
    df["p_holm"] = holm(df["p_one_sided"].fillna(1.0).to_numpy())  # undefined test -> not rejected
    df["reject_H0_at_0.05"] = df["p_holm"] < 0.05
    extra = pd.DataFrame([logistic_slope_h2a(scored, d, seed=seed) for d in singles + ["random_baseline"]] +
                         [segment_length_h3(scored, d) for d in singles + ["random_baseline"]])
    extra["p_holm"] = np.nan; extra["reject_H0_at_0.05"] = np.nan
    return pd.concat([df, extra], ignore_index=True), best_single


# --------------------------------------------------------------------------- 6. guardrails + summary
def guardrail_table(quality, lat):
    per = lat.groupby("detector").agg(p99_ms_per_doc=("p99_ms_per_doc", "max"), max_ram_mb=("max_ram_mb", "max"),
                                      errors=("errors", "sum")).reset_index()
    g = quality[["detector", "fpr_human"]].merge(per, on="detector")
    g["guard_fpr_le_5pct"] = g["fpr_human"] <= GUARDRAILS["max_fpr_human"]
    g["guard_latency"] = g["p99_ms_per_doc"] <= GUARDRAILS["max_ms_per_doc"]
    g["guard_ram"] = g["max_ram_mb"] <= GUARDRAILS["max_ram_mb"]
    g["guard_errors"] = g["errors"] <= GUARDRAILS["max_errors"]
    return g


def metrics_summary(quality, lat, bs_ref=16):
    """One row per detector: quality + latency at reference load + guardrail verdicts."""
    ref = lat[lat["batch_size"] == bs_ref][["detector", "median_ms", "p25_ms", "p75_ms", "p95_ms", "p99_ms",
                                            "ci95_lo_ms", "ci95_hi_ms", "median_ms_per_doc"]]
    ref = ref.rename(columns={c: f"lat_bs{bs_ref}_{c}" for c in ref.columns if c != "detector"})
    return quality.merge(ref, on="detector").merge(
        guardrail_table(quality, lat)[["detector", "guard_fpr_le_5pct", "guard_latency", "guard_ram", "guard_errors", "errors"]],
        on="detector")


def load_scored(raw_dir):
    """Per-document scores + frozen thresholds (validation) + predictions."""
    scores = load_jsonl(os.path.join(raw_dir, "doc_scores.jsonl"))
    thr = tune_thresholds(scores)
    return add_predictions(scores, thr), thr


def run_all(raw_dir, out_dir, n_boot=N_BOOT, seed=SEED):
    np.random.seed(seed)
    os.makedirs(out_dir, exist_ok=True)
    runs = load_jsonl(os.path.join(raw_dir, "test_run.jsonl"))
    ev, warm = split_warmup(runs)
    lat = latency_summary(ev, seed)
    scored, thr = load_scored(raw_dir)
    quality = quality_table(scored, n_boot, seed)
    tests, best = hypothesis_tests(scored, n_boot=n_boot, seed=seed)
    out = {
        "metrics_summary.csv": metrics_summary(quality, lat),
        "latency_summary.csv": lat,
        "flag_rate_by_proportion.csv": flag_rate_by_proportion(scored),
        "fnr_by_segment.csv": fnr_by_segment(scored),
        "hypothesis_tests.csv": tests,
    }
    for name, df in out.items():
        df.to_csv(os.path.join(out_dir, name), index=False)
    with open(os.path.join(out_dir, "thresholds.json"), "w") as f:
        json.dump({"thresholds": thr, "warmup_runs_discarded": int(len(warm)), "eval_runs_used": int(len(ev)),
                   "best_single_detector_on_val": best, "seed": seed}, f, indent=2)
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default="data/raw")
    ap.add_argument("--out", default="data/processed")
    ap.add_argument("--n-boot", type=int, default=N_BOOT)
    a = ap.parse_args()
    res = run_all(a.raw, a.out, a.n_boot)
    pd.set_option("display.width", 220, "display.max_columns", 40)
    print(res["metrics_summary.csv"].round(3).T.to_string())
    print(res["hypothesis_tests.csv"].round(4).to_string())