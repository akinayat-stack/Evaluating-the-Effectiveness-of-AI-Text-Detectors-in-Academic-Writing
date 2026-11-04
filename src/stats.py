"""Statistical tests for H1b/H2b: paired bootstrap on F1 difference between two detectors.
Usage: python src/stats.py results/scores.csv --a ensemble_mean --b stylometric_heuristic"""
import argparse
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score


def paired_bootstrap_f1(scores, a, b, n_boot=2000, seed=42):
    A = scores[scores.detector == a].sort_values("id").reset_index(drop=True)
    B = scores[scores.detector == b].sort_values("id").reset_index(drop=True)
    assert len(A) == len(B) and (A["id"] == B["id"]).all(), "detectors must cover the same documents"
    y, pa, pb = A["label"].to_numpy(), A["pred"].to_numpy(), B["pred"].to_numpy()
    rng = np.random.default_rng(seed)
    diffs = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(y), len(y))
        diffs.append(f1_score(y[idx], pa[idx], zero_division=0) - f1_score(y[idx], pb[idx], zero_division=0))
    diffs = np.array(diffs)
    obs = f1_score(y, pa, zero_division=0) - f1_score(y, pb, zero_division=0)
    return {"delta_f1": obs, "ci95": (np.percentile(diffs, 2.5), np.percentile(diffs, 97.5)),
            "p_one_sided": float((diffs <= 0).mean())}  # H1: F1(a) > F1(b)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("scores"); ap.add_argument("--a", required=True); ap.add_argument("--b", required=True)
    ap.add_argument("--n-boot", type=int, default=2000)
    args = ap.parse_args()
    print(paired_bootstrap_f1(pd.read_csv(args.scores), args.a, args.b, args.n_boot))
