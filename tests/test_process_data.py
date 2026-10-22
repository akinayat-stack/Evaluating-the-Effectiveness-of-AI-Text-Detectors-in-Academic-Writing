import os
import sys
import unittest

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

import process_data as pdx  # noqa: E402
from make_corpus import make_corpus  # noqa: E402


class TestStatistics(unittest.TestCase):
    def test_bootstrap_ci_reproducible_and_brackets_median(self):
        x = np.arange(1, 101, dtype=float)
        a, b = pdx.bootstrap_ci(x, seed=42), pdx.bootstrap_ci(x, seed=42)
        self.assertEqual(a, b)
        self.assertLess(a[0], np.median(x)); self.assertGreater(a[1], np.median(x))

    def test_wilson_ci_bounds(self):
        lo, hi = pdx.wilson_ci(0, 20)
        self.assertEqual(lo, 0.0); self.assertGreater(hi, 0.0)
        lo, hi = pdx.wilson_ci(20, 20)
        self.assertLess(lo, 1.0); self.assertEqual(hi, 1.0)

    def test_holm_monotone_and_bounded(self):
        adj = pdx.holm([0.01, 0.04, 0.03])
        self.assertTrue(np.all(adj >= np.array([0.01, 0.04, 0.03])) and np.all(adj <= 1))
        self.assertAlmostEqual(adj[0], 0.03)

    def test_split_warmup_removes_cold_runs(self):
        runs = pd.DataFrame([
            {"record": "meta"},
            {"record": "latency", "phase": "warmup", "latency_ms": 9.0},
            {"record": "latency", "phase": "eval", "latency_ms": 1.0},
        ])
        ev, warm = pdx.split_warmup(runs)
        self.assertEqual(list(ev["phase"]), ["eval"]); self.assertEqual(len(warm), 1)

    def test_threshold_respects_fpr_guardrail_on_validation(self):
        rng = np.random.default_rng(0)
        s = np.r_[rng.normal(0.3, 0.1, 200), rng.normal(0.7, 0.15, 200)].clip(0, 1)
        y = np.r_[np.zeros(200), np.ones(200)].astype(int)
        df = pd.DataFrame({"detector": "d", "split": "val", "score": s, "label": y})
        thr = pdx.tune_thresholds(df)["d"]
        fpr = ((s >= thr) & (y == 0)).sum() / 200
        self.assertLessEqual(fpr, 0.05)


class TestCorpusAndLogs(unittest.TestCase):
    def test_corpus_is_deterministic_and_well_formed(self):
        a, b = make_corpus(42), make_corpus(42)
        pd.testing.assert_frame_equal(a, b)
        self.assertEqual(set(a["category"]), {"pure-human", "pure-ai", "hybrid"})
        self.assertTrue(set(a["split"]) == {"val", "test"})
        self.assertTrue((a[a.series == "pure_human"]["label"] == 0).all())
        self.assertTrue((a[a.category == "hybrid"]["ai_fraction"].between(0.02, 0.95)).all())

    def test_committed_logs_have_warmup_and_eval_and_no_errors(self):
        raw = os.path.join(ROOT, "data/raw/test_run.jsonl")
        if not os.path.exists(raw):
            self.skipTest("raw logs not present")
        runs = pdx.load_jsonl(raw)
        ev, warm = pdx.split_warmup(runs)
        self.assertGreater(len(warm), 0); self.assertGreater(len(ev), 0)
        self.assertEqual(int(ev["error"].sum()), 0)


if __name__ == "__main__":
    unittest.main()