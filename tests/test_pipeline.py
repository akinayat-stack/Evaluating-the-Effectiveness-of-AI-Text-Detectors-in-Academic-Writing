import os
import subprocess
import sys
import tempfile
import unittest

import pandas as pd
import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from detectors import build  # noqa: E402


class TestPipeline(unittest.TestCase):
    def test_sample_run_produces_outputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            with open(os.path.join(ROOT, "configs/config.yaml"), encoding="utf-8") as f:
                cfg = yaml.safe_load(f)
            cfg["output"]["dir"] = tmp
            cfg["data"]["path"] = os.path.join(ROOT, cfg["data"]["path"])
            cpath = os.path.join(tmp, "cfg.yaml")
            with open(cpath, "w", encoding="utf-8") as f:
                yaml.safe_dump(cfg, f)
            r = subprocess.run([sys.executable, os.path.join(ROOT, "src/main.py"), "--config", cpath],
                               capture_output=True, text=True)
            self.assertEqual(r.returncode, 0, r.stderr)
            for name in ("results.csv", "results.json", "scores.csv", "summary.csv", "run.log"):
                self.assertTrue(os.path.exists(os.path.join(tmp, name)), name)
            summ = pd.read_csv(os.path.join(tmp, "summary.csv"), index_col=0)
            for det in ("stylometric_heuristic", "lexical_formality", "ensemble_mean"):
                self.assertIn(det, summ.index)
            self.assertTrue(summ["f1"].between(0, 1).all())
            # sanity: every real detector must beat the random baseline on AUROC
            self.assertGreater(summ.loc["ensemble_mean", "auroc"], summ.loc["random_baseline", "auroc"])

    def test_sample_data_schema(self):
        df = pd.read_csv(os.path.join(ROOT, "data/sample/sample.csv"))
        self.assertTrue(5 <= len(df) <= 10)
        self.assertEqual(set(df["category"]), {"pure-human", "pure-ai", "hybrid"})
        self.assertTrue(set(df["label"]) <= {0, 1})

    def test_ensemble_differs_from_single_member(self):
        df = pd.read_csv(os.path.join(ROOT, "data/sample/sample.csv"))
        dets = build(["stylometric_heuristic", "lexical_formality", "ensemble_mean"])
        a = dets["stylometric_heuristic"].score(df["text"])
        e = dets["ensemble_mean"].score(df["text"])
        self.assertNotEqual(a, e)

    def test_scores_in_unit_interval(self):
        dets = build(["random_baseline", "stylometric_heuristic", "lexical_formality"])
        for d in dets.values():
            s = d.score(["", "Short text.", "Furthermore, it is important to note this."])
            self.assertTrue(all(0.0 <= v <= 1.0 for v in s), d.name)

    def test_unknown_detector_raises(self):
        with self.assertRaises(ValueError):
            build(["does_not_exist"])


if __name__ == "__main__":
    unittest.main()
