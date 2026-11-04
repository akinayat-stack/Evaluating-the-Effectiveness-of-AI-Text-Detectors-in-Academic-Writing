"""Entry point:  python src/main.py --config configs/config.yaml"""
import argparse
import json
import os
import sys

import pandas as pd
import yaml

from benchmark import run, summarize


def main():
    ap = argparse.ArgumentParser(description="AI-text detector benchmark")
    ap.add_argument("--config", required=True, help="path to YAML config")
    args = ap.parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    runs, docs = run(cfg)
    summ, failed = summarize(runs, cfg)

    out = cfg["output"]["dir"]
    os.makedirs(out, exist_ok=True)
    runs.to_csv(f"{out}/results.csv", index=False)
    docs.to_csv(f"{out}/scores.csv", index=False)
    summ.to_csv(f"{out}/summary.csv")
    with open(f"{out}/results.json", "w") as f:
        json.dump(json.loads(runs.to_json(orient="records")), f, indent=2)
    with open(f"{out}/run.log", "w") as f:
        f.write(summ.round(4).to_string())

    pd.set_option("display.width", 200)
    print(f"== {cfg['experiment']['name']} | threshold={cfg['decision']['threshold']} ==")
    print(summ.round(4).to_string())
    if failed:
        print(f"ERROR: detectors failed: {failed}", file=sys.stderr)
        sys.exit(1)
    print(f"Results written to {out}/")


if __name__ == "__main__":
    main()
