# Evaluating AI-Text Detectors in Academic Writing: Performance on Hybrid Human–AI Texts

**Authors:** Aruzhan Abulgazinova, Aruzhan Kinayatova

![ci](https://github.com/akinayat-stack/Evaluating-the-Effectiveness-of-AI-Text-Detectors-in-Academic-Writing/actions/workflows/ci.yml/badge.svg)

## Research Question
How do watermarking, zero-shot and classifier-based AI-text detectors perform on **hybrid** (partially human-edited, partially AI-generated) academic texts, how do AI-content proportion and AI-segment length affect them, and does an ensemble outperform single detectors?
Hypotheses (H0/H1), the variable matrix, metrics and the baseline are specified in [`docs/methodology_card.md`](docs/methodology_card.md).

## Overview
A reproducible benchmarking harness. Independent variables (detectors, decision threshold, data path, number of runs) live in `configs/config.yaml`. `src/main.py` runs every configured detector on a labelled CSV and writes:
- quality metrics: **F1 (primary)**, AUROC, FNR;
- guardrail metrics: FPR on pure-human text, latency, peak RAM, errors;
- per-document scores (for the statistical tests in `src/stats.py`).

**Status (Week 6):** tested skeleton + 9-row sample. The built-in detectors are transparent CPU-only smoke-test baselines; their sample results are *not* scientific findings. Full corpus collection and real detectors (Binoculars, DetectGPT, Ghostbuster-like, RoBERTa) are Week 7.

## System Requirements
- Docker ≥ 24, **or** Python 3.11 (CI and Docker image use 3.11.9; also verified on 3.12)
- CPU only; ≥ 2 GB RAM for the sample run (≈ 8 GB for neural detectors)
- Internet only for the optional neural detector (`requirements-hf.txt`)

## Quickstart
```bash
git clone https://github.com/akinayat-stack/Evaluating-the-Effectiveness-of-AI-Text-Detectors-in-Academic-Writing.git && cd Evaluating-the-Effectiveness-of-AI-Text-Detectors-in-Academic-Writing
docker build -t ai-detect-bench .
docker run --rm -v "$(pwd)/results:/app/results" ai-detect-bench
```
Without Docker:
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt && python src/main.py --config configs/config.yaml
```
Outputs in `results/`: `summary.csv`, `results.csv`, `results.json`, `scores.csv`, `run.log`.

Tests: `python -m unittest discover -s tests`
Paired bootstrap for two detectors: `python src/stats.py results/scores.csv --a ensemble_mean --b stylometric_heuristic`

## Repository Structure
```
├── README.md  LICENSE  CITATION.cff  .gitignore  .dockerignore
├── Dockerfile  requirements.txt  requirements-hf.txt
├── .github/workflows/ci.yml   # tests + docker build on every push
├── configs/config.yaml        # experiment parameters (independent variables)
├── data/sample/sample.csv     # 9-row smoke-test sample
├── src/
│   ├── main.py                # entry point
│   ├── benchmark.py           # measurement + guardrails
│   ├── detectors.py           # detector implementations
│   └── stats.py               # paired bootstrap for H1b / H2b
├── tests/test_pipeline.py
├── notebooks/                 # analysis (Week 7+)
├── docs/methodology_card.md
└── results/                   # outputs (git-ignored)
```

## Planned Benchmarks
| ID | Varied (independent) | Measured (dependent) | Hypothesis / test |
|----|----------------------|----------------------|-------------------|
| B1 | Detector type × text category | F1, AUROC | H1, H1b — two-way ANOVA, paired bootstrap |
| B2 | AI-content proportion (0–100 %) | Probability of correct detection | H2a — logistic regression |
| B3 | Ensemble vs best single detector | F1 | H2b — paired bootstrap |
| B4 | AI-segment length (25–400 tokens) | FNR | H3 — χ² / Cochran–Armitage |

## Data format
`id,text,label,category,ai_fraction,ai_segment_tokens,split` — `label=1` means the text contains AI-generated content (pure-AI or hybrid); `label=0` is pure human.

## Citation / License
Released under the MIT License (see `LICENSE`). Cite via `CITATION.cff` or:
```bibtex
@software{abulgazinova_kinayatova_2026,
  author = {Abulgazinova, Aruzhan and Kinayatova, Aruzhan},
  title  = {Evaluating AI-text detectors on hybrid academic texts},
  year   = {2026},
  url    = {https://github.com/akinayat-stack/Evaluating-the-Effectiveness-of-AI-Text-Detectors-in-Academic-Writing}
}
```
