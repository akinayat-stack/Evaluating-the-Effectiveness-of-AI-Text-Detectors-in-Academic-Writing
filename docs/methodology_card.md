# Methodology Passport Card — Task 2

## 1. Project information
- **Title:** Evaluating the Effectiveness of AI-Text Detectors in Academic Writing: Performance on Hybrid Human–AI Texts
- **Authors:** Abulgazinova Aruzhan, Kinayatova Aruzhan
- **GitHub repository:** https://github.com/akinayat-stack/Evaluating-the-Effectiveness-of-AI-Text-Detectors-in-Academic-Writing

## 2. Research type
**Quantitative (experimental).** All outputs are numeric (F1, AUROC, FNR), conditions are varied and controlled on a constructed corpus, and conclusions come from statistical tests. A qualitative component is unnecessary: the question concerns measurable detection accuracy, not perception or usage.

## 3. Hypotheses (refined after Task 1)
Refinements relative to Task 1: (a) hybrid text is operationalised as the **positive class ("contains AI")**, pure-human is the negative class; (b) every F1 "on hybrid texts" is computed on the subset **hybrid + pure-human**, because F1 needs negatives; (c) for H2a, Task 1's "F1 vs AI proportion" is replaced by a per-document logistic model of correct detection (F1 is not defined for a single document); (d) the novelty claim is narrowed: closest prior work is CHEAT (Yu et al.), which covers Polish/Fusion texts for supervised detectors, whereas this study compares three detector paradigms on hybrids with *controlled* AI proportion and segment length. Significance level α = 0.05, Holm correction across the five tests.

| ID | RQ | H0 | H1 |
|---|---|---|---|
| H1 | RQ1 | Mean F1 does not differ between detector types (watermark / zero-shot / classifier) on hybrid texts | At least one detector type has a different mean F1 on hybrid texts |
| H1b | RQ1 | F1(classifier) ≤ F1(zero-shot) on hybrid texts | F1(classifier) > F1(zero-shot) on hybrid texts |
| H2a | RQ2 | The coefficient of AI-content proportion in a logistic model of correct detection is 0 | The coefficient is > 0 (detection improves as AI proportion grows) |
| H2b | RQ2 | F1(ensemble) ≤ F1(best single detector) | F1(ensemble) > F1(best single detector) |
| H3 | RQ3 | FNR does not depend on AI-segment length | FNR for short segments (≤ 50 tokens) > FNR for long segments (≥ 200 tokens) |

Planned tests: two-way ANOVA (or Friedman/Kruskal–Wallis if assumptions fail) for H1; paired bootstrap on F1 difference for H1b and H2b (`src/stats.py`); logistic regression for H2a; χ² / Cochran–Armitage trend test for H3.

## 4. Variable matrix
| Group | Variables |
|---|---|
| **Independent** | Detector type/configuration (random baseline, stylometric heuristic, zero-shot, classifier, ensemble); text category (pure-human / pure-AI / hybrid); AI-content proportion (0, 25, 50, 75, 100 %); AI-segment length (25, 50, 100, 200, 400 tokens) |
| **Dependent** | F1 at a fixed threshold; AUROC; FNR; FPR on pure-human; processing time (s/doc); peak RAM (MB) |
| **Controlled** | Document length (300–500 tokens); academic discipline mix; generating LLM and temperature; random seed (42); decision threshold (tuned once on validation split, then frozen); hardware (CPU-only, ≤ 8 GB RAM); OS and library versions (exact pins in `requirements.txt`, base image `python:3.11.9-slim`); fixed input dataset and splits; editing style of hybrid construction |

**Detector → type mapping (independent variable "detector type")**

| Detector | Type | Status |
|---|---|---|
| `random_baseline` | sanity baseline | Week 6, implemented |
| `stylometric_heuristic`, `lexical_formality` | transparent smoke-test heuristics | Week 6, implemented |
| `ensemble_mean` (mean of non-baseline members) | ensemble | Week 6, implemented |
| `roberta_openai` | classifier | optional, implemented (`requirements-hf.txt`) |
| Binoculars, DetectGPT | zero-shot | Week 7 |
| Kirchenbauer-style green-list detector | watermark | Week 7 (in-house watermarked generation) |
| Ghostbuster-like feature classifier | classifier | Week 7 |

Note: watermark detectors only work on text generated with the watermark, so the corpus part for them is generated in-house with an open model (a controlled condition).

## 5. Metrics and baseline
- **Primary metric:** F1 of the binary decision "contains AI" at a fixed threshold. It decides H1, H1b, H2b. AUROC and FNR are secondary (FNR is the target of H3).
- **Guardrail metrics:** FPR on pure-human texts ≤ 5 % (a false accusation is worse than a miss); no exceptions or NaN scores; ≤ 10 s per document on CPU; peak RAM ≤ 8 GB. All are checked automatically in `src/benchmark.py` and reported as `guard_*` columns.
- **Baseline:** (1) random-score baseline (sanity: detectors must beat chance; checked in `tests/`); (2) best single detector on the validation split as the reference for the ensemble (H2b); (3) from Week 7, a published detector (`roberta_openai`, Binoculars) as an external reference so the comparison is not only against toy heuristics. Fairness: identical data, splits, seed and threshold for all methods.

## 6. Status at end of Week 6
Working, tested pipeline (5 unit tests, CI with `docker build`): config → data → 4 detectors (random, stylometric heuristic, lexical formality, ensemble of the two heuristics) → metrics/guardrails → CSV/JSON/log. Real detectors (Binoculars, DetectGPT, Ghostbuster-like, RoBERTa) and the full corpus are Week 7 work. The stylometric heuristic is a transparent smoke-test baseline, not a claim of detector quality; its sample results are not scientific findings.
