# Domain Model: Formal Definition of Loss, Metrics and Notation

**Research track:** Machine Learning / AI.
**System under study:** binary AI-text detectors that decide whether an academic text *contains AI-generated content* (pure-AI or hybrid) or is *pure human*.

This document formalises (1) the data and the decision rule, (2) the **loss function** $`L`$ used to choose each detector's threshold, (3) the **target metrics** and **guardrail metrics**, (4) the statistics used to quantify uncertainty, and (5) the full **notation table**. Section 7 maps every formula to the function that implements it in `src/process_data.py`.

---

## 1. Mathematical notation

### 1.1 Data and decision rule

| Symbol | Meaning | Domain / unit | Code name |
|:---:|---|:---:|---|
| $`\mathcal{D}`$ | Full labelled corpus | set of documents | `corpus.csv` |
| $`\mathcal{D}_{\text{val}}, \mathcal{D}_{\text{test}}`$ | Validation split (threshold tuning) and test split (final metrics) | disjoint subsets of $`\mathcal{D}`$ | `split` |
| $`N`$ | Number of documents in the evaluated split (test: $`N = 329`$) | $`\mathbb{N}`$ | `n_test_docs` |
| $`N_{+},\ N_{-}`$ | Number of AI-containing (positive) and pure-human (negative) documents | $`\mathbb{N}`$ | `label` |
| $`x_i`$ | The $`i`$-th document (text) | string | `text` |
| $`y_i`$ | Ground truth: $`1`$ if the text contains AI-generated content, $`0`$ if pure human | $`\{0, 1\}`$ | `label` |
| $`a_i`$ | AI-content proportion of document $`i`$ (0, 0.25, 0.5, 0.75, 1) | $`[0, 1]`$ | `ai_fraction` |
| $`m_i`$ | Length of the single embedded AI segment (25, 50, 100, 200, 400) | tokens | `segment_target` |
| $`d`$ | Detector index: random baseline, stylometric heuristic, lexical formality, ensemble | categorical | `detector` |
| $`f_d`$ | Detector $`d`$ as a scoring function $`f_d : x \mapsto s`$ | text $`\to [0,1]`$ | `detectors.py` |
| $`s_i`$ | Detector score of document $`i`$ (higher means more AI-like) | $`[0, 1]`$ | `score` |
| $`t_d`$ | Decision threshold of detector $`d`$, tuned once on $`\mathcal{D}_{\text{val}}`$, then frozen | $`[0, 1]`$ | `thresholds.json` |
| $`\hat{y}_i`$ | Predicted label at threshold $`t_d`$ | $`\{0, 1\}`$ | `pred` |

### 1.2 Metrics and statistics

| Symbol | Meaning | Domain / unit | Code name |
|:---:|---|:---:|---|
| $`\mathrm{TP},\ \mathrm{FP},\ \mathrm{TN},\ \mathrm{FN}`$ | Confusion-matrix counts on the evaluated split | $`\mathbb{N}`$ | — |
| $`\mathrm{FPR}`$ | False positive rate on pure-human texts (false accusation rate) | $`[0, 1]`$ | `fpr_human` |
| $`\mathrm{FNR}`$ | False negative rate on AI-containing texts (miss rate) | $`[0, 1]`$ | `fnr` |
| $`F_1`$ | Primary target metric | $`[0, 1]`$, dimensionless | `f1` |
| $`\mathrm{AUROC}`$ | Area under the ROC curve, threshold-free quality | $`[0, 1]`$, dimensionless | `auroc` |
| $`L(t)`$ | Loss function of the threshold $`t`$ (Section 3) | $`[0, 1]`$ | — |
| $`\varepsilon`$ | Guardrail limit on FPR, $`\varepsilon = 0.05`$ | $`[0, 1]`$ | `max_fpr_human` |
| $`\mathcal{T}`$ | Set of candidate thresholds (midpoints between sorted validation scores) | finite subset of $`\mathbb{R}`$ | `cands` |
| $`\theta`$ | A generic metric (F1, AUROC, FNR, FPR or a latency median) | — | — |
| $`B`$ | Number of bootstrap replicates, $`B = 2000`$ | $`\mathbb{N}`$ | `N_BOOT` |
| $`\theta^{*}_{b}`$ | Value of $`\theta`$ on the $`b`$-th bootstrap resample | same as $`\theta`$ | — |
| $`\theta^{*}_{(q)}`$ | $`q`$-quantile of $`\{\theta^{*}_{b}\}_{b=1}^{B}`$ | same as $`\theta`$ | — |
| $`\alpha`$ | Significance level, $`\alpha = 0.05`$ (95 % intervals) | $`(0, 1)`$ | `alpha` |
| $`z`$ | Standard normal quantile, $`z = 1.96`$ | dimensionless | `z` |
| $`\hat{p},\ n`$ | Observed proportion and sample size (Wilson interval) | $`[0,1]`$, $`\mathbb{N}`$ | `k / n` |
| $`\beta_0,\ \beta_1`$ | Intercept and slope of the logistic model for hypothesis H2a | dimensionless | `coef_` |
| $`\sigma(\cdot)`$ | Logistic function $`\sigma(u) = 1 / (1 + e^{-u})`$ | $`\mathbb{R} \to (0, 1)`$ | — |
| $`\tau_{r,b}`$ | Latency of evaluation run $`r`$ at batch size $`b`$ | milliseconds (ms) | `latency_ms` |
| $`b`$ | Batch size (load): documents per request | $`\{1, 4, 16, 64, 256\}`$ | `batch_size` |
| $`R_w,\ R_e`$ | Number of warm-up runs (discarded) and evaluation runs ($`R_w = 3`$, $`R_e = 30`$ per setting) | $`\mathbb{N}`$ | `warmup_reps`, `eval_reps` |
| $`\mathrm{P}_q(\cdot)`$ | Empirical $`q`$-percentile; $`\mathrm{P}_{50}`$ is the median, also $`\mathrm{P}_{25}, \mathrm{P}_{75}, \mathrm{P}_{95}, \mathrm{P}_{99}`$ | same as input | `np.percentile` |
| $`\mathbb{1}[\cdot]`$ | Indicator function: 1 if the condition holds, otherwise 0 | $`\{0, 1\}`$ | — |

---

## 2. Decision rule

Each detector $`d`$ returns a score $`s_i = f_d(x_i)`$. The label is predicted with a single frozen threshold:

```math
\hat{y}_i = \mathbb{1}\left[\, s_i \ge t_d \,\right]
```

The confusion-matrix counts on a split of $`N`$ documents are:

```math
\mathrm{TP} = \sum_{i=1}^{N} \mathbb{1}\left[\hat{y}_i = 1 \wedge y_i = 1\right], \qquad
\mathrm{FP} = \sum_{i=1}^{N} \mathbb{1}\left[\hat{y}_i = 1 \wedge y_i = 0\right]
```

```math
\mathrm{FN} = \sum_{i=1}^{N} \mathbb{1}\left[\hat{y}_i = 0 \wedge y_i = 1\right], \qquad
\mathrm{TN} = \sum_{i=1}^{N} \mathbb{1}\left[\hat{y}_i = 0 \wedge y_i = 0\right]
```

Here $`N_{+} = \mathrm{TP} + \mathrm{FN}`$ and $`N_{-} = \mathrm{FP} + \mathrm{TN}`$.

---

## 3. Loss function

### 3.1 Loss used for threshold selection

The detectors are heuristic scoring functions, so the only parameter that is *fitted* is the threshold $`t`$. It is chosen on the validation split by minimising the **F1 loss**

```math
L(t) = 1 - F_1(t) = 1 - \frac{2\,\mathrm{TP}(t)}{2\,\mathrm{TP}(t) + \mathrm{FP}(t) + \mathrm{FN}(t)}
```

subject to the **false-accusation constraint** (a false accusation of a student is worse than a miss):

```math
t_d = \arg\min_{t \in \mathcal{T}} \; L(t) \qquad \text{s.t.} \qquad \mathrm{FPR}(t) \le \varepsilon, \quad \varepsilon = 0.05
```

If no candidate $`t \in \mathcal{T}`$ satisfies the constraint, the threshold with the lowest FPR is taken, and the detector is then reported as *failing* the guardrail. The threshold is computed once on $`\mathcal{D}_{\text{val}}`$ and never re-tuned on $`\mathcal{D}_{\text{test}}`$.

### 3.2 Training loss for classifier-type detectors (reference)

A neural classifier detector (for example the optional `roberta_openai` member) is trained with the binary cross-entropy loss, where $`s_i`$ is its predicted probability that $`x_i`$ contains AI-generated content:

```math
L_{\mathrm{BCE}} = -\frac{1}{N} \sum_{i=1}^{N} \left[\, y_i \log s_i + (1 - y_i) \log (1 - s_i) \,\right]
```

The four detectors evaluated in this study (random baseline, stylometric heuristic, lexical formality, ensemble mean) are not trained, so only the loss of Section 3.1 is used.

---

## 4. Target metrics

**Primary metric (decides H2b):**

```math
F_1 = \frac{2\,\mathrm{TP}}{2\,\mathrm{TP} + \mathrm{FP} + \mathrm{FN}}
```

**Secondary metrics:**

```math
\mathrm{FNR} = \frac{\mathrm{FN}}{\mathrm{FN} + \mathrm{TP}} = \frac{\mathrm{FN}}{N_{+}}
```

```math
\mathrm{AUROC} = \frac{1}{N_{+} N_{-}} \sum_{i:\, y_i = 1} \; \sum_{j:\, y_j = 0} \left( \mathbb{1}\left[s_i > s_j\right] + \tfrac{1}{2}\,\mathbb{1}\left[s_i = s_j\right] \right)
```

AUROC is the probability that a random AI-containing text receives a higher score than a random human text. A random detector gives $`\mathrm{AUROC} \approx 0.5`$.

**Hypothesis-test statistics:**

```math
\Delta F_1 = F_1(\text{ensemble}) - F_1(\text{best single detector}) \quad \text{(H2b)}
```

```math
\Delta \mathrm{FNR} = \mathrm{FNR}(m \le 50) - \mathrm{FNR}(m \ge 200) \quad \text{(H3)}
```

```math
P(\hat{y}_i = 1) = \sigma\left(\beta_0 + \beta_1 a_i\right), \qquad H_1 : \beta_1 > 0 \quad \text{(H2a)}
```

---

## 5. Guardrail metrics

A detector is accepted only if **all** guardrails hold. They are checked automatically and stored as `guard_*` columns in `data/processed/metrics_summary.csv`.

```math
\mathrm{FPR} = \frac{\mathrm{FP}}{\mathrm{FP} + \mathrm{TN}} \le 0.05
```

| Guardrail | Condition | Column |
|---|---|---|
| False accusations of human text | $`\mathrm{FPR} \le 0.05`$ | `guard_fpr_le_5pct` |
| Time per document | $`\mathrm{P}_{99}(\text{ms per document}) \le 10{,}000`$ ms | `guard_latency` |
| Peak memory | peak RAM $`\le 8192`$ MB | `guard_ram` |
| Errors | number of failed runs $`= 0`$ | `guard_errors` |

---

## 6. Performance metrics and uncertainty

### 6.1 Latency statistics (warm-up filtered)

For every detector and batch size $`b`$, the first $`R_w`$ (cold) runs are **discarded**; the statistics use only the $`R_e`$ evaluation runs $`\tau_{1,b}, \dots, \tau_{R_e,b}`$:

```math
\tilde{\tau}_b = \mathrm{P}_{50}(\tau_{\cdot,b}), \qquad \mathrm{IQR}_b = \left[\mathrm{P}_{25}(\tau_{\cdot,b}),\ \mathrm{P}_{75}(\tau_{\cdot,b})\right], \qquad \mathrm{P}_{95}(\tau_{\cdot,b}), \quad \mathrm{P}_{99}(\tau_{\cdot,b})
```

All latencies are in milliseconds (ms).

### 6.2 Bootstrap confidence interval

To quantify uncertainty, a 95 % percentile bootstrap confidence interval is computed for each metric $`\theta`$ (F1, AUROC, FNR, FPR, median latency). The data are resampled with replacement $`B = 2000`$ times with a fixed seed (`numpy` seed 42):

```math
\mathrm{CI}_{95\%}(\theta) = \left[\, \theta^{*}_{(0.025)},\ \theta^{*}_{(0.975)} \,\right]
```

The bootstrap mean and standard deviation are reported as

```math
\bar{\theta}^{*} = \frac{1}{B} \sum_{b=1}^{B} \theta^{*}_{b}, \qquad \mathrm{SD}^{*} = \sqrt{\frac{1}{B-1} \sum_{b=1}^{B} \left(\theta^{*}_{b} - \bar{\theta}^{*}\right)^{2}}
```

### 6.3 Wilson interval for proportions

Flag rates and FNR by AI proportion and by segment length are proportions $`\hat{p} = k / n`$ from small samples ($`n = 21`$ per cell). For them the Wilson 95 % interval is used, which stays inside $`[0, 1]`$ near 0 and 1:

```math
\left[\, \frac{\hat{p} + \frac{z^{2}}{2n}}{1 + \frac{z^{2}}{n}} \;\mp\; \frac{z}{1 + \frac{z^{2}}{n}} \sqrt{\frac{\hat{p}\,(1 - \hat{p})}{n} + \frac{z^{2}}{4 n^{2}}} \,\right], \qquad z = 1.96
```

---

## 7. Mapping of formulas to code

| Formula | Section | Function in `src/process_data.py` |
|---|---|---|
| Warm-up filtering, $`R_w`$ discarded | 6.1 | `split_warmup` |
| Median, $`\mathrm{P}_{25}`$, $`\mathrm{P}_{75}`$, $`\mathrm{P}_{95}`$, $`\mathrm{P}_{99}`$ of latency | 6.1 | `latency_summary` |
| Bootstrap $`\mathrm{CI}_{95\%}`$ of the median | 6.2 | `bootstrap_ci` |
| Threshold $`t_d`$ (loss $`L`$ with FPR constraint) | 3.1 | `tune_thresholds` |
| Predictions $`\hat{y}_i`$ | 2 | `add_predictions` |
| $`F_1`$, $`\mathrm{AUROC}`$, $`\mathrm{FNR}`$, $`\mathrm{FPR}`$ with bootstrap CI | 4, 6.2 | `quality_table` |
| Wilson interval | 6.3 | `wilson_ci` |
| Guardrails | 5 | `guardrail_table`, `metrics_summary` |
| H2a, H2b, H3 | 4 | `logistic_slope_h2a`, `hypothesis_tests`, `segment_length_h3` |
