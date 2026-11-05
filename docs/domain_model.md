# Domain Model: Formal Definition of Metrics and Notation

## 1. Mathematical notation

| Symbol | Meaning | Domain |
|:---:|---|:---:|
| $N$ | Number of documents in the test split | $\mathbb{N}$ |
| $y_i \in \{0, 1\}$ | Ground truth: 1 if the text contains AI-generated content, 0 if it is pure human text | $\{0, 1\}$ |
| $s_i \in [0, 1]$ | Detector score for document $i$ | $[0, 1]$ |
| $t$ | Fixed detector threshold selected on the validation split | $[0, 1]$ |
| $\hat{y}_i \in \{0, 1\}$ | Detector prediction at the fixed threshold $t$ | $\{0, 1\}$ |
| $p_i \in [0, 1]$ | AI-content proportion in a hybrid document | $\mathbb{R}$ |
| $\ell_i$ | AI-segment length in tokens | $\mathbb{N}$ |
| $L(t)$ | Threshold-selection loss | $\mathbb{R}_{\ge 0}$ |
| $\mathrm{TP},\mathrm{FP},\mathrm{FN},\mathrm{TN}$ | Confusion-matrix counts at threshold $t$ | $\mathbb{N}$ |

The detector prediction is $\hat{y}_i(t)=\mathbb{1}[s_i\ge t]$. Thresholds are
selected on validation data and then frozen before evaluating the test split.

## 2. Threshold-selection loss

The threshold objective penalizes both false accusations and missed AI content,
with a larger cost for false positives:

$$
L(t) = c_{FP}\cdot \mathrm{FPR}(t) + c_{FN}\cdot \mathrm{FNR}(t),
\qquad c_{FP} > c_{FN}.
$$

The implemented selection rule maximizes test-relevant F1 on validation data
subject to the human-text guardrail:

$$
t^{*} = \arg\max_{t}\; F_1(t)
\quad \text{s.t.}\quad \mathrm{FPR}(t) \le 0.05.
$$

## 3. Target metric: F1 score

The primary metric is the $F_1$ score, computed on the test split:

$$
F_1 = \frac{2 \cdot \mathrm{TP}}
{2 \cdot \mathrm{TP} + \mathrm{FP} + \mathrm{FN}}
$$

where:

- $\mathrm{TP} = \sum_{i=1}^{N} \mathbb{1}[\hat{y}_i = 1 \wedge y_i = 1]$
- $\mathrm{FP} = \sum_{i=1}^{N} \mathbb{1}[\hat{y}_i = 1 \wedge y_i = 0]$
- $\mathrm{FN} = \sum_{i=1}^{N} \mathbb{1}[\hat{y}_i = 0 \wedge y_i = 1]$
- $\mathrm{TN} = \sum_{i=1}^{N} \mathbb{1}[\hat{y}_i = 0 \wedge y_i = 0]$

## 4. Guardrail metrics

The system must minimize false accusations of human text. The false positive
rate (FPR) must satisfy the guardrail:

$$
\mathrm{FPR} = \frac{\mathrm{FP}}{\mathrm{FP} + \mathrm{TN}} \le 0.05
$$

The false-negative rate and AUROC are also reported:

$$
\mathrm{FNR} = \frac{\mathrm{FN}}{\mathrm{FN} + \mathrm{TP}},
\qquad
\mathrm{AUROC} = \Pr(s_i^+ > s_j^-),
$$

where $s_i^+$ and $s_j^-$ are scores from positive and pure-human documents.

## 5. Statistical confidence intervals

To quantify uncertainty, we compute a 95% percentile bootstrap confidence
interval for each metric $\theta$:

$$
\operatorname{CI}_{95\%}(\theta)
= \bigl[\theta^*_{0.025},\ \theta^*_{0.975}\bigr]
$$

where $\theta^*_{(q)}$ is the $q$-quantile of $B = 2000$ bootstrap
replicates.
