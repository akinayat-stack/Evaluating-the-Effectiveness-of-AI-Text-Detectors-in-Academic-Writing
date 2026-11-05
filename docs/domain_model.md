# Domain Model: Formal Definition of Metrics and Notation

## 1. Mathematical notation

| Symbol | Meaning | Domain |
|:---:|---|:---:|
| $N$ | Number of documents in the test split | $\mathbb{N}$ |
| $y_i \in \{0, 1\}$ | Ground truth: 1 if the text contains AI-generated content, 0 if it is pure human text | $\{0, 1\}$ |
| $\hat{y}_i \in \{0, 1\}$ | Detector prediction at the fixed threshold $t$ | $\{0, 1\}$ |
| $p_i \in [0, 1]$ | AI-content proportion in a hybrid document | $\mathbb{R}$ |
| $L_i$ | AI-segment length in tokens | $\mathbb{N}$ |

## 2. Target metric: F1 score

The primary metric is the $F_1$ score, computed on the test split:

$$
F_1 = \frac{2 \cdot \mathrm{TP}}
{2 \cdot \mathrm{TP} + \mathrm{FP} + \mathrm{FN}}
$$

where:

- $\mathrm{TP} = \sum_{i=1}^{N} \mathbb{1}[\hat{y}_i = 1 \wedge y_i = 1]$
- $\mathrm{FP} = \sum_{i=1}^{N} \mathbb{1}[\hat{y}_i = 1 \wedge y_i = 0]$
- $\mathrm{FN} = \sum_{i=1}^{N} \mathbb{1}[\hat{y}_i = 0 \wedge y_i = 1]$

## 3. Guardrail metric: false positive rate

The system must minimize false accusations of human text. The false positive
rate (FPR) must satisfy the guardrail:

$$
\mathrm{FPR} = \frac{\mathrm{FP}}{\mathrm{FP} + \mathrm{TN}} \le 0.05
$$

## 4. Statistical confidence intervals

To quantify uncertainty, we compute a 95% percentile bootstrap confidence
interval for each metric $\theta$:

$$
\mathrm{CI}_{95\%}(\theta) =
\left[\theta^*_{(0.025)}, \theta^*_{(0.975)}\right]
$$

where $\theta^*_{(q)}$ is the $q$-quantile of $B = 2000$ bootstrap
replicates.
