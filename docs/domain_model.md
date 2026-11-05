\# Domain Model: Formal Definition of Metrics and Notation



\## 1. Mathematical Notation

| Symbol | Meaning | Domain |

|---|---|---|

| $N$ | Number of documents in the test split | $\\mathbb{N}$ |

| $y\_i \\in \\{0,1\\}$ | Ground truth: 1 if text contains AI, 0 if pure human | $\\{0,1\\}$ |

| $\\hat{y}\_i \\in \\{0,1\\}$ | Detector prediction at fixed threshold $t$ | $\\{0,1\\}$ |

| $p\_i \\in \[0,1]$ | AI-content proportion in hybrid document | $\\mathbb{R}$ |

| $L\_i$ | AI-segment length (tokens) | $\\mathbb{N}$ |



\## 2. Target Metric: F1 Score

The primary metric is the $F\_1$ score, computed on the test split:

$$

F\_1 = \\frac{2 \\cdot \\mathrm{TP}}{2 \\cdot \\mathrm{TP} + \\mathrm{FP} + \\mathrm{FN}}

$$

Where:

\* $\\mathrm{TP} = \\sum\_{i=1}^{N} \\mathbb{1}\[\\hat{y}\_i = 1 \\wedge y\_i = 1]$

\* $\\mathrm{FP} = \\sum\_{i=1}^{N} \\mathbb{1}\[\\hat{y}\_i = 1 \\wedge y\_i = 0]$

\* $\\mathrm{FN} = \\sum\_{i=1}^{N} \\mathbb{1}\[\\hat{y}\_i = 0 \\wedge y\_i = 1]$



\## 3. Guardrail Metric: False Positive Rate (FPR)

The system must minimize false accusations of human text (FPR $\\le 5\\%$):

$$

\\mathrm{FPR} = \\frac{\\mathrm{FP}}{\\mathrm{FP} + \\mathrm{TN}} \\le 0.05

$$



\## 4. Statistical Confidence Intervals

To quantify uncertainty, we compute a 95% percentile bootstrap confidence interval for each metric $\\theta$:

$$

\\mathrm{CI}\_{95\\%}(\\theta) = \\left\[\\theta^\*\_{(0.025)}, \\theta^\*\_{(0.975)}\\right]

$$

where $\\theta^\*\_{(q)}$ is the $q$-quantile of $B=2000$ bootstrap replicates.

