"""Detector implementations. Every detector exposes `.name` and `.score(texts) -> list[float]`
where the score is P(text contains AI-generated content) in [0, 1].

Core detectors have no heavy dependencies so the pipeline runs offline on CPU.
Neural detectors are imported lazily (see requirements-hf.txt).
"""
import re
import numpy as np

STOCK_PHRASES = [
    "furthermore", "moreover", "in conclusion", "it is important to note",
    "plays a crucial role", "delve", "comprehensive", "multifaceted",
    "transformative", "robust framework", "it is essential", "represents a significant",
]


class RandomBaseline:
    """Trivial baseline: random scores. Any useful detector must beat this."""
    name = "random_baseline"

    def __init__(self, seed=42):
        self.rng = np.random.default_rng(seed)

    def score(self, texts):
        return [float(v) for v in self.rng.random(len(texts))]


class StylometricHeuristic:
    """Transparent toy detector: stock-phrase density + low sentence-length burstiness.
    NOT a state-of-the-art detector; a CPU-only smoke-test baseline for the pipeline."""
    name = "stylometric_heuristic"

    @staticmethod
    def _one(text):
        t = text.lower()
        sents = [s for s in re.split(r"(?<=[.!?])\s+", text.strip()) if s]
        words = max(len(t.split()), 1)
        hits = sum(t.count(p) for p in STOCK_PHRASES)
        phrase_score = min(hits / max(len(sents), 1) * 1.5, 1.0)
        lens = np.array([len(s.split()) for s in sents], dtype=float)
        burst = (lens.std() / lens.mean()) if len(lens) > 1 and lens.mean() > 0 else 0.5
        low_burst = float(np.clip(1.0 - burst / 0.6, 0.0, 1.0))
        return float(np.clip(0.8 * phrase_score + 0.2 * low_burst, 0.0, 1.0))

    def score(self, texts):
        return [self._one(t) for t in texts]


class LexicalFormality:
    """Second transparent CPU-only detector, independent of stock phrases: density of long,
    abstract (-tion/-ity/-ment/-ance/-ness) words and mean word length. Formal, nominalised
    prose scores higher. A smoke-test signal, NOT a state-of-the-art detector."""
    name = "lexical_formality"
    _SUFFIX = re.compile(r"(tion|tions|ity|ities|ment|ments|ance|ence|ness)$")

    @classmethod
    def _one(cls, text):
        words = re.findall(r"[a-zA-Z]+", text.lower())
        if not words:
            return 0.0
        abstract = sum(1 for w in words if len(w) > 6 and cls._SUFFIX.search(w)) / len(words)
        mean_len = float(np.mean([len(w) for w in words]))
        return float(np.clip(0.6 * min(abstract / 0.12, 1.0)
                             + 0.4 * np.clip((mean_len - 4.6) / 1.6, 0.0, 1.0), 0.0, 1.0))

    def score(self, texts):
        return [self._one(t) for t in texts]


class RobertaOpenAIDetector:
    """Optional neural classifier (needs requirements-hf.txt and internet on first run)."""
    name = "roberta_openai"

    def __init__(self):
        from transformers import pipeline  # lazy import
        self.pipe = pipeline("text-classification",
                             model="openai-community/roberta-base-openai-detector",
                             truncation=True, max_length=512)

    def score(self, texts):
        out = self.pipe(list(texts))
        return [o["score"] if o["label"].lower() == "fake" else 1 - o["score"] for o in out]


class EnsembleMean:
    """Mean of member scores (H2b: ensemble vs best single detector)."""
    name = "ensemble_mean"

    def __init__(self, members):
        self.members = members

    def score(self, texts):
        return [float(v) for v in np.mean([m.score(texts) for m in self.members], axis=0)]


def build(names, seed=42):
    """Instantiate detectors by name. The ensemble averages all non-baseline detectors."""
    registry = {
        "random_baseline": lambda: RandomBaseline(seed),
        "stylometric_heuristic": StylometricHeuristic,
        "lexical_formality": LexicalFormality,
        "roberta_openai": RobertaOpenAIDetector,
    }
    dets = {}
    for n in names:
        if n == "ensemble_mean":
            continue
        if n not in registry:
            raise ValueError(f"Unknown detector '{n}'. Available: {sorted(registry) + ['ensemble_mean']}")
        dets[n] = registry[n]()
    if "ensemble_mean" in names:
        members = [d for k, d in dets.items() if k != "random_baseline"]
        if not members:
            raise ValueError("ensemble_mean needs at least one non-baseline detector in the config")
        dets["ensemble_mean"] = EnsembleMean(members)
    return dets
