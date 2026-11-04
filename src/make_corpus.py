"""Deterministic SYNTHETIC benchmark corpus (pure-human / pure-AI / hybrid).

No real student or personal text is used (ethical note in docs/methodology_card.md).
Sentences are produced from templates: "human-like" ones are concrete, bursty and
first-person; "AI-like" ones are uniform, abstract and (60 %) contain stock phrases.
Deliberate overlap is built in (8 % of human sentences contain a stock phrase, 40 %
of AI sentences do not) so that toy detectors are not trivially perfect.

IMPORTANT: results on this corpus describe the *pipeline and the toy detectors*,
NOT real LLM output. Replace data/raw/corpus.csv with a real corpus to get findings.

Series (all documents ~ DOC_WORDS words):
  pure_human   : 0 % AI                               (negative class)
  pure_ai      : 100 % AI                             (positive)
  proportion   : sentence-interleaved hybrid, 25/50/75 % AI words     (RQ2, B2)
  segment      : ONE contiguous AI block of 25/50/100/200/400 words   (RQ3, B4)

Usage: python src/make_corpus.py --out data/raw/corpus.csv
"""
import argparse
import random

import numpy as np
import pandas as pd

SEED = 42
DOC_WORDS = 450
N_HUMAN, N_AI, N_PER_CELL = 200, 30, 30  # 200 negatives vs 270 positives (balanced F1)
PROPORTIONS = (0.25, 0.50, 0.75)
SEGMENT_LENGTHS = (25, 50, 100, 200, 400)
VAL_SHARE = 0.30  # validation split used once to tune/freeze the threshold

TOPICS = {
    "soil erosion": dict(thing="sediment loss", unit="plots", place="river terrace", who="field team",
                         method="rainfall simulator", factor="slope and vegetation cover"),
    "neural networks": dict(thing="validation accuracy", unit="runs", place="GPU cluster", who="our group",
                            method="grid search", factor="learning rate and batch size"),
    "student writing": dict(thing="draft quality", unit="students", place="first-year seminar", who="the instructors",
                            method="structured interviews", factor="deadline pressure and feedback timing"),
    "urban transport": dict(thing="commute time", unit="routes", place="city centre", who="the survey crew",
                            method="GPS tracking", factor="bus frequency and road congestion"),
    "coral reefs": dict(thing="bleaching rate", unit="transects", place="outer reef", who="the divers",
                        method="photo quadrats", factor="water temperature and turbidity"),
    "clinical trials": dict(thing="dropout rate", unit="patients", place="regional hospital", who="the nurses",
                            method="weekly follow-up calls", factor="dosage and visit schedule"),
    "inflation": dict(thing="price growth", unit="regions", place="national statistics office", who="the analysts",
                      method="panel regression", factor="wages and energy prices"),
    "renewable energy": dict(thing="capacity factor", unit="turbines", place="coastal wind farm", who="the engineers",
                             method="SCADA logging", factor="wind speed and grid curtailment"),
    "language acquisition": dict(thing="vocabulary growth", unit="children", place="kindergarten", who="the researchers",
                                 method="parent diaries", factor="home reading habits and age"),
    "protein folding": dict(thing="folding stability", unit="mutants", place="wet lab", who="the postdocs",
                            method="circular dichroism", factor="temperature and pH"),
}
MONTHS = ["March", "April", "May", "June", "September", "October"]

HUMAN = [
    "We measured {thing} on {n} {unit} at the {place} in {month}.",
    "{Who} expected a clean trend, but {k} of the {unit} behaved oddly.",
    "Honestly, the first week of {method} was a mess.",
    "One {unit1} ran twice as long as planned because the equipment froze.",
    "After we changed {factor0}, {thing} dropped by about {pct} percent, which surprised us.",
    "Nobody had logged the {place} data before 2019, so we digitised old paper notes by hand.",
    "{Who} argued for two hours about whether {n} {unit} were enough; we kept {n2}.",
    "It rained on day {k}, and the whole schedule slipped.",
    "The spread in {thing} was large, roughly {pct} percent between the best and worst {unit}, and we could not explain it with {factor0} alone.",
    "We reran everything overnight, then checked the {unit} again in {month} to be safe.",
    "A reviewer later pointed out that {method} can drift, so we added a recalibration step every {k} days.",
    "Small effects showed up only when {who} pooled the {place} records with the older ones.",
    "Why did {thing} rise in {month}? We still do not know.",
    "Measurement uncertainty remained substantial, and the documentation of the original protocol was incomplete.",
    "Most {unit} looked fine; {k} needed a second pass.",
]
HUMAN_FORMAL = [  # human but formal-sounding -> overlap with AI-like style
    "The calibration procedure and the documentation of the observation conditions are described in the appendix.",
    "Variation in {thing} across the {unit} indicates considerable heterogeneity in the underlying population.",
    "Further investigation of {factor} was beyond the scope of this preliminary study.",
]
HUMAN_STOCK = [  # 8 % of human sentences: contain a stock phrase
    "Moreover, {who} noticed that {thing} depended on {factor0}.",
    "It is important to note that we only had {n} {unit}.",
]

AI_STOCK = [
    "{Topic} represents a significant challenge that affects {thing} across many contexts worldwide.",
    "Furthermore, {factor} plays a crucial role in shaping {thing} and its long-term dynamics.",
    "Moreover, effective analysis of {thing} requires a comprehensive and multifaceted approach.",
    "It is important to note that {method} provides a robust framework for understanding {thing}.",
    "In conclusion, it is essential to consider {factor} when evaluating {thing}.",
    "Delving into {topic} reveals a transformative opportunity to improve our understanding of {thing}.",
]
AI_SMOOTH = [  # AI-like (uniform, abstract) but WITHOUT stock phrases
    "The implementation of {method} facilitates the systematic evaluation of {thing} across different conditions.",
    "Consideration of {factor} contributes to a more accurate interpretation of the observed variation.",
    "The relationship between {factor} and {thing} demonstrates consistent patterns within the available evidence.",
    "Continued attention to measurement quality supports the reliability and reproducibility of the findings.",
    "An integrated perspective on {topic} enables stakeholders to identify opportunities for improvement.",
]


def _fill(tpl, topic, rng):
    d = TOPICS[topic]
    unit = d["unit"]
    vals = dict(
        topic=topic, Topic=topic.capitalize(), thing=d["thing"], unit=unit,
        unit1=unit[:-1] if unit.endswith("s") else unit, place=d["place"], who=d["who"],
        Who=d["who"].capitalize(), method=d["method"], factor=d["factor"],
        factor0=d["factor"].split(" and ")[0], month=rng.choice(MONTHS),
        n=rng.choice([12, 18, 24, 40, 57, 83]), n2=rng.choice([9, 14, 21, 30]),
        k=rng.choice([2, 3, 4, 5, 7]), pct=rng.choice([4, 7, 12, 18, 26]),
    )
    return tpl.format(**vals)


def sentence(kind, topic, rng):
    if kind == "human":
        r = rng.random()
        pool = HUMAN_STOCK if r < 0.08 else HUMAN_FORMAL if r < 0.20 else HUMAN
    else:
        pool = AI_STOCK if rng.random() < 0.60 else AI_SMOOTH
    return _fill(rng.choice(pool), topic, rng)


def _wc(s):
    return len(s.split())


def build_proportion_doc(p, topic, rng):
    """Sentence-interleaved hybrid whose AI word share is close to p."""
    sents, kinds, words, ai_words = [], [], 0, 0
    while words < DOC_WORDS:
        want_ai = (ai_words / words < p) if words else rng.random() < p
        if rng.random() < 0.25:  # jitter -> irregular interleaving
            want_ai = not want_ai
        kind = "ai" if want_ai else "human"
        s = sentence(kind, topic, rng)
        sents.append(s); kinds.append(kind)
        words += _wc(s); ai_words += _wc(s) if kind == "ai" else 0
    return sents, kinds


def build_segment_doc(length, topic, rng):
    """One contiguous AI block of ~length words inside human text, random position."""
    block, bw = [], 0
    while bw < length:
        s = sentence("ai", topic, rng)
        block.append(s); bw += _wc(s)
    sents, kinds = [], []
    # human words to place before / after the block
    rest = max(DOC_WORDS - bw, 0)
    before_target = rng.random() * rest
    while sum(_wc(s) for s in sents) < before_target:
        sents.append(sentence("human", topic, rng)); kinds.append("human")
    sents += block; kinds += ["ai"] * len(block)
    while sum(_wc(s) for s in sents) < DOC_WORDS:
        sents.append(sentence("human", topic, rng)); kinds.append("human")
    return sents, kinds


def longest_ai_run(sents, kinds):
    best = cur = 0
    for s, k in zip(sents, kinds):
        cur = cur + _wc(s) if k == "ai" else 0
        best = max(best, cur)
    return best


def make_corpus(seed=SEED):
    rng = random.Random(seed)
    topics = list(TOPICS)
    rows = []

    def add(series, category, target_p, target_len, sents, kinds):
        total = sum(_wc(s) for s in sents)
        ai = sum(_wc(s) for s, k in zip(sents, kinds) if k == "ai")
        rows.append(dict(
            text=" ".join(sents), label=int(ai > 0), category=category,
            ai_fraction=round(ai / total, 3), ai_segment_tokens=longest_ai_run(sents, kinds),
            series=series, ai_fraction_target=target_p, segment_target=target_len, n_words=total))

    for _ in range(N_HUMAN):
        t = rng.choice(topics)
        sents, kinds = [], []
        while sum(_wc(x) for x in sents) < DOC_WORDS:
            sents.append(sentence("human", t, rng)); kinds.append("human")
        add("pure_human", "pure-human", 0.0, 0, sents, kinds)
    for _ in range(N_AI):
        t = rng.choice(topics)
        sents, kinds = [], []
        while sum(_wc(x) for x in sents) < DOC_WORDS:
            sents.append(sentence("ai", t, rng)); kinds.append("ai")
        add("pure_ai", "pure-ai", 1.0, DOC_WORDS, sents, kinds)
    for p in PROPORTIONS:
        for _ in range(N_PER_CELL):
            t = rng.choice(topics)
            sents, kinds = build_proportion_doc(p, t, rng)
            add("proportion", "hybrid", p, 0, sents, kinds)
    for L in SEGMENT_LENGTHS:
        for _ in range(N_PER_CELL):
            t = rng.choice(topics)
            sents, kinds = build_segment_doc(L, t, rng)
            add("segment", "hybrid", np.nan, L, sents, kinds)

    df = pd.DataFrame(rows)
    # stratified validation/test split (validation used once to freeze thresholds)
    nrng = np.random.default_rng(seed)
    df["split"] = "test"
    for _, idx in df.groupby(["series", "ai_fraction_target", "segment_target"], dropna=False).groups.items():
        idx = list(idx)
        val = nrng.choice(idx, size=max(1, round(VAL_SHARE * len(idx))), replace=False)
        df.loc[val, "split"] = "val"
    df.insert(0, "id", np.arange(1, len(df) + 1))
    return df[["id", "text", "label", "category", "ai_fraction", "ai_segment_tokens", "split",
               "series", "ai_fraction_target", "segment_target", "n_words"]]


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/raw/corpus.csv")
    ap.add_argument("--seed", type=int, default=SEED)
    a = ap.parse_args()
    df = make_corpus(a.seed)
    df.to_csv(a.out, index=False)
    print(f"{len(df)} documents -> {a.out}")
    print(df.groupby(["series", "category"]).size())