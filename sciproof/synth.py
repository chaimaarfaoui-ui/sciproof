"""Module 2: a reproducible SYNTHETIC corpus + gold evaluation set.

Why synthetic? (1) no internet in the build sandbox, (2) we know the ground truth
by construction, so tests can assert exact behaviour, (3) no fake medical claims:
every compound here is fictional.

Limit: because gold labels are generated from the same templates as the papers,
this benchmark is EASY. Treat its numbers as a regression test, not as proof of
real-world quality. A human-labelled gold set on real papers is the real test.
"""
from __future__ import annotations

import itertools
import json
import random
from pathlib import Path

COMPOUNDS = ["ZX-101", "QB-22", "MN-7", "RT-450", "VL-9"]
OUTCOMES = {
    # outcome: (unit, baseline range, effect range)
    "resting heart rate": ("beats per minute", (70, 80), (4, 9)),
    "systolic blood pressure": ("mmHg", (128, 140), (5, 12)),
    "sleep latency": ("minutes", (30, 45), (6, 15)),
    "fasting glucose": ("mg/dL", (105, 120), (6, 18)),
}
TEMPLATES = [
    "What was the effect of {c} on {o}?",
    "Does {c} reduce {o}?",
    "How much did {o} change with {c}?",
]
SECTIONS = ["Abstract", "Results"]


def _paper(rng, pid, c, o, design, supportive):
    unit, base_r, eff_r = OUTCOMES[o]
    base = rng.randint(*base_r)
    d = rng.randint(*eff_r)
    O = o[0].upper() + o[1:]
    title = f"SYNTHETIC: Effect of {c} on {o} ({design})"
    if design == "RCT":
        n, w = rng.choice([120, 200, 240, 400]), rng.choice([8, 12, 16])
        a1, a2 = rng.choice([(40, 65), (30, 60), (45, 70)])
        p = rng.choice(["0.001", "0.004", "0.01", "0.02", "0.03"])
        verb = rng.choice(["fell", "dropped", "decreased"])
        ae, r1, r2 = rng.choice(["nausea", "headache", "dizziness"]), rng.randint(6, 12), rng.randint(2, 5)
        secs = [
            ("Abstract", f"This synthetic trial tests a fictional compound, {c}. We randomized {n} fictional adults to {c} or placebo. {O} {verb} by {d} {unit} in the treatment arm (p = {p}). No serious adverse events were recorded."),
            ("Methods", f"Participants aged {a1} to {a2} with no prior cardiac disease were randomized 1:1 to {c} or placebo, as described in Fig. 1. {O} was measured at baseline and at week {w} by trained staff. Analysis followed the intention-to-treat principle, e.g. all randomized participants were included."),
            ("Results", f"Mean baseline {o} was {base} {unit} in both arms. At week {w}, {o} {verb} by {d} {unit} in the treatment arm relative to placebo (p = {p}). Mild {ae} occurred in {r1}% of treated participants versus {r2}% on placebo."),
            ("Discussion", f"These synthetic findings suggest {c} may modify {o} in this population. Longer follow-up is needed. Limitations include the single-region sample."),
            ("References", "1. Fictional A et al. Not a real paper. 2. Fictional B et al. Also not real."),
        ]
        year = rng.choice([2020, 2021, 2022])
    else:
        n = rng.choice([800, 1000, 1500])
        if supportive:
            p = "0.02"
            ab = f"In a fictional cohort of {n} adults, {c} users had lower {o} than non-users (p = {p})."
            res = f"{O} was {d} {unit} lower in users than non-users (p = {p}). Estimates were similar across sexes. Adherence was self-reported, which limits interpretation."
        else:
            p = rng.choice(["0.21", "0.34", "0.40", "0.55"])
            ab = f"In a fictional cohort of {n} adults, {c} users showed no meaningful change in {o}. Observational confounding cannot be excluded."
            res = f"{O} differed by {rng.randint(0, 2)} {unit} between users and non-users (p = {p}). Estimates were similar across sexes. Adherence was self-reported, which limits interpretation."
        secs = [
            ("Abstract", ab),
            ("Results", res),
            ("Discussion", "Residual confounding is possible. Differences in age range relative to trials may explain discordant findings."),
        ]
        year = rng.choice([2022, 2023, 2024])
    return {
        "paper_id": pid, "title": title, "year": year, "study_type": design,
        "sections": [{"name": a, "text": b} for a, b in secs],
    }


def build(seed: int = 7):
    rng = random.Random(seed)
    pairs = list(itertools.product(COMPOUNDS, OUTCOMES))
    rng.shuffle(pairs)
    present, missing = pairs[:14], pairs[14:]
    cohort_pairs = set(present[:5])
    supportive_cohort = present[0]  # one cohort agrees with its RCT, four disagree

    papers, gold, k = [], [], 0
    for i, (c, o) in enumerate(present):
        k += 1
        rid = f"SYN-{k:03d}"
        papers.append(_paper(rng, rid, c, o, "RCT", True))
        rel = [rid]
        if (c, o) in cohort_pairs:
            k += 1
            cid = f"SYN-{k:03d}"
            papers.append(_paper(rng, cid, c, o, "cohort", (c, o) == supportive_cohort))
            rel.append(cid)
        gold.append({"id": f"q{len(gold)+1:02d}", "question": TEMPLATES[i % 3].format(c=c, o=o),
                     "answerable": True, "relevant_papers": list(rel), "relevant_sections": SECTIONS})
        if len(rel) == 2:
            gold.append({"id": f"q{len(gold)+1:02d}",
                         "question": f"Is the evidence that {c} lowers {o} consistent across studies?",
                         "answerable": True, "relevant_papers": rel, "relevant_sections": SECTIONS,
                         "expected_verdict": "supported" if (c, o) == supportive_cohort else "conflicting"})
    for c, o in missing:
        gold.append({"id": f"q{len(gold)+1:02d}", "question": f"What was the effect of {c} on {o}?",
                     "answerable": False, "relevant_papers": [], "relevant_sections": []})
    for q in ["What was the effect of ZX-999 on resting heart rate?",
              "How does photosynthesis work in algae?",
              "Does QB-22 cure insomnia?"]:
        gold.append({"id": f"q{len(gold)+1:02d}", "question": q, "answerable": False,
                     "relevant_papers": [], "relevant_sections": []})
    return papers, gold


def write(root: str | Path = "."):
    root = Path(root)
    papers, gold = build()
    (root / "data/synthetic").mkdir(parents=True, exist_ok=True)
    (root / "data/eval").mkdir(parents=True, exist_ok=True)
    (root / "data/synthetic/papers.json").write_text(json.dumps(papers, indent=1))
    (root / "data/eval/gold.json").write_text(json.dumps(gold, indent=1))
    return len(papers), len(gold)


if __name__ == "__main__":
    print(write())
