"""Deterministic reasoning layer for VisualQA India.

The vision-language model supplies observations.  This module converts those
observations into a transparent heuristic score; it is not a calibrated safety
probability and must not be used as the sole basis for real-world decisions.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable, Mapping


@dataclass(frozen=True)
class Diagnostic:
    key: str
    label: str
    question: str
    penalty: int
    risk_terms: tuple[str, ...]
    safe_terms: tuple[str, ...]


DIAGNOSTICS: tuple[Diagnostic, ...] = (
    Diagnostic(
        "road_damage",
        "Road damage",
        "Is there visible road damage such as potholes, cracks, collapse, or erosion? Answer yes or no, then briefly explain.",
        25,
        ("pothole", "potholes", "crack", "cracks", "collapsed", "collapse", "erosion", "broken road", "road damage", "damaged road"),
        ("intact", "smooth", "well maintained", "no damage"),
    ),
    Diagnostic(
        "flooding",
        "Flooding or waterlogging",
        "Is the road or surrounding area flooded, submerged, or seriously waterlogged? Answer yes or no, then briefly explain.",
        30,
        ("flooded", "flooding", "submerged", "waterlogged", "deep water", "overflow"),
        ("dry", "no flooding", "not flooded"),
    ),
    Diagnostic(
        "obstruction",
        "Obstructions",
        "Is the usable path blocked by debris, a landslide, fallen objects, or another major obstruction? Answer yes or no, then briefly explain.",
        15,
        ("blocked", "debris", "landslide", "fallen tree", "obstruction", "obstructions", "barrier"),
        ("clear", "open", "unobstructed", "no obstruction"),
    ),
    Diagnostic(
        "surface_condition",
        "Surface condition",
        "Is the road surface severely uneven, washed out, or deteriorated? Answer yes or no, then briefly explain.",
        15,
        ("severely uneven", "washed out", "deteriorated", "large rut", "unusable surface"),
        ("even", "paved", "good condition", "normal surface"),
    ),
    Diagnostic(
        "access_safety",
        "Vehicle and pedestrian access",
        "Does the visible scene appear unsafe or impassable for vehicles or pedestrians? Answer yes or no, then briefly explain.",
        15,
        ("unsafe", "not safe", "impassable", "dangerous", "hazardous", "cannot pass"),
        ("safe", "passable", "accessible"),
    ),
)

_NEGATION = r"(?:no|not|without|free\s+of|absence\s+of|isn't|aren't|doesn't)"


def _has_term(text: str, term: str) -> bool:
    """Return True when a complete term occurs outside a nearby negation."""
    escaped = re.escape(term).replace(r"\ ", r"\s+")
    term_pattern = rf"(?<!\w){escaped}(?!\w)"
    if not re.search(term_pattern, text):
        return False

    negated = re.compile(
        rf"{_NEGATION}(?:[\s,;:-]+\w+){{0,3}}[\s,;:-]+{escaped}(?!\w)"
    )
    return negated.search(text) is None


def _starts_affirmative(text: str) -> bool:
    return re.match(r"^\s*(?:yes|yeah|there\s+(?:is|are))\b", text) is not None


def _starts_negative(text: str) -> bool:
    return re.match(r"^\s*(?:no|nope|there\s+(?:is|are)\s+no)\b", text) is not None


def classify_answer(answer: str, diagnostic: Diagnostic) -> tuple[bool, str | None]:
    """Classify a harmful-condition question answer as risk/no-risk."""
    text = " ".join(answer.lower().split())
    if not text:
        return False, None

    matching_risks = [term for term in diagnostic.risk_terms if _has_term(text, term)]
    if _starts_negative(text) and not matching_risks:
        return False, "explicit no"
    if matching_risks:
        return True, matching_risks[0]
    if _starts_affirmative(text):
        return True, "explicit yes"

    matching_safe = [term for term in diagnostic.safe_terms if _has_term(text, term)]
    if matching_safe:
        return False, matching_safe[0]
    return False, None


def score_answers(answers: Mapping[str, str] | Iterable[tuple[str, str]]) -> dict:
    """Score one answer per diagnostic, deducting each category at most once."""
    answer_map = dict(answers)
    score = 100
    findings: list[dict] = []
    unknown: list[str] = []

    for diagnostic in DIAGNOSTICS:
        answer = str(answer_map.get(diagnostic.key, "")).strip()
        is_risk, evidence = classify_answer(answer, diagnostic)
        if is_risk:
            score -= diagnostic.penalty
            findings.append(
                {
                    "key": diagnostic.key,
                    "label": diagnostic.label,
                    "penalty": diagnostic.penalty,
                    "evidence": evidence,
                    "answer": answer,
                }
            )
        elif evidence is None:
            unknown.append(diagnostic.label)

    score = max(0, min(100, score))
    if score >= 80:
        severity = "LOW OBSERVED RISK"
        color = "green"
    elif score >= 50:
        severity = "MODERATE OBSERVED RISK"
        color = "orange"
    else:
        severity = "HIGH OBSERVED RISK"
        color = "red"

    return {
        "score": score,
        "severity": severity,
        "color": color,
        "findings": findings,
        "unknown": unknown,
    }
