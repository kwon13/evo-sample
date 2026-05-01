"""Controlled mathematical concept taxonomy for MAP-Elites diversity."""

from __future__ import annotations

import re


CONCEPT_TYPE_TO_GROUP: dict[str, str] = {
    "number_theory.gcd_lcm_sync": "number_theory",
    "number_theory.crt_count": "number_theory",
    "combinatorics.committee_count": "combinatorics",
    "sequence.geometric_to_arithmetic_sum": "sequence",
    "algebra.quadratic_vieta_reciprocal": "algebra",
    "geometry.trig_area": "geometry",
    "geometry.line_intersection_distance": "geometry",
    "sequence.linear_recurrence": "sequence",
    "linear_algebra.linear_system_sum": "linear_algebra",
    "inequality.am_gm_product": "inequality",
    "number_theory.kth_root_mod_prime": "number_theory",
    "combinatorics.derangement_fixed_points": "combinatorics",
    "geometry.ptolemy_cyclic_quadrilateral": "geometry",
    "geometry.power_of_point_secants": "geometry",
    "number_theory.legendre_symbol": "number_theory",
}

CONCEPT_GROUPS: tuple[str, ...] = (
    "number_theory",
    "combinatorics",
    "sequence",
    "algebra",
    "geometry",
    "linear_algebra",
    "inequality",
)

CONCEPT_TYPES: tuple[str, ...] = tuple(CONCEPT_TYPE_TO_GROUP)


def concept_group_for_type(concept_type: str | None) -> str | None:
    if not concept_type:
        return None
    return CONCEPT_TYPE_TO_GROUP.get(str(concept_type).strip())


def validate_concept_decl(
    concept_type: str | None,
    concept_group: str | None,
) -> list[str]:
    reasons: list[str] = []
    if not concept_type:
        reasons.append("missing CONCEPT_TYPE")
        return reasons

    expected_group = concept_group_for_type(concept_type)
    if expected_group is None:
        reasons.append(f"unknown CONCEPT_TYPE: {concept_type}")
        return reasons

    if not concept_group:
        reasons.append("missing CONCEPT_GROUP")
    elif concept_group not in CONCEPT_GROUPS:
        reasons.append(f"unknown CONCEPT_GROUP: {concept_group}")
    elif concept_group != expected_group:
        reasons.append(
            f"CONCEPT_GROUP mismatch: expected {expected_group}, got {concept_group}"
        )
    return reasons


def _contains_all(text: str, needles: tuple[str, ...]) -> bool:
    return all(needle in text for needle in needles)


def _contains_any(text: str, needles: tuple[str, ...]) -> bool:
    return any(needle in text for needle in needles)


def _has_modular_power(text: str) -> bool:
    return ("x^" in text or "x**" in text) and ("mod" in text or "modulo" in text)


def validate_concept_contract(
    concept_type: str | None,
    problem: str,
    answer: str,
) -> list[str]:
    """Lightweight deterministic guard against concept-label mismatch.

    These checks intentionally remain shallow. Their job is to reject obvious
    label gaming, not to prove that the problem belongs to the concept.
    """
    if not concept_type:
        return ["missing CONCEPT_TYPE"]

    text = str(problem or "").lower()
    ans = str(answer or "").strip()
    reasons: list[str] = []

    if concept_type == "number_theory.gcd_lcm_sync":
        if not _contains_all(text, ("depart", "every", "together")):
            reasons.append("expected synchronized periodic departure wording")
    elif concept_type == "number_theory.crt_count":
        if not _contains_all(text, ("remainder", "divided by")):
            reasons.append("expected CRT remainder conditions")
        if not _contains_any(text, ("range", "inclusive", "<=", "between")):
            reasons.append("expected interval/counting bound")
    elif concept_type == "combinatorics.committee_count":
        if not _contains_all(text, ("committee", "men", "women")):
            reasons.append("expected mixed committee-count wording")
    elif concept_type == "sequence.geometric_to_arithmetic_sum":
        if not _contains_all(text, ("geometric sequence", "arithmetic sequence", "sum")):
            reasons.append("expected geometric-to-arithmetic sequence structure")
    elif concept_type == "algebra.quadratic_vieta_reciprocal":
        if not _contains_all(text, ("quadratic equation", "roots", "reciprocal")):
            reasons.append("expected quadratic roots reciprocal/Vieta structure")
    elif concept_type == "geometry.trig_area":
        if not _contains_all(text, ("triangle", "included angle", "area")):
            reasons.append("expected triangle area with included angle")
    elif concept_type == "geometry.line_intersection_distance":
        if not _contains_all(text, ("lines", "intersection", "origin")):
            reasons.append("expected line-intersection distance structure")
    elif concept_type == "sequence.linear_recurrence":
        if not _contains_all(text, ("sequence", "a(n)", "a(n-1)")):
            reasons.append("expected first-order linear recurrence")
    elif concept_type == "linear_algebra.linear_system_sum":
        if not _contains_all(text, ("solve the system", "x + y + z")):
            reasons.append("expected 3-variable linear system asking x+y+z")
    elif concept_type == "inequality.am_gm_product":
        if not _contains_all(text, ("positive real", "maximum", "product")):
            reasons.append("expected AM-GM product maximization structure")
    elif concept_type == "number_theory.kth_root_mod_prime":
        if not _contains_all(text, ("primitive root", "residue")):
            reasons.append("expected primitive-root residue setup")
        if not _has_modular_power(text):
            reasons.append("expected modular kth-root equation")
    elif concept_type == "combinatorics.derangement_fixed_points":
        if not _contains_all(text, ("redistributed", "exactly", "receive their own")):
            reasons.append("expected derangement with exactly fixed points")
    elif concept_type == "geometry.ptolemy_cyclic_quadrilateral":
        if not _contains_all(text, ("cyclic quadrilateral", "ptolemy", "diagonal")):
            reasons.append("expected Ptolemy cyclic quadrilateral structure")
    elif concept_type == "geometry.power_of_point_secants":
        if not _contains_all(text, ("external point", "secants", "circle")):
            reasons.append("expected power-of-point secant structure")
    elif concept_type == "number_theory.legendre_symbol":
        if not _contains_all(text, ("legendre symbol", "prime")):
            reasons.append("expected Legendre symbol over prime modulus")
        if ans not in {"-1", "0", "1"}:
            reasons.append("Legendre symbol answer must be -1, 0, or 1")
    else:
        reasons.append(f"no contract registered for {concept_type}")

    return reasons


def concept_axis_labels(diversity_axis: str) -> list[str]:
    if diversity_axis == "concept_group":
        return list(CONCEPT_GROUPS)
    if diversity_axis == "concept_type":
        return list(CONCEPT_TYPES)
    return []


def concept_prompt_block() -> str:
    type_lines = "\n".join(f"#   - {name}" for name in CONCEPT_TYPES)
    group_lines = "\n".join(f"#   - {name}" for name in CONCEPT_GROUPS)
    return (
        "# === CONTROLLED CONCEPT TAXONOMY ===\n"
        "# Every generator MUST define these top-level string constants before\n"
        "# `generate(seed)`:\n"
        "#   CONCEPT_TYPE = '<one allowed fine template label>'\n"
        "#   CONCEPT_GROUP = '<matching prefix/domain group>'\n"
        "# Allowed CONCEPT_TYPE values:\n"
        f"{type_lines}\n"
        "# Allowed CONCEPT_GROUP values:\n"
        f"{group_lines}\n"
        "# CONCEPT_GROUP must exactly match the prefix/domain group of\n"
        "# CONCEPT_TYPE. Do not invent new labels or synonyms.\n"
    )

