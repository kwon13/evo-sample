CONCEPT_TYPE_TO_GROUP: dict[str, str] = {
    "number_theory.gcd_lcm_sync": "number_theory",
    "number_theory.crt_count": "number_theory",
    "combinatorics.committee_count": "combinatorics",
    "sequence.geometric_to_arithmetic_sum": "sequence",
    "algebra.quadratic_vieta_reciprocal": "algebra",
    "geometry.trig_area": "geometry",
    "geometry.line_intersection_distance": "geometry",
    "sequence.linear_recurrence": "sequence",
    "linear_algebra.linear_system_sum": "algebra",
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
    "inequality",
)

CONCEPT_TYPES: tuple[str, ...] = tuple(CONCEPT_TYPE_TO_GROUP)


def concept_group_for_type(concept_type: str | None) -> str | None:
    if not concept_type:
        return None
    return CONCEPT_TYPE_TO_GROUP.get(str(concept_type).strip())


def validate_concept_decl(concept_type, concept_group):
    reasons = []
    if not concept_type:
        reasons.append("missing CONCEPT_TYPE")
        return reasons
    if not concept_group:
        reasons.append("missing CONCEPT_GROUP")
        return reasons
    if concept_group not in CONCEPT_GROUPS:
        reasons.append(f"unknown CONCEPT_GROUP: {concept_group}")
        return reasons
    # CONCEPT_TYPE is free-form. The prior prefix check rejected legitimate
    # cross-prefix labels (e.g. linear_algebra.* under group=algebra), so it
    # has been removed. CONCEPT_GROUP membership in CONCEPT_GROUPS is the
    # only structural invariant now.
    return reasons


def concept_axis_labels(diversity_axis: str) -> list[str]:
    if diversity_axis == "concept_group":
        return list(CONCEPT_GROUPS)
    if diversity_axis == "concept_type":
        return list(CONCEPT_TYPES)
    return []


def concept_prompt_block() -> str:
    """Group-only enumeration block: states which CONCEPT_GROUP values
    are valid and the CONCEPT_TYPE string format. How to *choose* a
    label is left to the mutation system prompt — this block only
    enumerates. CONCEPT_TYPE is intentionally free-form; only
    CONCEPT_GROUP membership is structurally enforced, and only
    CONCEPT_GROUP affects the MAP-Elites D-axis."""
    groups = ", ".join(CONCEPT_GROUPS)
    return (
        f"CONCEPT_GROUP must be exactly one of: {groups}\n"
        "CONCEPT_TYPE is a free-form '<group>.<snake_case_name>' string.\n"
    )


def nearest_concept_type(
    candidate: str | None, threshold: float = 0.6,
) -> str | None:
    """Best whitelisted CONCEPT_TYPE near ``candidate``, else None.

    Used to rescue model outputs that invent a sensible-looking but
    unwhitelisted label (``geometry.angle_trisector`` ->
    ``geometry.angle_bisector``-style match). Splits on '.' so the
    prefix-group must agree before a fuzzy suffix match is attempted.
    """
    import difflib
    if not candidate:
        return None
    text = str(candidate).strip()
    if "." not in text:
        return None
    group, _, suffix = text.partition(".")
    if group not in CONCEPT_GROUPS:
        return None
    in_group = [t for t in CONCEPT_TYPES if t.startswith(group + ".")]
    suffixes = [t.split(".", 1)[1] for t in in_group]
    matches = difflib.get_close_matches(suffix, suffixes, n=1, cutoff=threshold)
    if not matches:
        return None
    return f"{group}.{matches[0]}"
