def validate_grounding(decision: dict, evidence_pack: dict) -> dict:
    available_ids = {
        str(item.get("evidence_id"))
        for item in evidence_pack.get("items", [])
        if item.get("evidence_id") is not None
    }

    used_ids = decision.get("used_evidence_ids", [])
    if not isinstance(used_ids, list):
        used_ids = []

    used_ids = [str(x) for x in used_ids if x is not None]
    valid_used = [x for x in used_ids if x in available_ids]
    hallucinated = [x for x in used_ids if x not in available_ids]

    if not evidence_pack.get("items"):
        support_level = "none"
    elif not used_ids:
        support_level = "none"
    elif hallucinated:
        support_level = "invalid"
    elif len(valid_used) >= 2:
        support_level = "strong"
    else:
        support_level = "medium"

    return {
        "available_evidence_ids": sorted(list(available_ids)),
        "used_evidence_ids": used_ids,
        "valid_used_evidence_ids": valid_used,
        "hallucinated_evidence_ids": hallucinated,
        "evidence_support_level": support_level,
        "is_grounded": len(valid_used) > 0 and not hallucinated,
    }


def repair_decision_evidence_fields(decision: dict, evidence_pack: dict) -> dict:
    if not isinstance(decision, dict):
        decision = {}

    available_ids = {
        str(item.get("evidence_id"))
        for item in evidence_pack.get("items", [])
        if item.get("evidence_id") is not None
    }

    used = decision.get("used_evidence_ids")
    if not isinstance(used, list):
        used = []

    used = [str(x) for x in used if str(x) in available_ids]

    decision["used_evidence_ids"] = used

    if not evidence_pack.get("items"):
        decision["evidence_support_level"] = "none"
    elif not used:
        decision["evidence_support_level"] = "none"
    elif len(used) >= 2:
        decision["evidence_support_level"] = "strong"
    else:
        decision["evidence_support_level"] = "medium"

    return decision