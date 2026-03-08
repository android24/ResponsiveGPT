def derive_round_risk_label_from_summary_row(
    row: dict,
    ttc_threshold: float = 3.0,
    distance_threshold: float = 2.0,
) -> bool:
    try:
        min_ttc = float(row["min_ttc"]) if row.get("min_ttc") else None
    except Exception:
        min_ttc = None

    try:
        min_distance = float(row["min_distance"]) if row.get("min_distance") else None
    except Exception:
        min_distance = None

    if min_ttc is not None and min_ttc < ttc_threshold:
        return True
    if min_distance is not None and min_distance < distance_threshold:
        return True
    return False


def derive_round_risk_label_from_clip_meta(
    meta: dict,
    ttc_threshold: float = 3.0,
    distance_threshold: float = 2.0,
) -> bool:
    min_ttc = meta.get("event_min_ttc")
    min_distance = meta.get("event_min_distance")

    if min_ttc is not None and min_ttc < ttc_threshold:
        return True
    if min_distance is not None and min_distance < distance_threshold:
        return True
    return False