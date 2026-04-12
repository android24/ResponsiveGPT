def derive_ind_risk_label(
    row: dict,
    ttc_threshold: float = 3.0,
    distance_threshold: float = 2.5,
    drac_threshold: float = 8.0,
) -> bool:
    """
    用 inD summary 指标构造弱标签。
    """
    try:
        min_ttc = float(row["min_ttc"]) if row.get("min_ttc") else None
    except Exception:
        min_ttc = None

    try:
        min_dist = float(row["min_center_distance"]) if row.get("min_center_distance") else None
    except Exception:
        min_dist = None

    try:
        max_drac = float(row["max_drac"]) if row.get("max_drac") else None
    except Exception:
        max_drac = None

    if min_ttc is not None and min_ttc < ttc_threshold:
        return True
    if min_dist is not None and min_dist < distance_threshold:
        return True
    if max_drac is not None and max_drac > drac_threshold:
        return True
    return False