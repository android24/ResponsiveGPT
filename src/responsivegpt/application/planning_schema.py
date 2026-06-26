ALLOWED_PLANNING_STRATEGIES = {
    "keep_current",
    "increase_headway",
    "yield",
    "decelerate",
    "maintain_speed",
    "prepare_lane_change",
    "avoid_lane_change",
    "monitor_vru",
    "unknown",
}

ALLOWED_RISK_LEVELS = {"low", "medium", "high", "unknown"}

ALLOWED_VALIDITY = {"short", "medium", "long"}


def default_planning_output(reason: str = "default fallback") -> dict:
    return {
        "planning_schema_version": "planning_v2",
        "scene_summary": "",
        "scenario_assessment": {
            "dataset": "unknown",
            "scenario_type": "unknown",
            "dominant_conflict_mechanism": "unknown",
            "why_this_scenario_is_risky": "",
        },
        "risk_forecast": {
            "risk_level": "unknown",
            "risk_trend": "unknown",
            "time_horizon_s": 3.0,
            "main_risk_factors": [],
            "metric_evidence": {
                "ttc": None,
                "thw": None,
                "drac": None,
                "dcpa": None,
                "ttca": None,
                "min_future_distance": None,
                "physical_risk_index": None,
            },
            "expected_conflict_frames": [],
        },
        "focus_object": {
            "object_id": None,
            "object_type": "unknown",
            "reason": "",
        },
        "recommended_strategy": {
            "strategy": "unknown",
            "rationale": reason,
            "priority": "safety",
        },
        "reactive_guidance": {
            "must_check": [],
            "avoid_actions": [],
            "preferred_actions": [],
            "safety_constraints": [],
            "fast_rule_hint": "",
        },
        "staleness_control": {
            "valid_for_frames": 10,
            "refresh_if": ["risk changes"],
            "staleness_risk": "medium",
        },
        "confidence": 0.0,
        "diagnostics": {
            "used_frames": 0,
            "token_budget_class": "planning",
            "fallback": True,
        },
    }


def validate_planning_output(obj: dict) -> dict:
    """
    宽松校验 + 自动修复，保证 Planning Memory 永远有结构化内容。
    """
    if not isinstance(obj, dict):
        return default_planning_output("planning output is not dict")

    out = default_planning_output("auto repaired")
    out.update(obj)
    raw_risk_forecast = (
        dict(out.get("risk_forecast"))
        if isinstance(out.get("risk_forecast"), dict)
        else {}
    )
    raw_reactive_guidance = (
        dict(out.get("reactive_guidance"))
        if isinstance(out.get("reactive_guidance"), dict)
        else {}
    )

    # schema version
    out["planning_schema_version"] = str(out.get("planning_schema_version") or "planning_v1")

    # risk forecast
    rf = raw_risk_forecast
    risk_level = str(rf.get("risk_level", "unknown")).lower()
    if risk_level not in ALLOWED_RISK_LEVELS:
        risk_level = "unknown"

    out["risk_forecast"] = {
        "risk_level": risk_level,
        "risk_trend": str(rf.get("risk_trend", "unknown")),
        "time_horizon_s": _safe_float(rf.get("time_horizon_s"), 3.0),
        "main_risk_factors": _safe_list(rf.get("main_risk_factors")),
        "metric_evidence": {},
        "expected_conflict_frames": _safe_list(rf.get("expected_conflict_frames")),
    }

    # focus object
    fo = out.get("focus_object")
    if not isinstance(fo, dict):
        fo = {}
    out["focus_object"] = {
        "object_id": fo.get("object_id"),
        "object_type": str(fo.get("object_type", "unknown")),
        "reason": str(fo.get("reason", "")),
    }

    # strategy
    rs = out.get("recommended_strategy")
    if not isinstance(rs, dict):
        rs = {}
    strategy = str(rs.get("strategy", "unknown")).lower()
    if strategy not in ALLOWED_PLANNING_STRATEGIES:
        strategy = "unknown"
    out["recommended_strategy"] = {
        "strategy": strategy,
        "rationale": str(rs.get("rationale", "")),
        "priority": str(rs.get("priority", "safety")),
    }

    # reactive guidance
    rg = raw_reactive_guidance
    out["reactive_guidance"] = {
        "must_check": _safe_list(rg.get("must_check")),
        "avoid_actions": _safe_list(rg.get("avoid_actions")),
        "preferred_actions": _safe_list(rg.get("preferred_actions")),
        "safety_constraints": _safe_list(rg.get("safety_constraints")),
        "fast_rule_hint": str(rg.get("fast_rule_hint", "")),
    }

    # scenario assessment
    sa = out.get("scenario_assessment")
    if not isinstance(sa, dict):
        sa = {}
    out["scenario_assessment"] = {
        "dataset": str(sa.get("dataset", "unknown")),
        "scenario_type": str(sa.get("scenario_type", "unknown")),
        "dominant_conflict_mechanism": str(sa.get("dominant_conflict_mechanism", "unknown")),
        "why_this_scenario_is_risky": str(sa.get("why_this_scenario_is_risky", "")),
    }

    # metric evidence
    metric_evidence = raw_risk_forecast.get("metric_evidence")
    if not isinstance(metric_evidence, dict):
        metric_evidence = {}

    out["risk_forecast"]["metric_evidence"] = {
        "ttc": metric_evidence.get("ttc"),
        "thw": metric_evidence.get("thw"),
        "drac": metric_evidence.get("drac"),
        "dcpa": metric_evidence.get("dcpa"),
        "ttca": metric_evidence.get("ttca"),
        "min_future_distance": metric_evidence.get("min_future_distance"),
        "physical_risk_index": metric_evidence.get("physical_risk_index"),
    }

    # staleness control
    sc = out.get("staleness_control")
    if not isinstance(sc, dict):
        sc = {}
    out["staleness_control"] = {
        "valid_for_frames": int(_safe_float(sc.get("valid_for_frames"), 10)),
        "refresh_if": _safe_list(sc.get("refresh_if")),
        "staleness_risk": str(sc.get("staleness_risk", "medium")),
    }

    out["confidence"] = max(0.0, min(1.0, _safe_float(out.get("confidence"), 0.0)))

    validity = out.get("validity")
    if not isinstance(validity, dict):
        validity = {}
    validity_level = str(validity.get("validity_level", "short")).lower()
    if validity_level not in ALLOWED_VALIDITY:
        validity_level = "short"

    out["validity"] = {
        "valid_for_frames": int(_safe_float(validity.get("valid_for_frames"), 10)),
        "validity_level": validity_level,
        "refresh_condition": str(validity.get("refresh_condition", "risk changes")),
    }

    diagnostics = out.get("diagnostics")
    if not isinstance(diagnostics, dict):
        diagnostics = {}
    out["diagnostics"] = {
        "used_frames": int(_safe_float(diagnostics.get("used_frames"), 0)),
        "token_budget_class": str(diagnostics.get("token_budget_class", "planning")),
        "fallback": bool(diagnostics.get("fallback", False)),
    }

    return out


def _safe_float(x, default):
    try:
        return float(x)
    except Exception:
        return default


def _safe_list(x):
    if isinstance(x, list):
        return x
    if x is None:
        return []
    return [x]
