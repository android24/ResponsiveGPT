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
        "planning_schema_version": "planning_v1",
        "scene_summary": "",
        "risk_forecast": {
            "risk_level": "unknown",
            "risk_trend": "unknown",
            "time_horizon_s": 3.0,
            "main_risk_factors": [],
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
        },
        "confidence": 0.0,
        "validity": {
            "valid_for_frames": 10,
            "validity_level": "short",
            "refresh_condition": "risk changes or safety metrics cross threshold",
        },
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

    # schema version
    out["planning_schema_version"] = str(out.get("planning_schema_version") or "planning_v1")

    # risk forecast
    rf = out.get("risk_forecast")
    if not isinstance(rf, dict):
        rf = {}
    risk_level = str(rf.get("risk_level", "unknown")).lower()
    if risk_level not in ALLOWED_RISK_LEVELS:
        risk_level = "unknown"

    out["risk_forecast"] = {
        "risk_level": risk_level,
        "risk_trend": str(rf.get("risk_trend", "unknown")),
        "time_horizon_s": _safe_float(rf.get("time_horizon_s"), 3.0),
        "main_risk_factors": _safe_list(rf.get("main_risk_factors")),
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
    rg = out.get("reactive_guidance")
    if not isinstance(rg, dict):
        rg = {}
    out["reactive_guidance"] = {
        "must_check": _safe_list(rg.get("must_check")),
        "avoid_actions": _safe_list(rg.get("avoid_actions")),
        "preferred_actions": _safe_list(rg.get("preferred_actions")),
        "safety_constraints": _safe_list(rg.get("safety_constraints")),
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