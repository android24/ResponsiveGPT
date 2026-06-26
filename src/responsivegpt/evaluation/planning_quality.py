def planning_risk_to_bool(planning: dict) -> bool:
    rf = planning.get("risk_forecast", {})
    level = str(rf.get("risk_level", "unknown")).lower()
    return level == "high"


def physics_future_risk(frame_metrics_list, start_idx: int, horizon: int = 10) -> bool:
    end = min(len(frame_metrics_list), start_idx + horizon)
    for i in range(start_idx, end):
        m = frame_metrics_list[i]
        if (
            m.unsafe_ttc
            or m.unsafe_drac
            or m.unsafe_dcpa
            or m.unsafe_future_distance
            or (m.physical_risk_index is not None and m.physical_risk_index >= 0.65)
        ):
            return True
    return False


def compute_planning_quality(planning_records, frame_metrics_list, decision_list, horizon: int = 10):
    """
    planning_records:
      [
        {"frame_pos": int, "frame_index": int, "planning": dict}
      ]
    """
    successful_records = [
        record
        for record in planning_records
        if not bool(
            (record.get("planning") or {})
            .get("diagnostics", {})
            .get("fallback")
        )
    ]
    if not successful_records:
        return {
            "planning_call_count": 0,
            "planning_attempt_count": len(planning_records),
            "planning_failure_count": len(planning_records),
            "planning_hit_rate": None,
            "planning_precision": None,
            "planning_miss_rate": None,
            "planning_false_alarm_rate": None,
            "planning_reactive_consistency": None,
        }

    tp = fp = fn = tn = 0
    consistency = []

    for rec in successful_records:
        frame_pos = rec.get("frame_pos", 0)
        planning = rec.get("planning", {})

        pred_high = planning_risk_to_bool(planning)
        future_high = physics_future_risk(frame_metrics_list, frame_pos, horizon=horizon)

        if pred_high and future_high:
            tp += 1
        elif pred_high and not future_high:
            fp += 1
        elif (not pred_high) and future_high:
            fn += 1
        else:
            tn += 1

        consistency.append(
            _planning_reactive_consistent(planning, decision_list, frame_pos, horizon)
        )

    return {
        "planning_call_count": len(successful_records),
        "planning_attempt_count": len(planning_records),
        "planning_failure_count": (
            len(planning_records) - len(successful_records)
        ),
        "planning_hit_rate": _safe_div(tp, tp + fn),
        "planning_precision": _safe_div(tp, tp + fp),
        "planning_miss_rate": _safe_div(fn, tp + fn),
        "planning_false_alarm_rate": _safe_div(fp, fp + tn),
        "planning_reactive_consistency": _safe_mean(consistency),
        "planning_confusion": {
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "tn": tn,
        },
    }


def _planning_reactive_consistent(planning, decision_list, frame_pos, horizon):
    strategy = (
        planning
        .get("recommended_strategy", {})
        .get("strategy", "unknown")
    )

    conservative_strategies = {
        "increase_headway",
        "yield",
        "decelerate",
        "avoid_lane_change",
        "monitor_vru",
    }

    if strategy not in conservative_strategies:
        return None

    end = min(len(decision_list), frame_pos + horizon)
    if frame_pos >= end:
        return None

    # 如果规划建议保守，那么未来窗口中 reactive 至少应有中高风险或保守动作倾向
    for i in range(frame_pos, end):
        d = decision_list[i]
        risk = str(d.get("risk_level", "")).lower()
        action = str(d.get("recommended_action", "")).lower()

        if risk in {"medium", "high"}:
            return 1.0
        if any(k in action for k in ["slow", "decelerate", "yield", "safe", "headway", "brake"]):
            return 1.0

    return 0.0


def _safe_div(a, b):
    return a / b if b else None


def _safe_mean(values):
    vals = [v for v in values if isinstance(v, (int, float))]
    return sum(vals) / len(vals) if vals else None
