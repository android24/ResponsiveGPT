def is_risky_frame(frame_safety) -> bool:
    return (
        bool(frame_safety.unsafe_ttc)
        or bool(frame_safety.unsafe_drac)
        or bool(frame_safety.unsafe_dcpa)
        or bool(frame_safety.unsafe_future_distance)
        or (
            frame_safety.physical_risk_index is not None
            and frame_safety.physical_risk_index >= 0.35
        )
    )


def should_call_llm(
    policy: str,
    frame_pos: int,
    frame_safety,
    stride: int = 5,
    risk_threshold: float = 0.35,
    *,
    max_stale_frames: int = 30,
    risk_delta_threshold: float = 0.15,
    last_llm_frame_pos: int | None = None,
    last_llm_risk_level: str | None = None,
    last_llm_risk_index: float | None = None,
    evidence_changed: bool = False,
    grounding_refresh_required: bool = False,
    planning_hint_updated: bool = False,
) -> bool:
    policy = str(policy or "hybrid").lower()
    stride = max(1, int(stride))
    max_stale_frames = max(1, int(max_stale_frames))

    if policy == "none":
        return False

    if frame_pos == 0:
        return True

    if policy == "always":
        return True

    if policy == "stride":
        return frame_pos % stride == 0

    risky = (
        frame_safety.unsafe_ttc
        or frame_safety.unsafe_drac
        or frame_safety.unsafe_dcpa
        or frame_safety.unsafe_future_distance
        or (
            frame_safety.physical_risk_index is not None
            and frame_safety.physical_risk_index >= risk_threshold
        )
    )

    if policy == "risk_only":
        return risky

    if policy == "event_triggered":
        if last_llm_frame_pos is None:
            return True

        if frame_pos - last_llm_frame_pos >= max_stale_frames:
            return True

        current_risk_level = str(getattr(frame_safety, "physical_risk_level", "") or "")
        if current_risk_level and current_risk_level != str(last_llm_risk_level or ""):
            return True

        current_risk_index = getattr(frame_safety, "physical_risk_index", None)
        if current_risk_index is not None and last_llm_risk_index is not None:
            if abs(float(current_risk_index) - float(last_llm_risk_index)) >= risk_delta_threshold:
                return True

        if grounding_refresh_required:
            return True

        if risky and (evidence_changed or planning_hint_updated):
            return True

        return False

    if policy == "hybrid":
        return risky or frame_pos % stride == 0

    return True


def fallback_decision_from_physics(frame_safety) -> dict:
    risky = is_risky_frame(frame_safety)

    if frame_safety.physical_risk_index is not None:
        if frame_safety.physical_risk_index >= 0.65:
            level = "high"
        elif frame_safety.physical_risk_index >= 0.35:
            level = "medium"
        else:
            level = "low"
    else:
        level = "unknown"

    return {
        "is_potential_violation": risky,
        "risk_level": level,
        "recommended_action": "保持安全距离并持续监测" if risky else "保持当前策略并监测",
        "warning": "物理风险指标触发的规则判定" if risky else "",
        "reason": "physics-based fallback without LLM call",
        "confidence": 0.5,
        "source": "physics_fallback",
    }
