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


def risk_phase_from_safety(frame_safety, previous_phase: str | None = None) -> str:
    risk_index = getattr(frame_safety, "physical_risk_index", None)
    unsafe = (
        bool(getattr(frame_safety, "unsafe_ttc", False))
        or bool(getattr(frame_safety, "unsafe_drac", False))
        or bool(getattr(frame_safety, "unsafe_dcpa", False))
        or bool(getattr(frame_safety, "unsafe_future_distance", False))
    )
    if unsafe or (risk_index is not None and float(risk_index) >= 0.65):
        return "conflict"
    if risk_index is not None and float(risk_index) >= 0.35:
        return "approaching"
    if str(previous_phase or "") in {"conflict", "approaching"}:
        return "recovery"
    return "stable"


def llm_call_reasons(
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
    risk_phase_changed: bool = False,
    novelty_detected: bool = False,
    planning_reactive_conflict: bool = False,
) -> list[str]:
    policy = str(policy or "hybrid").lower()
    stride = max(1, int(stride))
    max_stale_frames = max(1, int(max_stale_frames))
    reasons = []

    if policy == "none":
        return reasons

    if frame_pos == 0:
        reasons.append("first_frame")

    if policy == "always":
        reasons.append("policy_always")
        return reasons

    if policy == "stride":
        if frame_pos % stride == 0:
            reasons.append("stride_refresh")
        return reasons

    risky = (
        bool(getattr(frame_safety, "unsafe_ttc", False))
        or bool(getattr(frame_safety, "unsafe_drac", False))
        or bool(getattr(frame_safety, "unsafe_dcpa", False))
        or bool(getattr(frame_safety, "unsafe_future_distance", False))
        or (
            frame_safety.physical_risk_index is not None
            and frame_safety.physical_risk_index >= risk_threshold
        )
    )

    if policy == "risk_only":
        if risky:
            reasons.append("risk_threshold")
        return reasons

    if policy == "event_triggered":
        if last_llm_frame_pos is None:
            reasons.append("no_previous_llm")
            return reasons

        if frame_pos - last_llm_frame_pos >= max_stale_frames:
            reasons.append("stale_decision")

        current_risk_level = str(getattr(frame_safety, "physical_risk_level", "") or "")
        if current_risk_level and current_risk_level != str(last_llm_risk_level or ""):
            reasons.append("risk_level_changed")

        current_risk_index = getattr(frame_safety, "physical_risk_index", None)
        if current_risk_index is not None and last_llm_risk_index is not None:
            if abs(float(current_risk_index) - float(last_llm_risk_index)) >= risk_delta_threshold:
                reasons.append("risk_delta_large")

        if grounding_refresh_required:
            reasons.append("grounding_refresh_required")

        if risky and evidence_changed:
            reasons.append("evidence_changed_under_risk")

        if risky and planning_hint_updated:
            reasons.append("planning_hint_updated_under_risk")

        if risky and risk_phase_changed:
            reasons.append("risk_phase_changed_under_risk")

        if risky and novelty_detected:
            reasons.append("novelty_under_risk")

        if risky and planning_reactive_conflict:
            reasons.append("planning_reactive_conflict_under_risk")

        return reasons

    if policy == "hybrid":
        if risky:
            reasons.append("risk_threshold")
        if frame_pos % stride == 0:
            reasons.append("stride_refresh")
        return reasons

    reasons.append("policy_default")
    return reasons


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
    risk_phase_changed: bool = False,
    novelty_detected: bool = False,
    planning_reactive_conflict: bool = False,
) -> bool:
    return bool(llm_call_reasons(
        policy=policy,
        frame_pos=frame_pos,
        frame_safety=frame_safety,
        stride=stride,
        risk_threshold=risk_threshold,
        max_stale_frames=max_stale_frames,
        risk_delta_threshold=risk_delta_threshold,
        last_llm_frame_pos=last_llm_frame_pos,
        last_llm_risk_level=last_llm_risk_level,
        last_llm_risk_index=last_llm_risk_index,
        evidence_changed=evidence_changed,
        grounding_refresh_required=grounding_refresh_required,
        planning_hint_updated=planning_hint_updated,
        risk_phase_changed=risk_phase_changed,
        novelty_detected=novelty_detected,
        planning_reactive_conflict=planning_reactive_conflict,
    ))


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
