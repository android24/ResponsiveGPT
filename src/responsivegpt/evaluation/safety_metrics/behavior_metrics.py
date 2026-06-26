from .base import BehaviorSafetyMetrics
from .utils import safe_var


def _decision_violation(decision):
    v = decision.get("is_potential_violation", None)
    return bool(v) if isinstance(v, bool) else False


def _risk_level_score(decision):
    level = str(decision.get("risk_level", "")).lower()
    if level == "high":
        return 3.0
    if level == "medium":
        return 2.0
    if level == "low":
        return 1.0
    return None


def compute_behavior_safety_metrics(frame_metrics_list, decision_list, trigger_list_by_frame=None):
    n = min(len(frame_metrics_list), len(decision_list))

    if n == 0:
        return BehaviorSafetyMetrics(
            num_frames=0,
            reaction_delay_frames=None,
            reaction_success_rate=None,
            reaction_censored=False,
            first_risky_frame_pos=None,
            reaction_observation_window_frames=None,
            trigger_delay_frames=None,
            decision_flip_count=0,
            decision_flip_rate=0.0,
            risk_level_variance=None,
        )

    # first physical risky frame
    first_risky_idx = None
    for i in range(n):
        m = frame_metrics_list[i]
        risky = (
            m.unsafe_ttc or
            m.unsafe_drac or
            m.unsafe_dcpa or
            m.unsafe_future_distance or
            (m.physical_risk_index is not None and m.physical_risk_index >= 0.65)
        )
        if risky:
            first_risky_idx = i
            break

    # first LLM violation after risk appears
    reaction_delay = None
    if first_risky_idx is not None:
        for j in range(first_risky_idx, n):
            if _decision_violation(decision_list[j]):
                reaction_delay = j - first_risky_idx
                break
    reaction_success_rate = (
        None
        if first_risky_idx is None
        else float(reaction_delay is not None)
    )
    reaction_censored = (
        first_risky_idx is not None and reaction_delay is None
    )
    reaction_observation_window = (
        None
        if first_risky_idx is None
        else max(0, n - 1 - first_risky_idx)
    )

    # first trigger after risk appears
    trigger_delay = None
    if first_risky_idx is not None and trigger_list_by_frame:
        for j in range(first_risky_idx, n):
            if trigger_list_by_frame.get(j):
                trigger_delay = j - first_risky_idx
                break

    # decision stability
    violation_seq = [_decision_violation(d) for d in decision_list[:n]]
    flips = 0
    for i in range(1, len(violation_seq)):
        if violation_seq[i] != violation_seq[i - 1]:
            flips += 1

    risk_scores = [_risk_level_score(d) for d in decision_list[:n]]

    return BehaviorSafetyMetrics(
        num_frames=n,
        reaction_delay_frames=reaction_delay,
        reaction_success_rate=reaction_success_rate,
        reaction_censored=reaction_censored,
        first_risky_frame_pos=first_risky_idx,
        reaction_observation_window_frames=reaction_observation_window,
        trigger_delay_frames=trigger_delay,
        decision_flip_count=flips,
        decision_flip_rate=flips / max(1, n - 1),
        risk_level_variance=safe_var(risk_scores),
    )
