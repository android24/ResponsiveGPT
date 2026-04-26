from .base import LLMPhysicsAlignment
from .utils import safe_mean, safe_max


def llm_risk_level_to_score(decision):
    level = str(decision.get("risk_level", "")).lower()
    if level == "high":
        return 3.0
    if level == "medium":
        return 2.0
    if level == "low":
        return 1.0

    # fallback: violation boolean
    if decision.get("is_potential_violation") is True:
        return 3.0
    if decision.get("is_potential_violation") is False:
        return 1.0
    return None


def physics_risk_index_to_score(x):
    """
    将 [0,1] 风险指数映射到 [1,3]
    """
    if x is None:
        return None
    return 1.0 + 2.0 * max(0.0, min(1.0, x))


def compute_llm_physics_alignment(frame_metrics_list, decision_list):
    n = min(len(frame_metrics_list), len(decision_list))
    if n == 0:
        return LLMPhysicsAlignment(
            num_frames=0,
            llm_violation_rate=0.0,
            avg_physical_risk_index=None,
            max_physical_risk_index=None,
            alignment_accuracy=None,
            overreaction_rate=None,
            underreaction_rate=None,
            mean_risk_level_error=None,
        )

    llm_violations = []
    matches = []
    overreactions = []
    underreactions = []
    errors = []
    risk_vals = []

    for i in range(n):
        m = frame_metrics_list[i]
        d = decision_list[i]

        llm_violation = bool(d.get("is_potential_violation", False))
        llm_violations.append(llm_violation)

        physics_high = (
            (m.physical_risk_index is not None and m.physical_risk_index >= 0.65)
            or m.unsafe_ttc
            or m.unsafe_drac
            or m.unsafe_dcpa
            or m.unsafe_future_distance
        )

        matches.append(llm_violation == physics_high)
        overreactions.append(llm_violation and not physics_high)
        underreactions.append((not llm_violation) and physics_high)

        llm_score = llm_risk_level_to_score(d)
        phy_score = physics_risk_index_to_score(m.physical_risk_index)
        if llm_score is not None and phy_score is not None:
            errors.append(abs(llm_score - phy_score))

        risk_vals.append(m.physical_risk_index)

    return LLMPhysicsAlignment(
        num_frames=n,
        llm_violation_rate=sum(llm_violations) / n,
        avg_physical_risk_index=safe_mean(risk_vals),
        max_physical_risk_index=safe_max(risk_vals),
        alignment_accuracy=sum(matches) / n,
        overreaction_rate=sum(overreactions) / n,
        underreaction_rate=sum(underreactions) / n,
        mean_risk_level_error=safe_mean(errors),
    )