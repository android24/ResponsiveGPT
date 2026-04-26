from dataclasses import dataclass
from typing import Optional
from .safety_metrics import SafetyThresholds, compute_frame_safety_metrics


DEFAULT_THRESHOLDS = SafetyThresholds()


@dataclass
class StepMetrics:
    ttc_s: Optional[float]
    thw_s: Optional[float]
    drac_mps2: Optional[float]
    dcpa_m: Optional[float]
    ttca_s: Optional[float]
    predicted_ttc_s: Optional[float]
    min_future_distance_m: Optional[float]
    physical_risk_index: Optional[float]
    physical_risk_level: str
    is_violation: Optional[bool]


def compute_step_metrics(scene, decision: dict, thresholds: SafetyThresholds = DEFAULT_THRESHOLDS) -> StepMetrics:
    physical = compute_frame_safety_metrics(scene, thresholds)

    v = decision.get("is_potential_violation", None)
    is_violation = v if isinstance(v, bool) else None

    return StepMetrics(
        ttc_s=physical.ttc_s,
        thw_s=physical.thw_s,
        drac_mps2=physical.drac_mps2,
        dcpa_m=physical.dcpa_m,
        ttca_s=physical.ttca_s,
        predicted_ttc_s=physical.predicted_ttc_s,
        min_future_distance_m=physical.min_future_distance_m,
        physical_risk_index=physical.physical_risk_index,
        physical_risk_level=physical.physical_risk_level,
        is_violation=is_violation,
    )