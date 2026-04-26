from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class FrameSafetyMetrics:
    frame_index: Optional[int]

    # raw physical states
    distance_m: Optional[float]
    rel_speed_mps: Optional[float]
    ego_speed_mps: Optional[float]
    lead_speed_mps: Optional[float]

    ego_x: Optional[float]
    ego_y: Optional[float]
    other_x: Optional[float]
    other_y: Optional[float]

    ego_vx: Optional[float]
    ego_vy: Optional[float]
    other_vx: Optional[float]
    other_vy: Optional[float]

    # instantaneous kinematic risk
    ttc_s: Optional[float]
    thw_s: Optional[float]
    drac_mps2: Optional[float]

    # future / spatial interaction risk
    ttca_s: Optional[float]
    dcpa_m: Optional[float]
    predicted_ttc_s: Optional[float]
    min_future_distance_m: Optional[float]

    # flags
    closing: bool
    unsafe_ttc: bool
    unsafe_thw: bool
    unsafe_drac: bool
    unsafe_dcpa: bool
    unsafe_future_distance: bool

    # composite
    physical_risk_index: Optional[float]
    physical_risk_level: str


@dataclass(frozen=True)
class EpisodeSafetyMetrics:
    num_frames: int

    # instantaneous aggregates
    min_ttc_s: Optional[float]
    avg_ttc_s: Optional[float]
    min_thw_s: Optional[float]
    avg_thw_s: Optional[float]
    max_drac_mps2: Optional[float]
    avg_drac_mps2: Optional[float]

    # future / spatial aggregates
    min_dcpa_m: Optional[float]
    avg_dcpa_m: Optional[float]
    min_predicted_ttc_s: Optional[float]
    min_future_distance_m: Optional[float]
    avg_future_distance_m: Optional[float]

    # exposure ratios
    unsafe_ttc_ratio: float
    unsafe_thw_ratio: float
    unsafe_drac_ratio: float
    unsafe_dcpa_ratio: float
    unsafe_future_distance_ratio: float

    # composite risk
    avg_physical_risk_index: Optional[float]
    max_physical_risk_index: Optional[float]
    physical_risk_exposure: float

    # event-level flags
    has_critical_ttc: bool
    has_critical_drac: bool
    has_critical_spatial_risk: bool


@dataclass(frozen=True)
class LLMPhysicsAlignment:
    num_frames: int
    llm_violation_rate: float
    avg_physical_risk_index: Optional[float]
    max_physical_risk_index: Optional[float]
    alignment_accuracy: Optional[float]
    overreaction_rate: Optional[float]
    underreaction_rate: Optional[float]
    mean_risk_level_error: Optional[float]


@dataclass(frozen=True)
class BehaviorSafetyMetrics:
    num_frames: int
    reaction_delay_frames: Optional[int]
    trigger_delay_frames: Optional[int]
    decision_flip_count: int
    decision_flip_rate: float
    risk_level_variance: Optional[float]