from dataclasses import dataclass, field
from typing import Literal, Optional

SceneType = Literal["highD", "inD", "rounD", "custom"]

@dataclass(frozen=True)
class SceneState:
    # ===== 通用字段 =====
    scene_type: SceneType
    ego_speed_mps: float
    headway_m: float
    lane_change: bool

    dist_to_intersection_m: float
    traffic_light: str
    vrus_present: bool

    lead_speed_mps: Optional[float] = None
    rel_speed_mps: Optional[float] = None

    # ===== highD/时序扩展字段 =====
    ego_x: Optional[float] = None
    ego_y: Optional[float] = None
    other_x: Optional[float] = None
    other_y: Optional[float] = None

    # 强烈建议补这几个字段，用于 DCPA/TCPA
    ego_vx: Optional[float] = None
    ego_vy: Optional[float] = None
    other_vx: Optional[float] = None
    other_vy: Optional[float] = None

    event_type: Optional[str] = None
    frame_index: Optional[int] = None
    duration_s: Optional[float] = None

    min_ttc_raw: Optional[float] = None
    min_thw_raw: Optional[float] = None
    min_dhw_raw: Optional[float] = None

    lane_change_direction: Optional[str] = None

@dataclass
class DriverProfile:
    driver_type: str = "unknown"
    risk_sensitivity: float = 0.5
    safety_weight: float = 0.65
    efficiency_weight: float = 0.35

@dataclass(frozen=True)
class RetrievedRule:
    id: str
    score: float
    text: str

@dataclass(frozen=True)
class StepResult:
    profile: DriverProfile
    rules: list[RetrievedRule]
    decision: dict
    triggers: list = field(default_factory=list)
    guardrails: dict = field(default_factory=dict)
    profile_update: dict = field(default_factory=dict)
    evidence: dict = field(default_factory=dict)
