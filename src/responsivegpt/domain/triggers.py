from dataclasses import dataclass, field
from typing import Optional, Literal

TriggerLevel = Literal["frame", "episode", "profile"]
TriggerType = Literal[
    "risk_threshold",
    "compliance_violation",
    "human_feedback",
    "persistent_high_risk",
    "preference_mismatch",
    "vru_protection",
]
TriggerAction = Literal[
    "increase_safety_weight",
    "decrease_efficiency_weight",
    "increase_risk_sensitivity",
    "apply_guardrail",
    "freeze_lane_change",
    "slowdown_bias",
    "profile_update",
]
TargetLayer = Literal["global", "longitudinal", "lateral", "interaction", "temporal"]

@dataclass(frozen=True)
class TriggerEvent:
    trigger_id: str
    trigger_type: TriggerType
    level: TriggerLevel
    source: str
    activated: bool
    score: float
    reason: str

    action: TriggerAction
    action_value: float

    target_layer: Optional[TargetLayer] = None
    parameter_key: Optional[str] = None
    priority: int = 0

    ttl_frames: int = 0
    ttl_episodes: int = 0

    scene_type: Optional[str] = None
    event_type: Optional[str] = None
    frame_index: Optional[int] = None

    metadata: dict = field(default_factory=dict)