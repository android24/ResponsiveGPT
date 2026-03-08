from __future__ import annotations
from dataclasses import dataclass
from typing import Optional
from ..domain.models import SceneState


@dataclass
class StepMetrics:
    ttc_s: Optional[float]
    is_violation: Optional[bool]


def compute_ttc(scene: SceneState) -> Optional[float]:
    """
    简化 TTC:
    - 优先用 rel_speed_mps 作为 closing speed
    - 否则用 ego_speed - lead_speed
    """
    if scene.headway_m is None:
        return None

    closing = None
    if scene.rel_speed_mps is not None:
        closing = scene.rel_speed_mps
    elif scene.lead_speed_mps is not None:
        closing = scene.ego_speed_mps - scene.lead_speed_mps

    if closing is None:
        return None
    if closing <= 0:
        return None
    if scene.headway_m <= 0:
        return 0.0

    return scene.headway_m / closing


def compute_violation(decision: dict) -> Optional[bool]:
    v = decision.get("is_potential_violation", None)
    if isinstance(v, bool):
        return v
    return None


def compute_step_metrics(scene: SceneState, decision: dict) -> StepMetrics:
    return StepMetrics(
        ttc_s=compute_ttc(scene),
        is_violation=compute_violation(decision),
    )