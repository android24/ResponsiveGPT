import uuid
from typing import List, Optional

from ..domain.models import SceneState, DriverProfile
from ..domain.triggers import TriggerEvent
from .guardrails import GuardrailState, default_guardrail_state, apply_guardrail_action


def _risk_level_to_num(risk_level: str) -> float:
    mapping = {
        "low": 0.2,
        "medium": 0.6,
        "high": 1.0,
    }
    return mapping.get(str(risk_level).lower(), 0.0)


def _safe_float(v, default=None):
    try:
        return float(v)
    except Exception:
        return default


class TriggerManager:
    """
    论文级最小实现：
    - 输入：scene / profile / llm decision / human feedback / recent history
    - 输出：trigger list + 更新后的 guardrail state + profile delta proposal
    """

    def __init__(
        self,
        ttc_threshold: float = 3.0,
        distance_threshold: float = 2.0,
        persistent_risk_ratio_threshold: float = 0.4,
        persistent_window: int = 5,
    ):
        self.ttc_threshold = ttc_threshold
        self.distance_threshold = distance_threshold
        self.persistent_risk_ratio_threshold = persistent_risk_ratio_threshold
        self.persistent_window = persistent_window

    def evaluate(
        self,
        scene: SceneState,
        profile: DriverProfile,
        decision: dict,
        human_feedback: str = "",
        recent_decisions: Optional[List[dict]] = None,
    ) -> tuple[list[TriggerEvent], GuardrailState, dict]:
        recent_decisions = recent_decisions or []
        triggers: list[TriggerEvent] = []
        guardrails = default_guardrail_state()
        profile_update = {}

        # ---------- 1) 即时风险 Trigger ----------
        ttc = self._estimate_ttc(scene)
        headway = scene.headway_m
        risk_level_num = _risk_level_to_num(decision.get("risk_level", "low"))
        violation_num = 1.0 if bool(decision.get("is_potential_violation", False)) else 0.0

        f_ttc = 0.0
        if ttc is not None:
            f_ttc = max(0.0, min(1.0, (self.ttc_threshold - ttc) / self.ttc_threshold))

        f_dist = 0.0
        if headway is not None:
            f_dist = max(0.0, min(1.0, (self.distance_threshold - headway) / self.distance_threshold))

        risk_score = 0.35 * f_ttc + 0.25 * f_dist + 0.25 * risk_level_num + 0.15 * violation_num

        if risk_score >= 0.7:
            trig = TriggerEvent(
                trigger_id=str(uuid.uuid4()),
                trigger_type="risk_threshold",
                level="frame",
                source="hybrid",
                activated=True,
                score=round(risk_score, 4),
                reason=f"high frame risk score={risk_score:.3f}, ttc={ttc}, headway={headway}",
                action="increase_safety_weight",
                action_value=0.1,
                ttl_frames=5,
                scene_type=scene.scene_type,
                event_type=scene.event_type,
                frame_index=scene.frame_index,
                metadata={"ttc": ttc, "headway": headway},
            )
            triggers.append(trig)

            guardrails = apply_guardrail_action(guardrails, "slowdown_bias", 0.1)
            profile_update["safety_weight_delta"] = profile_update.get("safety_weight_delta", 0.0) + 0.1
            profile_update["efficiency_weight_delta"] = profile_update.get("efficiency_weight_delta", 0.0) - 0.1

        # ---------- 2) 合规 Trigger ----------
        if bool(decision.get("is_potential_violation", False)):
            trig = TriggerEvent(
                trigger_id=str(uuid.uuid4()),
                trigger_type="compliance_violation",
                level="frame",
                source="llm",
                activated=True,
                score=1.0,
                reason="LLM detected potential compliance violation",
                action="apply_guardrail",
                action_value=1.0,
                ttl_frames=8,
                scene_type=scene.scene_type,
                event_type=scene.event_type,
                frame_index=scene.frame_index,
                metadata={"warning": decision.get("warning", "")},
            )
            triggers.append(trig)
            guardrails = apply_guardrail_action(guardrails, "apply_guardrail", 1.0)

            if scene.lane_change:
                freeze = TriggerEvent(
                    trigger_id=str(uuid.uuid4()),
                    trigger_type="compliance_violation",
                    level="frame",
                    source="llm",
                    activated=True,
                    score=0.9,
                    reason="Potential violation under lane change context",
                    action="freeze_lane_change",
                    action_value=1.0,
                    ttl_frames=5,
                    scene_type=scene.scene_type,
                    event_type=scene.event_type,
                    frame_index=scene.frame_index,
                )
                triggers.append(freeze)
                guardrails = apply_guardrail_action(guardrails, "freeze_lane_change", 1.0)

        # ---------- 3) VRU Trigger ----------
        if scene.vrus_present and risk_score >= 0.5:
            trig = TriggerEvent(
                trigger_id=str(uuid.uuid4()),
                trigger_type="vru_protection",
                level="frame",
                source="hybrid",
                activated=True,
                score=round(risk_score, 4),
                reason="VRU present under non-trivial risk",
                action="slowdown_bias",
                action_value=0.2,
                ttl_frames=10,
                scene_type=scene.scene_type,
                event_type=scene.event_type,
                frame_index=scene.frame_index,
            )
            triggers.append(trig)
            guardrails = apply_guardrail_action(guardrails, "slowdown_bias", 0.2)

        # ---------- 4) 持续高风险 Trigger ----------
        if recent_decisions:
            recent_flags = []
            for d in recent_decisions[-self.persistent_window:]:
                lvl = str(d.get("risk_level", "low")).lower()
                vio = bool(d.get("is_potential_violation", False))
                recent_flags.append(int(lvl == "high" or vio))

            if recent_flags:
                ratio = sum(recent_flags) / len(recent_flags)
                if ratio >= self.persistent_risk_ratio_threshold:
                    trig = TriggerEvent(
                        trigger_id=str(uuid.uuid4()),
                        trigger_type="persistent_high_risk",
                        level="episode",
                        source="history",
                        activated=True,
                        score=round(ratio, 4),
                        reason=f"persistent high-risk ratio={ratio:.3f} in recent window",
                        action="increase_risk_sensitivity",
                        action_value=0.1,
                        ttl_episodes=1,
                        scene_type=scene.scene_type,
                        event_type=scene.event_type,
                        frame_index=scene.frame_index,
                    )
                    triggers.append(trig)
                    profile_update["risk_sensitivity_delta"] = profile_update.get("risk_sensitivity_delta", 0.0) + 0.1

        # ---------- 5) 人类反馈 Trigger ----------
        fb = (human_feedback or "").lower()
        if any(k in fb for k in ["危险", "不安全", "不舒服", "too dangerous", "unsafe"]):
            trig = TriggerEvent(
                trigger_id=str(uuid.uuid4()),
                trigger_type="human_feedback",
                level="profile",
                source="human",
                activated=True,
                score=1.0,
                reason="human feedback indicates safety dissatisfaction",
                action="profile_update",
                action_value=0.1,
                ttl_episodes=3,
                scene_type=scene.scene_type,
                event_type=scene.event_type,
                frame_index=scene.frame_index,
            )
            triggers.append(trig)
            profile_update["risk_sensitivity_delta"] = profile_update.get("risk_sensitivity_delta", 0.0) + 0.1

        if any(k in fb for k in ["太慢", "低效", "too slow", "效率太低"]):
            trig = TriggerEvent(
                trigger_id=str(uuid.uuid4()),
                trigger_type="preference_mismatch",
                level="profile",
                source="human",
                activated=True,
                score=0.8,
                reason="human feedback indicates efficiency dissatisfaction",
                action="decrease_efficiency_weight",
                action_value=-0.1,
                ttl_episodes=3,
                scene_type=scene.scene_type,
                event_type=scene.event_type,
                frame_index=scene.frame_index,
            )
            triggers.append(trig)
            profile_update["safety_weight_delta"] = profile_update.get("safety_weight_delta", 0.0) - 0.1
            profile_update["efficiency_weight_delta"] = profile_update.get("efficiency_weight_delta", 0.0) + 0.1

        return triggers, guardrails, profile_update

    def _estimate_ttc(self, scene: SceneState):
        if scene.headway_m is None:
            return None

        closing = None
        if scene.rel_speed_mps is not None:
            closing = scene.rel_speed_mps
        elif scene.lead_speed_mps is not None:
            closing = scene.ego_speed_mps - scene.lead_speed_mps

        if closing is None or closing <= 0:
            return None
        if scene.headway_m <= 0:
            return 0.0
        return scene.headway_m / closing