from ..domain.logic import update_profile, make_evidence_prompts
from ..domain.models import StepResult
from ..infrastructure.hybrid_retriever import HybridRetriever
from .trigger_manager import TriggerManager

def _clamp(x: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, x))


class ResponsiveGPTService:
    """
    当前正式版主流程：

    1. 读取并更新 DriverProfile
    2. HybridRetriever 做 law/case/scenario 三路检索
    3. LLM 输出结构化 decision（含 evidence_ids）
    4. TriggerManager 生成 triggers / guardrails / profile_update
    5. 将 trigger 导致的参数变化写回 profile
    6. 返回统一 StepResult
    """
    def __init__(
        self,
        retriever: HybridRetriever,
        chat_model,
        profile_repo,
        trigger_manager: TriggerManager | None = None,
    ):
        self.retriever = retriever
        self.llm = chat_model
        self.repo = profile_repo
        self.trigger_manager = trigger_manager or TriggerManager()

    def step(
        self,
        scene,
        driver_type: str,
        feedback: str,
        recent_decisions=None,
    ) -> StepResult:
        """
        Args:
            scene: 当前场景 SceneState
            driver_type: 激进 / 保守 / ...
            feedback: 人类自然语言反馈
            recent_decisions: 最近若干步的 decision 列表，用于 persistent trigger

        Returns:
            StepResult
        """
        if recent_decisions is None:
            recent_decisions = []

        # --------------------------------------------------
        # 1) 基础画像更新（基于 driver_type + feedback）
        # --------------------------------------------------
        profile = self.repo.load()
        profile = update_profile(profile, driver_type, feedback)
        self.repo.save(profile)

        # --------------------------------------------------
        # 2) 三路检索：law / case / scenario
        # --------------------------------------------------
        evidence = self.retriever.retrieve(scene=scene, profile=profile)

        # --------------------------------------------------
        # 3) 生成 prompt 并调用 LLM
        # --------------------------------------------------
        system, user = make_evidence_prompts(
            profile=profile,
            scene=scene,
            human_feedback=feedback,
            evidence=evidence,
        )
        decision = self.llm.complete_json(system, user)

        # --------------------------------------------------
        # 4) Trigger 评估
        # --------------------------------------------------
        triggers, guardrails, profile_update = self.trigger_manager.evaluate(
            scene=scene,
            profile=profile,
            decision=decision,
            human_feedback=feedback,
            recent_decisions=recent_decisions,
        )

        # --------------------------------------------------
        # 5) 将 trigger 产生的 profile delta 应用回 profile
        # --------------------------------------------------
        profile = self._apply_profile_update(profile, profile_update)

        # 保存最终 profile
        self.repo.save(profile)

        # --------------------------------------------------
        # 6) 平铺 evidence，兼容 rules 字段
        # --------------------------------------------------
        flat_rules = self._flatten_evidence(evidence)

        # --------------------------------------------------
        # 7) 返回正式 StepResult
        # --------------------------------------------------
        return StepResult(
            profile=profile,
            rules=flat_rules,
            decision=decision,
            triggers=[t.__dict__ for t in triggers],
            guardrails=guardrails.__dict__,
            profile_update=profile_update,
            evidence={
                "laws": [self._serialize_evidence_item(x) for x in evidence.laws],
                "cases": [self._serialize_evidence_item(x) for x in evidence.cases],
                "scenarios": [self._serialize_evidence_item(x) for x in evidence.scenarios],
            },
        )

    def _apply_profile_update(self, profile, profile_update: dict):
        """
        把 trigger 产生的 delta 应用到 profile 上。
        """
        if not isinstance(profile_update, dict):
            return profile

        if "risk_sensitivity_delta" in profile_update:
            profile.risk_sensitivity = _clamp(
                profile.risk_sensitivity + float(profile_update["risk_sensitivity_delta"])
            )

        if "safety_weight_delta" in profile_update:
            profile.safety_weight = _clamp(
                profile.safety_weight + float(profile_update["safety_weight_delta"])
            )

        if "efficiency_weight_delta" in profile_update:
            profile.efficiency_weight = _clamp(
                profile.efficiency_weight + float(profile_update["efficiency_weight_delta"])
            )

        # 让 safety / efficiency 保持可解释
        total = profile.safety_weight + profile.efficiency_weight
        if total > 0:
            profile.safety_weight = round(profile.safety_weight / total, 3)
            profile.efficiency_weight = round(profile.efficiency_weight / total, 3)

        return profile

    def _flatten_evidence(self, evidence):
        """
        为保持 rules 字段兼容，把 laws/cases/scenarios 平铺。
        """
        flat_rules = []

        for item in evidence.laws + evidence.cases + evidence.scenarios:
            if hasattr(item, "doc"):
                flat_rules.append(item.doc)
            else:
                flat_rules.append(item)

        return flat_rules

    def _serialize_evidence_item(self, item):
        """
        把 evidence item 转成可写日志的 dict / 基础类型。
        """
        if hasattr(item, "__dict__"):
            data = dict(item.__dict__)
            if "doc" in data and hasattr(data["doc"], "__dict__"):
                data["doc"] = dict(data["doc"].__dict__)
            return data

        return str(item)
