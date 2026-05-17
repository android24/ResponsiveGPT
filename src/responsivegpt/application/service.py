from ..domain.logic import update_profile, make_evidence_prompts
from ..domain.models import StepResult

from ..infrastructure.hybrid_retriever import HybridRetriever

from .trigger_manager import TriggerManager
from .layered_profile_learner import LayeredProfileLearner
from ..infrastructure.profile_loader import materialize_profile_for_scene


class ResponsiveGPTService:
    """
    升级版闭环流程：

    1. load base_profile（runtime）
    2. update_profile（driver_type + feedback）
    3. materialize_profile_for_scene（场景适配）
    4. RAG 检索
    5. LLM decision
    6. TriggerManager
    7. LayeredProfileLearner（分层学习）
    8. 写回 runtime_profile
    """

    def __init__(
        self,
        retriever: HybridRetriever,
        chat_model,
        profile_repo,
        trigger_manager: TriggerManager,
        profile_learner: LayeredProfileLearner,
        trigger_state_store=None,
    ):
        self.retriever = retriever
        self.llm = chat_model
        self.repo = profile_repo

        self.trigger_manager = trigger_manager
        self.profile_learner = profile_learner
        self.trigger_state_store = trigger_state_store

    def step(
        self,
        scene,
        driver_type: str,
        feedback: str,
        recent_decisions=None,
        planning_hint: str = "",
        planning_metadata: dict | None = None,
        frame_safety=None,
    ) -> StepResult:

        if recent_decisions is None:
            recent_decisions = []

        # ==================================================
        # 1️⃣ load base profile（runtime）
        # ==================================================
        base_profile = self.repo.load()

        # ==================================================
        # 2️⃣ driver_type + feedback 更新（global层）
        # ==================================================
        base_profile = update_profile(base_profile, driver_type, feedback)

        # ==================================================
        # 3️⃣ 场景感知 profile（⚠️核心）
        # ==================================================
        scene_profile = materialize_profile_for_scene(base_profile, scene)

        # ==================================================
        # 4️⃣ RAG 检索
        # ==================================================
        evidence = self.retriever.retrieve(scene=scene, profile=scene_profile)

        # ==================================================
        # 5️⃣ LLM decision
        # ==================================================
        system, user = make_evidence_prompts(
            profile=scene_profile,          # ⚠️用 scene_profile
            scene=scene,
            human_feedback=feedback,
            evidence=evidence,
            planning_hint=planning_hint,
            planning_metadata=planning_metadata,
            frame_safety=frame_safety,
            token_budget_class="reactive_low",
        )
        # ==================================================
        # 5.1️⃣ Planning Hint Injection（新增）
        # ==================================================
        if planning_hint:
            user += f"""

        [Planning Thread Insight]

        The following planning insight comes from a slower long-horizon reasoning thread.

        Important constraints:
        - Planning may be stale.
        - Always prioritize the latest observation and physical safety metrics.
        - Use planning only as advisory guidance.
        - If planning conflicts with current scene risk, follow current scene risk.

        Planning insight:
        {planning_hint}
        """

        if planning_metadata:
            planning_age = planning_metadata.get("planning_age_frames")
            last_update_frame = planning_metadata.get("last_update_frame")

            user += f"""

        Planning metadata:
        - planning_age_frames: {planning_age}
        - last_update_frame: {last_update_frame}

        If planning_age_frames is large, reduce reliance on planning insight.
        """

        user += """

        Reactive reasoning constraints:
        - You are the fast Reactive Thread.
        - Operate under strict real-time constraints.
        - Prefer a safe feasible decision quickly.
        - The decision does not need to be globally optimal.
        - Return only valid JSON.
        """

        # ==================================================
        # 5.2️⃣ Reactive reasoning constraints（新增）
        # ==================================================
        user += """

        Reactive reasoning constraints:
        - Operate under strict real-time constraints.
        - Prefer concise and robust reasoning.
        - Produce a safe feasible decision quickly.
        - The decision does not need to be globally optimal.
        """

        decision = self.llm.complete_json(system, user)

        # ==================================================
        # 6️⃣ Trigger 评估
        # ==================================================
        triggers, guardrails, profile_update = self.trigger_manager.evaluate(
            scene=scene,
            profile=scene_profile,   # ⚠️用 scene_profile
            decision=decision,
            human_feedback=feedback,
            recent_decisions=recent_decisions,
        )

        # ==================================================
        # 7️⃣ 分层学习（⚠️核心升级）
        # ==================================================
        updated_profile = self.profile_learner.apply(
            profile=base_profile,        # ⚠️更新 base_profile
            triggers=triggers,
            profile_update=profile_update,
            decision=decision,
        )

        # ==================================================
        # 8️⃣ 写回 runtime profile
        # ==================================================
        self.repo.save(updated_profile)

        # ==================================================
        # 9️⃣ trigger 生命周期（可选）
        # ==================================================
        if self.trigger_state_store is not None:
            self.trigger_state_store.add(triggers)
            self.trigger_state_store.tick_frame()

        # ==================================================
        # 10️⃣ 平铺 evidence（兼容旧接口）
        # ==================================================
        flat_rules = self._flatten_evidence(evidence)

        # ==================================================
        # 11️⃣ 返回 StepResult
        # ==================================================
        return StepResult(
            profile=updated_profile,
            rules=flat_rules,
            decision=decision,
            triggers=[self._safe_dict(t) for t in triggers],
            guardrails=self._safe_dict(guardrails),
            profile_update=profile_update,
            evidence={
                "laws": [self._serialize_evidence_item(x) for x in evidence.laws],
                "cases": [self._serialize_evidence_item(x) for x in evidence.cases],
                "scenarios": [self._serialize_evidence_item(x) for x in evidence.scenarios],
            },
        )

    # ==================================================
    # utils
    # ==================================================
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

    def _safe_dict(self, obj):
        if isinstance(obj, dict):
            return obj
        if hasattr(obj, "__dict__"):
            return dict(obj.__dict__)
        return {"value": str(obj)}