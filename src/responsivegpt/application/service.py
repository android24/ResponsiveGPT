from ..domain.logic import update_profile, make_evidence_prompts, validate_decision_json
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
        evidence_pack: dict | None = None,
        planning_hint: str = "",
        planning_metadata: dict | None = None,
        frame_safety=None,
        require_grounded_decision: bool = False,
        allow_profile_update: bool = True,
    ) -> StepResult:

        if recent_decisions is None:
            recent_decisions = []

        # ==================================================
        # 1️⃣ load base profile（runtime）
        # ==================================================
        base_profile = self.repo.load()

        # ==================================================
        # 2️⃣ driver type selects the initial profile only. Human feedback is
        # handled by TriggerManager/ProfileLearner below so it is not applied
        # repeatedly before every LLM call.
        # ==================================================
        learner_enabled = (
            self.profile_learner is not None
            and bool(getattr(self.profile_learner, "enabled", True))
        )
        if learner_enabled:
            base_profile = update_profile(base_profile, driver_type, "")

        # ==================================================
        # 3️⃣ 场景感知 profile
        # ==================================================
        scene_profile = materialize_profile_for_scene(base_profile, scene)

        # ==================================================
        # 4️⃣ RAG v1 evidence pack
        # ==================================================
        # 新逻辑：RAG 检索已经在 runner_core 的 RAGOrchestrator 中完成。
        # service.step 只消费 evidence_pack，不再主动 retrieve。
        if evidence_pack is None:
            evidence_pack = {
                "budget": "reactive",
                "num_evidence": 0,
                "items": [],
                "evidence_text": "No RAG evidence provided.",
            }

        evidence_items = evidence_pack.get("items", []) or []
        evidence_text = evidence_pack.get("evidence_text", "No RAG evidence provided.")

        available_evidence_ids = [
            str(x.get("evidence_id"))
            for x in evidence_items
            if isinstance(x, dict) and x.get("evidence_id") is not None
        ]

        # ==================================================
        # 5️⃣ 构造 LLM prompt
        # ==================================================
        legacy_empty_evidence = {
            "laws": [],
            "cases": [],
            "scenarios": [],
        }

        system, user = make_evidence_prompts(
            profile=scene_profile,
            scene=scene,
            human_feedback=feedback,

            # 关键：不要传 None，避免旧 prompt 函数里 evidence["laws"] 崩溃
            evidence=legacy_empty_evidence,

            # 关键：这里仍然不让 make_evidence_prompts 注入 RAG v1，
            # RAG v1 evidence 继续由 service.step 后面手动追加
            evidence_pack=None,

            planning_hint="",
            planning_metadata=None,
            frame_safety=frame_safety,
            token_budget_class="reactive_low",
            require_grounded_decision=require_grounded_decision,
        )

        # ==================================================
        # 5.1️⃣ RAG v1 Evidence Injection
        # ==================================================
        user += f"""

    [RAG Evidence Pack]

    The following retrieved evidence may support the driving-risk decision.
    Use evidence only when it is relevant to the current scene.

    {evidence_text}

    Available evidence IDs:
    {available_evidence_ids}

    Evidence citation rules:
    - If you use any evidence, cite its evidence_id in used_evidence_ids.
    - used_evidence_ids must only contain IDs from Available evidence IDs.
    - Do not invent evidence IDs.
    - If no evidence is useful, set used_evidence_ids=[].
    - If evidence is weak or irrelevant, set evidence_support_level="none" or "weak".
    - The latest observation and physical safety metrics override stale or irrelevant evidence.
    """

        if require_grounded_decision and available_evidence_ids:
            user += """

    [Grounding Requirement]

    Relevant evidence is available.
    If it supports your decision, cite at least one valid evidence_id.
    If you choose not to use evidence, set used_evidence_ids=[] and explain why it is not relevant.
    """

        # ==================================================
        # 5.2️⃣ Planning Hint Injection
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

        # ==================================================
        # 5.3️⃣ Reactive + JSON Schema Constraints
        # ==================================================
        user += """

    [Reactive Thread Constraints]

    You are the fast Reactive Thread.
    You operate under strict real-time and token-budget constraints.
    Prefer a safe feasible decision quickly.
    The decision does not need to be globally optimal.
    Use compact reasoning.

    Return ONLY valid JSON.

    Required JSON schema:
    {
    "is_potential_violation": true or false,
    "risk_level": "low | medium | high | unknown",
    "recommended_action": "short action recommendation",
    "warning": "short warning if needed",
    "reason": "brief reason grounded in the latest scene, safety metrics, and useful evidence",
    "confidence": 0.0,
    "used_evidence_ids": [],
    "evidence_support_level": "strong | medium | weak | none"
    }

    Additional requirements:
    - used_evidence_ids must be a list.
    - evidence_support_level must be one of: strong, medium, weak, none.
    - Do not include markdown.
    - Do not include text outside JSON.
    """

        # ==================================================
        # 6️⃣ LLM decision
        # ==================================================
        decision = self.llm.complete_json(system, user)
        decision = validate_decision_json(decision)

        # ==================================================
        # 6.1️⃣ Decision schema repair for RAG fields
        # ==================================================
        # 注意：这里只做字段补齐，不做 grounding validation。
        # grounding validation 应在 runner_core 中完成，避免吞掉 hallucinated citations。
        if "used_evidence_ids" not in decision or not isinstance(decision.get("used_evidence_ids"), list):
            decision["used_evidence_ids"] = []
        else:
            decision["used_evidence_ids"] = [
                str(x) for x in decision["used_evidence_ids"] if x is not None
            ]

        support = str(decision.get("evidence_support_level", "none")).lower()
        if support not in {"strong", "medium", "weak", "none"}:
            support = "medium" if decision["used_evidence_ids"] else "none"
        decision["evidence_support_level"] = support

        # ==================================================
        # 7️⃣ Trigger 评估
        # ==================================================
        if self.trigger_manager is not None:
            triggers, guardrails, profile_update = self.trigger_manager.evaluate(
                scene=scene,
                profile=scene_profile,
                decision=decision,
                human_feedback=feedback,
                recent_decisions=recent_decisions,
            )
        else:
            triggers, guardrails, profile_update = [], {}, {}

        # ==================================================
        # 8️⃣ 分层学习
        # ==================================================
        if learner_enabled and allow_profile_update:
            updated_profile = self.profile_learner.apply(
                profile=base_profile,
                triggers=triggers,
                profile_update=profile_update,
                decision=decision,
            )
        else:
            updated_profile = base_profile

        # ==================================================
        # 9️⃣ 写回 runtime profile
        # ==================================================
        self.repo.save(updated_profile)

        # ==================================================
        # 🔟 trigger 生命周期
        # ==================================================
        if self.trigger_state_store is not None:
            self.trigger_state_store.add(triggers)
            self.trigger_state_store.tick_frame()

        # ==================================================
        # 1️⃣1️⃣ RAG evidence 平铺，兼容旧 rules 接口
        # ==================================================
        flat_rules = [
            self._serialize_rag_evidence_item(x)
            for x in evidence_items
            if isinstance(x, dict)
        ]

        laws = [
            self._serialize_rag_evidence_item(x)
            for x in evidence_items
            if isinstance(x, dict) and x.get("doc_type") == "law"
        ]

        cases = [
            self._serialize_rag_evidence_item(x)
            for x in evidence_items
            if isinstance(x, dict) and x.get("doc_type") == "case"
        ]

        scenarios = [
            self._serialize_rag_evidence_item(x)
            for x in evidence_items
            if isinstance(x, dict) and x.get("doc_type") in {"scenario", "safety", "policy"}
        ]

        # ==================================================
        # 1️⃣2️⃣ 返回 StepResult
        # ==================================================
        return StepResult(
            profile=updated_profile,
            rules=flat_rules,
            decision=decision,
            triggers=[self._safe_dict(t) for t in triggers],
            guardrails=self._safe_dict(guardrails),
            profile_update=profile_update,
            evidence={
                # 新 RAG v1 字段
                "rag_evidence_pack": evidence_pack,
                "num_rag_evidence": evidence_pack.get("num_evidence", 0),
                "available_evidence_ids": available_evidence_ids,
                "used_evidence_ids": decision.get("used_evidence_ids", []),
                "evidence_support_level": decision.get("evidence_support_level", "none"),

                # 兼容旧接口，避免 runner_core 读取 laws/cases/scenarios 报错
                "laws": laws,
                "cases": cases,
                "scenarios": scenarios,
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
        if item is None:
            return {}

        if isinstance(item, dict):
            return {
                "evidence_id": item.get("evidence_id"),
                "chunk_id": item.get("evidence_id"),
                "doc_type": item.get("doc_type"),
                "title": item.get("title"),
                "text": item.get("text"),
                "source": item.get("source"),
                "dataset_tags": item.get("dataset_tags", []),
                "scenario_tags": item.get("scenario_tags", []),
                "metric_tags": item.get("metric_tags", []),
                "risk_tags": item.get("risk_tags", []),
                "score": item.get("score"),
                "rerank_score": item.get("rerank_score"),
            }

        if hasattr(item, "__dict__"):
            data = dict(item.__dict__)
            if "doc" in data and hasattr(data["doc"], "__dict__"):
                data["doc"] = dict(data["doc"].__dict__)
            return data
        return {"text": str(item)}

    def _serialize_rag_evidence_item(self, item):
        """
        RAG v1 evidence pack item serializer.

        Keep the legacy log shape usable while preserving grounding fields.
        """
        if item is None:
            return {}

        if isinstance(item, dict):
            evidence_id = item.get("evidence_id") or item.get("chunk_id") or item.get("id")
            return {
                "id": evidence_id,
                "evidence_id": evidence_id,
                "chunk_id": item.get("chunk_id") or evidence_id,
                "doc_type": item.get("doc_type") or item.get("kb_type"),
                "title": item.get("title"),
                "text": item.get("text"),
                "source": item.get("source"),
                "jurisdiction": item.get("jurisdiction"),
                "dataset_tags": item.get("dataset_tags", []),
                "scenario_tags": item.get("scenario_tags", []),
                "metric_tags": item.get("metric_tags", []),
                "risk_tags": item.get("risk_tags", []),
                "condition": item.get("condition", {}),
                "risk_mechanism": item.get("risk_mechanism", ""),
                "recommended_action": item.get("recommended_action", []),
                "forbidden_action": item.get("forbidden_action", []),
                "severity": item.get("severity", ""),
                "score": item.get("score"),
                "rerank_score": item.get("rerank_score"),
            }

        return self._serialize_evidence_item(item)

    def _safe_dict(self, obj):
        if isinstance(obj, dict):
            return obj
        if hasattr(obj, "__dict__"):
            return dict(obj.__dict__)
        return {"value": str(obj)}
