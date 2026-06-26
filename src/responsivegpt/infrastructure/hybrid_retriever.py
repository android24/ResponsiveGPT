import math
from typing import Optional

from ..domain.evidence import KnowledgeDoc, RankedKnowledgeDoc, EvidenceBundle
from ..domain.models import SceneState, DriverProfile
from ..domain.ports import Embedder
from .knowledge_base import KnowledgeBase


RAG_DOC_TYPES = ("law", "case", "scenario", "policy", "safety")


def cosine(a: list[float], b: list[float]) -> float:
    dot = 0.0
    na = 0.0
    nb = 0.0
    for x, y in zip(a, b):
        dot += x * y
        na += x * x
        nb += y * y
    denom = math.sqrt(na) * math.sqrt(nb)
    return dot / denom if denom else 0.0


class HybridRetriever:
    def __init__(self, kb: KnowledgeBase, embedder: Embedder):
        self.kb = kb
        self.embedder = embedder
        self._doc_embeddings = {}   # doc_id -> embedding

    def _embed_doc(self, doc: KnowledgeDoc) -> list[float]:
        if doc.id not in self._doc_embeddings:
            text = f"{doc.title}\n{doc.text}"
            self._doc_embeddings[doc.id] = self.embedder.embed(text)
        return self._doc_embeddings[doc.id]

    def _metadata_score(
        self,
        doc: KnowledgeDoc,
        scene_type: Optional[str],
        event_type: Optional[str],
        pair_type: Optional[str],
    ) -> float:
        score = 0.0

        if scene_type and doc.scene_type in [scene_type, "all"]:
            score += 0.4
        elif doc.scene_type is None:
            score += 0.1

        if event_type and doc.event_type in [event_type, "all"]:
            score += 0.3
        elif doc.event_type is None:
            score += 0.05

        if pair_type and doc.pair_type in [pair_type, "all"]:
            score += 0.3
        elif doc.pair_type is None:
            score += 0.05

        return score

    def _priority_score(self, doc: KnowledgeDoc) -> float:
        # 归一化为 0~1 附近
        return max(0.0, min(1.0, float(doc.priority)))

    def _rank_docs(
        self,
        docs: list[KnowledgeDoc],
        semantic_text: str,
        scene_type: Optional[str],
        event_type: Optional[str],
        pair_type: Optional[str],
        top_k: int,
    ) -> list[RankedKnowledgeDoc]:
        if not docs:
            return []

        q_emb = self.embedder.embed(semantic_text)
        ranked = []

        for doc in docs:
            d_emb = self._embed_doc(doc)
            semantic_score = cosine(q_emb, d_emb)
            metadata_score = self._metadata_score(doc, scene_type, event_type, pair_type)
            priority_score = self._priority_score(doc)

            final_score = (
                0.65 * semantic_score +
                0.20 * metadata_score +
                0.15 * priority_score
            )

            ranked.append(
                RankedKnowledgeDoc(
                    doc=doc,
                    semantic_score=semantic_score,
                    metadata_score=metadata_score,
                    priority_score=priority_score,
                    final_score=final_score,
                )
            )

        ranked.sort(key=lambda x: x.final_score, reverse=True)
        return ranked[:top_k]
    
    def normalize_scene(self, scene: SceneState):
        event = str(getattr(scene, "event_type", None) or "").lower()
        dataset_raw = getattr(scene, "scene_type", None) or "custom"
        dataset_key = str(dataset_raw).lower()

        if dataset_key == "highd":
            dataset = "highD"
        elif dataset_key == "ind":
            dataset = "inD"
        elif dataset_key in {"round", "roundd", "roundabout"}:
            dataset = "rounD"
        else:
            dataset = str(dataset_raw)

        if dataset == "highD":
            pair_type = "vehicle_vehicle"
            if "following" in event:
                event_type = "car_following"
            elif "cut" in event:
                event_type = "cut_in"
            elif "lane" in event:
                event_type = "lane_change"
            else:
                event_type = "highway_interaction"

        elif dataset == "inD":
            if "pedestrian" in event:
                pair_type = "vehicle_pedestrian"
                event_type = "car_pedestrian"
            elif "bicycle" in event or "cyclist" in event:
                pair_type = "vehicle_cyclist"
                event_type = "car_bicycle"
            else:
                pair_type = "vehicle_vehicle"
                event_type = "intersection_vehicle"

        elif dataset == "rounD":
            if "pedestrian" in event:
                pair_type = "vehicle_pedestrian"
            elif "bicycle" in event or "cyclist" in event or getattr(scene, "vrus_present", False):
                pair_type = "vehicle_cyclist"
            else:
                pair_type = "vehicle_vehicle"
            event_type = event or "roundabout_interaction"
        else:
            event_type = event or None
            pair_type = None

        return dataset, event_type, pair_type

    def build_query(
        self,
        scene: SceneState,
        profile: DriverProfile,
    ) -> dict:
        scene_type, event_type, pair_type = self.normalize_scene(scene)
        profile_obj = profile or {}

        if isinstance(profile_obj, dict):
            driver_type = profile_obj.get("driver_type", "unknown")
            global_profile = profile_obj.get("global", {}) or {}
            safety_weight = global_profile.get("safety_weight", profile_obj.get("safety_weight", 0.65))
            efficiency_weight = global_profile.get("efficiency_weight", profile_obj.get("efficiency_weight", 0.35))
        else:
            driver_type = getattr(profile_obj, "driver_type", "unknown")
            safety_weight = getattr(profile_obj, "safety_weight", 0.65)
            efficiency_weight = getattr(profile_obj, "efficiency_weight", 0.35)

        semantic_text = (
            f"场景类型为 {scene_type}。"
            f"事件类型为 {event_type or 'unknown'}。"
            f"自车速度 {scene.ego_speed_mps:.2f} m/s。"
            f"与交互对象距离 {scene.headway_m:.2f} m。"
            f"相对速度 {scene.rel_speed_mps if scene.rel_speed_mps is not None else 'unknown'} m/s。"
            f"驾驶风格为 {driver_type}，"
            f"安全权重 {float(safety_weight):.2f}，效率权重 {float(efficiency_weight):.2f}。"
            f"请检索与该交通场景相关的法规、案例、场景知识、策略约束与安全机制，"
            f"用于判断是否存在潜在违规、高风险交互，以及推荐采取的驾驶策略。"
        )

        return {
            "semantic_text": semantic_text,
            "scene_type": scene_type,
            "event_type": event_type,
            "pair_type": pair_type,
        }

    def retrieve(
        self,
        scene: SceneState,
        profile: DriverProfile,
        law_top_k: int = 3,
        case_top_k: int = 3,
        scenario_top_k: int = 3,
    ) -> EvidenceBundle:
        q = self.build_query(scene, profile)

        laws = self.kb.filter_docs(
            kb_type="law",
            scene_type=q["scene_type"],
            event_type=q["event_type"],
            pair_type=q["pair_type"],
        )
        cases = self.kb.filter_docs(
            kb_type="case",
            scene_type=q["scene_type"],
            event_type=q["event_type"],
            pair_type=q["pair_type"],
        )
        scenarios = self.kb.filter_docs(
            kb_type="scenario",
            scene_type=q["scene_type"],
            event_type=q["event_type"],
            pair_type=q["pair_type"],
        )

        ranked_laws = self._rank_docs(
            docs=laws,
            semantic_text=q["semantic_text"],
            scene_type=q["scene_type"],
            event_type=q["event_type"],
            pair_type=q["pair_type"],
            top_k=law_top_k,
        )
        ranked_cases = self._rank_docs(
            docs=cases,
            semantic_text=q["semantic_text"],
            scene_type=q["scene_type"],
            event_type=q["event_type"],
            pair_type=q["pair_type"],
            top_k=case_top_k,
        )
        ranked_scenarios = self._rank_docs(
            docs=scenarios,
            semantic_text=q["semantic_text"],
            scene_type=q["scene_type"],
            event_type=q["event_type"],
            pair_type=q["pair_type"],
            top_k=scenario_top_k,
        )

        return EvidenceBundle(
            laws=ranked_laws,
            cases=ranked_cases,
            scenarios=ranked_scenarios,
        )
    
    def retrieve_docs_for_rag(
        self,
        scene: SceneState,
        profile: DriverProfile,
        top_k: int = 12,
    ) -> list[dict]:
        """
        RAG v1 专用接口，返回法规、案例、场景、策略、安全五类证据。
        """
        q = self.build_query(scene, profile)
        per_type_k = max(2, math.ceil(max(1, top_k) / len(RAG_DOC_TYPES)))
        docs = []

        for doc_type in RAG_DOC_TYPES:
            candidates = self._filter_docs_relaxed(
                kb_type=doc_type,
                scene_type=q["scene_type"],
                event_type=q["event_type"],
                pair_type=q["pair_type"],
            )
            ranked = self._rank_docs(
                docs=candidates,
                semantic_text=q["semantic_text"],
                scene_type=q["scene_type"],
                event_type=q["event_type"],
                pair_type=q["pair_type"],
                top_k=per_type_k,
            )
            docs.extend(self._docs_to_rag_items(ranked, doc_type=doc_type))

        docs.sort(key=lambda x: float(x.get("score") or 0.0), reverse=True)
        return docs[:top_k]

    def _filter_docs_relaxed(
        self,
        *,
        kb_type: str,
        scene_type: Optional[str],
        event_type: Optional[str],
        pair_type: Optional[str],
    ) -> list[KnowledgeDoc]:
        attempts = [
            {"scene_type": scene_type, "event_type": event_type, "pair_type": pair_type},
            {"scene_type": scene_type, "event_type": event_type, "pair_type": None},
            {"scene_type": scene_type, "event_type": None, "pair_type": None},
            {"scene_type": None, "event_type": None, "pair_type": None},
        ]

        for params in attempts:
            docs = self.kb.filter_docs(kb_type=kb_type, **params)
            if docs:
                return docs

        return []


    def _flatten_bundle(self, bundle) -> list[dict]:
        docs = []

        docs.extend(self._docs_to_rag_items(
            getattr(bundle, "laws", []),
            doc_type="law",
        ))

        docs.extend(self._docs_to_rag_items(
            getattr(bundle, "cases", []),
            doc_type="case",
        ))

        docs.extend(self._docs_to_rag_items(
            getattr(bundle, "scenarios", []),
            doc_type="scenario",
        ))

        return docs


    def _docs_to_rag_items(self, docs, doc_type: str) -> list[dict]:
        out = []

        for i, d in enumerate(docs or []):
            obj = self._doc_to_dict(d)
            item_doc_type = obj.get("doc_type") or obj.get("kb_type") or doc_type

            text = (
                obj.get("text")
                or obj.get("content")
                or obj.get("body")
                or obj.get("description")
                or obj.get("rule")
                or ""
            )

            title = (
                obj.get("title")
                or obj.get("name")
                or item_doc_type
            )

            chunk_id = (
                obj.get("chunk_id")
                or obj.get("doc_id")
                or obj.get("id")
                or f"{doc_type}_{abs(hash(str(text))) % 10_000_000}_{i}"
            )

            item = {
                "chunk_id": str(chunk_id),
                "doc_type": item_doc_type,
                "title": str(title),
                "text": str(text),

                # 这些 tag 如果原始 doc 有就保留，没有就给空
                "dataset_tags": obj.get("dataset_tags", []),
                "scenario_tags": obj.get("scenario_tags", []),
                "metric_tags": obj.get("metric_tags", []),
                "risk_tags": obj.get("risk_tags", []),

                "source": obj.get("source", ""),
                "jurisdiction": obj.get("jurisdiction", ""),
                "priority": obj.get("priority", 1),
                "condition": obj.get("condition", {}) or {},
                "risk_mechanism": obj.get("risk_mechanism", ""),
                "recommended_action": obj.get("recommended_action", []) or [],
                "forbidden_action": obj.get("forbidden_action", []) or [],
                "severity": obj.get("severity", ""),

                # 保留原始检索分数
                "score": obj.get("score") if obj.get("score") is not None else obj.get("final_score"),
                "semantic_score": obj.get("semantic_score"),
                "metadata_score": obj.get("metadata_score"),
                "priority_score": obj.get("priority_score"),
            }

            # 如果你的旧 KB 字段叫 scene_type / event_type / pair_type，
            # 这里顺手映射成 tags，方便 RAG v1 rerank。
            scene_type = obj.get("scene_type")
            event_type = obj.get("event_type")
            pair_type = obj.get("pair_type")

            if scene_type and not item["dataset_tags"]:
                item["dataset_tags"] = [scene_type]

            scenario_tags = []
            if event_type:
                scenario_tags.append(event_type)
            if pair_type:
                scenario_tags.append(pair_type)

            if scenario_tags and not item["scenario_tags"]:
                item["scenario_tags"] = scenario_tags

            out.append(item)

        return out


    def _doc_to_dict(self, doc) -> dict:
        if doc is None:
            return {}

        if isinstance(doc, dict):
            obj = dict(doc)
            inner = obj.get("doc")
            if inner is not None:
                payload = self._doc_to_dict(inner)
                for score_key in ("score", "semantic_score", "metadata_score", "priority_score", "final_score"):
                    if obj.get(score_key) is not None:
                        payload[score_key] = obj.get(score_key)
                return payload
            return obj

        if hasattr(doc, "doc"):
            payload = self._doc_to_dict(getattr(doc, "doc"))
            for attr in ("semantic_score", "metadata_score", "priority_score", "final_score"):
                if hasattr(doc, attr):
                    payload[attr] = getattr(doc, attr)
            payload["score"] = payload.get("final_score")
            return payload

        if hasattr(doc, "__dict__"):
            return dict(doc.__dict__)

        return {
            "text": str(doc),
        }
