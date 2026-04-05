import math
from typing import Optional

from ..domain.evidence import KnowledgeDoc, RankedKnowledgeDoc, EvidenceBundle
from ..domain.models import SceneState, DriverProfile
from ..domain.ports import Embedder
from .knowledge_base import KnowledgeBase


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

    def build_query(
        self,
        scene: SceneState,
        profile: DriverProfile,
    ) -> dict:
        pair_type = None
        if scene.scene_type == "rounD":
            if scene.vrus_present:
                pair_type = "vehicle_cyclist"
            else:
                pair_type = "vehicle_vehicle"

        semantic_text = (
            f"场景类型为 {scene.scene_type}。"
            f"事件类型为 {scene.event_type or 'unknown'}。"
            f"自车速度 {scene.ego_speed_mps:.2f} m/s。"
            f"与交互对象距离 {scene.headway_m:.2f} m。"
            f"相对速度 {scene.rel_speed_mps if scene.rel_speed_mps is not None else 'unknown'} m/s。"
            f"驾驶风格为 {profile.driver_type}，"
            f"安全权重 {profile.safety_weight:.2f}，效率权重 {profile.efficiency_weight:.2f}。"
            f"请检索与该交通场景相关的法规、案例与风险模式，"
            f"用于判断是否存在潜在违规、高风险交互，以及推荐采取的驾驶策略。"
        )

        return {
            "semantic_text": semantic_text,
            "scene_type": scene.scene_type,
            "event_type": scene.event_type,
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