from .rag_query_builder import build_rag_query
from .evidence_reranker import rerank_evidence
from .evidence_pack import build_evidence_pack
from ..infrastructure.json_disk_cache import JsonDiskCache


class RAGOrchestrator:
    def __init__(
        self,
        retriever,
        *,
        rag_mode: str = "full",
        budget: str = "reactive",
        top_k: int = 12,
        cache_dir: str | None = None,
        cache_enabled: bool = True,
    ):
        self.retriever = retriever
        self.rag_mode = rag_mode
        self.budget = budget
        self.top_k = top_k
        self.cache = JsonDiskCache(cache_dir, enabled=cache_enabled)
        self.last_cache_hit = False

    def run(
        self,
        *,
        dataset: str,
        scene,
        frame_safety=None,
        metadata=None,
        driver_type: str = "",
        feedback: str = "",
        planning_hint: str = "",
        profile=None,
    ) -> dict:
        if self.rag_mode == "none" or self.retriever is None:
            return self._empty_result(dataset, scene, metadata)

        rag_query = build_rag_query(
            dataset=dataset,
            scene=scene,
            frame_safety=frame_safety,
            metadata=metadata,
            driver_type=driver_type,
            feedback=feedback,
            planning_hint=planning_hint,
            rag_mode=self.rag_mode,
        )
        cache_key = self._cache_key(
            dataset=dataset,
            rag_query=rag_query,
            scene=scene,
            frame_safety=frame_safety,
            metadata=metadata,
            driver_type=driver_type,
            feedback=feedback,
            planning_hint=planning_hint,
            profile=profile,
        )
        self.last_cache_hit = False
        cached = self.cache.get(cache_key)
        if isinstance(cached, dict):
            self.last_cache_hit = True
            cached["cache"] = {
                "hit": True,
                "key": cache_key,
                "type": "rag_evidence",
            }
            return cached

        retrieved = self._retrieve(
            query_text=rag_query["query_text"],
            scene=scene,
            profile=profile,
            top_k=self.top_k,
        )
        reranked = rerank_evidence(retrieved, rag_query)
        evidence_pack = build_evidence_pack(reranked, budget=self.budget)

        result = {
            "rag_mode": self.rag_mode,
            "rag_query": rag_query,
            "retrieved": _compact_docs(retrieved),
            "reranked": _compact_docs(reranked),
            "evidence_pack": evidence_pack,
            "cache": {
                "hit": False,
                "key": cache_key,
                "type": "rag_evidence",
            },
        }
        self.cache.set(
            cache_key,
            result,
            metadata={
                "cache_version": "rag_evidence_v1",
                "dataset": dataset,
                "rag_mode": self.rag_mode,
                "budget": self.budget,
                "top_k": self.top_k,
                "query_chars": len(rag_query.get("query_text", "") or ""),
            },
        )
        return result

    def _cache_key(
        self,
        *,
        dataset: str,
        rag_query: dict,
        scene,
        frame_safety=None,
        metadata=None,
        driver_type: str = "",
        feedback: str = "",
        planning_hint: str = "",
        profile=None,
    ) -> str:
        safety_payload = {}
        if frame_safety is not None:
            safety_payload = {
                "ttc_s": getattr(frame_safety, "ttc_s", None),
                "thw_s": getattr(frame_safety, "thw_s", None),
                "drac_mps2": getattr(frame_safety, "drac_mps2", None),
                "dcpa_m": getattr(frame_safety, "dcpa_m", None),
                "physical_risk_index": getattr(
                    frame_safety, "physical_risk_index", None
                ),
                "physical_risk_level": getattr(
                    frame_safety, "physical_risk_level", None
                ),
            }
        return JsonDiskCache.stable_hash({
            "cache_version": "rag_evidence_v1",
            "retrieval_fingerprint": getattr(
                self.retriever, "retrieval_fingerprint", ""
            ),
            "dataset": dataset,
            "rag_mode": self.rag_mode,
            "budget": self.budget,
            "top_k": self.top_k,
            "query_text": rag_query.get("query_text", ""),
            "driver_type": driver_type,
            "profile_signature": self._profile_signature(profile),
            "feedback": feedback,
            "planning_hint": planning_hint,
            "metadata": metadata or {},
            "scene": {
                "event_type": getattr(scene, "event_type", None),
                "frame_index": getattr(scene, "frame_index", None),
                "ego_speed_mps": getattr(scene, "ego_speed_mps", None),
                "rel_speed_mps": getattr(scene, "rel_speed_mps", None),
                "headway_m": getattr(scene, "headway_m", None),
                "vrus_present": getattr(scene, "vrus_present", None),
            },
            "frame_safety": safety_payload,
        })

    def _profile_signature(self, profile) -> dict:
        if profile is None:
            return {}
        if isinstance(profile, dict):
            global_profile = profile.get("global", {}) or {}
            return {
                "driver_type": profile.get("driver_type", "unknown"),
                "safety_weight": global_profile.get(
                    "safety_weight",
                    profile.get("safety_weight", 0.65),
                ),
                "efficiency_weight": global_profile.get(
                    "efficiency_weight",
                    profile.get("efficiency_weight", 0.35),
                ),
                "risk_sensitivity": global_profile.get(
                    "risk_sensitivity",
                    profile.get("risk_sensitivity"),
                ),
            }
        return {
            "driver_type": getattr(profile, "driver_type", "unknown"),
            "safety_weight": getattr(profile, "safety_weight", 0.65),
            "efficiency_weight": getattr(profile, "efficiency_weight", 0.35),
            "risk_sensitivity": getattr(profile, "risk_sensitivity", None),
        }

    def cache_stats(self) -> dict:
        return self.cache.stats()

    def _retrieve(self, *, query_text: str, scene, profile=None, top_k: int = 12):
        r = self.retriever

        if r is None:
            return []

        if r.__class__.__name__.lower().startswith("null"):
            return []

        # 最高优先级：适配你现在的 HybridRetriever
        if hasattr(r, "retrieve_docs_for_rag"):
            return r.retrieve_docs_for_rag(
                scene=scene,
                profile=profile,
                top_k=top_k,
            )

        # 兼容旧的 retrieve(scene, profile)
        if hasattr(r, "retrieve"):
            method = getattr(r, "retrieve")

            try:
                bundle = method(scene, profile)
                return self._flatten_external_bundle(bundle)
            except TypeError:
                pass

            # 兼容 query 风格 retriever
            try:
                return method(query_text)
            except TypeError:
                pass

        # 兼容 search 风格
        if hasattr(r, "search"):
            try:
                return r.search(query_text, top_k=top_k)
            except TypeError:
                return r.search(query_text)

        return []

    def _empty_result(self, dataset, scene, metadata):
        return {
            "rag_mode": "none",
            "rag_query": {
                "query_text": "",
                "dataset": dataset,
                "event_type": getattr(scene, "event_type", None),
                "metadata": metadata or {},
            },
            "retrieved": [],
            "reranked": [],
            "evidence_pack": {
                "budget": self.budget,
                "num_evidence": 0,
                "items": [],
                "evidence_text": "No RAG evidence because rag_mode=none.",
            },
        }
    
    def _flatten_external_bundle(self, bundle):
        if bundle is None:
            return []

        # 如果本来就是 list
        if isinstance(bundle, list):
            return bundle

        docs = []

        for doc_type, attr in [
            ("law", "laws"),
            ("case", "cases"),
            ("scenario", "scenarios"),
        ]:
            items = getattr(bundle, attr, []) or []

            for i, d in enumerate(items):
                if isinstance(d, dict):
                    obj = dict(d)
                elif hasattr(d, "__dict__"):
                    obj = dict(d.__dict__)
                else:
                    obj = {"text": str(d)}

                text = (
                    obj.get("text")
                    or obj.get("content")
                    or obj.get("body")
                    or obj.get("description")
                    or obj.get("rule")
                    or ""
                )

                docs.append({
                    "chunk_id": str(
                        obj.get("chunk_id")
                        or obj.get("doc_id")
                        or obj.get("id")
                        or f"{doc_type}_{abs(hash(str(text))) % 10_000_000}_{i}"
                    ),
                    "doc_type": doc_type,
                    "title": obj.get("title", doc_type),
                    "text": str(text),
                    "dataset_tags": obj.get("dataset_tags", []),
                    "scenario_tags": obj.get("scenario_tags", []),
                    "metric_tags": obj.get("metric_tags", []),
                    "risk_tags": obj.get("risk_tags", []),
                    "source": obj.get("source", ""),
                    "priority": obj.get("priority", 1),
                    "score": obj.get("score"),
                })

        return docs


def _compact_docs(docs):
    out = []
    for d in docs or []:
        if isinstance(d, dict):
            obj = dict(d)
        elif hasattr(d, "__dict__"):
            obj = dict(d.__dict__)
        else:
            obj = {"text": str(d)}

        text = str(obj.get("text") or obj.get("content") or obj.get("value") or "")
        if len(text) > 240:
            text = text[:240] + "..."

        out.append({
            "chunk_id": obj.get("chunk_id") or obj.get("doc_id") or obj.get("id"),
            "doc_type": obj.get("doc_type"),
            "title": obj.get("title"),
            "text": text,
            "score": obj.get("score"),
            "rerank_score": obj.get("rerank_score"),
            "dataset_tags": obj.get("dataset_tags", []),
            "scenario_tags": obj.get("scenario_tags", []),
            "metric_tags": obj.get("metric_tags", []),
        })
    return out
