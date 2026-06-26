from .rag_query_builder import build_rag_query
from .evidence_reranker import rerank_evidence
from .evidence_pack import build_evidence_pack


class RAGOrchestrator:
    def __init__(
        self,
        retriever,
        *,
        rag_mode: str = "full",
        budget: str = "reactive",
        top_k: int = 12,
    ):
        self.retriever = retriever
        self.rag_mode = rag_mode
        self.budget = budget
        self.top_k = top_k

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

        retrieved = self._retrieve(
            query_text=rag_query["query_text"],
            scene=scene,
            profile=profile,
            top_k=self.top_k,
        )
        reranked = rerank_evidence(retrieved, rag_query)
        evidence_pack = build_evidence_pack(reranked, budget=self.budget)

        return {
            "rag_mode": self.rag_mode,
            "rag_query": rag_query,
            "retrieved": _compact_docs(retrieved),
            "reranked": _compact_docs(reranked),
            "evidence_pack": evidence_pack,
        }

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