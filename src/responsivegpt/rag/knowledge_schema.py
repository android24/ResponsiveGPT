from dataclasses import dataclass, asdict
from typing import Any, Dict, List


@dataclass
class KnowledgeChunk:
    chunk_id: str
    doc_type: str          # law / scenario / safety / case / policy
    title: str
    text: str

    dataset_tags: List[str]
    scenario_tags: List[str]
    metric_tags: List[str]
    risk_tags: List[str]

    source: str = ""
    jurisdiction: str = ""
    priority: int = 1

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def normalize_doc(doc) -> Dict[str, Any]:
    """
    兼容已有 KnowledgeBase / HybridRetriever 返回的 dict 或对象。
    """
    if doc is None:
        return {}

    if isinstance(doc, dict):
        obj = dict(doc)
    elif hasattr(doc, "__dict__"):
        obj = dict(doc.__dict__)
    else:
        obj = {"text": str(doc)}

    if obj.get("doc") is not None:
        inner = normalize_doc(obj.get("doc"))
        for score_key in ("score", "semantic_score", "metadata_score", "priority_score", "final_score", "rerank_score"):
            if obj.get(score_key) is not None:
                inner[score_key] = obj.get(score_key)
        if inner.get("score") is None and inner.get("final_score") is not None:
            inner["score"] = inner.get("final_score")
        return inner

    text = (
        obj.get("text")
        or obj.get("content")
        or obj.get("value")
        or obj.get("body")
        or ""
    )

    chunk_id = (
        obj.get("chunk_id")
        or obj.get("doc_id")
        or obj.get("id")
        or obj.get("title")
        or str(abs(hash(text)) % 10_000_000)
    )

    return {
        "chunk_id": str(chunk_id),
        "doc_type": obj.get("doc_type") or obj.get("kb_type") or "unknown",
        "title": obj.get("title", ""),
        "text": str(text),
        "dataset_tags": obj.get("dataset_tags", []) or [],
        "scenario_tags": obj.get("scenario_tags", []) or [],
        "metric_tags": obj.get("metric_tags", []) or [],
        "risk_tags": obj.get("risk_tags", []) or [],
        "source": obj.get("source", ""),
        "jurisdiction": obj.get("jurisdiction", ""),
        "priority": int(obj.get("priority", 1) or 1),
        "condition": obj.get("condition", {}) or {},
        "risk_mechanism": obj.get("risk_mechanism", "") or "",
        "recommended_action": obj.get("recommended_action", []) or [],
        "forbidden_action": obj.get("forbidden_action", []) or [],
        "severity": obj.get("severity", "") or "",
        "score": obj.get("score") if obj.get("score") is not None else obj.get("final_score"),
        "semantic_score": obj.get("semantic_score"),
        "metadata_score": obj.get("metadata_score"),
        "priority_score": obj.get("priority_score"),
        "rerank_score": obj.get("rerank_score"),
    }
