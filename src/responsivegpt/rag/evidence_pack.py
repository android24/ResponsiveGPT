from .knowledge_schema import normalize_doc


PREFERRED_DOC_TYPES = ("law", "case", "scenario", "safety", "policy")


def build_evidence_pack(docs, *, budget: str = "reactive") -> dict:
    if budget == "reactive_low":
        max_docs = 3
        max_chars_each = 220
    elif budget == "reactive_medium":
        max_docs = 4
        max_chars_each = 280
    elif budget == "reactive":
        max_docs = 5
        max_chars_each = 360
    elif budget == "planning":
        max_docs = 8
        max_chars_each = 650
    elif budget == "episode":
        max_docs = 12
        max_chars_each = 850
    else:
        max_docs = 5
        max_chars_each = 360

    normalized_docs = [normalize_doc(d) for d in (docs or [])]
    selected_docs = _select_diverse_docs(normalized_docs, max_docs=max_docs)
    items = []

    for obj in selected_docs:
        text = obj.get("text", "")
        if len(text) > max_chars_each:
            text = text[:max_chars_each] + "..."

        items.append({
            "evidence_id": obj.get("chunk_id"),
            "doc_type": obj.get("doc_type"),
            "title": obj.get("title"),
            "text": text,
            "source": obj.get("source"),
            "jurisdiction": obj.get("jurisdiction"),
            "dataset_tags": obj.get("dataset_tags", []),
            "scenario_tags": obj.get("scenario_tags", []),
            "metric_tags": obj.get("metric_tags", []),
            "risk_tags": obj.get("risk_tags", []),
            "condition": obj.get("condition", {}),
            "risk_mechanism": obj.get("risk_mechanism", ""),
            "recommended_action": obj.get("recommended_action", []),
            "forbidden_action": obj.get("forbidden_action", []),
            "severity": obj.get("severity", ""),
            "score": obj.get("score"),
            "rerank_score": obj.get("rerank_score"),
        })

    return {
        "budget": budget,
        "num_evidence": len(items),
        "items": items,
        "evidence_text": format_evidence_for_prompt(
            items,
            compact=budget in {"reactive_low", "reactive_medium"},
        ),
    }


def _select_diverse_docs(docs: list[dict], *, max_docs: int) -> list[dict]:
    selected = []
    selected_ids = set()

    for doc_type in PREFERRED_DOC_TYPES:
        for obj in docs:
            chunk_id = obj.get("chunk_id")
            if chunk_id in selected_ids:
                continue
            if obj.get("doc_type") == doc_type:
                selected.append(obj)
                selected_ids.add(chunk_id)
                break
        if len(selected) >= max_docs:
            return selected

    for obj in docs:
        chunk_id = obj.get("chunk_id")
        if chunk_id in selected_ids:
            continue
        selected.append(obj)
        selected_ids.add(chunk_id)
        if len(selected) >= max_docs:
            break

    return selected


def format_evidence_for_prompt(items, *, compact: bool = False) -> str:
    if not items:
        return "No external evidence retrieved."

    lines = []
    for i, item in enumerate(items, 1):
        evidence_id = item.get("evidence_id")
        doc_type = item.get("doc_type")
        title = item.get("title")
        text = item.get("text")
        risk_mechanism = item.get("risk_mechanism")
        recommended_action = item.get("recommended_action") or []
        forbidden_action = item.get("forbidden_action") or []
        severity = item.get("severity")

        details = []
        if compact:
            if recommended_action:
                details.append(f"recommended_action={recommended_action}")
        elif severity:
            details.append(f"severity={severity}")
        if not compact and risk_mechanism:
            details.append(f"risk_mechanism={risk_mechanism}")
        if not compact and recommended_action:
            details.append(f"recommended_action={recommended_action}")
        if not compact and forbidden_action:
            details.append(f"forbidden_action={forbidden_action}")

        detail_text = "\n" + "\n".join(details) if details else ""

        lines.append(
            f"[E{i}] evidence_id={evidence_id}; type={doc_type}; title={title}\n{text}{detail_text}"
        )

    return "\n\n".join(lines)
