from .knowledge_schema import normalize_doc


def rerank_evidence(docs, rag_query: dict):
    dataset = str(rag_query.get("dataset", "")).lower()
    scenario = str(rag_query.get("scenario_type", "")).lower()
    metrics = {str(x).lower() for x in rag_query.get("metric_terms", [])}
    primary_metrics = {str(x).lower() for x in rag_query.get("primary_metrics", [])}
    risk_focus = {str(x).lower() for x in rag_query.get("risk_focus", [])}

    scored = []

    for raw in docs or []:
        obj = normalize_doc(raw)

        score = 0.0
        if obj.get("score") is not None:
            try:
                score += float(obj.get("score"))
            except Exception:
                pass

        dataset_tags = {str(x).lower() for x in obj.get("dataset_tags", [])}
        scenario_tags = {str(x).lower() for x in obj.get("scenario_tags", [])}
        metric_tags = {str(x).lower() for x in obj.get("metric_tags", [])}
        risk_tags = {str(x).lower() for x in obj.get("risk_tags", [])}

        if dataset and dataset in dataset_tags:
            score += 0.25
        if scenario and scenario in scenario_tags:
            score += 0.35

        if metrics and metric_tags.intersection(metrics):
            score += 0.35
        elif primary_metrics and metric_tags.intersection(primary_metrics):
            score += 0.20

        if risk_focus and risk_tags.intersection(risk_focus):
            score += 0.20

        if obj.get("doc_type") == "law":
            score += 0.15
        if obj.get("doc_type") == "safety":
            score += 0.15
        if obj.get("doc_type") == "scenario":
            score += 0.10

        try:
            score += 0.03 * int(obj.get("priority", 1))
        except Exception:
            pass

        obj["rerank_score"] = score
        scored.append(obj)

    scored.sort(key=lambda x: x.get("rerank_score", 0.0), reverse=True)
    return scored