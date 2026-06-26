def compute_rag_metrics(frame_records):
    total = len(frame_records)
    if total == 0:
        return {
            "retrieval_coverage": 0.0,
            "evidence_usage_rate": 0.0,
            "grounded_decision_rate": 0.0,
            "hallucinated_citation_rate": 0.0,
            "raw_invalid_citation_attempt_rate": 0.0,
            "output_invalid_citation_frame_rate": 0.0,
            "citation_precision": 0.0,
            "avg_evidence_per_frame": 0.0,
        }

    retrieved = 0
    used = 0
    grounded = 0
    raw_invalid_frames = 0
    output_invalid_frames = 0
    output_valid_citations = 0
    output_citations = 0
    evidence_total = 0

    for r in frame_records:
        ep = r.get("evidence_pack", {}) or {}
        raw_grounding = r.get("grounding", {}) or {}
        output_grounding = r.get("output_grounding", raw_grounding) or {}

        n = ep.get("num_evidence", 0) or 0
        evidence_total += n

        if n > 0:
            retrieved += 1
        if output_grounding.get("used_evidence_ids"):
            used += 1
        if output_grounding.get("is_grounded"):
            grounded += 1
        if raw_grounding.get("hallucinated_evidence_ids"):
            raw_invalid_frames += 1
        if output_grounding.get("hallucinated_evidence_ids"):
            output_invalid_frames += 1

        output_valid_citations += len(output_grounding.get("valid_used_evidence_ids") or [])
        output_citations += len(output_grounding.get("used_evidence_ids") or [])

    output_invalid_rate = output_invalid_frames / total

    return {
        "retrieval_coverage": retrieved / total,
        "evidence_usage_rate": used / total,
        "grounded_decision_rate": grounded / total,
        # Backward-compatible name: this now describes invalid citations that
        # remain in the final decision after any grounding repair.
        "hallucinated_citation_rate": output_invalid_rate,
        "raw_invalid_citation_attempt_rate": raw_invalid_frames / total,
        "output_invalid_citation_frame_rate": output_invalid_rate,
        "citation_precision": (
            output_valid_citations / output_citations
            if output_citations
            else 0.0
        ),
        "avg_evidence_per_frame": evidence_total / total,
    }
