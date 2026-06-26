import argparse
import json
from pathlib import Path

from .io_utils import load_json


def _grounding_from_output(record: dict) -> dict:
    evidence_pack = record.get("evidence_pack", {}) or {}
    decision = record.get("decision", {}) or {}
    available_ids = {
        str(item.get("evidence_id"))
        for item in evidence_pack.get("items", [])
        if isinstance(item, dict) and item.get("evidence_id") is not None
    }
    used_ids = [
        str(value)
        for value in decision.get("used_evidence_ids", [])
        if value is not None
    ]
    valid_ids = [value for value in used_ids if value in available_ids]
    invalid_ids = [value for value in used_ids if value not in available_ids]
    return {
        "used_evidence_ids": used_ids,
        "valid_used_evidence_ids": valid_ids,
        "hallucinated_evidence_ids": invalid_ids,
        "is_grounded": bool(valid_ids) and not invalid_ids,
    }


def compute_metrics_from_decisions(decisions_path: str | Path) -> dict:
    decisions_path = Path(decisions_path)
    total = 0
    retrieved = 0
    used = 0
    grounded = 0
    raw_invalid_frames = 0
    output_invalid_frames = 0
    output_valid_citations = 0
    output_citations = 0
    evidence_total = 0

    with decisions_path.open("r", encoding="utf-8") as stream:
        for line in stream:
            if not line.strip():
                continue
            record = json.loads(line)
            total += 1
            evidence_pack = record.get("evidence_pack", {}) or {}
            raw_grounding = record.get("grounding", {}) or {}
            output_grounding = (
                record.get("output_grounding")
                or _grounding_from_output(record)
            )

            evidence_count = int(evidence_pack.get("num_evidence", 0) or 0)
            evidence_total += evidence_count
            retrieved += int(evidence_count > 0)
            used += int(bool(output_grounding.get("used_evidence_ids")))
            grounded += int(bool(output_grounding.get("is_grounded")))
            raw_invalid_frames += int(
                bool(raw_grounding.get("hallucinated_evidence_ids"))
            )
            output_invalid_frames += int(
                bool(output_grounding.get("hallucinated_evidence_ids"))
            )
            output_valid_citations += len(
                output_grounding.get("valid_used_evidence_ids") or []
            )
            output_citations += len(
                output_grounding.get("used_evidence_ids") or []
            )

    if total == 0:
        raise ValueError(f"No decision records found: {decisions_path}")

    output_invalid_rate = output_invalid_frames / total
    return {
        "retrieval_coverage": retrieved / total,
        "evidence_usage_rate": used / total,
        "grounded_decision_rate": grounded / total,
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


def backfill_run(run_dir: str | Path) -> dict:
    run_dir = Path(run_dir)
    summary_path = run_dir / "summary.json"
    decisions_path = run_dir / "decisions.jsonl"
    if not summary_path.exists():
        raise FileNotFoundError(summary_path)
    if not decisions_path.exists():
        raise FileNotFoundError(decisions_path)

    summary = load_json(summary_path)
    metrics = compute_metrics_from_decisions(decisions_path)
    summary["global_rag"] = metrics

    temp_path = summary_path.with_suffix(".json.tmp")
    temp_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    temp_path.replace(summary_path)
    return metrics


def main():
    parser = argparse.ArgumentParser(
        description="Backfill final-output RAG metrics from an existing run."
    )
    parser.add_argument("--run_dir", required=True)
    args = parser.parse_args()
    metrics = backfill_run(args.run_dir)
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
