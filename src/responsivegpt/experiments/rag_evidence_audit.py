import argparse
import csv
from collections import Counter, defaultdict
from pathlib import Path

from .io_utils import read_jsonl, write_csv


DOC_TYPES = ["law", "case", "scenario", "safety", "policy"]

RAG_EVIDENCE_SUMMARY_FIELDS = [
    "job_id",
    "dataset",
    "profile_name",
    "rag_variant",
    "planning_variant",
    "frames",
    "frames_with_evidence",
    "evidence_coverage",
    "grounded_frames",
    "grounded_rate",
    "law_coverage",
    "case_coverage",
    "scenario_coverage",
    "safety_coverage",
    "policy_coverage",
    "core_law_case_scenario_coverage",
    "used_law_rate",
    "used_case_rate",
    "used_scenario_rate",
    "used_safety_rate",
    "used_policy_rate",
    "avg_evidence_per_frame",
    "avg_used_evidence_per_frame",
    "unique_evidence_count",
    "top_used_evidence_ids",
]

RAG_TOP_EVIDENCE_FIELDS = [
    "evidence_id",
    "doc_type",
    "title",
    "total_pack_count",
    "used_count",
    "datasets",
    "rag_variants",
]


def _read_aggregate_csv(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as f:
        return [dict(row) for row in csv.DictReader(f)]


def _safe_div(num: int | float, den: int | float) -> float:
    return (num / den) if den else 0.0


def _items_by_id(items: list[dict]) -> dict[str, dict]:
    out = {}
    for item in items or []:
        if not isinstance(item, dict):
            continue
        evidence_id = str(item.get("evidence_id") or item.get("chunk_id") or item.get("id") or "")
        if evidence_id:
            out[evidence_id] = item
    return out


def _doc_type(item: dict | None) -> str:
    if not isinstance(item, dict):
        return "unknown"
    return str(item.get("doc_type") or item.get("kb_type") or "unknown")


def _title(item: dict | None) -> str:
    if not isinstance(item, dict):
        return ""
    return str(item.get("title") or item.get("evidence_id") or item.get("chunk_id") or "")


def _audit_run(row: dict, global_counter: dict) -> dict | None:
    run_dir = Path(str(row.get("run_dir") or ""))
    rag_trace_path = run_dir / "rag_trace.jsonl"
    if not rag_trace_path.exists():
        return None

    frames = 0
    frames_with_evidence = 0
    grounded_frames = 0
    evidence_total = 0
    used_total = 0
    type_frame_counts = Counter()
    used_type_frame_counts = Counter()
    unique_evidence = {}
    used_counter = Counter()

    for trace in read_jsonl(rag_trace_path):
        frames += 1
        evidence_pack = trace.get("evidence_pack", {}) or {}
        items = evidence_pack.get("items", []) or []
        item_by_id = _items_by_id(items)
        types_in_frame = {_doc_type(item) for item in items if isinstance(item, dict)}

        if items:
            frames_with_evidence += 1
        evidence_total += len(items)

        for item_id, item in item_by_id.items():
            unique_evidence[item_id] = item
            doc_type = _doc_type(item)
            key = (item_id, doc_type, _title(item))
            global_counter["pack"][key] += 1
            global_counter["datasets"][key].add(str(row.get("dataset") or ""))
            global_counter["variants"][key].add(str(row.get("rag_variant") or ""))

        for doc_type in types_in_frame:
            type_frame_counts[doc_type] += 1

        grounding = (
            trace.get("output_grounding")
            or trace.get("grounding")
            or {}
        )
        if grounding.get("is_grounded"):
            grounded_frames += 1

        used_ids = [
            str(x)
            for x in (
                trace.get("decision_used_evidence_ids")
                or grounding.get("valid_used_evidence_ids")
                or grounding.get("used_evidence_ids")
                or []
            )
        ]
        used_total += len(used_ids)
        used_types = set()
        for evidence_id in used_ids:
            item = item_by_id.get(evidence_id) or unique_evidence.get(evidence_id)
            doc_type = _doc_type(item)
            used_types.add(doc_type)
            used_counter[evidence_id] += 1
            key = (evidence_id, doc_type, _title(item))
            global_counter["used"][key] += 1
            global_counter["datasets"][key].add(str(row.get("dataset") or ""))
            global_counter["variants"][key].add(str(row.get("rag_variant") or ""))

        for doc_type in used_types:
            used_type_frame_counts[doc_type] += 1

    if frames == 0:
        return None

    top_used = ",".join(eid for eid, _ in used_counter.most_common(8))
    return {
        "job_id": row.get("job_id"),
        "dataset": row.get("dataset"),
        "profile_name": row.get("profile_name"),
        "rag_variant": row.get("rag_variant"),
        "planning_variant": row.get("planning_variant"),
        "frames": frames,
        "frames_with_evidence": frames_with_evidence,
        "evidence_coverage": _safe_div(frames_with_evidence, frames),
        "grounded_frames": grounded_frames,
        "grounded_rate": _safe_div(grounded_frames, frames),
        "law_coverage": _safe_div(type_frame_counts["law"], frames),
        "case_coverage": _safe_div(type_frame_counts["case"], frames),
        "scenario_coverage": _safe_div(type_frame_counts["scenario"], frames),
        "safety_coverage": _safe_div(type_frame_counts["safety"], frames),
        "policy_coverage": _safe_div(type_frame_counts["policy"], frames),
        "core_law_case_scenario_coverage": _safe_div(
            sum(
                1
                for trace in read_jsonl(rag_trace_path)
                if {"law", "case", "scenario"}.issubset(
                    {
                        _doc_type(item)
                        for item in ((trace.get("evidence_pack", {}) or {}).get("items", []) or [])
                        if isinstance(item, dict)
                    }
                )
            ),
            frames,
        ),
        "used_law_rate": _safe_div(used_type_frame_counts["law"], frames),
        "used_case_rate": _safe_div(used_type_frame_counts["case"], frames),
        "used_scenario_rate": _safe_div(used_type_frame_counts["scenario"], frames),
        "used_safety_rate": _safe_div(used_type_frame_counts["safety"], frames),
        "used_policy_rate": _safe_div(used_type_frame_counts["policy"], frames),
        "avg_evidence_per_frame": _safe_div(evidence_total, frames),
        "avg_used_evidence_per_frame": _safe_div(used_total, frames),
        "unique_evidence_count": len(unique_evidence),
        "top_used_evidence_ids": top_used,
    }


def _write_examples(rows: list[dict], output_dir: Path, max_examples: int = 12) -> Path:
    path = output_dir / "rag_evidence_examples.md"
    lines = [
        "# RAG Evidence Examples",
        "",
        "Examples require law, case, and scenario evidence in the same evidence pack.",
        "",
    ]
    count = 0
    for row in rows:
        if count >= max_examples:
            break
        run_dir = Path(str(row.get("run_dir") or ""))
        rag_trace_path = run_dir / "rag_trace.jsonl"
        if not rag_trace_path.exists():
            continue
        for trace in read_jsonl(rag_trace_path):
            evidence_pack = trace.get("evidence_pack", {}) or {}
            items = evidence_pack.get("items", []) or []
            types = {_doc_type(item) for item in items if isinstance(item, dict)}
            grounding = (
                trace.get("output_grounding")
                or trace.get("grounding")
                or {}
            )
            if not {"law", "case", "scenario"}.issubset(types):
                continue
            if not grounding.get("is_grounded"):
                continue
            lines.extend([
                f"## Example {count + 1}",
                "",
                f"- job_id: `{row.get('job_id')}`",
                f"- dataset/profile/rag: `{row.get('dataset')}` / `{row.get('profile_name')}` / `{row.get('rag_variant')}`",
                f"- event/frame: `{trace.get('event_index')}` / `{trace.get('frame_index')}`",
                f"- used_evidence_ids: `{','.join(trace.get('decision_used_evidence_ids') or [])}`",
                "",
                "| type | evidence_id | title |",
                "| --- | --- | --- |",
            ])
            for item in items[:8]:
                evidence_id = item.get("evidence_id") or item.get("chunk_id") or item.get("id")
                lines.append(f"| {_doc_type(item)} | `{evidence_id}` | {_title(item)} |")
            lines.append("")
            count += 1
            break

    if count == 0:
        lines.append("No grounded examples with law+case+scenario evidence were found.")
        lines.append("")

    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def make_rag_evidence_tables(rows: list[dict], output_dir: str | Path) -> tuple[list[dict], list[dict]]:
    output_dir = Path(output_dir)
    global_counter = {
        "pack": Counter(),
        "used": Counter(),
        "datasets": defaultdict(set),
        "variants": defaultdict(set),
    }

    summary_rows = []
    for row in rows:
        audited = _audit_run(row, global_counter)
        if audited is not None:
            summary_rows.append(audited)

    write_csv(output_dir / "rag_evidence_summary.csv", summary_rows, fieldnames=RAG_EVIDENCE_SUMMARY_FIELDS)

    keys = set(global_counter["pack"].keys()) | set(global_counter["used"].keys())
    top_rows = []
    for key in keys:
        evidence_id, doc_type, title = key
        top_rows.append({
            "evidence_id": evidence_id,
            "doc_type": doc_type,
            "title": title,
            "total_pack_count": global_counter["pack"][key],
            "used_count": global_counter["used"][key],
            "datasets": ",".join(sorted(x for x in global_counter["datasets"][key] if x)),
            "rag_variants": ",".join(sorted(x for x in global_counter["variants"][key] if x)),
        })

    top_rows.sort(key=lambda r: (int(r["used_count"]), int(r["total_pack_count"])), reverse=True)
    write_csv(output_dir / "rag_top_evidence.csv", top_rows, fieldnames=RAG_TOP_EVIDENCE_FIELDS)
    _write_examples(rows, output_dir)
    return summary_rows, top_rows


def main():
    parser = argparse.ArgumentParser(description="Audit RAG evidence usage in an experiment matrix.")
    parser.add_argument("--experiment_dir", required=True)
    args = parser.parse_args()

    experiment_dir = Path(args.experiment_dir)
    rows = _read_aggregate_csv(experiment_dir / "aggregate_summary.csv")
    if not rows:
        raise SystemExit(f"aggregate_summary.csv not found or empty: {experiment_dir}")

    summary_rows, top_rows = make_rag_evidence_tables(rows, experiment_dir)
    print(f"Wrote {len(summary_rows)} evidence summary rows and {len(top_rows)} top evidence rows.")


if __name__ == "__main__":
    main()
