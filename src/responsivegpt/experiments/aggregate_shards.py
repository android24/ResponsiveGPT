import argparse
import csv
from collections import defaultdict
from pathlib import Path

from .io_utils import load_json, write_csv


ROLLUP_FIELDS = [
    "dataset",
    "profile_name",
    "rag_variant",
    "rag_mode",
    "planning_variant",
    "llm_policy_variant",
    "frame_selection",
    "num_shards_completed",
    "num_shards_expected",
    "shard_coverage",
    "total_events",
    "total_frames",
    "candidate_frames",
    "selected_frames",
    "reactive_frames",
    "llm_calls",
    "llm_attempts",
    "non_llm_frames",
    "planning_calls",
    "planning_llm_attempts",
    "llm_error_count",
    "timeout_count",
    "connection_error_count",
    "rate_limit_count",
    "fallback_frame_count",
    "fallback_frame_rate",
    "max_reactive_api_attempts",
    "max_reactive_tokens",
    "max_planning_api_attempts",
    "max_planning_tokens",
    "reactive_request_budget_exhausted",
    "reactive_token_budget_exhausted",
    "planning_request_budget_exhausted",
    "planning_token_budget_exhausted",
    "reactive_token_overshoot",
    "planning_token_overshoot",
    "llm_budget_exhausted_frames",
    "planning_budget_exhausted_frames",
    "precision",
    "recall",
    "f1",
    "accuracy",
    "avg_underreaction_rate",
    "avg_overreaction_rate",
    "avg_reaction_delay_frames",
    "avg_decision_flip_rate",
    "reactive_llm_call_rate",
    "reactive_llm_attempt_rate",
    "planning_call_rate",
    "non_llm_frame_rate",
    "reactive_total_tokens",
    "planning_total_tokens",
    "reactive_latency_ms_p50",
    "reactive_latency_ms_p95",
    "planning_latency_ms_p50",
    "planning_latency_ms_p95",
    "retrieval_coverage",
    "evidence_usage_rate",
    "grounded_decision_rate",
    "hallucinated_citation_rate",
    "raw_invalid_citation_attempt_rate",
    "output_invalid_citation_frame_rate",
    "citation_precision",
    "avg_evidence_per_frame",
]


def _read_csv(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as f:
        return [dict(row) for row in csv.DictReader(f)]


def _to_float(value, default=0.0) -> float:
    if value in (None, ""):
        return default
    try:
        return float(value)
    except Exception:
        return default


def _to_int(value, default=0) -> int:
    if value in (None, ""):
        return default
    try:
        return int(float(value))
    except Exception:
        return default


def _safe_div(num: float, den: float) -> float:
    return num / den if den else 0.0


def _weighted_mean(rows: list[dict], metric: str, weight: str = "total_frames") -> float:
    num = 0.0
    den = 0.0
    for row in rows:
        value = row.get(metric)
        if value in (None, ""):
            continue
        w = _to_float(row.get(weight), 0.0)
        num += _to_float(value) * w
        den += w
    return _safe_div(num, den)


def _group_key(row: dict) -> tuple:
    return (
        row.get("dataset", ""),
        row.get("profile_name", ""),
        row.get("rag_variant", ""),
        row.get("rag_mode", ""),
        row.get("planning_variant", ""),
        row.get("llm_policy_variant", ""),
        row.get("frame_selection", ""),
    )


def _confusion_from_run(row: dict) -> dict:
    run_dir = row.get("run_dir")
    if not run_dir:
        return {"tp": 0, "fp": 0, "fn": 0, "tn": 0}
    summary_path = Path(run_dir) / "summary.json"
    if not summary_path.exists():
        return {"tp": 0, "fp": 0, "fn": 0, "tn": 0}
    cm = (load_json(summary_path).get("confusion_matrix") or {})
    return {
        "tp": _to_int(cm.get("tp")),
        "fp": _to_int(cm.get("fp")),
        "fn": _to_int(cm.get("fn")),
        "tn": _to_int(cm.get("tn")),
    }


def _rollup_group(rows: list[dict]) -> dict:
    first = rows[0]
    total_events = sum(_to_int(r.get("total_events")) for r in rows)
    total_frames = sum(_to_int(r.get("total_frames")) for r in rows)
    candidate_frames = sum(_to_int(r.get("candidate_frames")) for r in rows)
    selected_frames = sum(_to_int(r.get("selected_frames")) for r in rows)
    reactive_frames = sum(_to_int(r.get("reactive_frames")) for r in rows)
    llm_calls = sum(_to_int(r.get("llm_calls")) for r in rows)
    llm_attempts = sum(_to_int(r.get("llm_attempts")) for r in rows)
    non_llm_frames = sum(_to_int(r.get("non_llm_frames")) for r in rows)
    planning_calls = sum(_to_int(r.get("planning_calls")) for r in rows)
    planning_llm_attempts = sum(
        _to_int(r.get("planning_llm_attempts")) for r in rows
    )
    fallback_frame_count = sum(_to_int(r.get("fallback_frame_count")) for r in rows)

    cm = {"tp": 0, "fp": 0, "fn": 0, "tn": 0}
    for row in rows:
        item = _confusion_from_run(row)
        for key in cm:
            cm[key] += item[key]

    precision = _safe_div(cm["tp"], cm["tp"] + cm["fp"])
    recall = _safe_div(cm["tp"], cm["tp"] + cm["fn"])
    f1 = _safe_div(2 * precision * recall, precision + recall)
    accuracy = _safe_div(cm["tp"] + cm["tn"], sum(cm.values()))

    expected = max(_to_int(r.get("num_shards"), 0) for r in rows)
    completed = len(rows)

    return {
        "dataset": first.get("dataset", ""),
        "profile_name": first.get("profile_name", ""),
        "rag_variant": first.get("rag_variant", ""),
        "rag_mode": first.get("rag_mode", ""),
        "planning_variant": first.get("planning_variant", ""),
        "llm_policy_variant": first.get("llm_policy_variant", ""),
        "frame_selection": first.get("frame_selection", ""),
        "num_shards_completed": completed,
        "num_shards_expected": expected,
        "shard_coverage": _safe_div(completed, expected),
        "total_events": total_events,
        "total_frames": total_frames,
        "candidate_frames": candidate_frames,
        "selected_frames": selected_frames,
        "reactive_frames": reactive_frames,
        "llm_calls": llm_calls,
        "llm_attempts": llm_attempts,
        "non_llm_frames": non_llm_frames,
        "planning_calls": planning_calls,
        "planning_llm_attempts": planning_llm_attempts,
        "llm_error_count": sum(_to_int(r.get("llm_error_count")) for r in rows),
        "timeout_count": sum(_to_int(r.get("timeout_count")) for r in rows),
        "connection_error_count": sum(_to_int(r.get("connection_error_count")) for r in rows),
        "rate_limit_count": sum(_to_int(r.get("rate_limit_count")) for r in rows),
        "fallback_frame_count": fallback_frame_count,
        "fallback_frame_rate": _safe_div(fallback_frame_count, reactive_frames),
        "max_reactive_api_attempts": sum(
            _to_int(r.get("max_reactive_api_attempts")) for r in rows
        ),
        "max_reactive_tokens": sum(
            _to_int(r.get("max_reactive_tokens")) for r in rows
        ),
        "max_planning_api_attempts": sum(
            _to_int(r.get("max_planning_api_attempts")) for r in rows
        ),
        "max_planning_tokens": sum(
            _to_int(r.get("max_planning_tokens")) for r in rows
        ),
        "reactive_request_budget_exhausted": any(
            str(r.get("reactive_request_budget_exhausted")).lower()
            in {"true", "1"}
            for r in rows
        ),
        "reactive_token_budget_exhausted": any(
            str(r.get("reactive_token_budget_exhausted")).lower()
            in {"true", "1"}
            for r in rows
        ),
        "planning_request_budget_exhausted": any(
            str(r.get("planning_request_budget_exhausted")).lower()
            in {"true", "1"}
            for r in rows
        ),
        "planning_token_budget_exhausted": any(
            str(r.get("planning_token_budget_exhausted")).lower()
            in {"true", "1"}
            for r in rows
        ),
        "reactive_token_overshoot": sum(
            _to_int(r.get("reactive_token_overshoot")) for r in rows
        ),
        "planning_token_overshoot": sum(
            _to_int(r.get("planning_token_overshoot")) for r in rows
        ),
        "llm_budget_exhausted_frames": sum(_to_int(r.get("llm_budget_exhausted_frames")) for r in rows),
        "planning_budget_exhausted_frames": sum(_to_int(r.get("planning_budget_exhausted_frames")) for r in rows),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "accuracy": accuracy,
        "avg_underreaction_rate": _weighted_mean(rows, "avg_underreaction_rate"),
        "avg_overreaction_rate": _weighted_mean(rows, "avg_overreaction_rate"),
        "avg_reaction_delay_frames": _weighted_mean(rows, "avg_reaction_delay_frames", "total_events"),
        "avg_decision_flip_rate": _weighted_mean(rows, "avg_decision_flip_rate"),
        "reactive_llm_call_rate": _safe_div(llm_calls, reactive_frames),
        "reactive_llm_attempt_rate": _safe_div(llm_attempts, reactive_frames),
        "planning_call_rate": _safe_div(planning_calls, reactive_frames),
        "non_llm_frame_rate": _safe_div(non_llm_frames, reactive_frames),
        "reactive_total_tokens": sum(
            _to_int(r.get("reactive_total_tokens")) for r in rows
        ),
        "planning_total_tokens": sum(
            _to_int(r.get("planning_total_tokens")) for r in rows
        ),
        "reactive_latency_ms_p50": _weighted_mean(
            rows, "reactive_latency_ms_p50", "llm_attempts"
        ),
        "reactive_latency_ms_p95": _weighted_mean(
            rows, "reactive_latency_ms_p95", "llm_attempts"
        ),
        "planning_latency_ms_p50": _weighted_mean(
            rows, "planning_latency_ms_p50", "planning_llm_attempts"
        ),
        "planning_latency_ms_p95": _weighted_mean(
            rows, "planning_latency_ms_p95", "planning_llm_attempts"
        ),
        "retrieval_coverage": _weighted_mean(rows, "retrieval_coverage"),
        "evidence_usage_rate": _weighted_mean(rows, "evidence_usage_rate"),
        "grounded_decision_rate": _weighted_mean(rows, "grounded_decision_rate"),
        "hallucinated_citation_rate": _weighted_mean(rows, "hallucinated_citation_rate"),
        "raw_invalid_citation_attempt_rate": _weighted_mean(
            rows, "raw_invalid_citation_attempt_rate"
        ),
        "output_invalid_citation_frame_rate": _weighted_mean(
            rows, "output_invalid_citation_frame_rate"
        ),
        "citation_precision": _weighted_mean(rows, "citation_precision"),
        "avg_evidence_per_frame": _weighted_mean(rows, "avg_evidence_per_frame"),
    }


def aggregate_shards(experiment_dir: str | Path) -> list[dict]:
    experiment_dir = Path(experiment_dir)
    rows = _read_csv(experiment_dir / "aggregate_summary.csv")
    if not rows:
        raise SystemExit(f"aggregate_summary.csv not found or empty: {experiment_dir}")

    grouped = defaultdict(list)
    for row in rows:
        grouped[_group_key(row)].append(row)

    out = [_rollup_group(group) for _, group in sorted(grouped.items())]
    write_csv(experiment_dir / "shard_rollup_summary.csv", out, fieldnames=ROLLUP_FIELDS)
    return out


def main():
    parser = argparse.ArgumentParser(description="Aggregate ResponsiveGPT shard runs into full-pass rows.")
    parser.add_argument("--experiment_dir", required=True)
    args = parser.parse_args()

    rows = aggregate_shards(args.experiment_dir)
    print(f"Wrote {len(rows)} shard rollup rows to {Path(args.experiment_dir) / 'shard_rollup_summary.csv'}")


if __name__ == "__main__":
    main()
