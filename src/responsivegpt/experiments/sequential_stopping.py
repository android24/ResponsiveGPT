import argparse
import csv
import json
import math
from pathlib import Path

from .experiment_matrix import expand_jobs
from .io_utils import load_json
from .io_utils import write_csv, write_json


DEFAULT_METRICS = [
    "underreaction_rate",
    "overreaction_rate",
    "rag_grounded_decision_rate",
    "rag_output_invalid_citation_frame_rate",
]


def _read_weighted(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as stream:
        return [dict(row) for row in csv.DictReader(stream)]


def _key(row: dict) -> tuple:
    return (
        row.get("dataset"),
        row.get("profile_name"),
        row.get("rag_variant"),
        row.get("planning_variant"),
        row.get("llm_policy_variant"),
        row.get("metric"),
    )


def _row_is_valid(row: dict) -> bool:
    explicit = str(row.get("estimate_valid", "")).strip().lower()
    if explicit in {"false", "0", "no"}:
        return False
    try:
        values = [
            float(row["weighted_mean"]),
            float(row["ci95_low"]),
            float(row["ci95_high"]),
        ]
    except (KeyError, TypeError, ValueError):
        return False
    return all(math.isfinite(value) for value in values)


def _serialize_keys(keys: set[tuple]) -> list[list[str]]:
    return [list(key) for key in sorted(keys)]


def _expected_keys_for_round(
    directory: Path,
    metrics: set[str],
    observed_keys: set[tuple],
) -> set[tuple]:
    snapshot = directory / "config.snapshot.json"
    if not snapshot.exists():
        return {
            (*key[:5], metric)
            for key in observed_keys
            for metric in metrics
        }
    jobs = expand_jobs(load_json(snapshot))
    configurations = {
        (
            job.dataset,
            job.profile_name,
            job.rag_variant,
            job.planning_variant,
            job.llm_policy_variant,
        )
        for job in jobs
    }
    return {
        (*configuration, metric)
        for configuration in configurations
        for metric in metrics
    }


def evaluate_stopping(
    experiment_dirs: list[str | Path],
    *,
    metrics: list[str] | None = None,
    max_ci_half_width: float = 0.03,
    max_round_drift: float = 0.02,
) -> dict:
    metrics = set(metrics or DEFAULT_METRICS)
    rounds = []
    for index, directory in enumerate(experiment_dirs, 1):
        directory = Path(directory)
        rows = [
            row
            for row in _read_weighted(
                directory / "weighted_metric_summary.csv"
            )
            if row.get("metric") in metrics
        ]
        keyed_rows = {}
        duplicate_keys = set()
        for row in rows:
            key = _key(row)
            if key in keyed_rows:
                duplicate_keys.add(key)
            keyed_rows[key] = row
        rounds.append({
            "round": index,
            "experiment_dir": str(directory),
            "rows": keyed_rows,
            "duplicate_keys": duplicate_keys,
            "expected_keys": _expected_keys_for_round(
                directory, metrics, set(keyed_rows)
            ),
        })

    findings = []
    latest = rounds[-1]["rows"] if rounds else {}
    previous = rounds[-2]["rows"] if len(rounds) >= 2 else {}
    latest_keys = set(latest)
    previous_keys = set(previous)
    latest_expected = (
        rounds[-1]["expected_keys"] if rounds else set()
    )
    previous_expected = (
        rounds[-2]["expected_keys"] if len(rounds) >= 2 else set()
    )
    expected_keys = latest_expected | previous_expected
    expected_key_sets_match = latest_expected == previous_expected
    missing_in_latest = latest_expected - latest_keys
    missing_in_previous = previous_expected - previous_keys
    invalid_estimate_keys = {
        key
        for key in expected_keys
        if (
            key in latest and not _row_is_valid(latest[key])
        ) or (
            key in previous and not _row_is_valid(previous[key])
        )
    }
    duplicate_keys = set()
    if rounds:
        duplicate_keys.update(rounds[-1]["duplicate_keys"])
    if len(rounds) >= 2:
        duplicate_keys.update(rounds[-2]["duplicate_keys"])

    for key in sorted(expected_keys):
        row = latest.get(key)
        previous_row = previous.get(key)
        if (
            row is None
            or previous_row is None
            or key in invalid_estimate_keys
        ):
            continue
        mean = float(row["weighted_mean"])
        low = float(row["ci95_low"])
        high = float(row["ci95_high"])
        half_width = (high - low) / 2.0
        drift = abs(mean - float(previous_row["weighted_mean"]))
        precision_ok = half_width <= max_ci_half_width
        stability_ok = drift <= max_round_drift
        findings.append({
            "dataset": key[0],
            "profile_name": key[1],
            "rag_variant": key[2],
            "planning_variant": key[3],
            "llm_policy_variant": key[4],
            "metric": key[5],
            "weighted_mean": mean,
            "ci95_half_width": half_width,
            "round_drift": drift,
            "precision_ok": precision_ok,
            "stability_ok": stability_ok,
            "stop_ok": precision_ok and stability_ok,
        })

    enough_rounds = len(rounds) >= 2
    key_sets_match = latest_keys == previous_keys
    coverage_complete = bool(expected_keys) and not (
        missing_in_latest
        or missing_in_previous
        or invalid_estimate_keys
        or duplicate_keys
    ) and expected_key_sets_match
    stop = (
        enough_rounds
        and key_sets_match
        and expected_key_sets_match
        and coverage_complete
        and len(findings) == len(expected_keys)
        and all(row["stop_ok"] for row in findings)
    )
    reasons = []
    if not enough_rounds:
        reasons.append("at_least_two_rounds_required")
    if not key_sets_match:
        reasons.append("round_key_sets_do_not_match")
    if not expected_key_sets_match:
        reasons.append("round_expected_matrices_do_not_match")
    if missing_in_latest or missing_in_previous:
        reasons.append("requested_metric_coverage_incomplete")
    if invalid_estimate_keys:
        reasons.append("invalid_or_partial_weighted_estimates")
    if duplicate_keys:
        reasons.append("duplicate_weighted_estimate_keys")
    if coverage_complete and findings and not all(
        row["stop_ok"] for row in findings
    ):
        reasons.append("precision_or_stability_threshold_not_met")
    return {
        "decision": "stop" if stop else "continue",
        "num_rounds": len(rounds),
        "max_ci_half_width": max_ci_half_width,
        "max_round_drift": max_round_drift,
        "coverage_complete": coverage_complete,
        "key_sets_match": key_sets_match,
        "expected_key_sets_match": expected_key_sets_match,
        "expected_key_count": len(expected_keys),
        "missing_in_latest": _serialize_keys(missing_in_latest),
        "missing_in_previous": _serialize_keys(missing_in_previous),
        "invalid_estimate_keys": _serialize_keys(invalid_estimate_keys),
        "duplicate_keys": _serialize_keys(duplicate_keys),
        "continue_reasons": reasons,
        "findings": findings,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate precision/stability stopping for sequential runs."
    )
    parser.add_argument("--experiment_dirs", nargs="+", required=True)
    parser.add_argument("--metrics", nargs="*", default=DEFAULT_METRICS)
    parser.add_argument("--max_ci_half_width", type=float, default=0.03)
    parser.add_argument("--max_round_drift", type=float, default=0.02)
    parser.add_argument("--out_dir", required=True)
    args = parser.parse_args()

    result = evaluate_stopping(
        args.experiment_dirs,
        metrics=args.metrics,
        max_ci_half_width=args.max_ci_half_width,
        max_round_drift=args.max_round_drift,
    )
    out = Path(args.out_dir)
    write_json(out / "sequential_stopping_decision.json", result)
    write_csv(out / "sequential_stopping_findings.csv", result["findings"])
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
