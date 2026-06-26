import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path

from .io_utils import read_jsonl, write_csv
from .experiment_fingerprint import (
    expected_fingerprints_for_experiment,
    fingerprint_is_compatible,
)
from .validate_runs import validate_run_dir


PAIR_FIELDS = [
    "dataset",
    "profile_name",
    "rag_variant",
    "planning_variant",
    "dense_job_id",
    "sparse_job_id",
    "event_index",
    "dense_frames",
    "sparse_frames",
    "frame_reduction_rate",
    "violation_agreement",
    "trigger_delta",
    "physical_risk_exposure_delta",
    "alignment_accuracy_delta",
    "underreaction_rate_delta",
    "planning_hit_rate_delta",
    "planning_miss_rate_delta",
]

SUMMARY_FIELDS = [
    "dataset",
    "profile_name",
    "rag_variant",
    "planning_variant",
    "dense_job_id",
    "sparse_job_id",
    "num_pairs",
    "avg_frame_reduction_rate",
    "violation_agreement_rate",
    "avg_abs_trigger_delta",
    "avg_abs_physical_risk_exposure_delta",
    "avg_abs_alignment_accuracy_delta",
    "avg_abs_underreaction_rate_delta",
    "avg_abs_planning_hit_rate_delta",
    "avg_abs_planning_miss_rate_delta",
]


def _read_csv(path: str | Path) -> list[dict]:
    path = Path(path)
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as f:
        return [dict(row) for row in csv.DictReader(f)]


def _to_float(value):
    if value in (None, ""):
        return None
    try:
        out = float(value)
    except Exception:
        return None
    if math.isnan(out) or math.isinf(out):
        return None
    return out


def _to_bool(value):
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes"}


def _mean(values: list[float]) -> float | None:
    values = [value for value in values if value is not None]
    return sum(values) / len(values) if values else None


def _abs_mean(values: list[float]) -> float | None:
    values = [abs(value) for value in values if value is not None]
    return sum(values) / len(values) if values else None


def _latest_completed_statuses(experiment_dir: Path) -> list[dict]:
    latest = {}
    expected = expected_fingerprints_for_experiment(experiment_dir)
    for row in read_jsonl(experiment_dir / "job_status.jsonl"):
        if row.get("status") != "completed":
            continue
        job_id = row.get("job_id")
        if job_id and fingerprint_is_compatible(
            row, expected.get(str(job_id))
        ):
            latest[job_id] = row
    return [
        row
        for row in latest.values()
        if row.get("run_dir")
        and validate_run_dir(row["run_dir"], row.get("job", {})).get(
            "execution_valid", False
        )
    ]


def _job_key(job: dict) -> tuple:
    return (
        job.get("dataset", ""),
        job.get("profile_name", ""),
        job.get("rag_variant", ""),
        job.get("planning_variant", ""),
    )


def _is_dense(row: dict, dense_variant: str) -> bool:
    job = row.get("job", {}) or {}
    extra = job.get("extra_args", {}) or {}
    return (
        job.get("llm_policy_variant") == dense_variant
        or extra.get("frame_selection") == "all"
        and dense_variant == "all"
    )


def _is_sparse(row: dict, sparse_variant: str) -> bool:
    job = row.get("job", {}) or {}
    extra = job.get("extra_args", {}) or {}
    return (
        job.get("llm_policy_variant") == sparse_variant
        or extra.get("frame_selection") == "critical"
        and sparse_variant == "critical"
    )


def _episode_map(run_dir: str | Path) -> dict[int, dict]:
    out = {}
    for row in read_jsonl(Path(run_dir) / "episode_summary.jsonl"):
        try:
            out[int(row.get("event_index"))] = row
        except Exception:
            continue
    return out


def _nested(row: dict, section: str, key: str):
    obj = row.get(section) or {}
    if isinstance(obj, dict):
        return _to_float(obj.get(key))
    return None


def _diff(a, b):
    av = _to_float(a)
    bv = _to_float(b)
    if av is None or bv is None:
        return None
    return bv - av


def _build_pair_rows(dense_status: dict, sparse_status: dict) -> list[dict]:
    dense_job = dense_status.get("job", {}) or {}
    sparse_job = sparse_status.get("job", {}) or {}
    dense_episodes = _episode_map(dense_status.get("run_dir", ""))
    sparse_episodes = _episode_map(sparse_status.get("run_dir", ""))

    rows = []
    for event_index in sorted(set(dense_episodes).intersection(sparse_episodes)):
        dense = dense_episodes[event_index]
        sparse = sparse_episodes[event_index]
        dense_frames = _to_float(dense.get("episode_evaluated_num_frames") or dense.get("episode_num_frames"))
        sparse_frames = _to_float(sparse.get("episode_evaluated_num_frames") or sparse.get("episode_num_frames"))
        frame_reduction = None
        if dense_frames and sparse_frames is not None:
            frame_reduction = 1.0 - sparse_frames / dense_frames

        rows.append({
            "dataset": dense_job.get("dataset", sparse_job.get("dataset", "")),
            "profile_name": dense_job.get("profile_name", sparse_job.get("profile_name", "")),
            "rag_variant": dense_job.get("rag_variant", sparse_job.get("rag_variant", "")),
            "planning_variant": dense_job.get("planning_variant", sparse_job.get("planning_variant", "")),
            "dense_job_id": dense_status.get("job_id"),
            "sparse_job_id": sparse_status.get("job_id"),
            "event_index": event_index,
            "dense_frames": dense_frames,
            "sparse_frames": sparse_frames,
            "frame_reduction_rate": frame_reduction,
            "violation_agreement": int(_to_bool(dense.get("episode_llm_violation")) == _to_bool(sparse.get("episode_llm_violation"))),
            "trigger_delta": _diff(dense.get("trigger_count"), sparse.get("trigger_count")),
            "physical_risk_exposure_delta": _diff(
                _nested(dense, "episode_safety", "physical_risk_exposure"),
                _nested(sparse, "episode_safety", "physical_risk_exposure"),
            ),
            "alignment_accuracy_delta": _diff(
                _nested(dense, "llm_physics_alignment", "alignment_accuracy"),
                _nested(sparse, "llm_physics_alignment", "alignment_accuracy"),
            ),
            "underreaction_rate_delta": _diff(
                _nested(dense, "llm_physics_alignment", "underreaction_rate"),
                _nested(sparse, "llm_physics_alignment", "underreaction_rate"),
            ),
            "planning_hit_rate_delta": _diff(
                _nested(dense, "planning_quality", "planning_hit_rate"),
                _nested(sparse, "planning_quality", "planning_hit_rate"),
            ),
            "planning_miss_rate_delta": _diff(
                _nested(dense, "planning_quality", "planning_miss_rate"),
                _nested(sparse, "planning_quality", "planning_miss_rate"),
            ),
        })
    return rows


def build_calibration_report(
    experiment_dir: str | Path,
    *,
    dense_variant: str = "dense_all",
    sparse_variant: str = "sparse_critical",
) -> tuple[list[dict], list[dict]]:
    experiment_dir = Path(experiment_dir)
    statuses = _latest_completed_statuses(experiment_dir)

    dense_by_key = {}
    sparse_by_key = {}
    for status in statuses:
        job = status.get("job", {}) or {}
        key = _job_key(job)
        if _is_dense(status, dense_variant):
            dense_by_key[key] = status
        if _is_sparse(status, sparse_variant):
            sparse_by_key[key] = status

    pair_rows = []
    for key, dense_status in sorted(dense_by_key.items()):
        sparse_status = sparse_by_key.get(key)
        if sparse_status is None:
            continue
        pair_rows.extend(_build_pair_rows(dense_status, sparse_status))

    grouped = defaultdict(list)
    for row in pair_rows:
        grouped[(
            row["dataset"],
            row["profile_name"],
            row["rag_variant"],
            row["planning_variant"],
            row["dense_job_id"],
            row["sparse_job_id"],
        )].append(row)

    summary_rows = []
    for key, rows in sorted(grouped.items()):
        dataset, profile, rag, planning, dense_job_id, sparse_job_id = key
        summary_rows.append({
            "dataset": dataset,
            "profile_name": profile,
            "rag_variant": rag,
            "planning_variant": planning,
            "dense_job_id": dense_job_id,
            "sparse_job_id": sparse_job_id,
            "num_pairs": len(rows),
            "avg_frame_reduction_rate": _mean([_to_float(r.get("frame_reduction_rate")) for r in rows]),
            "violation_agreement_rate": _mean([_to_float(r.get("violation_agreement")) for r in rows]),
            "avg_abs_trigger_delta": _abs_mean([_to_float(r.get("trigger_delta")) for r in rows]),
            "avg_abs_physical_risk_exposure_delta": _abs_mean([_to_float(r.get("physical_risk_exposure_delta")) for r in rows]),
            "avg_abs_alignment_accuracy_delta": _abs_mean([_to_float(r.get("alignment_accuracy_delta")) for r in rows]),
            "avg_abs_underreaction_rate_delta": _abs_mean([_to_float(r.get("underreaction_rate_delta")) for r in rows]),
            "avg_abs_planning_hit_rate_delta": _abs_mean([_to_float(r.get("planning_hit_rate_delta")) for r in rows]),
            "avg_abs_planning_miss_rate_delta": _abs_mean([_to_float(r.get("planning_miss_rate_delta")) for r in rows]),
        })

    write_csv(experiment_dir / "dense_sparse_episode_calibration.csv", pair_rows, fieldnames=PAIR_FIELDS)
    write_csv(experiment_dir / "dense_sparse_calibration_summary.csv", summary_rows, fieldnames=SUMMARY_FIELDS)
    return pair_rows, summary_rows


def main():
    parser = argparse.ArgumentParser(description="Compare dense all-frame and sparse critical-frame ResponsiveGPT runs.")
    parser.add_argument("--experiment_dir", required=True)
    parser.add_argument("--dense_variant", default="dense_all")
    parser.add_argument("--sparse_variant", default="sparse_critical")
    args = parser.parse_args()

    pair_rows, summary_rows = build_calibration_report(
        args.experiment_dir,
        dense_variant=args.dense_variant,
        sparse_variant=args.sparse_variant,
    )
    print(
        f"Wrote {len(pair_rows)} episode calibration rows and {len(summary_rows)} "
        f"summary rows to {args.experiment_dir}"
    )


if __name__ == "__main__":
    main()
