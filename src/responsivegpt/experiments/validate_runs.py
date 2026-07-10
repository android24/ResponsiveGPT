import argparse
import csv
import hashlib
import json
from pathlib import Path

from .io_utils import load_json, read_jsonl, write_csv, write_json
from .experiment_fingerprint import (
    expected_fingerprints_for_experiment,
    fingerprint_is_compatible,
)

VALIDATION_SUMMARY_FIELDS = [
    "job_id",
    "run_dir",
    "valid",
    "failure_reasons",
    "execution_valid",
    "execution_failure_reasons",
    "quality_gate_pass",
    "quality_failures",
    "quality_observations",
    "current_method_compatible",
    "usable_for_current_method",
]


def _file_sha256(path: str | Path) -> str:
    path = Path(path)
    if not path.is_file():
        return ""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _expected_event_count(job: dict) -> int | None:
    summary_csv = job.get("summary_csv")
    if not summary_csv or not Path(summary_csv).exists():
        return None
    with Path(summary_csv).open("r", encoding="utf-8-sig", newline="") as stream:
        row_count = sum(1 for _ in csv.DictReader(stream))

    extra = job.get("extra_args", {}) or {}
    start = max(0, int(extra.get("start_index", 0) or 0))
    end = int(extra.get("end_index", -1) or -1)
    end = row_count if end < 0 else min(row_count, end)
    shard_id = int(extra.get("shard_id", -1) or -1)
    num_shards = int(extra.get("num_shards", 0) or 0)
    indices = range(start, max(start, end))
    if shard_id >= 0 and num_shards > 0:
        count = sum(1 for index in indices if index % num_shards == shard_id)
    else:
        count = max(0, end - start)
    if bool(int(extra.get("profile_protocol_enabled", 0) or 0)):
        adaptation = max(
            0, int(extra.get("profile_adaptation_episodes", 0) or 0)
        )
        pool = max(
            adaptation,
            int(
                extra.get("profile_adaptation_pool_episodes", 0)
                or adaptation
            ),
        )
        actual_pool = min(pool, count)
        actual_adaptation = min(adaptation, actual_pool)
        count = max(
            0, count - actual_pool + actual_adaptation
        )
    limit = int(job.get("limit", 0) or 0)
    count = min(count, limit) if limit > 0 else count
    return count


def validate_summary(summary: dict, job: dict | None = None) -> dict:
    job = job or {}
    execution_reasons = []
    quality_failures = []
    quality_observations = []

    total_frames = int(summary.get("total_frames") or 0)
    reactive_frames = int(summary.get("reactive_frames") or 0)
    llm_calls = int(summary.get("llm_calls") or 0)
    llm_attempts = int(summary.get("llm_attempts") or 0)
    planning_llm_attempts = int(
        summary.get("planning_llm_attempts") or 0
    )
    non_llm_frames = int(summary.get("non_llm_frames") or 0)
    use_retriever = int(job.get("use_retriever", 0) or 0)
    rag_mode = str(job.get("rag_mode", summary.get("rag_mode", "none")))
    extra = job.get("extra_args", {}) or {}
    profile_protocol_enabled = bool(
        int(extra.get("profile_protocol_enabled", 0) or 0)
    )
    protocol = summary.get("profile_protocol") or {}
    pilot_without_evaluation = (
        profile_protocol_enabled
        and bool(protocol.get("pilot_limited"))
        and int(protocol.get("evaluation_events", 0) or 0) == 0
        and protocol.get("formal_inference_eligible") is False
    )

    if not Path(job.get("summary_csv", "")).exists() and job.get("summary_csv"):
        execution_reasons.append("job summary_csv path does not exist")

    if summary.get("classification_skipped") is True:
        if pilot_without_evaluation:
            quality_observations.append(
                "classification metrics are not applicable to an "
                "adaptation-only pilot"
            )
        else:
            execution_reasons.append("classification was skipped")

    if int(summary.get("total_events") or 0) <= 0:
        execution_reasons.append("total_events <= 0")

    if profile_protocol_enabled:
        requested_adaptation = int(
            extra.get("profile_adaptation_episodes", 0) or 0
        )
        requested_pool = max(
            requested_adaptation,
            int(
                extra.get("profile_adaptation_pool_episodes", 0)
                or requested_adaptation
            ),
        )
        pilot_limited = bool(protocol.get("pilot_limited"))
        selected_adaptation = int(
            protocol.get("adaptation_episodes_actual", 0) or 0
        )
        execution_limit = int(
            protocol.get("execution_limit", 0) or 0
        )
        expected_adaptation_events = selected_adaptation
        if pilot_limited and execution_limit > 0:
            expected_adaptation_events = min(
                selected_adaptation, execution_limit
            )
        if not protocol.get("adaptation_strata_available"):
            execution_reasons.append(
                "profile protocol census unavailable"
            )
        if int(
            protocol.get("adaptation_pool_episodes_actual", -1)
        ) != requested_pool:
            execution_reasons.append(
                "adaptation pool size != requested pool size"
            )
        if int(protocol.get("adaptation_events", -1)) != (
            expected_adaptation_events
        ):
            execution_reasons.append(
                "adaptation events != expected executed adaptation episodes"
            )
        if (
            not pilot_limited
            and int(protocol.get("adaptation_events", -1))
            != requested_adaptation
        ):
            execution_reasons.append(
                "adaptation events != requested adaptation budget"
            )
        if pilot_limited:
            quality_observations.append(
                "limited profile run is pilot-only and excluded from "
                "formal weighted inference"
            )
            if protocol.get("formal_inference_eligible") is not False:
                execution_reasons.append(
                    "limited profile run marked inference eligible"
                )
        if (
            int(protocol.get("adaptation_events", 0))
            + int(protocol.get("evaluation_events", 0))
            != int(summary.get("total_events") or 0)
        ):
            execution_reasons.append(
                "profile protocol phase events != total_events"
            )

    expected_events = _expected_event_count(job)
    if (
        expected_events is not None
        and int(summary.get("total_events") or 0) != expected_events
    ):
        execution_reasons.append(
            f"total_events != expected_events ({expected_events})"
        )

    if total_frames <= 0:
        execution_reasons.append("total_frames <= 0")

    for counter in (
        "missing_files",
        "missing_clips",
        "missing_scenes",
        "empty_sequences",
    ):
        if int(summary.get(counter) or 0) != 0:
            execution_reasons.append(f"{counter} != 0")

    if not summary.get("dry_run") and not summary.get("inspect_only"):
        if reactive_frames <= 0:
            execution_reasons.append("reactive_frames <= 0")
        if reactive_frames != llm_calls + non_llm_frames:
            execution_reasons.append("reactive_frames != llm_calls + non_llm_frames")

    max_reactive_attempts = int(
        summary.get("max_reactive_api_attempts") or 0
    )
    max_planning_attempts = int(
        summary.get("max_planning_api_attempts") or 0
    )
    if (
        max_reactive_attempts > 0
        and llm_attempts > max_reactive_attempts
    ):
        execution_reasons.append(
            "llm_attempts exceeds max_reactive_api_attempts"
        )
    if (
        max_planning_attempts > 0
        and planning_llm_attempts > max_planning_attempts
    ):
        execution_reasons.append(
            "planning_llm_attempts exceeds max_planning_api_attempts"
        )
    if int(summary.get("reactive_token_overshoot") or 0) > 0:
        quality_observations.append(
            "reactive token cap overshot by the final admitted request"
        )
    if int(summary.get("planning_token_overshoot") or 0) > 0:
        quality_observations.append(
            "planning token cap overshot by the final admitted request"
        )

    for metric in ("precision", "recall", "f1", "accuracy"):
        if summary.get(metric) is None and not pilot_without_evaluation:
            execution_reasons.append(f"{metric} is null")

    global_rag = summary.get("global_rag")
    if use_retriever and rag_mode != "none":
        if not isinstance(global_rag, dict):
            execution_reasons.append("global_rag missing")
        else:
            retrieval_coverage = float(global_rag.get("retrieval_coverage") or 0.0)
            output_invalid_rate = float(
                global_rag.get(
                    "output_invalid_citation_frame_rate",
                    global_rag.get("hallucinated_citation_rate"),
                )
                or 0.0
            )
            if rag_mode == "full" and retrieval_coverage < 0.95:
                quality_failures.append("full RAG retrieval_coverage < 0.95")
            if rag_mode == "full" and output_invalid_rate > 0.05:
                quality_failures.append("full RAG output_invalid_citation_frame_rate > 0.05")
            elif rag_mode != "full" and output_invalid_rate > 0.05:
                quality_observations.append(
                    "baseline output_invalid_citation_frame_rate > 0.05"
                )

    execution_valid = len(execution_reasons) == 0
    quality_gate_pass = len(quality_failures) == 0

    return {
        # Keep valid/failure_reasons for older callers. They now describe
        # execution integrity only; model quality is reported separately.
        "valid": execution_valid,
        "failure_reasons": execution_reasons,
        "execution_valid": execution_valid,
        "execution_failure_reasons": execution_reasons,
        "quality_gate_pass": quality_gate_pass,
        "quality_failures": quality_failures,
        "quality_observations": quality_observations,
        "reactive_frames_check": {
            "reactive_frames": reactive_frames,
            "llm_calls": llm_calls,
            "non_llm_frames": non_llm_frames,
        },
    }


def validate_run_dir(run_dir: str | Path, job: dict | None = None) -> dict:
    run_dir = Path(run_dir)
    summary_path = run_dir / "summary.json"
    if not summary_path.exists():
        return {
            "valid": False,
            "failure_reasons": [f"summary.json not found: {summary_path}"],
            "execution_valid": False,
            "execution_failure_reasons": [f"summary.json not found: {summary_path}"],
            "quality_gate_pass": False,
            "quality_failures": [],
            "quality_observations": [],
        }
    summary = load_json(summary_path)
    result = validate_summary(summary, job=job)
    config_path = run_dir / "config.json"
    config = load_json(config_path) if config_path.exists() else {}
    episode_path = run_dir / "episode_summary.jsonl"
    episodes = []
    event_ids = []
    if not episode_path.exists():
        result["execution_failure_reasons"].append(
            "episode_summary.jsonl missing"
        )
    else:
        episodes = read_jsonl(episode_path)
        total_events = int(summary.get("total_events") or 0)
        if len(episodes) != total_events:
            result["execution_failure_reasons"].append(
                "episode_summary row count != total_events"
            )
        event_ids = [
            int(row.get("event_index", -1))
            for row in episodes
        ]
        if len(event_ids) != len(set(event_ids)):
            result["execution_failure_reasons"].append(
                "duplicate event_index in episode_summary"
            )
    if (summary.get("profile_protocol") or {}).get("enabled"):
        manifest_path = run_dir / "profile_split_manifest.json"
        if not manifest_path.exists():
            result["execution_failure_reasons"].append(
                "profile_split_manifest.json missing"
            )
        else:
            manifest = load_json(manifest_path)
            manifest_payload = dict(manifest)
            observed_hash = manifest_payload.pop("sha256", "")
            payload = json.dumps(
                manifest_payload,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            )
            expected_hash = hashlib.sha256(
                payload.encode("utf-8")
            ).hexdigest()
            if observed_hash != expected_hash:
                result["execution_failure_reasons"].append(
                    "profile split manifest hash mismatch"
                )
            expected_fingerprint = str(
                config.get("experiment_fingerprint", "") or ""
            )
            if (
                expected_fingerprint
                and manifest.get("experiment_fingerprint")
                != expected_fingerprint
            ):
                result["execution_failure_reasons"].append(
                    "profile split manifest fingerprint mismatch"
                )
            summary_csv = str(manifest.get("summary_csv", "") or "")
            summary_csv_hash = str(
                manifest.get("summary_csv_sha256", "") or ""
            )
            if summary_csv_hash and _file_sha256(summary_csv) != summary_csv_hash:
                result["execution_failure_reasons"].append(
                    "profile split source summary hash mismatch"
                )
            census_path = str(
                manifest.get("adaptation_census_path", "") or ""
            )
            census_hash = str(
                manifest.get("adaptation_census_sha256", "") or ""
            )
            if census_hash and _file_sha256(census_path) != census_hash:
                result["execution_failure_reasons"].append(
                    "profile split census hash mismatch"
                )
            adaptation = {
                int(value)
                for value in manifest.get("adaptation_indices", [])
            }
            adaptation_pool = {
                int(value)
                for value in manifest.get(
                    "adaptation_pool_indices", []
                )
            }
            evaluation = {
                int(value)
                for value in manifest.get("evaluation_indices", [])
            }
            if not adaptation.issubset(adaptation_pool):
                result["execution_failure_reasons"].append(
                    "adaptation indices are not a subset of adaptation pool"
                )
            if adaptation & evaluation:
                result["execution_failure_reasons"].append(
                    "adaptation and evaluation indices overlap"
                )
            for episode in episodes:
                event_index = int(episode.get("event_index", -1))
                phase = str(episode.get("experiment_phase", ""))
                if phase == "adaptation" and event_index not in adaptation:
                    result["execution_failure_reasons"].append(
                        "episode adaptation phase contradicts split manifest"
                    )
                    break
                if phase == "evaluation" and event_index not in evaluation:
                    result["execution_failure_reasons"].append(
                        "episode evaluation phase contradicts split manifest"
                    )
                    break
    checkpoint_path = run_dir / "episode_checkpoint.json"
    if checkpoint_path.exists():
        checkpoint = load_json(checkpoint_path)
        if checkpoint.get("completed") is not True:
            result["execution_failure_reasons"].append(
                "episode checkpoint is not complete"
            )
        checkpoint_ids = {
            int(value)
            for value in checkpoint.get(
                "completed_event_indices", []
            )
        }
        failed_events = checkpoint.get("failed_events") or {}
        if failed_events:
            result["execution_failure_reasons"].append(
                "episode checkpoint contains failed events"
            )
        if checkpoint_ids != set(event_ids):
            result["execution_failure_reasons"].append(
                "episode checkpoint indices != episode_summary indices"
            )
        if len(checkpoint_ids) != int(summary.get("total_events") or 0):
            result["execution_failure_reasons"].append(
                "episode checkpoint count != total_events"
            )
        checkpoint_summary = checkpoint.get("summary") or {}
        if int(checkpoint_summary.get("total_events") or 0) != int(
            summary.get("total_events") or 0
        ):
            result["execution_failure_reasons"].append(
                "episode checkpoint summary != final summary"
            )
        expected_fingerprint = str(
            config.get("experiment_fingerprint", "") or ""
        )
        if (
            expected_fingerprint
            and checkpoint.get("experiment_fingerprint")
            != expected_fingerprint
        ):
            result["execution_failure_reasons"].append(
                "episode checkpoint fingerprint mismatch"
            )
        expected_method = str(config.get("method_version", "") or "")
        if (
            expected_method
            and checkpoint.get("method_version") != expected_method
        ):
            result["execution_failure_reasons"].append(
                "episode checkpoint method version mismatch"
            )
        protocol = summary.get("profile_protocol") or {}
        split_hash = str(
            protocol.get("split_manifest_sha256", "") or ""
        )
        checkpoint_split_hash = str(
            checkpoint.get("split_manifest_sha256", "") or ""
        )
        if split_hash and checkpoint_split_hash != split_hash:
            result["execution_failure_reasons"].append(
                "episode checkpoint split manifest hash mismatch"
            )
    elif config.get("experiment_fingerprint"):
        result["execution_failure_reasons"].append(
            "episode_checkpoint.json missing for fingerprinted run"
        )

    result["execution_valid"] = not result["execution_failure_reasons"]
    result["valid"] = result["execution_valid"]
    result["failure_reasons"] = list(result["execution_failure_reasons"])
    return result


def latest_usable_completed_statuses(
    experiment_dir: str | Path,
    *,
    require_quality: bool = False,
) -> list[dict]:
    from .experiment_matrix import expand_jobs

    experiment_dir = Path(experiment_dir)
    snapshot = experiment_dir / "config.snapshot.json"
    if not snapshot.exists():
        return []
    jobs = expand_jobs(load_json(snapshot))
    expected_by_id = {job.job_id: job.to_dict() for job in jobs}
    fingerprints = expected_fingerprints_for_experiment(experiment_dir)
    candidates = {}
    for status in read_jsonl(experiment_dir / "job_status.jsonl"):
        job_id = status.get("job_id")
        if job_id and status.get("status") == "completed":
            candidates.setdefault(str(job_id), []).append(status)

    selected = []
    for job_id, job in expected_by_id.items():
        for status in reversed(candidates.get(job_id, [])):
            if not fingerprint_is_compatible(
                status, fingerprints.get(job_id)
            ):
                continue
            run_dir = status.get("run_dir")
            if not run_dir:
                continue
            validation = validate_run_dir(run_dir, job)
            if not validation.get("execution_valid", False):
                continue
            if require_quality and not validation.get(
                "quality_gate_pass", False
            ):
                continue
            normalized = dict(status)
            normalized["job"] = job
            normalized["validation"] = validation
            selected.append(normalized)
            break
    return selected


def _latest_completed_statuses_for_audit(
    experiment_dir: str | Path,
) -> list[dict]:
    latest = {}
    for status in read_jsonl(Path(experiment_dir) / "job_status.jsonl"):
        if status.get("status") != "completed":
            continue
        job_id = status.get("job_id")
        if job_id:
            latest[str(job_id)] = status
    return list(latest.values())


def matrix_completion_status(experiment_dir: str | Path) -> dict:
    from .experiment_matrix import expand_jobs

    experiment_dir = Path(experiment_dir)
    snapshot = experiment_dir / "config.snapshot.json"
    if not snapshot.exists():
        result = {
            "matrix_complete": False,
            "primary_matrix_ready": False,
            "reason": "config.snapshot.json missing",
            "expected_jobs": 0,
            "usable_jobs": 0,
            "missing_job_ids": [],
            "invalid_job_ids": [],
            "quality_failed_job_ids": [],
        }
        write_json(experiment_dir / "matrix_completion.json", result)
        return result

    jobs = expand_jobs(load_json(snapshot))
    expected_by_id = {job.job_id: job.to_dict() for job in jobs}
    expected_fingerprints = expected_fingerprints_for_experiment(
        experiment_dir
    )
    completed_by_job = {}
    for status in read_jsonl(experiment_dir / "job_status.jsonl"):
        job_id = status.get("job_id")
        if job_id and status.get("status") == "completed":
            completed_by_job.setdefault(str(job_id), []).append(status)

    missing = []
    invalid = []
    usable = []
    quality_failed = []
    for job_id, job in expected_by_id.items():
        candidates = completed_by_job.get(job_id, [])
        if not candidates:
            missing.append(job_id)
            continue

        selected_validation = None
        for status in reversed(candidates):
            if not fingerprint_is_compatible(
                status, expected_fingerprints.get(job_id)
            ):
                continue
            run_dir = status.get("run_dir")
            if not run_dir:
                continue
            validation = validate_run_dir(run_dir, job)
            if validation.get("execution_valid", False):
                selected_validation = validation
                break

        if selected_validation is None:
            invalid.append(job_id)
            continue
        usable.append(job_id)
        if selected_validation.get("quality_gate_pass") is False:
            quality_failed.append(job_id)

    matrix_complete = (
        len(usable) == len(expected_by_id)
        and not missing
        and not invalid
    )
    result = {
        "matrix_complete": matrix_complete,
        "primary_matrix_ready": matrix_complete and not quality_failed,
        "expected_jobs": len(expected_by_id),
        "usable_jobs": len(usable),
        "missing_jobs": len(missing),
        "invalid_jobs": len(invalid),
        "quality_failed_jobs": len(quality_failed),
        "missing_job_ids": sorted(missing),
        "invalid_job_ids": sorted(invalid),
        "quality_failed_job_ids": sorted(quality_failed),
    }
    write_json(experiment_dir / "matrix_completion.json", result)
    return result


def validate_experiment_dir(experiment_dir: str | Path) -> list[dict]:
    experiment_dir = Path(experiment_dir)
    expected = expected_fingerprints_for_experiment(experiment_dir)
    rows = []
    for status in _latest_completed_statuses_for_audit(experiment_dir):
        run_dir = status.get("run_dir")
        job_id = str(status.get("job_id"))
        job = status.get("job", {})
        validation = validate_run_dir(run_dir, job) if run_dir else {
            "valid": False,
            "failure_reasons": ["missing run_dir"],
            "execution_valid": False,
            "execution_failure_reasons": ["missing run_dir"],
            "quality_gate_pass": False,
            "quality_failures": [],
            "quality_observations": [],
        }
        compatible = bool(
            expected
            and job_id in expected
            and fingerprint_is_compatible(status, expected.get(job_id))
        )
        rows.append({
            "job_id": status.get("job_id"),
            "run_dir": run_dir,
            "valid": validation.get("valid"),
            "failure_reasons": "; ".join(validation.get("failure_reasons", [])),
            "execution_valid": validation.get("execution_valid"),
            "execution_failure_reasons": "; ".join(
                validation.get("execution_failure_reasons", [])
            ),
            "quality_gate_pass": validation.get("quality_gate_pass"),
            "quality_failures": "; ".join(validation.get("quality_failures", [])),
            "quality_observations": "; ".join(
                validation.get("quality_observations", [])
            ),
            "current_method_compatible": compatible,
            "usable_for_current_method": bool(
                validation.get("execution_valid") and compatible
            ),
        })

    completion = matrix_completion_status(experiment_dir)
    represented = {str(row.get("job_id")) for row in rows}
    for job_id in completion.get("missing_job_ids", []):
        if job_id in represented:
            continue
        rows.append({
            "job_id": job_id,
            "run_dir": "",
            "valid": False,
            "failure_reasons": "expected matrix job missing",
            "execution_valid": False,
            "execution_failure_reasons": "expected matrix job missing",
            "quality_gate_pass": False,
            "quality_failures": "",
            "quality_observations": "",
            "current_method_compatible": False,
            "usable_for_current_method": False,
        })

    write_csv(
        experiment_dir / "validation_summary.csv",
        rows,
        fieldnames=VALIDATION_SUMMARY_FIELDS,
    )
    return rows


def main():
    parser = argparse.ArgumentParser(description="Validate ResponsiveGPT experiment runs.")
    parser.add_argument("--experiment_dir", required=True)
    args = parser.parse_args()
    rows = validate_experiment_dir(args.experiment_dir)
    completion = matrix_completion_status(args.experiment_dir)
    valid_count = sum(1 for row in rows if row.get("execution_valid"))
    usable_count = sum(
        1 for row in rows if row.get("usable_for_current_method")
    )
    print(
        f"Validated {completion.get('expected_jobs', len(rows))} "
        f"expected matrix jobs; "
        f"execution_valid={valid_count}; current_method_usable={usable_count}"
    )


if __name__ == "__main__":
    main()
