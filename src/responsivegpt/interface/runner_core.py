import os
import json
import csv
import hashlib
import random
import time
from collections import Counter
from copy import deepcopy
from pathlib import Path

from dataclasses import asdict

from ..evaluation.metrics import compute_step_metrics
from ..evaluation.classification import compute_confusion_and_scores
from ..evaluation.trigger_plotter import TriggerPlotter
from .llm_call_policy import should_call_llm, fallback_decision_from_physics

from ..evaluation.safety_metrics import (
    thresholds_for_dataset,
    compute_frame_safety_metrics,
    aggregate_episode_safety_metrics,
    compute_llm_physics_alignment,
    compute_behavior_safety_metrics,
)

from .adapters.adapter_factory import build_event_adapter, build_sequence_adapter
from ..evaluation.round_labels import derive_round_risk_label_from_summary_row
from ..evaluation.ind_labels import derive_ind_risk_label

from ..application.planning_memory import PlanningMemory
from ..application.planning_service import PlanningService
from ..infrastructure.llm_jiekou import LLMBudgetExceeded
from ..experiments.stratified_sampler import _allocate_quotas
from ..application.planning_formatter import (
    summarize_scene,
    summarize_safety,
    summarize_decision,
    compact_json,
)
from ..evaluation.planning_quality import compute_planning_quality

from ..rag import (
    RAGOrchestrator,
    validate_grounding,
    repair_decision_evidence_fields,
    compute_rag_metrics,
)


def append_jsonl(path: str, obj: dict):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def _read_jsonl(path: str) -> list[dict]:
    if not os.path.exists(path):
        return []
    rows = []
    with open(path, "r", encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _atomic_write_json(path: str, obj: dict) -> None:
    temp_path = f"{path}.tmp"
    with open(temp_path, "w", encoding="utf-8") as stream:
        json.dump(obj, stream, ensure_ascii=False, indent=2)
    os.replace(temp_path, path)


def _file_sha256(path: str) -> str:
    file_path = Path(path)
    if not path or not file_path.is_file():
        return ""
    digest = hashlib.sha256()
    with file_path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _profile_stratum_key(row: dict) -> tuple[str, str, str, str]:
    return (
        str(row.get("risk_stratum", "unknown")),
        str(row.get("event_type", row.get("eventType", "unknown"))),
        str(row.get("dataset_risk_label", row.get("risk_label", ""))),
        str(row.get("vru_present", row.get("vrus_present", ""))),
    )


def _serialized_stratum_counts(
    event_indices: list[int] | set[int],
    strata_rows: list[dict] | None,
) -> list[dict]:
    if not strata_rows:
        return []
    counts = Counter(
        _profile_stratum_key(strata_rows[event_index])
        for event_index in event_indices
    )
    return [
        {
            "risk_stratum": key[0],
            "event_type": key[1],
            "dataset_risk_label": key[2],
            "vru_present": key[3],
            "count": count,
        }
        for key, count in sorted(counts.items())
    ]


def _capture_episode_state(
    summary: dict,
    service,
    rag_phase_stats: dict,
    *,
    processed_events: int,
    adapted_profile_saved: bool,
) -> dict:
    return {
        "summary": deepcopy(summary),
        "profile": deepcopy(service.repo.load()),
        "llm_usage_state": deepcopy(
            service.llm.export_usage_state()
        ),
        "rag_phase_stats": deepcopy(rag_phase_stats),
        "processed_events": processed_events,
        "adapted_profile_saved": adapted_profile_saved,
    }


def _restore_episode_state(
    snapshot: dict,
    summary: dict,
    service,
    rag_phase_stats: dict,
) -> tuple[int, bool]:
    summary.clear()
    summary.update(deepcopy(snapshot["summary"]))
    service.repo.save(deepcopy(snapshot["profile"]))
    service.llm.import_usage_state(
        deepcopy(snapshot["llm_usage_state"])
    )
    rag_phase_stats.clear()
    rag_phase_stats.update(
        deepcopy(snapshot["rag_phase_stats"])
    )
    return (
        int(snapshot["processed_events"]),
        bool(snapshot["adapted_profile_saved"]),
    )


def _prune_jsonl_to_events(path: str, completed: set[int]) -> None:
    if not os.path.exists(path):
        return
    rows = [
        row
        for row in _read_jsonl(path)
        if int(row.get("event_index", -1)) in completed
    ]
    with open(path, "w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")


def _prune_frame_csv_to_events(path: str, completed: set[int]) -> None:
    if not os.path.exists(path):
        return
    with open(path, "r", encoding="utf-8", newline="") as stream:
        rows = list(csv.reader(stream))
    if not rows:
        return
    kept = [rows[0]]
    for row in rows[1:]:
        try:
            event_index = int(row[0])
        except Exception:
            continue
        if event_index in completed:
            kept.append(row)
    with open(path, "w", encoding="utf-8", newline="") as stream:
        csv.writer(stream).writerows(kept)


def safe_len(x) -> int:
    if x is None:
        return 0
    try:
        return len(x)
    except Exception:
        return 0


def safe_trigger_type(trigger_item) -> str:
    if isinstance(trigger_item, dict):
        return str(trigger_item.get("trigger_type", "unknown"))
    if hasattr(trigger_item, "trigger_type"):
        return str(trigger_item.trigger_type)
    return "unknown"


def safe_dict(obj):
    if isinstance(obj, dict):
        return obj
    if hasattr(obj, "__dict__"):
        return dict(obj.__dict__)
    return {"value": str(obj)}


def safe_list_dict(items):
    return [safe_dict(x) for x in items] if items else []


def derive_dataset_risk_label(dataset: str, row: dict, args) -> bool:
    dataset = dataset.lower()

    if dataset == "highd":
        try:
            min_ttc = float(row["minTTC"]) if row.get("minTTC") else None
        except Exception:
            min_ttc = None

        try:
            min_thw = float(row["minTHW"]) if row.get("minTHW") else None
        except Exception:
            min_thw = None

        if min_ttc is not None and min_ttc < 3.0:
            return True
        if min_thw is not None and min_thw < 0.5:
            return True
        return False

    if dataset == "round":
        return derive_round_risk_label_from_summary_row(
            row,
            ttc_threshold=args.ttc_threshold,
            distance_threshold=args.distance_threshold,
        )

    if dataset == "ind":
        return derive_ind_risk_label(
            row,
            ttc_threshold=args.ttc_threshold,
            distance_threshold=args.distance_threshold,
            drac_threshold=args.drac_threshold,
        )

    raise ValueError("dataset must be one of: highd / round / ind")


def init_summary(args, template_profile_path: str):
    return {
        "dataset": args.dataset,
        "mode": args.mode,
        "dry_run": bool(getattr(args, "dry_run", False)),
        "inspect_only": bool(getattr(args, "inspect_only", False)),
        "total_events": 0,
        "total_frames": 0,
        "dataset_risk_true": 0,
        "episode_llm_violation_true": 0,
        "episode_agreement": 0,
        "missing_files": 0,
        "missing_clips": 0,
        "missing_scenes": 0,
        "empty_sequences": 0,
        "rows_seen": 0,
        "row_start_index": int(getattr(args, "start_index", 0) or 0),
        "row_end_index": int(getattr(args, "end_index", -1) or -1),
        "shard_id": int(getattr(args, "shard_id", -1) or -1),
        "num_shards": int(getattr(args, "num_shards", 0) or 0),
        "rows_skipped_by_range": 0,
        "rows_skipped_by_shard": 0,
        "frame_selection": str(getattr(args, "frame_selection", "all") or "all"),
        "critical_top_k": int(getattr(args, "critical_top_k", 5) or 5),
        "candidate_frames": 0,
        "selected_frames": 0,
        "max_llm_calls": int(getattr(args, "max_llm_calls", 0) or 0),
        "max_planning_calls": int(getattr(args, "max_planning_calls", 0) or 0),
        "max_reactive_api_attempts": int(
            getattr(args, "max_reactive_api_attempts", 0) or 0
        ),
        "max_reactive_tokens": int(
            getattr(args, "max_reactive_tokens", 0) or 0
        ),
        "max_planning_api_attempts": int(
            getattr(args, "max_planning_api_attempts", 0) or 0
        ),
        "max_planning_tokens": int(
            getattr(args, "max_planning_tokens", 0) or 0
        ),
        "llm_budget_exhausted_frames": 0,
        "planning_budget_exhausted_frames": 0,
        "llm_budget_exhausted": False,
        "planning_budget_exhausted": False,
        
        # LLM 调用统计
        "llm_calls": 0,
        "llm_attempts": 0,
        "llm_error_count": 0,
        "timeout_count": 0,
        "connection_error_count": 0,
        "rate_limit_count": 0,
        "other_llm_error_count": 0,
        "non_llm_frames": 0,
        "llm_call_rate": 0.0,
        "fallback_frame_count": 0,
        "fallback_frame_rate": 0.0,
        "reactive_frames": 0,
        "dry_run_frames": 0,
        "inspect_frames": 0,

        "planning_calls": 0,
        "planning_failures": 0,
        "planning_llm_attempts": 0,

        "profile_name": args.profile_name,
        "template_profile_path": template_profile_path,
        "experiment_fingerprint": str(
            getattr(args, "experiment_fingerprint", "") or ""
        ),
        "method_version": str(getattr(args, "method_version", "") or ""),
        "repeat_seed": int(getattr(args, "repeat_seed", 0) or 0),
        "episode_order_seed": int(
            getattr(args, "episode_order_seed", 0) or 0
        ),
        "ablation": {
            "use_trigger": bool(args.use_trigger),
            "use_profile_learner": bool(args.use_profile_learner),
            "use_retriever": bool(args.use_retriever),
            "history_window": args.history_window,
        },
    }


def make_empty_rag_result(dataset, scene, metadata=None, budget="reactive"):
    return {
        "rag_mode": "none",
        "rag_query": {
            "query_text": "",
            "dataset": dataset,
            "event_type": getattr(scene, "event_type", None),
            "metadata": metadata or {},
        },
        "retrieved": [],
        "reranked": [],
        "evidence_pack": {
            "budget": budget,
            "num_evidence": 0,
            "items": [],
            "evidence_text": "No RAG evidence available.",
        },
    }


def make_empty_grounding():
    return {
        "available_evidence_ids": [],
        "used_evidence_ids": [],
        "valid_used_evidence_ids": [],
        "hallucinated_evidence_ids": [],
        "evidence_support_level": "none",
        "is_grounded": False,
    }


def select_critical_frame_positions(scenes, frame_safety_metrics, top_k: int = 5) -> list[int]:
    if not scenes:
        return []

    top_k = max(1, int(top_k or 1))
    selected = {0, len(scenes) - 1}

    scored = []
    for pos, safety in enumerate(frame_safety_metrics):
        risk = getattr(safety, "physical_risk_index", None)
        if isinstance(risk, (int, float)):
            scored.append((float(risk), pos))

    scored.sort(key=lambda item: (-item[0], item[1]))
    for _, pos in scored[:top_k]:
        selected.add(pos)

    # Keep local context around the most critical frames so sparse runs still
    # preserve short-term interaction dynamics instead of isolated snapshots.
    for _, pos in scored[: min(top_k, 3)]:
        if pos > 0:
            selected.add(pos - 1)
        if pos + 1 < len(scenes):
            selected.add(pos + 1)

    return sorted(selected)


def compact_evidence_pack_for_trace(evidence_pack: dict) -> dict:
    evidence_pack = evidence_pack or {}
    compact_items = []
    for item in evidence_pack.get("items", []) or []:
        if not isinstance(item, dict):
            continue
        compact_items.append({
            "evidence_id": item.get("evidence_id"),
            "doc_type": item.get("doc_type"),
            "title": item.get("title"),
            "source": item.get("source"),
            "severity": item.get("severity"),
            "score": item.get("score"),
            "rerank_score": item.get("rerank_score"),
        })
    return {
        "budget": evidence_pack.get("budget"),
        "num_evidence": evidence_pack.get("num_evidence", len(compact_items)),
        "items": compact_items,
        "evidence_text": "",
    }


def error_kind(exc: Exception) -> str:
    text = str(exc).lower()
    name = exc.__class__.__name__.lower()
    if "timeout" in text or "timed out" in text or "timeout" in name:
        return "timeout"
    if "connection" in text or "connection" in name:
        return "connection"
    if "rate" in text and "limit" in text:
        return "rate_limit"
    return "other"


def select_experiment_rows(
    rows,
    *,
    start_index: int = 0,
    end_index: int = -1,
    shard_id: int = -1,
    num_shards: int = 0,
    episode_order_seed: int = 0,
) -> tuple[list[tuple[int, object]], dict[str, int]]:
    selected = []
    skipped_by_range = 0
    skipped_by_shard = 0
    shard_enabled = shard_id >= 0 and num_shards > 0
    for idx, row in enumerate(rows):
        if idx < start_index or (end_index >= 0 and idx >= end_index):
            skipped_by_range += 1
            continue
        if shard_enabled and (idx % num_shards) != shard_id:
            skipped_by_shard += 1
            continue
        selected.append((idx, row))
    if episode_order_seed:
        random.Random(episode_order_seed).shuffle(selected)
    return selected, {
        "rows_skipped_by_range": skipped_by_range,
        "rows_skipped_by_shard": skipped_by_shard,
    }


def select_profile_adaptation_indices(
    indexed_rows: list[tuple[int, object]],
    requested: int,
    *,
    strata_rows: list[dict] | None = None,
) -> set[int]:
    requested = max(0, int(requested or 0))
    if requested == 0:
        return set()

    def stratum_key(row):
        getter = row.get if hasattr(row, "get") else lambda _key, default=None: default
        return (
            str(getter("risk_stratum", "unknown")),
            str(getter("event_type", getter("eventType", "unknown"))),
            str(getter("dataset_risk_label", getter("risk_label", ""))),
            str(getter("vru_present", getter("vrus_present", ""))),
        )

    def row_for_stratum(event_index, row):
        if (
            strata_rows is not None
            and 0 <= event_index < len(strata_rows)
        ):
            return strata_rows[event_index]
        return row

    remaining = Counter(
        stratum_key(row_for_stratum(event_index, row))
        for event_index, row in indexed_rows
    )
    selected = set()
    for event_index, row in indexed_rows:
        if len(selected) >= requested:
            break
        key = stratum_key(row_for_stratum(event_index, row))
        if remaining[key] <= 1:
            continue
        selected.add(event_index)
        remaining[key] -= 1
    return selected


def select_profile_adaptation_pool(
    indexed_rows: list[tuple[int, object]],
    requested: int,
    *,
    strata_rows: list[dict],
    allocation: str = "neyman",
) -> list[int]:
    requested = max(0, int(requested or 0))
    if requested == 0:
        return []

    def stratum_key(row):
        return (
            str(row.get("risk_stratum", "unknown")),
            str(row.get("event_type", "unknown")),
            str(row.get("dataset_risk_label", "")),
            str(row.get("vru_present", "")),
        )

    groups = {}
    event_rows = {}
    for event_index, _ in indexed_rows:
        row = strata_rows[event_index]
        key = stratum_key(row)
        groups.setdefault(key, []).append(row)
        event_rows.setdefault(key, []).append(event_index)

    # Reserve one episode from every observed stratum for evaluation.
    quota_groups = {
        key: rows[:-1]
        for key, rows in groups.items()
        if len(rows) > 1
    }
    capacity = sum(len(rows) for rows in quota_groups.values())
    target = min(requested, capacity)
    quotas = _allocate_quotas(
        quota_groups, target, method=allocation
    )
    chosen = set()
    for key, quota in quotas.items():
        chosen.update(event_rows[key][:int(quota or 0)])
    return [
        event_index
        for event_index, _ in indexed_rows
        if event_index in chosen
    ]


def _canonical_identity(row: dict) -> tuple[str, str, str, str]:
    def first(*keys):
        for key in keys:
            value = row.get(key)
            if value not in (None, ""):
                text = str(value)
                if text.lstrip("-").isdigit():
                    return str(int(text))
                return text
        return ""

    return (
        first("recording_id", "recordingId", "recording_prefix", "prefix"),
        first("event_type", "eventType", "pair_type"),
        first("start_frame", "startFrame"),
        first("end_frame", "endFrame"),
    )


def load_profile_adaptation_strata(
    summary_csv: str,
    dataset: str,
) -> tuple[list[dict] | None, str]:
    summary_path = Path(summary_csv)
    with summary_path.open(
        "r", encoding="utf-8-sig", newline=""
    ) as stream:
        summary_rows = [dict(row) for row in csv.DictReader(stream)]
    expected_rows = len(summary_rows)
    seed_suffix = ""
    if "_seed" in summary_path.stem:
        seed_suffix = summary_path.stem.rsplit("_seed", 1)[1]
    candidates = [
        summary_path.parent / f"{dataset}_cumulative_census.csv",
    ]
    if seed_suffix:
        candidates.append(
            summary_path.parent
            / f"{dataset}_core_sample_census_seed{seed_suffix}.csv"
        )
    for path in candidates:
        if not path.exists():
            continue
        with path.open("r", encoding="utf-8-sig", newline="") as stream:
            rows = [dict(row) for row in csv.DictReader(stream)]
        if len(rows) != expected_rows:
            continue
        row_indexes = [
            int(row.get("row_index", -1)) for row in rows
        ]
        if (
            any(index < 0 for index in row_indexes)
            or len(set(row_indexes)) != expected_rows
        ):
            raise ValueError(
                f"Adaptation census row_index is missing or duplicated: {path}"
            )
        for index, (summary_row, census_row) in enumerate(
            zip(summary_rows, rows)
        ):
            summary_identity = _canonical_identity(summary_row)
            census_identity = _canonical_identity(census_row)
            comparable = [
                (left, right)
                for left, right in zip(summary_identity, census_identity)
                if left and right
            ]
            if comparable and any(
                left != right for left, right in comparable
            ):
                raise ValueError(
                    "Adaptation census identity mismatch at row "
                    f"{index}: {path}"
                )
        return rows, str(path)
    return None, ""


def run_interaction_experiment(args, ctx):
    logger = ctx["logger"]
    service = ctx["service"]
    effective_driver_type = ctx["effective_driver_type"]

    planning_interval = getattr(args, "planning_interval", 20)
    planning_risk_threshold = getattr(args, "planning_risk_threshold", 0.45)
    planning_time_horizon_s = getattr(args, "planning_time_horizon_s", 3.0)
    use_planning_thread = bool(getattr(args, "use_planning_thread", 1))

    inspect_only = bool(getattr(args, "inspect_only", False))
    dry_run = bool(getattr(args, "dry_run", False))
    event_adapter = build_event_adapter(args.dataset, args.summary_csv)
    thresholds = thresholds_for_dataset(args.dataset)
    planning_service = PlanningService(service.llm)

    frame_metrics_path = os.path.join(logger.run_dir, "frame_metrics.csv")
    episode_summary_path = os.path.join(logger.run_dir, "episode_summary.jsonl")
    profile_trace_path = os.path.join(logger.run_dir, "profile_trace.jsonl")
    trigger_trace_path = os.path.join(logger.run_dir, "trigger_trace.jsonl")
    profile_delta_path = os.path.join(logger.run_dir, "profile_delta.jsonl")
    guardrail_trace_path = os.path.join(logger.run_dir, "guardrail_trace.jsonl")
    planning_trace_path = os.path.join(logger.run_dir, "planning_trace.jsonl")
    rag_trace_path = os.path.join(logger.run_dir, "rag_trace.jsonl")
    checkpoint_path = os.path.join(
        logger.run_dir, "episode_checkpoint.json"
    )
    checkpoint = {}
    completed_event_indices = set()
    if os.path.exists(checkpoint_path):
        with open(checkpoint_path, "r", encoding="utf-8") as stream:
            checkpoint = json.load(stream)
        expected_fingerprint = str(
            getattr(args, "experiment_fingerprint", "") or ""
        )
        if (
            expected_fingerprint
            and checkpoint.get("experiment_fingerprint")
            != expected_fingerprint
        ):
            raise ValueError(
                "Episode checkpoint fingerprint does not match this job"
            )
        completed_event_indices = {
            int(value)
            for value in checkpoint.get(
                "completed_event_indices", []
            )
        }
        for path in (
            logger.decisions_path,
            episode_summary_path,
            profile_trace_path,
            trigger_trace_path,
            profile_delta_path,
            guardrail_trace_path,
            planning_trace_path,
            rag_trace_path,
        ):
            _prune_jsonl_to_events(path, completed_event_indices)
        _prune_frame_csv_to_events(
            frame_metrics_path, completed_event_indices
        )
        if isinstance(checkpoint.get("profile"), dict):
            service.repo.save(checkpoint["profile"])
        service.llm.import_usage_state(
            checkpoint.get("llm_usage_state", {})
        )

    rag = RAGOrchestrator(
        retriever=getattr(service, "retriever", None),
        rag_mode=getattr(args, "rag_mode", "full"),
        budget=getattr(args, "rag_budget", "reactive"),
        top_k=getattr(args, "rag_top_k", 12),
    )

    global_frame_records_for_rag = []

    if not os.path.exists(frame_metrics_path):
        with open(
            frame_metrics_path, "w", newline="", encoding="utf-8"
        ) as f:
            writer = csv.writer(f)
            writer.writerow([
            "event_index",
            "recordingId",
            "event_id",
            "eventType",
            "pair_type",
            "location_id",
            "ego_id",
            "other_id",
            "frame_index",

            "ttc_s",
            "thw_s",
            "drac_mps2",
            "dcpa_m",
            "ttca_s",
            "predicted_ttc_s",
            "min_future_distance_m",
            "physical_risk_index",
            "physical_risk_level",

            "is_violation",
            "ego_speed_mps",
            "rel_speed_mps",
            "headway_m",
            "vrus_present",
            "num_triggers",
            "num_rules",
            "num_law_evidence",
            "num_case_evidence",
            "num_scenario_evidence",

            "llm_called",
            "decision_source",
            "planning_active",
            "planning_age_frames",
            "planning_risk_level",
            "planning_strategy",

            # ===== RAG v1 grounding columns =====
            "rag_mode",
            "num_rag_evidence",
            "used_evidence_count",
            "is_grounded",
            "evidence_support_level",
            "hallucinated_evidence_count",
            ])

    summary = checkpoint.get("summary") or init_summary(
        args, template_profile_path=ctx.get("template_profile_path", "")
    )
    summary.setdefault("planning_calls", 0)
    summary.setdefault("planning_enabled", bool(getattr(args, "use_planning_thread", 0)))

    all_y_true = []
    all_y_pred = []
    global_trigger_stats = Counter()
    global_episode_safety_records = []
    global_alignment_records = []
    global_behavior_records = []
    global_planning_records = []
    global_planning_quality_records = []
    active_profile_snapshot = safe_dict(service.repo.load())

    processed_events = len(completed_event_indices)
    start_index = int(getattr(args, "start_index", 0) or 0)
    end_index = int(getattr(args, "end_index", -1) or -1)
    shard_id = int(getattr(args, "shard_id", -1) or -1)
    num_shards = int(getattr(args, "num_shards", 0) or 0)

    episode_order_seed = int(getattr(args, "episode_order_seed", 0) or 0)
    indexed_rows, selection_stats = select_experiment_rows(
        event_adapter.iter_rows(),
        start_index=start_index,
        end_index=end_index,
        shard_id=shard_id,
        num_shards=num_shards,
        episode_order_seed=episode_order_seed,
    )
    summary.update(selection_stats)

    profile_adaptation_episodes = max(
        0, int(getattr(args, "profile_adaptation_episodes", 0) or 0)
    )
    profile_adaptation_pool_episodes = max(
        profile_adaptation_episodes,
        int(
            getattr(args, "profile_adaptation_pool_episodes", 0)
            or profile_adaptation_episodes
        ),
    )
    profile_protocol_enabled = bool(
        getattr(args, "profile_protocol_enabled", 0)
    )
    previous_protocol = summary.get("profile_protocol") or {}
    summary["profile_protocol"] = {
        "adaptation_episodes_requested": profile_adaptation_episodes,
        "adaptation_pool_episodes_requested": (
            profile_adaptation_pool_episodes
        ),
        "adaptation_events": int(
            previous_protocol.get("adaptation_events", 0) or 0
        ),
        "evaluation_events": int(
            previous_protocol.get("evaluation_events", 0) or 0
        ),
        "adaptation_frames": int(
            previous_protocol.get("adaptation_frames", 0) or 0
        ),
        "evaluation_frames": int(
            previous_protocol.get("evaluation_frames", 0) or 0
        ),
        "profile_frozen_during_evaluation": bool(
            profile_protocol_enabled
        ),
        "enabled": profile_protocol_enabled,
    }
    adaptation_strata_rows, adaptation_census_path = (None, "")
    if profile_protocol_enabled:
        adaptation_strata_rows, adaptation_census_path = (
            load_profile_adaptation_strata(
                args.summary_csv,
                args.dataset,
            )
        )
        if adaptation_strata_rows is None:
            raise FileNotFoundError(
                "Profile protocol requires an identity-aligned sample census "
                f"for {args.summary_csv}"
            )
    summary["profile_protocol"]["adaptation_census_path"] = (
        adaptation_census_path
    )
    summary["profile_protocol"]["adaptation_strata_available"] = bool(
        adaptation_strata_rows
    )
    adaptation_pool_order = (
        select_profile_adaptation_pool(
            indexed_rows,
            profile_adaptation_pool_episodes,
            strata_rows=adaptation_strata_rows,
            allocation=str(
                getattr(args, "profile_adaptation_allocation", "neyman")
            ),
        )
        if profile_protocol_enabled
        else []
    )
    adaptation_pool_indices = set(adaptation_pool_order)
    adaptation_indices = set(
        adaptation_pool_order[:profile_adaptation_episodes]
    )
    summary["profile_protocol"]["adaptation_allocation"] = str(
        getattr(args, "profile_adaptation_allocation", "neyman")
    )
    summary["profile_protocol"]["adaptation_pool_episodes_actual"] = len(
        adaptation_pool_indices
    )
    summary["profile_protocol"]["adaptation_episodes_actual"] = len(
        adaptation_indices
    )
    evaluation_indices = [
        event_index
        for event_index, _ in indexed_rows
        if event_index not in adaptation_pool_indices
    ]
    eligible_indices = sorted(
        set(adaptation_indices) | set(evaluation_indices)
    )
    split_manifest = {
        "dataset": args.dataset,
        "episode_order_seed": episode_order_seed,
        "experiment_fingerprint": str(
            getattr(args, "experiment_fingerprint", "") or ""
        ),
        "method_version": str(
            getattr(args, "method_version", "") or ""
        ),
        "summary_csv": str(args.summary_csv),
        "summary_csv_sha256": _file_sha256(args.summary_csv),
        "adaptation_census_path": adaptation_census_path,
        "adaptation_census_sha256": _file_sha256(
            adaptation_census_path
        ),
        "allocation": summary["profile_protocol"][
            "adaptation_allocation"
        ],
        "adaptation_pool_episodes_requested": (
            profile_adaptation_pool_episodes
        ),
        "adaptation_pool_episodes_actual": len(
            adaptation_pool_indices
        ),
        "adaptation_episodes_requested": (
            profile_adaptation_episodes
        ),
        "adaptation_episodes_actual": len(adaptation_indices),
        "evaluation_episodes_selected": len(evaluation_indices),
        "adaptation_pool_indices": adaptation_pool_order,
        "adaptation_indices": sorted(adaptation_indices),
        "evaluation_indices": evaluation_indices,
        "eligible_indices": eligible_indices,
        "adaptation_pool_stratum_counts": (
            _serialized_stratum_counts(
                adaptation_pool_indices, adaptation_strata_rows
            )
        ),
        "adaptation_stratum_counts": _serialized_stratum_counts(
            adaptation_indices, adaptation_strata_rows
        ),
        "evaluation_stratum_counts": _serialized_stratum_counts(
            evaluation_indices, adaptation_strata_rows
        ),
    }
    split_payload = json.dumps(
        split_manifest,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    split_manifest["sha256"] = hashlib.sha256(
        split_payload.encode("utf-8")
    ).hexdigest()
    split_manifest_path = os.path.join(
        logger.run_dir, "profile_split_manifest.json"
    )
    with open(split_manifest_path, "w", encoding="utf-8") as stream:
        json.dump(split_manifest, stream, ensure_ascii=False, indent=2)
    summary["profile_protocol"]["split_manifest_path"] = (
        split_manifest_path
    )
    summary["profile_protocol"]["split_manifest_sha256"] = (
        split_manifest["sha256"]
    )
    indexed_rows = [
        item for item in indexed_rows if item[0] in adaptation_indices
    ] + [
        item
        for item in indexed_rows
        if item[0] not in adaptation_pool_indices
    ]
    profile_pilot_limited = bool(
        profile_protocol_enabled
        and int(getattr(args, "limit", 0) or 0) > 0
        and int(getattr(args, "limit", 0) or 0) < len(indexed_rows)
    )
    summary["profile_protocol"]["pilot_limited"] = (
        profile_pilot_limited
    )
    summary["profile_protocol"]["formal_inference_eligible"] = not (
        profile_pilot_limited
    )
    summary["profile_protocol"]["execution_limit"] = int(
        getattr(args, "limit", 0) or 0
    )
    adapted_profile_saved = False
    rag_phase_stats = {
        "adaptation": {"calls": 0, "latencies_ms": []},
        "evaluation": {"calls": 0, "latencies_ms": []},
    }
    if checkpoint.get("rag_phase_stats"):
        rag_phase_stats = checkpoint["rag_phase_stats"]
    failed_events = dict(checkpoint.get("failed_events") or {})

    def commit_episode_checkpoint(
        event_index: int | None = None,
        *,
        completed: bool = False,
    ) -> None:
        if event_index is not None:
            completed_event_indices.add(event_index)
        _atomic_write_json(checkpoint_path, {
            "experiment_fingerprint": str(
                getattr(args, "experiment_fingerprint", "") or ""
            ),
            "method_version": str(
                getattr(args, "method_version", "") or ""
            ),
            "completed_event_indices": sorted(
                completed_event_indices
            ),
            "summary": summary,
            "profile": service.repo.load(),
            "llm_usage_state": service.llm.export_usage_state(),
            "rag_phase_stats": rag_phase_stats,
            "failed_events": failed_events,
            "split_manifest_sha256": split_manifest["sha256"],
            "completed": completed,
        })

    def fail_episode(event_index: int, reason: str) -> None:
        nonlocal processed_events, adapted_profile_saved
        processed_events, adapted_profile_saved = (
            _restore_episode_state(
                episode_snapshot,
                summary,
                service,
                rag_phase_stats,
            )
        )
        failed_events[str(event_index)] = reason
        commit_episode_checkpoint()
        raise RuntimeError(
            f"Episode {event_index} cannot be evaluated: {reason}"
        )

    evaluation_index_set = set(evaluation_indices)
    for episode in _read_jsonl(episode_summary_path):
        if int(episode.get("event_index", -1)) not in evaluation_index_set:
            continue
        global_episode_safety_records.append(
            episode.get("episode_safety") or {}
        )
        global_alignment_records.append(
            episode.get("llm_physics_alignment") or {}
        )
        global_behavior_records.append(
            episode.get("behavior_safety") or {}
        )
        planning_quality = episode.get("planning_quality") or {}
        if planning_quality.get("planning_enabled"):
            global_planning_quality_records.append(planning_quality)
        dataset_risk = bool(episode.get("dataset_risk_label"))
        prediction = bool(episode.get("episode_llm_violation"))
        all_y_true.append(dataset_risk)
        all_y_pred.append(prediction)
        global_trigger_stats.update(
            episode.get("trigger_distribution") or {}
        )
    global_frame_records_for_rag.extend([
        row
        for row in _read_jsonl(logger.decisions_path)
        if int(row.get("event_index", -1)) in evaluation_index_set
    ])

    for idx, row in indexed_rows:
        if idx in completed_event_indices:
            continue
        if args.limit > 0 and processed_events >= args.limit:
            break
        failed_events.pop(str(idx), None)
        episode_snapshot = _capture_episode_state(
            summary,
            service,
            rag_phase_stats,
            processed_events=processed_events,
            adapted_profile_saved=adapted_profile_saved,
        )

        experiment_phase = (
            "adaptation"
            if idx in adaptation_indices
            else "evaluation"
        )
        is_evaluation_phase = experiment_phase == "evaluation"
        episode_feedback = (
            args.feedback
            if not profile_protocol_enabled
            or experiment_phase == "adaptation"
            else ""
        )
        if (
            profile_protocol_enabled
            and is_evaluation_phase
            and not adapted_profile_saved
        ):
            adapted_profile_path = os.path.join(
                logger.run_dir, "adapted_profile.json"
            )
            with open(adapted_profile_path, "w", encoding="utf-8") as stream:
                json.dump(
                    service.repo.load(),
                    stream,
                    ensure_ascii=False,
                    indent=2,
                )
            summary["profile_protocol"][
                "adapted_profile_path"
            ] = adapted_profile_path
            adapted_profile_saved = True
        summary["rows_seen"] += 1
        processed_events += 1
        metadata = event_adapter.row_metadata(row)

        if args.mode == "batch":
            scenes = [event_adapter.row_to_scene(row)]
            sequence_path = None
        else:
            seq_adapter, sequence_path, missing_key = build_sequence_adapter(
                dataset=args.dataset,
                metadata=metadata,
                args=args,
            )

            if hasattr(seq_adapter, "validate_schema"):
                try:
                    seq_adapter.validate_schema()
                except Exception as e:
                    summary["missing_files"] += 1
                    print("[SCHEMA ERROR]", e)
                    fail_episode(idx, f"sequence schema error: {e}")

            if seq_adapter is None:
                summary["missing_files"] += 1
                if missing_key:
                    summary[missing_key] += 1
                if summary["missing_files"] <= 5:
                    print(f"[WARN] sequence file missing: {sequence_path}")
                fail_episode(
                    idx,
                    f"sequence file missing: {sequence_path}",
                )

            scenes = list(seq_adapter.iter_scenes())

        if not scenes:
            summary["empty_sequences"] += 1
            if summary["empty_sequences"] <= 5:
                print("[WARN] empty sequence:", {
                    "event_index": idx,
                    "dataset": args.dataset,
                    "metadata": metadata,
                    "sequence_path": sequence_path,
                })
            fail_episode(
                idx,
                f"empty sequence: {sequence_path}",
            )
        summary["profile_protocol"][
            f"{experiment_phase}_events"
        ] += 1

        dataset_risk = derive_dataset_risk_label(args.dataset, row, args)

        ttc_values = []
        violation_flags = []
        recent_decisions = []
        feedback_consumed = False

        episode_trigger_stats = Counter()
        episode_trigger_count = 0

        frame_safety_metrics_list = []
        decision_list = []
        trigger_list_by_frame = {}
        
        planning_memory = PlanningMemory()
        planning_records = []

        scene_history_for_planning = []
        safety_history_for_planning = []

        last_planning_frame_pos = None
        last_llm_frame_pos = None
        last_llm_risk_level = None
        last_llm_risk_index = None
        last_llm_evidence_ids = ()
        last_llm_planning_update_frame = None

        # ============================================================
        # inspect_only:
        # 只检查 summary → sequence path → scenes 是否能打通
        # 不计算完整 safety metrics
        # 不调用 LLM
        # ============================================================
        if inspect_only:
            append_jsonl(episode_summary_path, {
                "event_index": idx,
                "experiment_phase": experiment_phase,
                "dataset": args.dataset,
                "mode": args.mode,
                "metadata": metadata,
                "sequence_path": sequence_path,
                "dataset_risk_label": dataset_risk,
                "episode_num_frames": len(scenes),
                "inspect_only": True,
                "dry_run": False,
            })

            summary["total_events"] += 1
            summary["total_frames"] += len(scenes)
            summary["candidate_frames"] += len(scenes)
            summary["selected_frames"] += len(scenes)
            summary["inspect_frames"] += len(scenes)
            summary["profile_protocol"][
                f"{experiment_phase}_frames"
            ] += len(scenes)
            if is_evaluation_phase:
                summary["dataset_risk_true"] += int(dataset_risk)
            summary["precision"] = None
            summary["recall"] = None
            summary["f1"] = None
            summary["accuracy"] = None

            print(
                f"[INSPECT] event={idx} "
                f"dataset={args.dataset} "
                f"mode={args.mode} "
                f"frames={len(scenes)} "
                f"risk={dataset_risk} "
                f"sequence_path={sequence_path}"
            )

            commit_episode_checkpoint(idx)
            continue


        # ============================================================
        # dry_run:
        # 计算完整 safety metrics
        # 不调用 LLM
        # ============================================================
        if dry_run:
            frame_safety_metrics_list = []

            for scene in scenes:
                frame_safety = compute_frame_safety_metrics(scene, thresholds)
                frame_safety_metrics_list.append(frame_safety)

                with open(frame_metrics_path, "a", newline="", encoding="utf-8") as f:
                    writer = csv.writer(f)
                    writer.writerow([
                        idx,
                        metadata.get("recordingId"),
                        metadata.get("event_id"),
                        metadata.get("eventType"),
                        metadata.get("pair_type"),
                        metadata.get("location_id"),
                        metadata.get("egoId") or metadata.get("egoTrackId") or metadata.get("ego_track_id"),
                        metadata.get("otherId") or metadata.get("otherTrackId") or metadata.get("trackId_2"),
                        scene.frame_index,

                        "" if frame_safety.ttc_s is None else round(frame_safety.ttc_s, 4),
                        "" if frame_safety.thw_s is None else round(frame_safety.thw_s, 4),
                        "" if frame_safety.drac_mps2 is None else round(frame_safety.drac_mps2, 4),
                        "" if frame_safety.dcpa_m is None else round(frame_safety.dcpa_m, 4),
                        "" if frame_safety.ttca_s is None else round(frame_safety.ttca_s, 4),
                        "" if frame_safety.predicted_ttc_s is None else round(frame_safety.predicted_ttc_s, 4),
                        "" if frame_safety.min_future_distance_m is None else round(frame_safety.min_future_distance_m, 4),
                        "" if frame_safety.physical_risk_index is None else round(frame_safety.physical_risk_index, 4),
                        frame_safety.physical_risk_level,

                        "",  # is_violation
                        scene.ego_speed_mps,
                        scene.rel_speed_mps,
                        scene.headway_m,
                        int(scene.vrus_present),
                        0,
                        0,
                        0,
                        0,
                        0,

                        0,              # llm_called
                        "dry_run",      # decision_source
                        0,              # planning_active
                        "",             # planning_age_frames
                        "",             # planning_risk_level
                        "",             # planning_strategy
                        "none",         # rag_mode
                        0,              # num_rag_evidence
                        0,              # used_evidence_count
                        0,              # is_grounded
                        "none",         # evidence_support_level
                        0,              # hallucinated_evidence_count
                    ])

            episode_safety = aggregate_episode_safety_metrics(frame_safety_metrics_list)

            # dry-run 下也进入 global safety 汇总
            if is_evaluation_phase:
                global_episode_safety_records.append(asdict(episode_safety))

            append_jsonl(episode_summary_path, {
                "event_index": idx,
                "experiment_phase": experiment_phase,
                "dataset": args.dataset,
                "mode": args.mode,
                "metadata": metadata,
                "sequence_path": sequence_path,
                "dataset_risk_label": dataset_risk,
                "episode_num_frames": len(scenes),
                "inspect_only": False,
                "dry_run": True,
                "episode_safety": asdict(episode_safety),
            })

            summary["total_events"] += 1
            summary["total_frames"] += len(scenes)
            summary["candidate_frames"] += len(scenes)
            summary["selected_frames"] += len(scenes)
            summary["dry_run_frames"] += len(scenes)
            summary["profile_protocol"][
                f"{experiment_phase}_frames"
            ] += len(scenes)
            if is_evaluation_phase:
                summary["dataset_risk_true"] += int(dataset_risk)

            print(
                f"[DRY-RUN] event={idx} "
                f"dataset={args.dataset} "
                f"frames={len(scenes)} "
                f"risk={dataset_risk} "
                f"min_ttc={episode_safety.min_ttc_s} "
                f"max_drac={episode_safety.max_drac_mps2} "
                f"min_dcpa={episode_safety.min_dcpa_m} "
                f"risk_exposure={episode_safety.physical_risk_exposure}"
            )

            commit_episode_checkpoint(idx)
            continue

        frame_selection = str(getattr(args, "frame_selection", "all") or "all")
        if frame_selection == "critical":
            all_frame_safety = [compute_frame_safety_metrics(scene, thresholds) for scene in scenes]
            selected_positions = select_critical_frame_positions(
                scenes,
                all_frame_safety,
                top_k=int(getattr(args, "critical_top_k", 5) or 5),
            )
            frame_iter = [
                (frame_pos, scenes[frame_pos], all_frame_safety[frame_pos])
                for frame_pos in selected_positions
            ]
        else:
            frame_iter = [(frame_pos, scene, None) for frame_pos, scene in enumerate(scenes)]

        evaluated_frame_count = len(frame_iter)
        summary["candidate_frames"] += len(scenes)
        summary["selected_frames"] += evaluated_frame_count

        for frame_pos, scene, precomputed_frame_safety in frame_iter:
            summary["reactive_frames"] += 1

            # ==================================================
            # 0. 每帧默认值，保证后面任何地方都能安全访问
            # ==================================================
            rag_result = make_empty_rag_result(
                dataset=args.dataset,
                scene=scene,
                metadata=metadata,
                budget=getattr(args, "rag_budget", "reactive"),
            )

            evidence_pack = rag_result["evidence_pack"]
            grounding = make_empty_grounding()
            output_grounding = make_empty_grounding()

            result = None
            decision = None
            llm_called = False
            # ============================================================
            # 1. 先计算当前帧物理安全指标
            # ============================================================
            frame_safety = (
                precomputed_frame_safety
                if precomputed_frame_safety is not None
                else compute_frame_safety_metrics(scene, thresholds)
            )
            frame_safety_metrics_list.append(frame_safety)

            scene_history_for_planning.append(summarize_scene(scene))
            safety_history_for_planning.append(summarize_safety(frame_safety))

            # ============================================================
            # 3. Planning Thread 调度
            # ============================================================
            planning_enabled = bool(getattr(args, "use_planning_thread", 0))
            planning_mode = str(getattr(args, "planning_mode", "interval_risk") or "interval_risk")
            planning_interval = int(getattr(args, "planning_interval", 20))
            planning_min_gap = int(getattr(args, "planning_min_gap", 10))
            planning_risk_threshold = float(getattr(args, "planning_risk_threshold", 0.45))
            planning_time_horizon_s = float(getattr(args, "planning_time_horizon_s", 3.0))
            planning_max_history = int(getattr(args, "planning_max_history", 12))

            current_frame_id = scene.frame_index if scene.frame_index is not None else frame_pos

            should_plan = False

            if planning_enabled and not dry_run and not inspect_only:
                if frame_pos == 0:
                    should_plan = True
                else:
                    interval_due = planning_interval > 0 and frame_pos % planning_interval == 0
                    risk_due = (
                    frame_safety.physical_risk_index is not None
                    and frame_safety.physical_risk_index >= planning_risk_threshold
                    )
                    stale_due = planning_memory.is_stale(current_frame_id)

                    if planning_mode == "interval":
                        should_plan = interval_due
                    elif planning_mode == "risk":
                        should_plan = risk_due
                    elif planning_mode == "interval_risk":
                        should_plan = interval_due or risk_due or stale_due
                    else:
                        should_plan = False

            max_planning_calls = int(getattr(args, "max_planning_calls", 0) or 0)
            if should_plan and max_planning_calls > 0 and summary.get("planning_calls", 0) >= max_planning_calls:
                should_plan = False
                summary["planning_budget_exhausted"] = True
                summary["planning_budget_exhausted_frames"] += 1
            if should_plan and service.llm.budget_exhausted("planning"):
                should_plan = False
                summary["planning_budget_exhausted"] = True
                summary["planning_budget_exhausted_frames"] += 1

            can_plan_by_gap = (
                last_planning_frame_pos is None
                or frame_pos - last_planning_frame_pos >= planning_min_gap
            )

            if should_plan and can_plan_by_gap:
                event_type = metadata.get("eventType") or metadata.get("pair_type") or ""
                pair_type = metadata.get("pair_type") or ""
                vrus_present = bool(scene.vrus_present)
                with (
                    service.llm.usage_context("planning"),
                    service.llm.phase_usage_context(
                        f"{experiment_phase}_planning"
                    ),
                ):
                    planning_output = planning_service.plan(
                        dataset=args.dataset,
                        driver_type=effective_driver_type,
                        feedback=episode_feedback,
                        planning_interval=planning_interval,
                        current_frame=scene.frame_index or frame_pos,
                        time_horizon_s=args.planning_time_horizon_s,
                        recent_scene_summaries=compact_json(scene_history_for_planning, max_items=args.planning_max_history),
                        recent_safety_summaries=compact_json(safety_history_for_planning, max_items=args.planning_max_history),
                        recent_decision_summaries=compact_json(
                            [summarize_decision(d) for d in decision_list],
                            max_items=args.planning_max_history,
                        ),
                        current_safety_snapshot=json.dumps(
                            summarize_safety(frame_safety),
                            ensure_ascii=False,
                        ),
                        event_type=event_type,
                        pair_type=pair_type,
                        vrus_present=vrus_present,
                    )

                planning_fallback = bool(
                    planning_output.get("diagnostics", {}).get("fallback")
                )
                if not planning_fallback:
                    planning_memory.update(
                        planning_output, current_frame_id
                    )
                last_planning_frame_pos = frame_pos

                if planning_fallback:
                    summary["planning_failures"] = summary.get("planning_failures", 0) + 1

                planning_record = {
                    "event_index": idx,
                    "experiment_phase": experiment_phase,
                    "frame_pos": frame_pos,
                    "frame_index": scene.frame_index,
                    "planning": planning_output,
                    "planning_success": not planning_fallback,
                }

                planning_records.append(planning_record)
                if is_evaluation_phase:
                    global_planning_records.append(planning_record)

                summary["planning_calls"] = summary.get("planning_calls", 0) + 1
                append_jsonl(planning_trace_path, planning_record)
                if service.llm.budget_exhausted("planning"):
                    summary["planning_budget_exhausted"] = True

            # ============================================================
            # 2.1 Planning Hint 给 Reactive Thread
            # ============================================================
            planning_hint = ""
            planning_metadata = {}

            planning_peek = bool(getattr(args, "planning_peek", 1))
            if (
                planning_enabled
                and planning_peek
                and planning_memory.last_update_frame is not None
            ):
                planning_hint = planning_memory.to_reactive_hint(
                    current_frame=current_frame_id
                )
                planning_metadata = {
                    "planning_age_frames": current_frame_id - planning_memory.last_update_frame,
                    "last_update_frame": planning_memory.last_update_frame,
                }

            # ==================================================
            # 3. RAG v1：当前帧只执行一次，并使用当前帧安全指标
            # ==================================================
            rag_started = time.perf_counter()
            try:
                profile_for_rag = None

                try:
                    profile_for_rag = service.repo.load()
                except Exception:
                    profile_for_rag = None

                rag_result = rag.run(
                    dataset=args.dataset,
                    scene=scene,
                    frame_safety=frame_safety,
                    metadata=metadata,
                    driver_type=effective_driver_type,
                    feedback=episode_feedback,
                    planning_hint=planning_hint,
                    profile=profile_for_rag,
                )

                if not isinstance(rag_result, dict):
                    rag_result = make_empty_rag_result(
                        dataset=args.dataset,
                        scene=scene,
                        metadata=metadata,
                        budget=getattr(args, "rag_budget", "reactive"),
                    )

                evidence_pack = rag_result.get("evidence_pack") or evidence_pack

            except Exception as e:
                print(f"[WARN] RAG failed: event={idx}, frame={scene.frame_index}, error={e}")
            finally:
                rag_phase_stats[experiment_phase]["calls"] += 1
                rag_phase_stats[experiment_phase]["latencies_ms"].append(
                    (time.perf_counter() - rag_started) * 1000.0
                )

            # ============================================================
            # 3. Reactive Thread / LLM 调用策略
            # ============================================================
            call_llm = should_call_llm(
                policy=getattr(args, "llm_policy", "hybrid"),
                frame_pos=frame_pos,
                frame_safety=frame_safety,
                stride=int(getattr(args, "llm_stride", 5)),
                risk_threshold=float(getattr(args, "llm_risk_threshold", 0.35)),
                max_stale_frames=int(getattr(args, "llm_max_stale_frames", 30)),
                risk_delta_threshold=float(getattr(args, "llm_risk_delta_threshold", 0.15)),
                last_llm_frame_pos=last_llm_frame_pos,
                last_llm_risk_level=last_llm_risk_level,
                last_llm_risk_index=last_llm_risk_index,
                evidence_changed=tuple(
                    item.get("evidence_id")
                    for item in evidence_pack.get("items", [])
                    if isinstance(item, dict)
                ) != last_llm_evidence_ids,
                grounding_refresh_required=bool(
                    getattr(args, "require_grounded_decision", 0)
                    and recent_decisions
                    and (
                        set(
                            str(x)
                            for x in recent_decisions[-1].get(
                                "used_evidence_ids", []
                            )
                        )
                        - {
                            str(item.get("evidence_id"))
                            for item in evidence_pack.get("items", [])
                            if isinstance(item, dict)
                            and item.get("evidence_id") is not None
                        }
                    )
                ),
                planning_hint_updated=(
                    planning_metadata.get("last_update_frame") is not None
                    and planning_metadata.get("last_update_frame") != last_llm_planning_update_frame
                ),
            )

            # dry_run / inspect_only 理论上前面已经 continue，这里再防御一次
            if dry_run or inspect_only:
                call_llm = False

            max_llm_calls = int(getattr(args, "max_llm_calls", 0) or 0)
            if call_llm and max_llm_calls > 0 and summary.get("llm_calls", 0) >= max_llm_calls:
                call_llm = False
                summary["llm_budget_exhausted"] = True
                summary["llm_budget_exhausted_frames"] += 1
            if call_llm and service.llm.budget_exhausted("reactive"):
                call_llm = False
                summary["llm_budget_exhausted"] = True
                summary["llm_budget_exhausted_frames"] += 1

            llm_succeeded = False
            if call_llm:
                step_feedback = episode_feedback
                if (
                    bool(getattr(args, "feedback_once_per_episode", 0))
                    and feedback_consumed
                ):
                    step_feedback = ""
                try:
                    with (
                        service.llm.usage_context("reactive"),
                        service.llm.phase_usage_context(
                            f"{experiment_phase}_reactive"
                        ),
                    ):
                        result = service.step(
                            scene=scene,
                            driver_type=effective_driver_type,
                            feedback=step_feedback,
                            recent_decisions=recent_decisions,
                            planning_hint=planning_hint,
                            planning_metadata=planning_metadata,
                            evidence_pack=evidence_pack,
                            frame_safety=frame_safety,
                            require_grounded_decision=bool(getattr(args, "require_grounded_decision", 0)),
                            allow_profile_update=(
                                not profile_protocol_enabled
                                or experiment_phase == "adaptation"
                            ),
                        )

                    raw_decision = dict(result.decision) if isinstance(result.decision, dict) else safe_dict(result.decision)
                    grounding = validate_grounding(raw_decision, evidence_pack)
                    decision = repair_decision_evidence_fields(raw_decision, evidence_pack)
                    output_grounding = validate_grounding(decision, evidence_pack)
                    decision_source = "llm"
                    summary["llm_calls"] += 1
                    llm_succeeded = True
                    if step_feedback:
                        feedback_consumed = True
                    if service.llm.budget_exhausted("reactive"):
                        summary["llm_budget_exhausted"] = True

                except LLMBudgetExceeded as e:
                    summary["llm_budget_exhausted"] = True
                    summary["llm_budget_exhausted_frames"] += 1
                    result = None
                    decision = fallback_decision_from_physics(frame_safety)
                    decision_source = "physics_fallback_after_llm_budget"
                    summary["non_llm_frames"] += 1
                    print(
                        f"[WARN] reactive LLM budget exhausted: "
                        f"event={idx}, frame={scene.frame_index}, error={e}"
                    )
                except Exception as e:
                    kind = error_kind(e)
                    summary["llm_error_count"] = summary.get("llm_error_count", 0) + 1
                    summary["fallback_frame_count"] = summary.get("fallback_frame_count", 0) + 1
                    if kind == "timeout":
                        summary["timeout_count"] = summary.get("timeout_count", 0) + 1
                    elif kind == "connection":
                        summary["connection_error_count"] = summary.get("connection_error_count", 0) + 1
                    elif kind == "rate_limit":
                        summary["rate_limit_count"] = summary.get("rate_limit_count", 0) + 1
                    else:
                        summary["other_llm_error_count"] = summary.get("other_llm_error_count", 0) + 1
                    print(
                        f"[WARN] service.step failed: "
                        f"event={idx}, frame={scene.frame_index}, error={e}"
                    )
                    result = None
                    decision = fallback_decision_from_physics(frame_safety)
                    decision_source = "physics_fallback_after_llm_error"
                    summary["non_llm_frames"] += 1

            else:
                result = None

                if bool(getattr(args, "reuse_last_decision", 1)) and recent_decisions:
                    decision = dict(recent_decisions[-1])
                    decision["source"] = "reused_last_decision"
                else:
                    decision = fallback_decision_from_physics(frame_safety)

                decision_source = decision.get("source", "physics_fallback")
                summary["non_llm_frames"] += 1

            # 给 decision 统一补充来源字段，后续 planning quality / CSV 会用到
            decision["_source"] = decision_source
            decision["_planning_hint_used"] = bool(planning_hint)
            decision["_planning_age_frames"] = planning_metadata.get("planning_age_frames")

            if llm_succeeded:
                last_llm_frame_pos = frame_pos
                last_llm_risk_level = getattr(frame_safety, "physical_risk_level", None)
                last_llm_risk_index = getattr(frame_safety, "physical_risk_index", None)
                last_llm_evidence_ids = tuple(
                    item.get("evidence_id")
                    for item in evidence_pack.get("items", [])
                    if isinstance(item, dict)
                )
                last_llm_planning_update_frame = planning_metadata.get("last_update_frame")

            decision_list.append(decision)

            # ==================================================
            # 3. grounding validation
            # ==================================================
            if decision_source != "llm":
                try:
                    grounding = validate_grounding(
                        decision=decision,
                        evidence_pack=evidence_pack,
                    )
                    if bool(getattr(args, "require_grounded_decision", 0)):
                        decision = repair_decision_evidence_fields(decision, evidence_pack)
                    output_grounding = validate_grounding(
                        decision=decision,
                        evidence_pack=evidence_pack,
                    )
                except Exception as e:
                    print(f"[WARN] grounding validation failed: event={idx}, frame={scene.frame_index}, error={e}")
                    grounding = make_empty_grounding()
                    output_grounding = make_empty_grounding()

            # ============================================================
            # 4. 基于 decision + scene 计算 step metrics
            # ============================================================
            m = compute_step_metrics(scene, decision, thresholds=thresholds)

            if result is not None:
                trigger_dicts = safe_list_dict(getattr(result, "triggers", []))
                guardrail_dict = safe_dict(getattr(result, "guardrails", {}))
                profile_dict = safe_dict(result.profile)
                active_profile_snapshot = profile_dict
                evidence_dict = getattr(result, "evidence", {}) or {}
                rules_list = safe_list_dict(getattr(result, "rules", []))
                profile_update = getattr(result, "profile_update", {})
            else:
                trigger_dicts = []
                guardrail_dict = {}
                profile_dict = active_profile_snapshot
                evidence_dict = {}
                rules_list = []
                profile_update = {}

            if trigger_dicts:
                trigger_list_by_frame[frame_pos] = trigger_dicts

            if m.ttc_s is not None:
                ttc_values.append(m.ttc_s)
            if m.is_violation is not None:
                violation_flags.append(bool(m.is_violation))

            recent_decisions.append(decision)
            if args.mode == "episode" and args.history_window > 0:
                recent_decisions = recent_decisions[-args.history_window:]
            elif args.mode == "batch":
                recent_decisions = []

            for trig in trigger_dicts:
                t_type = safe_trigger_type(trig)
                episode_trigger_stats[t_type] += 1
                if is_evaluation_phase:
                    global_trigger_stats[t_type] += 1
                episode_trigger_count += 1

            trace_evidence_pack = (
                evidence_pack
                if bool(getattr(args, "trace_detail", True))
                else compact_evidence_pack_for_trace(evidence_pack)
            )

            frame_record = {
                "event_index": idx,
                "experiment_phase": experiment_phase,
                "metadata": metadata,
                "sequence_path": sequence_path,
                "scene": scene.__dict__,
                "profile": profile_dict,
                "decision": decision,
                "decision_source": decision_source,
                "planning_hint": planning_hint,
                "planning_memory": planning_memory.get() if bool(getattr(args, "use_planning_thread", 0)) else None,
                "triggers": trigger_dicts,
                "trigger_count": len(trigger_dicts),
                "guardrails": guardrail_dict,
                "profile_update": profile_update,
                "evidence": evidence_dict,
                "rules": rules_list,
                # ===== RAG v1 =====
                "rag_mode": getattr(args, "rag_mode", "none"),
                "rag_query": (rag_result or {}).get("rag_query", {}),
                "evidence_pack": trace_evidence_pack,
                "grounding": grounding,
                "output_grounding": output_grounding,
                "step_metrics": {
                    "ttc_s": m.ttc_s,
                    "thw_s": m.thw_s,
                    "drac_mps2": m.drac_mps2,
                    "dcpa_m": m.dcpa_m,
                    "ttca_s": m.ttca_s,
                    "predicted_ttc_s": m.predicted_ttc_s,
                    "min_future_distance_m": m.min_future_distance_m,
                    "physical_risk_index": m.physical_risk_index,
                    "physical_risk_level": m.physical_risk_level,
                    "is_violation": m.is_violation,
                },
                "frame_safety": asdict(frame_safety),
                "dataset_risk_label": dataset_risk,
            }
            if is_evaluation_phase:
                global_frame_records_for_rag.append(frame_record)
            logger.append_decision(frame_record)

            append_jsonl(profile_trace_path, {
                "event_index": idx,
                "frame_index": scene.frame_index,
                "profile": profile_dict,
            })

            append_jsonl(profile_delta_path, {
                "event_index": idx,
                "frame_index": scene.frame_index,
                "profile_update": profile_update,
            })

            append_jsonl(guardrail_trace_path, {
                "event_index": idx,
                "frame_index": scene.frame_index,
                "guardrails": guardrail_dict,
            })

            append_jsonl(rag_trace_path, {
                "event_index": idx,
                "frame_index": scene.frame_index,
                "dataset": args.dataset,
                "metadata": metadata,
                "rag_mode": getattr(args, "rag_mode", "full"),
                "rag_query": (rag_result or {}).get("rag_query", {}),
                "retrieved": rag_result.get("retrieved"),
                "reranked": rag_result.get("reranked"),
                "evidence_pack": trace_evidence_pack,
                "decision_used_evidence_ids": decision.get("used_evidence_ids", []),
                "grounding": grounding,
                "output_grounding": output_grounding,
            })

            for trig in trigger_dicts:
                append_jsonl(trigger_trace_path, {
                    "event_index": idx,
                    "frame_index": scene.frame_index,
                    "trigger": trig,
                })

            used_evidence_ids = output_grounding.get("used_evidence_ids", [])
            hallucinated_ids = output_grounding.get("hallucinated_evidence_ids", [])

            with open(frame_metrics_path, "a", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow([
                    idx,
                    metadata.get("recordingId"),
                    metadata.get("event_id"),
                    metadata.get("eventType"),
                    metadata.get("pair_type"),
                    metadata.get("location_id"),
                    metadata.get("egoId") or metadata.get("egoTrackId") or metadata.get("ego_track_id"),
                    metadata.get("otherId") or metadata.get("otherTrackId") or metadata.get("trackId_2"),
                    scene.frame_index,
                    "" if m.ttc_s is None else round(m.ttc_s, 4),
                    "" if m.thw_s is None else round(m.thw_s, 4),
                    "" if m.drac_mps2 is None else round(m.drac_mps2, 4),
                    "" if m.dcpa_m is None else round(m.dcpa_m, 4),
                    "" if m.ttca_s is None else round(m.ttca_s, 4),
                    "" if m.predicted_ttc_s is None else round(m.predicted_ttc_s, 4),
                    "" if m.min_future_distance_m is None else round(m.min_future_distance_m, 4),
                    "" if m.physical_risk_index is None else round(m.physical_risk_index, 4),
                    m.physical_risk_level,

                    "" if m.is_violation is None else int(m.is_violation),
                    scene.ego_speed_mps,
                    scene.rel_speed_mps,
                    scene.headway_m,
                    int(scene.vrus_present),
                    safe_len(trigger_dicts),
                    safe_len(rules_list),
                    safe_len(evidence_dict.get("laws", [])),
                    safe_len(evidence_dict.get("cases", [])),
                    safe_len(evidence_dict.get("scenarios", [])),

                    int(call_llm),
                    decision_source,
                    int(planning_memory.last_update_frame is not None),
                    "" if planning_memory.last_update_frame is None else current_frame_id - planning_memory.last_update_frame,
                    planning_memory.get()
                        .get("risk_forecast", {})
                        .get("risk_level"),
                    planning_memory.get()
                        .get("recommended_strategy", {})
                        .get("strategy"),

                     # ===== RAG v1 grounding metrics =====
                    getattr(args, "rag_mode", "none"),
                    evidence_pack.get("num_evidence", 0),
                    safe_len(used_evidence_ids),
                    int(bool(output_grounding.get("is_grounded"))),
                    output_grounding.get("evidence_support_level", "none"),
                    safe_len(hallucinated_ids),
                ])

        episode_llm_violation = (sum(violation_flags) > 0) if violation_flags else False
        episode_safety = aggregate_episode_safety_metrics(frame_safety_metrics_list)
        alignment = compute_llm_physics_alignment(
            frame_safety_metrics_list,
            decision_list,
        )
        behavior = compute_behavior_safety_metrics(
            frame_safety_metrics_list,
            decision_list,
            trigger_list_by_frame=trigger_list_by_frame,
        )
        episode_planning_enabled = bool(getattr(args, "use_planning_thread", 0))
        if episode_planning_enabled:
            planning_quality = compute_planning_quality(
                planning_records,
                frame_safety_metrics_list,
                decision_list,
                horizon=int(getattr(args, "planning_quality_horizon", 10)),
            )
            if is_evaluation_phase:
                global_planning_quality_records.append(planning_quality)
        else:
            planning_quality = {
                "planning_enabled": False,
                "planning_skip_reason": "planning thread disabled",
            }
        min_ttc_est = min(ttc_values) if ttc_values else None
        avg_ttc_est = (sum(ttc_values) / len(ttc_values)) if ttc_values else None
        violation_rate = (
            sum(1 for x in violation_flags if x) / len(violation_flags)
            if violation_flags else None
        )

        if is_evaluation_phase:
            global_episode_safety_records.append(asdict(episode_safety))
            global_alignment_records.append(asdict(alignment))
            global_behavior_records.append(asdict(behavior))
        episode_summary = {
            "event_index": idx,
            "experiment_phase": experiment_phase,
            "dataset": args.dataset,
            "mode": args.mode,
            "metadata": metadata,
            "sequence_path": sequence_path,
            "dataset_risk_label": dataset_risk,
            "episode_num_frames": len(scenes),
            "episode_evaluated_num_frames": evaluated_frame_count,
            "frame_selection": frame_selection,
            "episode_llm_violation": episode_llm_violation,
            "episode_violation_rate": violation_rate,
            "episode_min_ttc_estimated": min_ttc_est,
            "episode_avg_ttc_estimated": avg_ttc_est,

            "episode_safety": asdict(episode_safety),
            "llm_physics_alignment": asdict(alignment),
            "behavior_safety": asdict(behavior),

            "planning_quality": planning_quality,
            "planning_call_count": len(planning_records),

            "trigger_count": episode_trigger_count,
            "trigger_distribution": dict(episode_trigger_stats)
        }
        append_jsonl(episode_summary_path, episode_summary)

        summary["total_events"] += 1
        summary["total_frames"] += evaluated_frame_count
        summary["profile_protocol"][
            f"{experiment_phase}_frames"
        ] += evaluated_frame_count
        if is_evaluation_phase:
            summary["dataset_risk_true"] += int(dataset_risk)
            summary["episode_llm_violation_true"] += int(
                episode_llm_violation
            )
            summary["episode_agreement"] += int(
                episode_llm_violation == dataset_risk
            )
            all_y_true.append(bool(dataset_risk))
            all_y_pred.append(bool(episode_llm_violation))

        print(
            f"[{idx}] profile={args.profile_name} "
            f"{args.dataset}/{args.mode} "
            f"frames={evaluated_frame_count}/{len(scenes)} "
            f"dataset_risk={dataset_risk} "
            f"episode_llm_violation={episode_llm_violation} "
            f"triggers={episode_trigger_count}"
        )
        commit_episode_checkpoint(idx)

    no_evaluation_episodes = (
        bool(summary["profile_protocol"].get("enabled"))
        and int(
            summary["profile_protocol"].get("evaluation_events", 0)
            or 0
        ) == 0
    )
    if dry_run or inspect_only or no_evaluation_episodes:
        summary["classification_skipped"] = True
        if no_evaluation_episodes and not (dry_run or inspect_only):
            summary["classification_skip_reason"] = (
                "No evaluation episodes were executed; adaptation-only "
                "pilot runs do not produce formal classification metrics."
            )
        else:
            summary["classification_skip_reason"] = (
                "dry_run/inspect_only does not call LLM, so no valid "
                "model predictions exist."
            )

        summary["episode_llm_violation_true"] = None
        summary["episode_agreement"] = None
        summary["episode_agreement_rate"] = None

        summary["confusion_matrix"] = None
        summary["precision"] = None
        summary["recall"] = None
        summary["f1"] = None
        summary["accuracy"] = None
        summary["total"] = 0

    else:
        summary["classification_skipped"] = False

        summary["episode_agreement_rate"] = (
            summary["episode_agreement"]
            / summary["profile_protocol"]["evaluation_events"]
            if summary["profile_protocol"]["evaluation_events"] else 0.0
        )

        cls = compute_confusion_and_scores(all_y_true, all_y_pred)
        summary.update(cls)

    summary["trigger_distribution"] = dict(global_trigger_stats)
    summary["total_triggers"] = sum(global_trigger_stats.values())
    summary["avg_triggers_per_event"] = (
        summary["total_triggers"]
        / summary["profile_protocol"]["evaluation_events"]
        if summary["profile_protocol"]["evaluation_events"] else 0.0
    )
    summary["avg_triggers_per_frame"] = (
        summary["total_triggers"]
        / summary["profile_protocol"]["evaluation_frames"]
        if summary["profile_protocol"]["evaluation_frames"] else 0.0
    )

    summary["llm_call_rate"] = (
        summary["llm_calls"] / summary["total_frames"]
        if summary["total_frames"] else 0.0
    )
    summary["overall_llm_call_rate"] = summary["llm_call_rate"]
    summary["llm_call_rate_scope"] = "overall_executed_phases"
    evaluation_frames = int(
        summary["profile_protocol"].get("evaluation_frames", 0) or 0
    )
    summary["fallback_frame_rate"] = (
        summary.get("fallback_frame_count", 0) / summary["reactive_frames"]
        if summary.get("reactive_frames", 0) else 0.0
    )

    def _avg(records, key):
        vals = [r.get(key) for r in records if isinstance(r.get(key), (int, float))]
        return sum(vals) / len(vals) if vals else None

    summary["global_safety"] = {
        "avg_min_ttc_s": _avg(global_episode_safety_records, "min_ttc_s"),
        "avg_avg_ttc_s": _avg(global_episode_safety_records, "avg_ttc_s"),
        "avg_max_drac_mps2": _avg(global_episode_safety_records, "max_drac_mps2"),
        "avg_min_dcpa_m": _avg(global_episode_safety_records, "min_dcpa_m"),
        "avg_min_future_distance_m": _avg(global_episode_safety_records, "min_future_distance_m"),
        "avg_unsafe_ttc_ratio": _avg(global_episode_safety_records, "unsafe_ttc_ratio"),
        "avg_unsafe_thw_ratio": _avg(global_episode_safety_records, "unsafe_thw_ratio"),
        "avg_unsafe_drac_ratio": _avg(global_episode_safety_records, "unsafe_drac_ratio"),
        "avg_unsafe_dcpa_ratio": _avg(global_episode_safety_records, "unsafe_dcpa_ratio"),
        "avg_unsafe_future_distance_ratio": _avg(global_episode_safety_records, "unsafe_future_distance_ratio"),
        "avg_physical_risk_exposure": _avg(global_episode_safety_records, "physical_risk_exposure"),
        "avg_max_physical_risk_index": _avg(global_episode_safety_records, "max_physical_risk_index"),
    }

    summary["global_alignment"] = {
        "avg_alignment_accuracy": _avg(global_alignment_records, "alignment_accuracy"),
        "avg_overreaction_rate": _avg(global_alignment_records, "overreaction_rate"),
        "avg_underreaction_rate": _avg(global_alignment_records, "underreaction_rate"),
        "avg_mean_risk_level_error": _avg(global_alignment_records, "mean_risk_level_error"),
        "avg_llm_violation_rate": _avg(global_alignment_records, "llm_violation_rate"),
    }

    summary["global_behavior"] = {
        "avg_reaction_delay_frames": _avg(global_behavior_records, "reaction_delay_frames"),
        "avg_reaction_success_rate": _avg(
            global_behavior_records, "reaction_success_rate"
        ),
        "reaction_censored_episodes": sum(
            int(bool(record.get("reaction_censored")))
            for record in global_behavior_records
        ),
        "avg_trigger_delay_frames": _avg(global_behavior_records, "trigger_delay_frames"),
        "avg_decision_flip_rate": _avg(global_behavior_records, "decision_flip_rate"),
        "avg_risk_level_variance": _avg(global_behavior_records, "risk_level_variance"),
    }
    try:
        initial_profile = service.repo.get_template()
        final_profile = service.repo.load()
        profile_paths = [
            ("global", "risk_sensitivity"),
            ("global", "safety_weight"),
            ("global", "efficiency_weight"),
            ("longitudinal", "preferred_time_headway"),
            ("longitudinal", "min_time_headway"),
            ("lateral", "lane_change_aggressiveness"),
            ("lateral", "min_gap_acceptance"),
            ("interaction", "vehicle_vehicle_assertiveness"),
            ("interaction", "vehicle_cyclist_yield_bias"),
            ("interaction", "vehicle_pedestrian_yield_bias"),
        ]
        parameter_deltas = {}
        for section, key in profile_paths:
            before = (
                initial_profile.get(section, {}).get(key)
                if isinstance(initial_profile, dict)
                else None
            )
            after = (
                final_profile.get(section, {}).get(key)
                if isinstance(final_profile, dict)
                else None
            )
            if isinstance(before, (int, float)) and isinstance(
                after, (int, float)
            ):
                parameter_deltas[f"{section}.{key}"] = after - before
        summary["global_profile_adaptation"] = {
            "enabled": bool(getattr(args, "use_profile_learner", 0)),
            "changed_parameter_count": sum(
                abs(value) > 1e-12
                for value in parameter_deltas.values()
            ),
            "parameter_delta_l1": sum(
                abs(value) for value in parameter_deltas.values()
            ),
            "parameter_deltas": parameter_deltas,
            "initial_global": initial_profile.get("global", {}),
            "final_global": final_profile.get("global", {}),
        }
    except Exception as exc:
        summary["global_profile_adaptation"] = {
            "enabled": bool(getattr(args, "use_profile_learner", 0)),
            "error": str(exc),
        }
    api_usage = service.llm.usage_summary()
    phase_usage = service.llm.phase_usage_summary()
    reactive_usage = api_usage.get("reactive", {})
    planning_usage = api_usage.get("planning", {})
    phase_costs = {}
    for phase in ("adaptation", "evaluation"):
        reactive_phase = phase_usage.get(
            f"{phase}_reactive", {}
        )
        planning_phase = phase_usage.get(
            f"{phase}_planning", {}
        )
        phase_frames = int(
            summary["profile_protocol"].get(
                f"{phase}_frames", 0
            ) or 0
        )
        phase_costs[phase] = {
            "frames": phase_frames,
            "reactive_attempts": int(
                reactive_phase.get("attempts", 0) or 0
            ),
            "reactive_successes": int(
                reactive_phase.get("successes", 0) or 0
            ),
            "reactive_total_tokens": int(
                reactive_phase.get("total_tokens", 0) or 0
            ),
            "reactive_latency_ms_p50": reactive_phase.get(
                "latency_ms_p50", 0.0
            ),
            "reactive_latency_ms_p95": reactive_phase.get(
                "latency_ms_p95", 0.0
            ),
            "reactive_attempt_rate": (
                int(reactive_phase.get("attempts", 0) or 0)
                / phase_frames if phase_frames else 0.0
            ),
            "planning_attempts": int(
                planning_phase.get("attempts", 0) or 0
            ),
            "planning_successes": int(
                planning_phase.get("successes", 0) or 0
            ),
            "planning_total_tokens": int(
                planning_phase.get("total_tokens", 0) or 0
            ),
            "planning_latency_ms_p50": planning_phase.get(
                "latency_ms_p50", 0.0
            ),
            "planning_latency_ms_p95": planning_phase.get(
                "latency_ms_p95", 0.0
            ),
            "planning_attempt_rate": (
                int(planning_phase.get("attempts", 0) or 0)
                / phase_frames if phase_frames else 0.0
            ),
            "rag_calls": int(
                rag_phase_stats[phase]["calls"]
            ),
            "rag_latency_ms_mean": (
                sum(rag_phase_stats[phase]["latencies_ms"])
                / len(rag_phase_stats[phase]["latencies_ms"])
                if rag_phase_stats[phase]["latencies_ms"] else 0.0
            ),
            "rag_latency_ms_p95": service.llm._percentile(
                rag_phase_stats[phase]["latencies_ms"], 0.95
            ),
        }
    summary["phase_costs"] = phase_costs
    summary["primary_metric_scope"] = "evaluation"
    summary["overall_total_events"] = int(
        summary.get("total_events", 0) or 0
    )
    summary["overall_total_frames"] = int(
        summary.get("total_frames", 0) or 0
    )
    summary["evaluation_total_events"] = int(
        summary["profile_protocol"].get("evaluation_events", 0) or 0
    )
    summary["evaluation_total_frames"] = evaluation_frames
    summary["evaluation_llm_call_rate"] = (
        phase_costs["evaluation"]["reactive_successes"]
        / evaluation_frames if evaluation_frames else 0.0
    )
    summary["llm_attempts"] = int(reactive_usage.get("attempts", 0) or 0)
    summary["planning_llm_attempts"] = int(
        planning_usage.get("attempts", 0) or 0
    )
    summary["api_usage"] = api_usage
    reactive_budget = service.llm.budget_status("reactive")
    planning_budget = service.llm.budget_status("planning")
    summary["llm_budget_exhausted"] = bool(
        summary.get("llm_budget_exhausted")
        or reactive_budget["exhausted"]
    )
    summary["planning_budget_exhausted"] = bool(
        summary.get("planning_budget_exhausted")
        or planning_budget["exhausted"]
    )
    summary["reactive_request_budget_exhausted"] = bool(
        reactive_budget["attempts_exhausted"]
    )
    summary["reactive_token_budget_exhausted"] = bool(
        reactive_budget["tokens_exhausted"]
    )
    summary["planning_request_budget_exhausted"] = bool(
        planning_budget["attempts_exhausted"]
    )
    summary["planning_token_budget_exhausted"] = bool(
        planning_budget["tokens_exhausted"]
    )
    summary["reactive_token_overshoot"] = int(
        reactive_budget["token_overshoot"]
    )
    summary["planning_token_overshoot"] = int(
        planning_budget["token_overshoot"]
    )
    summary["token_time_efficiency"] = {
        "llm_calls": summary.get("llm_calls", 0),
        "llm_attempts": summary.get("llm_attempts", 0),
        "planning_calls": summary.get("planning_calls", 0),
        "planning_llm_attempts": summary.get("planning_llm_attempts", 0),
        "non_llm_frames": summary.get("non_llm_frames", 0),
        "dry_run_frames": summary.get("dry_run_frames", 0),
        "inspect_frames": summary.get("inspect_frames", 0),
        "reactive_frames": summary.get("reactive_frames", 0),
        "total_frames": summary.get("total_frames", 0),
        "max_llm_calls": summary.get("max_llm_calls", 0),
        "max_planning_calls": summary.get("max_planning_calls", 0),
        "max_reactive_api_attempts": summary.get(
            "max_reactive_api_attempts", 0
        ),
        "max_reactive_tokens": summary.get("max_reactive_tokens", 0),
        "max_planning_api_attempts": summary.get(
            "max_planning_api_attempts", 0
        ),
        "max_planning_tokens": summary.get("max_planning_tokens", 0),
        "reactive_budget": reactive_budget,
        "planning_budget": planning_budget,
        "llm_budget_exhausted": summary.get("llm_budget_exhausted", False),
        "planning_budget_exhausted": summary.get("planning_budget_exhausted", False),
        "llm_budget_exhausted_frames": summary.get("llm_budget_exhausted_frames", 0),
        "planning_budget_exhausted_frames": summary.get("planning_budget_exhausted_frames", 0),

        "reactive_llm_call_rate": (
            summary.get("llm_calls", 0) / summary["reactive_frames"]
            if summary.get("reactive_frames", 0) else 0.0
        ),
        "reactive_llm_attempt_rate": (
            summary.get("llm_attempts", 0) / summary["reactive_frames"]
            if summary.get("reactive_frames", 0) else 0.0
        ),
        "reactive_total_tokens": int(
            reactive_usage.get("total_tokens", 0) or 0
        ),
        "planning_total_tokens": int(
            planning_usage.get("total_tokens", 0) or 0
        ),
        "reactive_latency_ms_p50": reactive_usage.get("latency_ms_p50", 0.0),
        "reactive_latency_ms_p95": reactive_usage.get("latency_ms_p95", 0.0),
        "planning_latency_ms_p50": planning_usage.get("latency_ms_p50", 0.0),
        "planning_latency_ms_p95": planning_usage.get("latency_ms_p95", 0.0),
        "phase_costs": phase_costs,
        "evaluation_reactive_llm_attempt_rate": (
            phase_costs["evaluation"]["reactive_attempts"]
            / evaluation_frames if evaluation_frames else 0.0
        ),
        "evaluation_reactive_llm_call_rate": (
            phase_costs["evaluation"]["reactive_successes"]
            / evaluation_frames if evaluation_frames else 0.0
        ),
        "evaluation_planning_call_rate": (
            phase_costs["evaluation"]["planning_attempts"]
            / evaluation_frames if evaluation_frames else 0.0
        ),
        "evaluation_non_llm_frame_rate": (
            max(
                0,
                evaluation_frames
                - phase_costs["evaluation"]["reactive_successes"],
            )
            / evaluation_frames if evaluation_frames else 0.0
        ),
        "planning_call_rate": (
            summary.get("planning_calls", 0) / summary["reactive_frames"]
            if summary.get("reactive_frames", 0) else 0.0
        ),
        "non_llm_frame_rate": (
            summary.get("non_llm_frames", 0) / summary["reactive_frames"]
            if summary.get("reactive_frames", 0) else 0.0
        ),
    }
    planning_enabled = bool(getattr(args, "use_planning_thread", 0))
    planning_quality_available = (
        planning_enabled
        and not dry_run
        and not inspect_only
        and any(
            int(record.get("planning_call_count", 0) or 0) > 0
            for record in global_planning_quality_records
        )
    )

    if dry_run:
        planning_skip_reason = "dry_run skips Planning Thread and Reactive decisions"
    elif inspect_only:
        planning_skip_reason = "inspect_only only validates data loading"
    elif not planning_enabled:
        planning_skip_reason = "planning thread disabled"
    elif not planning_quality_available:
        planning_skip_reason = "no successful planning records generated"
    else:
        planning_skip_reason = None

    summary["global_planning"] = {
        "planning_enabled": planning_enabled,
        "planning_quality_available": planning_quality_available,
        "planning_skip_reason": planning_skip_reason,

        "total_planning_calls": summary.get("planning_calls", 0),
        "overall_planning_calls": summary.get("planning_calls", 0),
        "evaluation_planning_calls": phase_costs[
            "evaluation"
        ]["planning_attempts"],
        "planning_failures": summary.get("planning_failures", 0),

        "avg_planning_calls_per_event": (
            summary.get("planning_calls", 0) / summary["total_events"]
            if summary["total_events"] else 0.0
        ),
        "overall_avg_planning_calls_per_event": (
            summary.get("planning_calls", 0) / summary["total_events"]
            if summary["total_events"] else 0.0
        ),
        "avg_evaluation_planning_calls_per_event": (
            phase_costs["evaluation"]["planning_attempts"]
            / summary["profile_protocol"]["evaluation_events"]
            if summary["profile_protocol"]["evaluation_events"] else 0.0
        ),

        "num_planning_quality_records": sum(
            1
            for record in global_planning_quality_records
            if int(record.get("planning_call_count", 0) or 0) > 0
        ),

        "avg_planning_hit_rate": (
            _avg(global_planning_quality_records, "planning_hit_rate")
            if planning_quality_available else None
        ),
        "avg_planning_precision": (
            _avg(global_planning_quality_records, "planning_precision")
            if planning_quality_available else None
        ),
        "avg_planning_miss_rate": (
            _avg(global_planning_quality_records, "planning_miss_rate")
            if planning_quality_available else None
        ),
        "avg_planning_false_alarm_rate": (
            _avg(global_planning_quality_records, "planning_false_alarm_rate")
            if planning_quality_available else None
        ),
        "avg_planning_reactive_consistency": (
            _avg(global_planning_quality_records, "planning_reactive_consistency")
            if planning_quality_available else None
        ),
    }

    summary["global_rag"] = compute_rag_metrics(global_frame_records_for_rag)

    with open(os.path.join(logger.run_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    commit_episode_checkpoint(completed=True)

    with open(os.path.join(logger.run_dir, "summary.csv"), "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["metric", "value"])
        for k, v in summary.items():
            if isinstance(v, (dict, list)):
                writer.writerow([k, json.dumps(v, ensure_ascii=False)])
            else:
                writer.writerow([k, v])

    trigger_csv_path = os.path.join(logger.run_dir, "trigger_summary.csv")
    with open(trigger_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["trigger_type", "count"])
        for trigger_type, count in sorted(global_trigger_stats.items(), key=lambda x: x[0]):
            writer.writerow([trigger_type, count])

    try:
        plotter = TriggerPlotter(run_dir=logger.run_dir)
        plotter.plot_all()
    except Exception as e:
        print("[WARN] Trigger plotting failed:", e)

    return summary
