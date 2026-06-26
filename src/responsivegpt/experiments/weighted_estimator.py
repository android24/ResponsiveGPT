import argparse
import csv
import hashlib
import json
import math
import random
from collections import defaultdict
from pathlib import Path

from .io_utils import read_jsonl, write_csv
from .validate_runs import latest_usable_completed_statuses


DEFAULT_CENSUS = "data/full_pool_census/cornercase_v1/full_pool_episode_census.csv"
DEFAULT_SAMPLE_DIR = "data/eval_samples/core_v1"

METRICS = [
    "episode_llm_violation",
    "episode_violation_rate",
    "trigger_count",
    "planning_call_count",
    "episode_evaluated_num_frames",
    "physical_risk_exposure",
    "max_physical_risk_index",
    "unsafe_ttc_ratio",
    "unsafe_drac_ratio",
    "unsafe_dcpa_ratio",
    "alignment_accuracy",
    "underreaction_rate",
    "overreaction_rate",
    "reaction_delay_frames",
    "reaction_success_rate",
    "decision_flip_rate",
    "planning_hit_rate",
    "planning_precision",
    "planning_miss_rate",
    "planning_reactive_consistency",
    "rag_retrieval_coverage",
    "rag_evidence_usage_rate",
    "rag_grounded_decision_rate",
    "rag_output_invalid_citation_frame_rate",
    "rag_citation_precision",
    "decision_intervention_rate",
    "unnecessary_intervention_rate",
    "missed_intervention_rate",
    "safety_action_appropriateness",
    "offline_profile_utility",
]

CONDITIONAL_METRICS = {
    "reaction_delay_frames",
    "reaction_success_rate",
    "planning_hit_rate",
    "planning_precision",
    "planning_miss_rate",
    "planning_reactive_consistency",
    "missed_intervention_rate",
    "safety_action_appropriateness",
}

WEIGHTED_FIELDS = [
    "job_id",
    "run_dir",
    "dataset",
    "mode",
    "profile_name",
    "use_profile_learner",
    "profile_protocol_enabled",
    "profile_adaptation_episodes",
    "profile_adaptation_pool_episodes",
    "rag_variant",
    "planning_variant",
    "llm_policy_variant",
    "sample_census_path",
    "estimand",
    "inclusion_probability_method",
    "metric",
    "population_total",
    "sample_total",
    "expected_sample_total",
    "num_strata",
    "estimate_valid",
    "population_coverage",
    "metric_completeness",
    "missing_metric_rows",
    "not_applicable_rows",
    "censored_rows",
    "missingness_policy",
    "observed_weight_sum",
    "missing_strata_count",
    "missing_strata_json",
    "variance_method",
    "num_recording_clusters",
    "singleton_strata_count",
    "cluster_bootstrap_rounds",
    "weighted_mean",
    "weighted_se",
    "ci95_low",
    "ci95_high",
    "unweighted_mean",
]

OBSERVATION_FIELDS = [
    "job_id",
    "run_dir",
    "dataset",
    "mode",
    "profile_name",
    "use_profile_learner",
    "profile_protocol_enabled",
    "profile_adaptation_episodes",
    "profile_adaptation_pool_episodes",
    "rag_variant",
    "planning_variant",
    "llm_policy_variant",
    "sample_census_path",
    "event_index",
    "recording_cluster",
    "risk_stratum",
    "event_type",
    "dataset_risk_label",
    "vru_present",
    "metric",
    "value",
    "core_sample_rows",
    "evaluation_sample_rows",
    "core_inclusion_probability",
    "evaluation_given_core_probability",
    "combined_inclusion_probability",
    "design_weight",
]

STRATUM_FIELDS = [
    "job_id",
    "dataset",
    "mode",
    "profile_name",
    "use_profile_learner",
    "profile_protocol_enabled",
    "profile_adaptation_episodes",
    "profile_adaptation_pool_episodes",
    "rag_variant",
    "planning_variant",
    "llm_policy_variant",
    "sample_census_path",
    "metric",
    "risk_stratum",
    "event_type",
    "dataset_risk_label",
    "vru_present",
    "population_rows",
    "core_sample_rows",
    "evaluation_sample_rows",
    "sample_rows",
    "core_inclusion_probability",
    "evaluation_given_core_probability",
    "combined_inclusion_probability",
    "stratum_weight",
    "stratum_mean",
    "stratum_variance",
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
    if isinstance(value, bool):
        return float(value)
    try:
        out = float(value)
    except Exception:
        return None
    if math.isnan(out) or math.isinf(out):
        return None
    return out


def _mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def _variance(values: list[float]) -> float:
    if len(values) <= 1:
        return 0.0
    mean = sum(values) / len(values)
    return sum((value - mean) ** 2 for value in values) / (len(values) - 1)


def _cluster_id(row: dict) -> str:
    return str(
        row.get("recording_id")
        or row.get("recordingId")
        or row.get("recording_prefix")
        or row.get("prefix")
        or f"row:{row.get('row_index', '')}"
    )


def _stable_seed(*parts) -> int:
    raw = "|".join(str(part) for part in parts)
    return int(hashlib.sha1(raw.encode("utf-8")).hexdigest()[:12], 16)


def _cluster_bootstrap_uncertainty(
    observations_by_key: dict,
    population_by_key: dict,
    population_total: int,
    point_estimate: float,
    *,
    seed: int,
    rounds: int = 1000,
) -> dict:
    all_observations = [
        observation
        for observations in observations_by_key.values()
        for observation in observations
    ]
    clusters = sorted({
        observation[1] for observation in all_observations
    })
    all_values = [observation[0] for observation in all_observations]
    pooled_variance = _variance(all_values)
    bounded_unit_interval = bool(all_values) and all(
        0.0 <= value <= 1.0 for value in all_values
    )

    bootstrap_values = []
    if len(clusters) >= 2:
        rng = random.Random(seed)
        for _ in range(rounds):
            cluster_weights = {
                cluster: rng.expovariate(1.0)
                for cluster in clusters
            }
            estimate = 0.0
            valid = True
            for key, population_rows in population_by_key.items():
                observations = observations_by_key.get(key, [])
                if population_rows <= 0 or not observations:
                    valid = False
                    break
                denominator = sum(
                    cluster_weights[observation[1]]
                    for observation in observations
                )
                if denominator <= 0:
                    valid = False
                    break
                mean_h = sum(
                    observation[0] * cluster_weights[observation[1]]
                    for observation in observations
                ) / denominator
                estimate += (
                    population_rows / population_total
                ) * mean_h
            if valid:
                bootstrap_values.append(estimate)

    bootstrap_values.sort()
    bootstrap_variance = _variance(bootstrap_values)
    singleton_strata_count = 0
    conservative_variance_floor = 0.0
    for key, population_rows in population_by_key.items():
        observations = observations_by_key.get(key, [])
        cluster_count = len({
            observation[1] for observation in observations
        })
        if len(observations) >= 2 and cluster_count >= 2:
            continue
        singleton_strata_count += 1
        fallback_variance = pooled_variance
        if bounded_unit_interval:
            fallback_variance = max(fallback_variance, 0.25)
        weight = population_rows / population_total
        conservative_variance_floor += (
            weight ** 2
            * fallback_variance
            / max(1, len(observations))
        )

    variance = max(bootstrap_variance, conservative_variance_floor)
    se = math.sqrt(max(0.0, variance))
    if bootstrap_values:
        lo_index = int(0.025 * (len(bootstrap_values) - 1))
        hi_index = int(0.975 * (len(bootstrap_values) - 1))
        bootstrap_low = bootstrap_values[lo_index]
        bootstrap_high = bootstrap_values[hi_index]
    else:
        bootstrap_low = point_estimate
        bootstrap_high = point_estimate
    ci95_low = min(
        bootstrap_low, point_estimate - 1.96 * se
    )
    ci95_high = max(
        bootstrap_high, point_estimate + 1.96 * se
    )
    if bounded_unit_interval:
        ci95_low = max(0.0, ci95_low)
        ci95_high = min(1.0, ci95_high)
    return {
        "se": se,
        "ci95_low": ci95_low,
        "ci95_high": ci95_high,
        "variance_method": (
            "recording_cluster_bayesian_bootstrap"
            + (
                "_with_singleton_variance_floor"
                if singleton_strata_count else ""
            )
        ),
        "num_recording_clusters": len(clusters),
        "singleton_strata_count": singleton_strata_count,
        "cluster_bootstrap_rounds": len(bootstrap_values),
    }


def _stratum_key(row: dict) -> tuple[str, str, str, str]:
    return (
        str(row.get("risk_stratum", "unknown")),
        str(row.get("event_type", "unknown")),
        str(row.get("dataset_risk_label", "")),
        str(row.get("vru_present", "")),
    )


def _metric_value(episode: dict, metric: str):
    if metric == "episode_llm_violation":
        value = episode.get("episode_llm_violation")
        if value in (None, ""):
            return None
        return 1.0 if bool(value) else 0.0
    if metric in episode:
        return _to_float(episode.get(metric))

    safety = episode.get("episode_safety") or {}
    if metric in safety:
        return _to_float(safety.get(metric))

    alignment = episode.get("llm_physics_alignment") or {}
    if metric in alignment:
        return _to_float(alignment.get(metric))

    behavior = episode.get("behavior_safety") or {}
    if metric in behavior:
        return _to_float(behavior.get(metric))

    planning = episode.get("planning_quality") or {}
    if metric in planning:
        return _to_float(planning.get(metric))

    return None


def _episode_has_physical_risk(episode: dict) -> bool:
    safety = episode.get("episode_safety") or {}
    return bool(
        safety.get("has_critical_ttc")
        or safety.get("has_critical_drac")
        or safety.get("has_critical_spatial_risk")
        or (_to_float(safety.get("max_physical_risk_index")) or 0.0) >= 0.65
    )


def _metric_value_with_status(episode: dict, metric: str):
    if metric == "reaction_delay_frames":
        behavior = episode.get("behavior_safety") or {}
        value = _to_float(behavior.get("reaction_delay_frames"))
        if value is not None:
            return value, "observed"
        if behavior.get("reaction_censored") is True:
            return None, "right_censored"
        if _episode_has_physical_risk(episode):
            return None, "right_censored"
        return None, "not_applicable"

    if metric == "reaction_success_rate":
        behavior = episode.get("behavior_safety") or {}
        value = _to_float(behavior.get("reaction_success_rate"))
        if value is not None:
            return value, "observed"
        if _episode_has_physical_risk(episode):
            return (
                float(
                    _to_float(behavior.get("reaction_delay_frames"))
                    is not None
                ),
                "observed",
            )
        return None, "not_applicable"

    if metric in {
        "planning_hit_rate",
        "planning_precision",
        "planning_miss_rate",
        "planning_reactive_consistency",
    }:
        planning = episode.get("planning_quality") or {}
        confusion = planning.get("planning_confusion") or {}
        tp = int(confusion.get("tp", 0) or 0)
        fp = int(confusion.get("fp", 0) or 0)
        fn = int(confusion.get("fn", 0) or 0)
        if metric in {"planning_hit_rate", "planning_miss_rate"}:
            positives = tp + fn
            if positives == 0:
                return None, "not_applicable"
            if metric == "planning_hit_rate":
                return tp / positives, "observed"
            return fn / positives, "observed"
        if metric == "planning_precision":
            predicted_positive = tp + fp
            if predicted_positive == 0:
                return None, "not_applicable"
            return tp / predicted_positive, "observed"
        elif int(planning.get("planning_call_count", 0) or 0) == 0:
            return None, "not_applicable"

    value = _metric_value(episode, metric)
    if value is not None:
        return value, "observed"
    return None, "missing"


def _is_intervention_action(action: str) -> bool:
    action = str(action or "").lower()
    return any(token in action for token in (
        "decelerate",
        "slow",
        "brake",
        "yield",
        "increase_headway",
        "avoid_lane_change",
        "stop",
        "减速",
        "制动",
        "让行",
        "停车",
        "增加车距",
    ))


def _action_family(action: str) -> str:
    action = str(action or "").lower()
    if any(token in action for token in ("stop", "停车")):
        return "stop"
    if any(token in action for token in ("brake", "制动", "decelerate", "减速", "slow")):
        return "decelerate"
    if any(token in action for token in ("yield", "让行")):
        return "yield"
    if any(token in action for token in ("increase_headway", "增加车距")):
        return "increase_headway"
    if any(token in action for token in ("avoid_lane_change", "保持车道", "禁止换道")):
        return "hold_lane"
    if any(token in action for token in ("maintain", "保持速度", "monitor", "观察")):
        return "maintain"
    return "other"


def _risk_action_appropriateness(
    action: str,
    frame: dict,
    scene: dict,
) -> float:
    family = _action_family(action)
    event_type = str(scene.get("event_type") or "").lower()
    longitudinal_risk = bool(
        frame.get("unsafe_ttc")
        or frame.get("unsafe_drac")
        or any(token in event_type for token in (
            "following",
            "rear",
            "cutin",
            "cut_in",
            "lane_change",
        ))
    )
    crossing_risk = bool(
        frame.get("unsafe_dcpa")
        or frame.get("unsafe_future_distance")
        or scene.get("vrus_present")
        or any(token in event_type for token in (
            "cross",
            "intersection",
            "pedestrian",
            "cyclist",
            "roundabout",
        ))
    )

    if family in {"maintain", "other"}:
        return 0.0
    if longitudinal_risk and crossing_risk:
        return 1.0 if family in {
            "stop", "decelerate", "yield", "hold_lane"
        } else 0.75
    if longitudinal_risk:
        if family in {
            "stop", "decelerate", "increase_headway", "hold_lane"
        }:
            return 1.0
        return 0.6
    if crossing_risk:
        if family in {"stop", "decelerate", "yield", "hold_lane"}:
            return 1.0
        return 0.6
    return 0.6 if _is_intervention_action(action) else 0.0


def _decision_episode_metrics(run_dir: Path) -> dict[int, dict[str, float]]:
    accum = defaultdict(lambda: {
        "frames": 0,
        "interventions": 0,
        "safe_frames": 0,
        "unnecessary_interventions": 0,
        "risky_frames": 0,
        "missed_interventions": 0,
        "appropriateness_sum": 0.0,
        "utility_sum": 0.0,
    })
    decisions_path = run_dir / "decisions.jsonl"
    if not decisions_path.exists():
        return {}
    target_profile = {}
    initial_profile_path = run_dir / "initial_profile.json"
    if initial_profile_path.exists():
        try:
            target_profile = json.loads(
                initial_profile_path.read_text(encoding="utf-8")
            )
        except Exception:
            target_profile = {}

    with decisions_path.open("r", encoding="utf-8") as stream:
        for line in stream:
            if not line.strip():
                continue
            record = json.loads(line)
            if record.get("experiment_phase") == "adaptation":
                continue
            event_index = int(record.get("event_index", -1))
            if event_index < 0:
                continue
            decision = record.get("decision") or {}
            frame = record.get("frame_safety") or {}
            scene = record.get("scene") or {}
            profile = record.get("profile") or {}
            if (
                not isinstance(target_profile.get("global"), dict)
                and isinstance(profile.get("global"), dict)
            ):
                # Legacy fallback only. Freeze the first available profile;
                # never let a method's learned profile redefine its own score.
                target_profile = profile
            global_profile = target_profile.get("global") or {}
            safety_weight = _to_float(
                global_profile.get("safety_weight")
            )
            efficiency_weight = _to_float(
                global_profile.get("efficiency_weight")
            )
            safety_weight = 0.5 if safety_weight is None else safety_weight
            efficiency_weight = (
                0.5 if efficiency_weight is None else efficiency_weight
            )
            total_weight = max(1e-12, safety_weight + efficiency_weight)
            safety_weight /= total_weight
            efficiency_weight /= total_weight

            risky = bool(
                frame.get("unsafe_ttc")
                or frame.get("unsafe_drac")
                or frame.get("unsafe_dcpa")
                or frame.get("unsafe_future_distance")
                or (_to_float(frame.get("physical_risk_index")) or 0.0)
                >= 0.65
            )
            intervention = _is_intervention_action(
                decision.get("recommended_action", "")
            )
            action_appropriateness = (
                _risk_action_appropriateness(
                    decision.get("recommended_action", ""),
                    frame,
                    scene,
                )
                if risky else float(not intervention)
            )
            effective_intervention = (
                intervention and action_appropriateness >= 0.75
            )
            item = accum[event_index]
            item["frames"] += 1
            item["interventions"] += int(intervention)
            item["safe_frames"] += int(not risky)
            item["unnecessary_interventions"] += int(
                intervention and not risky
            )
            item["risky_frames"] += int(risky)
            item["missed_interventions"] += int(
                risky and not effective_intervention
            )
            item["appropriateness_sum"] += (
                action_appropriateness if risky else 0.0
            )
            if risky:
                # Safety is a hard constraint. Efficiency preference cannot
                # reward inaction in a physically unsafe frame.
                frame_utility = action_appropriateness
            else:
                frame_utility = (
                    safety_weight
                    + efficiency_weight * float(not intervention)
                )
            item["utility_sum"] += frame_utility

    out = {}
    for event_index, item in accum.items():
        frames = item["frames"]
        safe_frames = item["safe_frames"]
        risky_frames = item["risky_frames"]
        out[event_index] = {
            "decision_intervention_rate": (
                item["interventions"] / frames if frames else 0.0
            ),
            "unnecessary_intervention_rate": (
                item["unnecessary_interventions"] / safe_frames
                if safe_frames else 0.0
            ),
            "missed_intervention_rate": (
                item["missed_interventions"] / risky_frames
                if risky_frames else None
            ),
            "safety_action_appropriateness": (
                item["appropriateness_sum"] / risky_frames
                if risky_frames else None
            ),
            "offline_profile_utility": (
                item["utility_sum"] / frames if frames else 0.0
            ),
        }
    return out


def _load_sample_rows(
    sample_dir: Path,
    dataset: str,
    seed: int,
    job: dict,
) -> tuple[list[dict], Path]:
    summary_path = Path(str(job.get("summary_csv", "")))
    summary_rows = _read_csv(summary_path)
    expected_rows = len(summary_rows)
    sequential_path = (
        summary_path.parent / f"{dataset}_cumulative_census.csv"
    )
    candidates = [
        sequential_path,
        sample_dir / f"{dataset}_core_sample_census_seed{seed}.csv",
    ]
    for path in candidates:
        rows = _read_csv(path)
        if rows and len(rows) == expected_rows:
            return rows, path
    raise FileNotFoundError(
        "Sample census not found or row count does not match summary CSV "
        f"({expected_rows} rows). Tried: "
        + ", ".join(str(path) for path in candidates)
    )


def _rag_episode_metrics(run_dir: Path) -> dict[int, dict[str, float]]:
    accum = defaultdict(lambda: {
        "frames": 0,
        "retrieved_frames": 0,
        "used_frames": 0,
        "grounded_frames": 0,
        "invalid_frames": 0,
        "valid_citations": 0,
        "used_citations": 0,
    })
    decisions_path = run_dir / "decisions.jsonl"
    if not decisions_path.exists():
        return {}
    with decisions_path.open("r", encoding="utf-8") as stream:
        for line in stream:
            if not line.strip():
                continue
            record = json.loads(line)
            if record.get("experiment_phase") == "adaptation":
                continue
            event_index = int(record.get("event_index", -1))
            if event_index < 0:
                continue
            evidence_pack = record.get("evidence_pack", {}) or {}
            grounding = (
                record.get("output_grounding")
                or record.get("grounding")
                or {}
            )
            item = accum[event_index]
            item["frames"] += 1
            item["retrieved_frames"] += int(
                int(evidence_pack.get("num_evidence", 0) or 0) > 0
            )
            item["used_frames"] += int(
                bool(grounding.get("used_evidence_ids"))
            )
            item["grounded_frames"] += int(
                bool(grounding.get("is_grounded"))
            )
            item["invalid_frames"] += int(
                bool(grounding.get("hallucinated_evidence_ids"))
            )
            item["valid_citations"] += len(
                grounding.get("valid_used_evidence_ids") or []
            )
            item["used_citations"] += len(
                grounding.get("used_evidence_ids") or []
            )

    out = {}
    for event_index, item in accum.items():
        frames = item["frames"]
        used_citations = item["used_citations"]
        out[event_index] = {
            "rag_retrieval_coverage": item["retrieved_frames"] / frames,
            "rag_evidence_usage_rate": item["used_frames"] / frames,
            "rag_grounded_decision_rate": item["grounded_frames"] / frames,
            "rag_output_invalid_citation_frame_rate": (
                item["invalid_frames"] / frames
            ),
            "rag_citation_precision": (
                item["valid_citations"] / used_citations
                if used_citations
                else 0.0
            ),
        }
    return out


def _population_counts(census_csv: str | Path) -> dict[str, dict[tuple, int]]:
    out = defaultdict(lambda: defaultdict(int))
    for row in _read_csv(census_csv):
        dataset = str(row.get("dataset"))
        out[dataset][_stratum_key(row)] += 1
    return out


def _estimate_job(
    *,
    status: dict,
    census_counts: dict[str, dict[tuple, int]],
    sample_dir: Path,
    seed: int,
    observation_sink: list[dict] | None = None,
) -> tuple[list[dict], list[dict]]:
    run_dir = Path(status.get("run_dir", ""))
    if not run_dir.exists():
        return [], []

    job = status.get("job", {}) or {}
    summary_path = run_dir / "summary.json"
    if summary_path.exists():
        with summary_path.open("r", encoding="utf-8") as stream:
            run_summary = json.load(stream)
        protocol = run_summary.get("profile_protocol") or {}
        if (
            protocol.get("enabled")
            and protocol.get("formal_inference_eligible") is False
        ):
            return [], []
    dataset = str(job.get("dataset", ""))
    sample_rows, sample_census_path = _load_sample_rows(
        sample_dir, dataset, seed, job
    )
    sample_by_event_index = {
        idx: row for idx, row in enumerate(sample_rows)
    }
    episodes = read_jsonl(run_dir / "episode_summary.jsonl")
    if not episodes:
        return [], []
    episodes = [
        episode
        for episode in episodes
        if episode.get("experiment_phase") != "adaptation"
    ]
    if not episodes:
        return [], []
    evaluation_event_indexes = {
        int(episode.get("event_index", -1))
        for episode in episodes
    }
    sample_counts_by_key = defaultdict(int)
    core_counts_by_key = defaultdict(int)
    for row in sample_by_event_index.values():
        core_counts_by_key[_stratum_key(row)] += 1
    for event_index, row in sample_by_event_index.items():
        if event_index in evaluation_event_indexes:
            sample_counts_by_key[_stratum_key(row)] += 1
    expected_sample_total = len(evaluation_event_indexes)

    grouped_values = {
        metric: defaultdict(list)
        for metric in METRICS
    }
    unweighted_values = defaultdict(list)
    rag_by_event = _rag_episode_metrics(run_dir)
    decision_by_event = _decision_episode_metrics(run_dir)
    not_applicable_by_metric_key = {
        metric: defaultdict(int) for metric in METRICS
    }
    censored_by_metric_key = {
        metric: defaultdict(int) for metric in METRICS
    }
    missing_by_metric_key = {
        metric: defaultdict(int) for metric in METRICS
    }

    for episode in episodes:
        event_index = int(episode.get("event_index", -1))
        sample_row = sample_by_event_index.get(event_index)
        if sample_row is None:
            continue
        key = _stratum_key(sample_row)
        for metric in METRICS:
            if metric.startswith("rag_"):
                value = rag_by_event.get(event_index, {}).get(metric)
                status_name = "observed" if value is not None else "missing"
            elif metric in decision_by_event.get(event_index, {}):
                value = decision_by_event[event_index].get(metric)
                status_name = (
                    "observed"
                    if value is not None
                    else "not_applicable"
                )
            else:
                value, status_name = _metric_value_with_status(
                    episode, metric
                )
            if value is None:
                if status_name == "not_applicable":
                    target = not_applicable_by_metric_key
                elif status_name == "right_censored":
                    target = censored_by_metric_key
                else:
                    target = missing_by_metric_key
                target[metric][key] += 1
                continue
            grouped_values[metric][key].append(
                (value, _cluster_id(sample_row), event_index)
            )
            unweighted_values[metric].append(value)

    population_by_key = census_counts.get(dataset, {})
    population_total = sum(population_by_key.values())
    job_fields = {
        "job_id": status.get("job_id"),
        "run_dir": str(run_dir),
        "dataset": dataset,
        "mode": job.get("mode", ""),
        "profile_name": job.get("profile_name", ""),
        "use_profile_learner": (
            job.get("extra_args", {}) or {}
        ).get("use_profile_learner", ""),
        "profile_protocol_enabled": (
            job.get("extra_args", {}) or {}
        ).get("profile_protocol_enabled", ""),
        "profile_adaptation_episodes": (
            job.get("extra_args", {}) or {}
        ).get("profile_adaptation_episodes", 0),
        "profile_adaptation_pool_episodes": (
            job.get("extra_args", {}) or {}
        ).get("profile_adaptation_pool_episodes", 0),
        "rag_variant": job.get("rag_variant", ""),
        "planning_variant": job.get("planning_variant", ""),
        "llm_policy_variant": job.get("llm_policy_variant", ""),
        "sample_census_path": str(sample_census_path),
        "estimand": (
            "full_cornercase_population_mean_under_two_stage_"
            "stratified_sampling"
        ),
        "inclusion_probability_method": (
            "pi_core_times_pi_evaluation_given_core"
        ),
    }

    weighted_rows = []
    stratum_rows = []
    for metric, values_by_key in grouped_values.items():
        if not values_by_key or population_total <= 0:
            continue

        conditional = metric in CONDITIONAL_METRICS
        estimation_population_by_key = {}
        for key, population_rows in population_by_key.items():
            if population_rows <= 0:
                continue
            if conditional:
                sampled_rows = sample_counts_by_key.get(key, 0)
                observed_rows = len(values_by_key.get(key, []))
                if sampled_rows <= 0 or observed_rows <= 0:
                    continue
                estimation_population_by_key[key] = (
                    population_rows * observed_rows / sampled_rows
                )
            else:
                estimation_population_by_key[key] = population_rows
        estimation_population_total = sum(
            estimation_population_by_key.values()
        )
        if estimation_population_total <= 0:
            continue

        weighted_mean = 0.0
        variance = 0.0
        sample_total = 0
        used_strata = 0
        observed_population = 0

        for key, observations in sorted(values_by_key.items()):
            values = [value for value, _, _ in observations]
            population_rows = int(population_by_key.get(key, 0))
            core_sample_rows = int(core_counts_by_key.get(key, 0))
            evaluation_sample_rows = int(
                sample_counts_by_key.get(key, 0)
            )
            sample_rows_n = len(values)
            if population_rows <= 0 or sample_rows_n <= 0:
                continue

            estimation_population = estimation_population_by_key.get(
                key, 0.0
            )
            if estimation_population <= 0:
                continue
            weight = (
                estimation_population / estimation_population_total
            )
            mean_h = _mean(values)
            var_h = _variance(values)
            finite_population_correction = max(0.0, 1.0 - sample_rows_n / population_rows)
            weighted_mean += weight * mean_h
            variance += (weight ** 2) * finite_population_correction * var_h / sample_rows_n
            sample_total += sample_rows_n
            used_strata += 1
            observed_population += estimation_population

            risk_stratum, event_type, risk_label, vru_present = key
            core_pi = (
                core_sample_rows / population_rows
                if population_rows else 0.0
            )
            eval_given_core_pi = (
                evaluation_sample_rows / core_sample_rows
                if core_sample_rows else 0.0
            )
            stratum_rows.append({
                **job_fields,
                "metric": metric,
                "risk_stratum": risk_stratum,
                "event_type": event_type,
                "dataset_risk_label": risk_label,
                "vru_present": vru_present,
                "population_rows": population_rows,
                "core_sample_rows": core_sample_rows,
                "evaluation_sample_rows": evaluation_sample_rows,
                "sample_rows": sample_rows_n,
                "core_inclusion_probability": core_pi,
                "evaluation_given_core_probability": (
                    eval_given_core_pi
                ),
                "combined_inclusion_probability": (
                    core_pi * eval_given_core_pi
                ),
                "stratum_weight": weight,
                "stratum_mean": mean_h,
                "stratum_variance": var_h,
            })

        observed_keys = {
            key
            for key, observations in values_by_key.items()
            if observations and population_by_key.get(key, 0) > 0
        }
        expected_keys = {
            key
            for key, count in population_by_key.items()
            if count > 0
        }
        missing_keys = sorted(expected_keys - observed_keys)
        missing_metric_rows = sum(
            missing_by_metric_key[metric].values()
        )
        not_applicable_rows = sum(
            not_applicable_by_metric_key[metric].values()
        )
        censored_rows = sum(censored_by_metric_key[metric].values())
        expected_applicable_total = max(
            0,
            expected_sample_total
            - not_applicable_rows
            - censored_rows,
        )
        metric_completeness = (
            sample_total / expected_applicable_total
            if expected_applicable_total else 1.0
        )
        population_coverage = (
            observed_population / population_total
            if population_total else 0.0
        )
        complete_applicable_rows = (
            missing_metric_rows == 0
            and sample_total == expected_applicable_total
        )
        estimate_valid = bool(observed_keys) and complete_applicable_rows
        if not conditional:
            estimate_valid = (
                estimate_valid
                and not missing_keys
                and math.isclose(
                    population_coverage, 1.0, abs_tol=1e-12
                )
            )

        if observation_sink is not None:
            for key, observations in values_by_key.items():
                population_rows = int(population_by_key.get(key, 0))
                core_sample_rows = int(core_counts_by_key.get(key, 0))
                sample_rows_n = sample_counts_by_key.get(key, 0)
                if population_rows <= 0 or sample_rows_n <= 0:
                    continue
                for value, cluster, event_index in observations:
                    observation_sink.append({
                        **job_fields,
                        "event_index": event_index,
                        "recording_cluster": cluster,
                        "risk_stratum": key[0],
                        "event_type": key[1],
                        "dataset_risk_label": key[2],
                        "vru_present": key[3],
                        "metric": metric,
                        "value": value,
                        "core_sample_rows": core_sample_rows,
                        "evaluation_sample_rows": sample_rows_n,
                        "core_inclusion_probability": (
                            core_sample_rows / population_rows
                        ),
                        "evaluation_given_core_probability": (
                            sample_rows_n / core_sample_rows
                            if core_sample_rows else 0.0
                        ),
                        "combined_inclusion_probability": (
                            sample_rows_n / population_rows
                        ),
                        "design_weight": population_rows / sample_rows_n,
                    })
        uncertainty = (
            _cluster_bootstrap_uncertainty(
                values_by_key,
                estimation_population_by_key,
                estimation_population_total,
                weighted_mean,
                seed=_stable_seed(
                    status.get("job_id"), dataset, metric, seed
                ),
            )
            if estimate_valid
            else {}
        )
        se = uncertainty.get("se") if estimate_valid else None
        weighted_rows.append({
            **job_fields,
            "metric": metric,
            "population_total": population_total,
            "sample_total": sample_total,
            "expected_sample_total": expected_sample_total,
            "num_strata": used_strata,
            "estimate_valid": estimate_valid,
            "population_coverage": population_coverage,
            "metric_completeness": metric_completeness,
            "missing_metric_rows": missing_metric_rows,
            "not_applicable_rows": not_applicable_rows,
            "censored_rows": censored_rows,
            "missingness_policy": (
                "observed_reactions_only_with_censor_count"
                if metric == "reaction_delay_frames"
                else "conditional_structural_na"
                if conditional
                else "complete_case_required"
            ),
            "observed_weight_sum": population_coverage,
            "missing_strata_count": len(missing_keys),
            "missing_strata_json": json.dumps(
                [
                    {
                        "risk_stratum": key[0],
                        "event_type": key[1],
                        "dataset_risk_label": key[2],
                        "vru_present": key[3],
                    }
                    for key in missing_keys
                ],
                ensure_ascii=False,
                sort_keys=True,
            ),
            "variance_method": uncertainty.get(
                "variance_method", ""
            ),
            "num_recording_clusters": uncertainty.get(
                "num_recording_clusters", 0
            ),
            "singleton_strata_count": uncertainty.get(
                "singleton_strata_count", 0
            ),
            "cluster_bootstrap_rounds": uncertainty.get(
                "cluster_bootstrap_rounds", 0
            ),
            "weighted_mean": weighted_mean if estimate_valid else None,
            "weighted_se": se,
            "ci95_low": uncertainty.get("ci95_low"),
            "ci95_high": uncertainty.get("ci95_high"),
            "unweighted_mean": _mean(unweighted_values.get(metric, [])),
        })

    return weighted_rows, stratum_rows


def build_weighted_estimates(
    experiment_dir: str | Path,
    *,
    census_csv: str | Path = DEFAULT_CENSUS,
    sample_dir: str | Path = DEFAULT_SAMPLE_DIR,
    seed: int = 20260613,
) -> tuple[list[dict], list[dict]]:
    experiment_dir = Path(experiment_dir)
    sample_dir = Path(sample_dir)
    census_counts = _population_counts(census_csv)

    weighted_rows = []
    stratum_rows = []
    observation_rows = []
    for status in latest_usable_completed_statuses(experiment_dir):
        job_weighted, job_strata = _estimate_job(
            status=status,
            census_counts=census_counts,
            sample_dir=sample_dir,
            seed=seed,
            observation_sink=observation_rows,
        )
        weighted_rows.extend(job_weighted)
        stratum_rows.extend(job_strata)

    write_csv(experiment_dir / "weighted_metric_summary.csv", weighted_rows, fieldnames=WEIGHTED_FIELDS)
    write_csv(experiment_dir / "weighted_stratum_metric_summary.csv", stratum_rows, fieldnames=STRATUM_FIELDS)
    write_csv(
        experiment_dir / "weighted_episode_observations.csv",
        observation_rows,
        fieldnames=OBSERVATION_FIELDS,
    )
    return weighted_rows, stratum_rows


def clear_weighted_outputs(experiment_dir: str | Path) -> None:
    experiment_dir = Path(experiment_dir)
    write_csv(
        experiment_dir / "weighted_metric_summary.csv",
        [],
        fieldnames=WEIGHTED_FIELDS,
    )
    write_csv(
        experiment_dir / "weighted_stratum_metric_summary.csv",
        [],
        fieldnames=STRATUM_FIELDS,
    )
    write_csv(
        experiment_dir / "weighted_episode_observations.csv",
        [],
        fieldnames=OBSERVATION_FIELDS,
    )


def main():
    parser = argparse.ArgumentParser(description="Build design-weighted estimates for Neyman-stratified ResponsiveGPT samples.")
    parser.add_argument("--experiment_dir", required=True)
    parser.add_argument("--census_csv", default=DEFAULT_CENSUS)
    parser.add_argument("--sample_dir", default=DEFAULT_SAMPLE_DIR)
    parser.add_argument("--seed", type=int, default=20260613)
    args = parser.parse_args()

    weighted_rows, stratum_rows = build_weighted_estimates(
        args.experiment_dir,
        census_csv=args.census_csv,
        sample_dir=args.sample_dir,
        seed=args.seed,
    )
    print(
        f"Wrote {len(weighted_rows)} weighted metric rows and {len(stratum_rows)} "
        f"stratum rows to {args.experiment_dir}"
    )


if __name__ == "__main__":
    main()
