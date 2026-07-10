from collections import defaultdict
from pathlib import Path
import re

from .io_utils import write_csv


RAG_TABLE_FIELDS = [
    "estimator",
    "dataset",
    "mode",
    "profile_name",
    "use_profile_learner",
    "rag_variant",
    "planning_variant",
    "llm_policy_family",
    "profile_adaptation_episodes",
    "frame_selection",
    "critical_top_k",
    "num_runs",
    "total_events",
    "total_frames",
    "precision",
    "recall",
    "f1",
    "accuracy",
    "avg_underreaction_rate",
    "avg_overreaction_rate",
    "reactive_llm_call_rate",
    "reactive_llm_attempt_rate",
    "planning_reuse_rate",
    "risk_phase_transition_rate",
    "llm_cache_hit_rate",
    "planning_cache_hit_rate",
    "rag_cache_hit_rate",
    "retrieval_coverage",
    "evidence_usage_rate",
    "grounded_decision_rate",
    "hallucinated_citation_rate",
    "raw_invalid_citation_attempt_rate",
    "output_invalid_citation_frame_rate",
    "citation_precision",
    "avg_evidence_per_frame",
]

CROSS_DATASET_FIELDS = [
    "estimator",
    "dataset",
    "mode",
    "profile_name",
    "use_profile_learner",
    "rag_variant",
    "planning_variant",
    "llm_policy_family",
    "profile_adaptation_episodes",
    "frame_selection",
    "critical_top_k",
    "num_runs",
    "total_events",
    "total_frames",
    "precision",
    "recall",
    "f1",
    "accuracy",
    "avg_underreaction_rate",
    "avg_overreaction_rate",
    "reactive_llm_call_rate",
    "reactive_llm_attempt_rate",
    "planning_reuse_rate",
    "risk_phase_transition_rate",
    "llm_cache_hit_rate",
    "planning_cache_hit_rate",
    "rag_cache_hit_rate",
    "retrieval_coverage",
    "evidence_usage_rate",
    "grounded_decision_rate",
    "hallucinated_citation_rate",
    "raw_invalid_citation_attempt_rate",
    "output_invalid_citation_frame_rate",
    "citation_precision",
    "avg_evidence_per_frame",
]

PROFILE_TABLE_FIELDS = [
    "estimator",
    "dataset",
    "mode",
    "profile_name",
    "use_profile_learner",
    "rag_variant",
    "planning_variant",
    "llm_policy_family",
    "profile_adaptation_episodes",
    "frame_selection",
    "critical_top_k",
    "num_runs",
    "total_events",
    "total_frames",
    "f1",
    "accuracy",
    "avg_underreaction_rate",
    "avg_overreaction_rate",
    "avg_reaction_delay_frames",
    "avg_decision_flip_rate",
    "planning_reuse_rate",
    "risk_phase_transition_rate",
    "grounded_decision_rate",
]

MODE_TABLE_FIELDS = [
    "estimator",
    "dataset",
    "mode",
    "profile_name",
    "use_profile_learner",
    "rag_variant",
    "planning_variant",
    "llm_policy_family",
    "profile_adaptation_episodes",
    "frame_selection",
    "critical_top_k",
    "num_runs",
    "total_events",
    "total_frames",
    "selected_frames",
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
    "planning_reuse_rate",
    "risk_phase_transition_rate",
    "llm_cache_hit_rate",
    "planning_cache_hit_rate",
    "rag_cache_hit_rate",
    "retrieval_coverage",
    "grounded_decision_rate",
]

WEIGHTED_PRIMARY_FIELDS = [
    "estimator",
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
    "metric",
    "weighted_mean",
    "weighted_se",
    "ci95_low",
    "ci95_high",
    "population_total",
    "sample_total",
    "population_coverage",
    "metric_completeness",
    "not_applicable_rows",
    "censored_rows",
    "missingness_policy",
    "variance_method",
    "num_recording_clusters",
    "singleton_strata_count",
]

PROFILE_LEARNING_FIELDS = [
    "estimator",
    "dataset",
    "mode",
    "profile_name",
    "use_profile_learner",
    "rag_variant",
    "planning_variant",
    "llm_policy_family",
    "profile_adaptation_episodes",
    "frame_selection",
    "critical_top_k",
    "num_runs",
    "avg_underreaction_rate",
    "avg_overreaction_rate",
    "avg_reaction_success_rate",
    "avg_reaction_delay_frames",
    "avg_decision_flip_rate",
    "avg_decision_intervention_rate",
    "avg_unnecessary_intervention_rate",
    "avg_missed_intervention_rate",
    "avg_offline_profile_utility",
    "profile_changed_parameter_count",
    "profile_parameter_delta_l1",
]

PROFILE_ADAPTATION_BUDGET_FIELDS = [
    "estimator",
    "dataset",
    "mode",
    "profile_name",
    "use_profile_learner",
    "rag_variant",
    "planning_variant",
    "llm_policy_family",
    "profile_adaptation_episodes",
    "num_runs",
    "avg_underreaction_rate",
    "avg_overreaction_rate",
    "avg_reaction_success_rate",
    "avg_offline_profile_utility",
    "profile_changed_parameter_count",
    "profile_parameter_delta_l1",
    "adaptation_frames",
    "adaptation_reactive_attempts",
    "adaptation_reactive_total_tokens",
    "adaptation_planning_attempts",
    "adaptation_planning_total_tokens",
    "adaptation_rag_calls",
    "adaptation_rag_latency_ms_p95",
    "evaluation_frames",
    "evaluation_reactive_attempts",
    "evaluation_reactive_total_tokens",
    "evaluation_planning_attempts",
    "evaluation_planning_total_tokens",
    "evaluation_rag_calls",
    "evaluation_rag_latency_ms_p95",
]


def _llm_policy_family(value) -> str:
    text = str(value or "")
    normalized = re.sub(
        r"(?:_?(?:order_)?seed_?)\d+$",
        "",
        text,
    ).rstrip("_")
    return normalized or "seed_family"

BUDGET_AUDIT_FIELDS = [
    "dataset",
    "mode",
    "profile_name",
    "use_profile_learner",
    "planning_variant",
    "llm_policy_variant",
    "comparison_dimension",
    "compared_variants",
    "num_variants",
    "equal_request_caps",
    "equal_token_caps",
    "reactive_attempt_ratio",
    "reactive_token_ratio",
    "planning_attempt_ratio",
    "planning_token_ratio",
    "actual_usage_matched_5pct",
]


def _mean(values):
    vals = [float(v) for v in values if isinstance(v, (int, float)) or str(v).replace(".", "", 1).isdigit()]
    return sum(vals) / len(vals) if vals else ""


def _is_formal_performance_row(row: dict) -> bool:
    enabled = str(
        row.get("profile_protocol_enabled", "")
    ).strip().lower() in {"1", "true"}
    eligible = str(
        row.get("profile_formal_inference_eligible", "")
    ).strip().lower()
    return not (enabled and eligible in {"0", "false"})


def _treatment_cell(row: dict) -> tuple:
    return (
        str(row.get("dataset", "unknown")),
        str(row.get("mode", "unknown")),
        str(row.get("profile_name", "unknown")),
        str(row.get("use_profile_learner", "")),
        str(row.get("planning_variant", "unknown")),
        _llm_policy_family(row.get("llm_policy_variant", "")),
        str(row.get("profile_adaptation_episodes", "")),
        str(row.get("frame_selection", "")),
        str(row.get("critical_top_k", "")),
    )


def _cell_fields(key: tuple) -> dict:
    return dict(zip((
        "dataset",
        "mode",
        "profile_name",
        "use_profile_learner",
        "planning_variant",
        "llm_policy_family",
        "profile_adaptation_episodes",
        "frame_selection",
        "critical_top_k",
    ), key))


def make_rag_ablation_table(rows: list[dict], output_dir: str | Path) -> list[dict]:
    grouped = defaultdict(list)
    for row in rows:
        if not _is_formal_performance_row(row):
            continue
        key = _treatment_cell(row) + (
            str(row.get("rag_variant", "unknown")),
        )
        grouped[key].append(row)

    out = []
    for key, group in sorted(grouped.items()):
        rag_variant = key[-1]
        item = {
            "estimator": "unweighted_descriptive",
            "rag_variant": rag_variant,
            "num_runs": len(group),
            "total_events": sum(int(r.get("total_events") or 0) for r in group),
            "total_frames": sum(int(r.get("total_frames") or 0) for r in group),
            "precision": _mean([r.get("precision") for r in group]),
            "recall": _mean([r.get("recall") for r in group]),
            "f1": _mean([r.get("f1") for r in group]),
            "accuracy": _mean([r.get("accuracy") for r in group]),
            "avg_underreaction_rate": _mean([r.get("avg_underreaction_rate") for r in group]),
            "avg_overreaction_rate": _mean([r.get("avg_overreaction_rate") for r in group]),
            "reactive_llm_call_rate": _mean([r.get("reactive_llm_call_rate") for r in group]),
            "reactive_llm_attempt_rate": _mean([r.get("reactive_llm_attempt_rate") for r in group]),
            "planning_reuse_rate": _mean([r.get("planning_reuse_rate") for r in group]),
            "risk_phase_transition_rate": _mean(
                [r.get("risk_phase_transition_rate") for r in group]
            ),
            "llm_cache_hit_rate": _mean([r.get("llm_cache_hit_rate") for r in group]),
            "planning_cache_hit_rate": _mean([r.get("planning_cache_hit_rate") for r in group]),
            "rag_cache_hit_rate": _mean([r.get("rag_cache_hit_rate") for r in group]),
            "retrieval_coverage": _mean([r.get("retrieval_coverage") for r in group]),
            "evidence_usage_rate": _mean([r.get("evidence_usage_rate") for r in group]),
            "grounded_decision_rate": _mean([r.get("grounded_decision_rate") for r in group]),
            "hallucinated_citation_rate": _mean([r.get("hallucinated_citation_rate") for r in group]),
            "raw_invalid_citation_attempt_rate": _mean(
                [r.get("raw_invalid_citation_attempt_rate") for r in group]
            ),
            "output_invalid_citation_frame_rate": _mean(
                [r.get("output_invalid_citation_frame_rate") for r in group]
            ),
            "citation_precision": _mean([r.get("citation_precision") for r in group]),
            "avg_evidence_per_frame": _mean([r.get("avg_evidence_per_frame") for r in group]),
        }
        item.update(_cell_fields(key[:-1]))
        out.append(item)

    write_csv(Path(output_dir) / "rag_ablation_table.csv", out, fieldnames=RAG_TABLE_FIELDS)
    return out


def _summarize_group(group: list[dict]) -> dict:
    return {
        "num_runs": len(group),
        "total_events": sum(int(r.get("total_events") or 0) for r in group),
        "total_frames": sum(int(r.get("total_frames") or 0) for r in group),
        "selected_frames": sum(int(r.get("selected_frames") or 0) for r in group),
        "precision": _mean([r.get("precision") for r in group]),
        "recall": _mean([r.get("recall") for r in group]),
        "f1": _mean([r.get("f1") for r in group]),
        "accuracy": _mean([r.get("accuracy") for r in group]),
        "avg_underreaction_rate": _mean([r.get("avg_underreaction_rate") for r in group]),
        "avg_overreaction_rate": _mean([r.get("avg_overreaction_rate") for r in group]),
        "avg_reaction_delay_frames": _mean([r.get("avg_reaction_delay_frames") for r in group]),
        "avg_decision_flip_rate": _mean([r.get("avg_decision_flip_rate") for r in group]),
        "reactive_llm_call_rate": _mean([r.get("reactive_llm_call_rate") for r in group]),
        "reactive_llm_attempt_rate": _mean([r.get("reactive_llm_attempt_rate") for r in group]),
        "planning_reuse_rate": _mean([r.get("planning_reuse_rate") for r in group]),
        "risk_phase_transition_rate": _mean(
            [r.get("risk_phase_transition_rate") for r in group]
        ),
        "llm_cache_hit_rate": _mean([r.get("llm_cache_hit_rate") for r in group]),
        "planning_cache_hit_rate": _mean([r.get("planning_cache_hit_rate") for r in group]),
        "rag_cache_hit_rate": _mean([r.get("rag_cache_hit_rate") for r in group]),
        "retrieval_coverage": _mean([r.get("retrieval_coverage") for r in group]),
        "evidence_usage_rate": _mean([r.get("evidence_usage_rate") for r in group]),
        "grounded_decision_rate": _mean([r.get("grounded_decision_rate") for r in group]),
        "hallucinated_citation_rate": _mean([r.get("hallucinated_citation_rate") for r in group]),
        "raw_invalid_citation_attempt_rate": _mean(
            [r.get("raw_invalid_citation_attempt_rate") for r in group]
        ),
        "output_invalid_citation_frame_rate": _mean(
            [r.get("output_invalid_citation_frame_rate") for r in group]
        ),
        "citation_precision": _mean([r.get("citation_precision") for r in group]),
        "avg_evidence_per_frame": _mean([r.get("avg_evidence_per_frame") for r in group]),
    }


def make_cross_dataset_table(rows: list[dict], output_dir: str | Path) -> list[dict]:
    grouped = defaultdict(list)
    for row in rows:
        if not _is_formal_performance_row(row):
            continue
        key = _treatment_cell(row) + (
            row.get("rag_variant", "unknown"),
        )
        grouped[key].append(row)

    out = []
    for key, group in sorted(grouped.items()):
        item = {
            "estimator": "unweighted_descriptive",
            "rag_variant": key[-1],
        }
        item.update(_cell_fields(key[:-1]))
        item.update(_summarize_group(group))
        out.append(item)

    write_csv(Path(output_dir) / "cross_dataset_table.csv", out, fieldnames=CROSS_DATASET_FIELDS)
    return out


def make_profile_adaptation_table(rows: list[dict], output_dir: str | Path) -> list[dict]:
    grouped = defaultdict(list)
    for row in rows:
        if not _is_formal_performance_row(row):
            continue
        key = _treatment_cell(row) + (
            row.get("rag_variant", "unknown"),
        )
        grouped[key].append(row)

    out = []
    for key, group in sorted(grouped.items()):
        item = {
            "estimator": "unweighted_descriptive",
            "rag_variant": key[-1],
        }
        item.update(_cell_fields(key[:-1]))
        item.update(_summarize_group(group))
        out.append(item)

    write_csv(Path(output_dir) / "profile_adaptation_table.csv", out, fieldnames=PROFILE_TABLE_FIELDS)
    return out


def make_mode_comparison_table(rows: list[dict], output_dir: str | Path) -> list[dict]:
    grouped = defaultdict(list)
    for row in rows:
        if not _is_formal_performance_row(row):
            continue
        key = _treatment_cell(row) + (
            row.get("rag_variant", "unknown"),
        )
        grouped[key].append(row)

    out = []
    for key, group in sorted(grouped.items()):
        item = {
            "estimator": "unweighted_descriptive",
            "rag_variant": key[-1],
        }
        item.update(_cell_fields(key[:-1]))
        item.update(_summarize_group(group))
        out.append(item)

    write_csv(Path(output_dir) / "mode_comparison_table.csv", out, fieldnames=MODE_TABLE_FIELDS)
    return out


def make_weighted_primary_table(
    weighted_rows: list[dict],
    output_dir: str | Path,
) -> list[dict]:
    out = []
    for row in weighted_rows:
        if str(row.get("estimate_valid", "")).lower() not in {
            "true", "1"
        } and row.get("estimate_valid") is not True:
            continue
        out.append({
            "estimator": "design_weighted_primary",
            "dataset": row.get("dataset"),
            "mode": row.get("mode"),
            "profile_name": row.get("profile_name"),
            "use_profile_learner": row.get("use_profile_learner"),
            "profile_protocol_enabled": row.get(
                "profile_protocol_enabled"
            ),
            "profile_adaptation_episodes": row.get(
                "profile_adaptation_episodes"
            ),
            "profile_adaptation_pool_episodes": row.get(
                "profile_adaptation_pool_episodes"
            ),
            "rag_variant": row.get("rag_variant"),
            "planning_variant": row.get("planning_variant"),
            "llm_policy_variant": row.get("llm_policy_variant"),
            "metric": row.get("metric"),
            "weighted_mean": row.get("weighted_mean"),
            "weighted_se": row.get("weighted_se"),
            "ci95_low": row.get("ci95_low"),
            "ci95_high": row.get("ci95_high"),
            "population_total": row.get("population_total"),
            "sample_total": row.get("sample_total"),
            "population_coverage": row.get("population_coverage"),
            "metric_completeness": row.get("metric_completeness"),
            "not_applicable_rows": row.get("not_applicable_rows"),
            "censored_rows": row.get("censored_rows"),
            "missingness_policy": row.get("missingness_policy"),
            "variance_method": row.get("variance_method"),
            "num_recording_clusters": row.get(
                "num_recording_clusters"
            ),
            "singleton_strata_count": row.get(
                "singleton_strata_count"
            ),
        })
    write_csv(
        Path(output_dir) / "paper_primary_weighted_table.csv",
        out,
        fieldnames=WEIGHTED_PRIMARY_FIELDS,
    )
    return out


def make_profile_learning_ablation_table(
    rows: list[dict],
    output_dir: str | Path,
) -> list[dict]:
    grouped = defaultdict(list)
    for row in rows:
        if not _is_formal_performance_row(row):
            continue
        key = _treatment_cell(row) + (
            str(row.get("rag_variant", "unknown")),
        )
        grouped[key].append(row)

    out = []
    for key, group in sorted(grouped.items()):
        item = {
            "estimator": "unweighted_descriptive",
            "rag_variant": key[-1],
            "num_runs": len(group),
            "avg_underreaction_rate": _mean([
                row.get("avg_underreaction_rate") for row in group
            ]),
            "avg_overreaction_rate": _mean([
                row.get("avg_overreaction_rate") for row in group
            ]),
            "avg_reaction_success_rate": _mean([
                row.get("avg_reaction_success_rate") for row in group
            ]),
            "avg_reaction_delay_frames": _mean([
                row.get("avg_reaction_delay_frames") for row in group
            ]),
            "avg_decision_flip_rate": _mean([
                row.get("avg_decision_flip_rate") for row in group
            ]),
            "avg_decision_intervention_rate": _mean([
                row.get("avg_decision_intervention_rate")
                for row in group
            ]),
            "avg_unnecessary_intervention_rate": _mean([
                row.get("avg_unnecessary_intervention_rate")
                for row in group
            ]),
            "avg_missed_intervention_rate": _mean([
                row.get("avg_missed_intervention_rate")
                for row in group
            ]),
            "avg_offline_profile_utility": _mean([
                row.get("avg_offline_profile_utility")
                for row in group
            ]),
            "profile_changed_parameter_count": _mean([
                row.get("profile_changed_parameter_count") for row in group
            ]),
            "profile_parameter_delta_l1": _mean([
                row.get("profile_parameter_delta_l1") for row in group
            ]),
        }
        item.update(_cell_fields(key[:-1]))
        out.append(item)
    write_csv(
        Path(output_dir) / "profile_learning_ablation_table.csv",
        out,
        fieldnames=PROFILE_LEARNING_FIELDS,
    )
    return out


def make_profile_adaptation_budget_curve(
    rows: list[dict],
    weighted_rows: list[dict],
    output_dir: str | Path,
) -> list[dict]:
    grouped = defaultdict(list)
    weighted_grouped = defaultdict(lambda: defaultdict(list))
    metric_names = {
        "underreaction_rate": "avg_underreaction_rate",
        "overreaction_rate": "avg_overreaction_rate",
        "reaction_success_rate": "avg_reaction_success_rate",
        "offline_profile_utility": "avg_offline_profile_utility",
    }
    for row in rows:
        if str(row.get("profile_protocol_enabled", "")).lower() in {
            "1", "true"
        } and str(
            row.get("profile_formal_inference_eligible", "true")
        ).lower() not in {"0", "false"}:
            key = (
                row.get("dataset", "unknown"),
                row.get("mode", "unknown"),
                row.get("profile_name", "unknown"),
                row.get("use_profile_learner", ""),
                row.get("rag_variant", "unknown"),
                row.get("planning_variant", "unknown"),
                _llm_policy_family(
                    row.get("llm_policy_variant", "")
                ),
                int(row.get("profile_adaptation_episodes") or 0),
            )
            grouped[key].append(row)
    for row in weighted_rows:
        if str(row.get("estimate_valid", "")).lower() not in {
            "1", "true"
        } and row.get("estimate_valid") is not True:
            continue
        metric = str(row.get("metric", ""))
        if metric not in metric_names:
            continue
        key = (
            row.get("dataset", "unknown"),
            row.get("mode", "unknown"),
            row.get("profile_name", "unknown"),
            row.get("use_profile_learner", ""),
            row.get("rag_variant", "unknown"),
            row.get("planning_variant", "unknown"),
            _llm_policy_family(row.get("llm_policy_variant", "")),
            int(row.get("profile_adaptation_episodes") or 0),
        )
        weighted_grouped[key][metric_names[metric]].append(
            row.get("weighted_mean")
        )

    out = []
    for (
        dataset,
        mode,
        profile_name,
        use_profile_learner,
        rag_variant,
        planning_variant,
        llm_policy_family,
        adaptation_episodes,
    ), group in sorted(grouped.items()):
        weighted = weighted_grouped.get(
            (
                dataset,
                mode,
                profile_name,
                use_profile_learner,
                rag_variant,
                planning_variant,
                llm_policy_family,
                adaptation_episodes,
            ),
            {},
        )
        if not weighted:
            continue
        out.append({
            "estimator": "design_weighted_primary",
            "dataset": dataset,
            "mode": mode,
            "profile_name": profile_name,
            "use_profile_learner": use_profile_learner,
            "rag_variant": rag_variant,
            "planning_variant": planning_variant,
            "llm_policy_family": llm_policy_family,
            "profile_adaptation_episodes": adaptation_episodes,
            "num_runs": len(group),
            "avg_underreaction_rate": _mean(
                weighted.get("avg_underreaction_rate", [])
            ),
            "avg_overreaction_rate": _mean(
                weighted.get("avg_overreaction_rate", [])
            ),
            "avg_reaction_success_rate": _mean(
                weighted.get("avg_reaction_success_rate", [])
            ),
            "avg_offline_profile_utility": _mean(
                weighted.get("avg_offline_profile_utility", [])
            ),
            "profile_changed_parameter_count": _mean([
                row.get("profile_changed_parameter_count") for row in group
            ]),
            "profile_parameter_delta_l1": _mean([
                row.get("profile_parameter_delta_l1") for row in group
            ]),
            "adaptation_frames": _mean([
                row.get("adaptation_frames") for row in group
            ]),
            "adaptation_reactive_attempts": _mean([
                row.get("adaptation_reactive_attempts") for row in group
            ]),
            "adaptation_reactive_total_tokens": _mean([
                row.get("adaptation_reactive_total_tokens") for row in group
            ]),
            "adaptation_planning_attempts": _mean([
                row.get("adaptation_planning_attempts") for row in group
            ]),
            "adaptation_planning_total_tokens": _mean([
                row.get("adaptation_planning_total_tokens") for row in group
            ]),
            "adaptation_rag_calls": _mean([
                row.get("adaptation_rag_calls") for row in group
            ]),
            "adaptation_rag_latency_ms_p95": _mean([
                row.get("adaptation_rag_latency_ms_p95") for row in group
            ]),
            "evaluation_frames": _mean([
                row.get("evaluation_frames") for row in group
            ]),
            "evaluation_reactive_attempts": _mean([
                row.get("evaluation_reactive_attempts") for row in group
            ]),
            "evaluation_reactive_total_tokens": _mean([
                row.get("evaluation_reactive_total_tokens") for row in group
            ]),
            "evaluation_planning_attempts": _mean([
                row.get("evaluation_planning_attempts") for row in group
            ]),
            "evaluation_planning_total_tokens": _mean([
                row.get("evaluation_planning_total_tokens") for row in group
            ]),
            "evaluation_rag_calls": _mean([
                row.get("evaluation_rag_calls") for row in group
            ]),
            "evaluation_rag_latency_ms_p95": _mean([
                row.get("evaluation_rag_latency_ms_p95") for row in group
            ]),
        })

    write_csv(
        Path(output_dir) / "profile_adaptation_budget_curve.csv",
        out,
        fieldnames=PROFILE_ADAPTATION_BUDGET_FIELDS,
    )
    return out


def _max_min_ratio(values) -> float | None:
    values = [
        float(value)
        for value in values
        if value not in (None, "")
    ]
    if not values:
        return None
    minimum = min(values)
    maximum = max(values)
    if minimum == 0:
        return 1.0 if maximum == 0 else float("inf")
    return maximum / minimum


def make_budget_match_audit(
    rows: list[dict],
    output_dir: str | Path,
) -> list[dict]:
    grouped = defaultdict(list)
    for row in rows:
        key = (
            row.get("dataset"),
            row.get("mode"),
            row.get("profile_name"),
            row.get("use_profile_learner"),
            row.get("planning_variant"),
            row.get("llm_policy_variant"),
        )
        grouped[key].append(row)

    out = []
    for key, group in sorted(grouped.items()):
        rag_variants = sorted({
            str(row.get("rag_variant"))
            for row in group
            if row.get("rag_variant") not in (None, "")
        })
        request_caps = {
            str(row.get("max_reactive_api_attempts"))
            for row in group
        }
        planning_request_caps = {
            str(row.get("max_planning_api_attempts"))
            for row in group
        }
        token_caps = {
            str(row.get("max_reactive_tokens"))
            for row in group
        }
        planning_token_caps = {
            str(row.get("max_planning_tokens"))
            for row in group
        }
        ratios = {
            "reactive_attempt_ratio": _max_min_ratio(
                row.get("llm_attempts") for row in group
            ),
            "reactive_token_ratio": _max_min_ratio(
                row.get("reactive_total_tokens") for row in group
            ),
            "planning_attempt_ratio": _max_min_ratio(
                row.get("planning_llm_attempts") for row in group
            ),
            "planning_token_ratio": _max_min_ratio(
                row.get("planning_total_tokens") for row in group
            ),
        }
        equal_request_caps = (
            len(request_caps) == 1
            and len(planning_request_caps) == 1
        )
        equal_token_caps = (
            len(token_caps) == 1
            and len(planning_token_caps) == 1
        )
        is_rag_comparison = len(rag_variants) >= 2
        actual_matched = is_rag_comparison and all(
            value is not None and value <= 1.05
            for value in ratios.values()
        )
        out.append({
            "dataset": key[0],
            "mode": key[1],
            "profile_name": key[2],
            "use_profile_learner": key[3],
            "planning_variant": key[4],
            "llm_policy_variant": key[5],
            "comparison_dimension": (
                "rag_variant" if is_rag_comparison else "not_applicable"
            ),
            "compared_variants": ";".join(rag_variants),
            "num_variants": len(rag_variants),
            "equal_request_caps": equal_request_caps,
            "equal_token_caps": equal_token_caps,
            **ratios,
            "actual_usage_matched_5pct": actual_matched,
        })
    write_csv(
        Path(output_dir) / "budget_match_audit.csv",
        out,
        fieldnames=BUDGET_AUDIT_FIELDS,
    )
    return out
