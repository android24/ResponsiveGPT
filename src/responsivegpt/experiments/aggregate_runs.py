import argparse
import csv
from pathlib import Path

from .io_utils import load_json, read_jsonl, write_csv
from .make_paper_tables import (
    make_cross_dataset_table,
    make_mode_comparison_table,
    make_profile_adaptation_budget_curve,
    make_profile_adaptation_table,
    make_rag_ablation_table,
    make_profile_learning_ablation_table,
    make_budget_match_audit,
    make_weighted_primary_table,
)
from .rag_evidence_audit import make_rag_evidence_tables
from .report_writer import write_report
from .statistical_tests import (
    make_profile_adaptation_budget_significance,
    make_significance_tables,
    make_profile_learning_significance_tables,
    make_weighted_significance_tables,
)
from .validate_runs import (
    matrix_completion_status,
    latest_usable_completed_statuses,
    validate_experiment_dir,
    validate_run_dir,
)
from .weighted_estimator import (
    _decision_episode_metrics,
    build_weighted_estimates,
    clear_weighted_outputs,
)
from .analysis_provenance import write_analysis_provenance


def _read_csv_rows(path: str | Path) -> list[dict]:
    path = Path(path)
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as stream:
        return [dict(row) for row in csv.DictReader(stream)]


AGGREGATE_FIELDS = [
    "job_id",
    "valid",
    "execution_valid",
    "quality_gate_pass",
    "quality_failures",
    "quality_observations",
    "run_dir",
    "dataset",
    "mode",
    "profile_name",
    "use_profile_learner",
    "profile_protocol_enabled",
    "profile_protocol_pilot_limited",
    "profile_formal_inference_eligible",
    "profile_adaptation_episodes",
    "profile_adaptation_pool_episodes",
    "profile_adaptation_allocation",
    "adaptation_census_path",
    "rag_variant",
    "rag_mode",
    "use_retriever",
    "require_grounded_decision",
    "planning_variant",
    "use_planning_thread",
    "llm_policy_variant",
    "llm_policy",
    "limit",
    "start_index",
    "end_index",
    "shard_id",
    "num_shards",
    "frame_selection",
    "critical_top_k",
    "candidate_frames",
    "selected_frames",
    "overall_total_events",
    "overall_total_frames",
    "total_events",
    "total_frames",
    "reactive_frames",
    "overall_llm_calls",
    "llm_calls",
    "llm_cache_hits",
    "llm_cache_misses",
    "llm_cache_hit_rate",
    "llm_attempts",
    "non_llm_frames",
    "planning_calls",
    "planning_cache_hits",
    "planning_cache_misses",
    "planning_cache_hit_rate",
    "overall_planning_calls",
    "planning_llm_attempts",
    "planning_reuse_frames",
    "planning_reuse_rate",
    "phase_transition_count",
    "risk_phase_transition_rate",
    "rag_cache_hits",
    "rag_cache_misses",
    "rag_cache_hit_rate",
    "llm_error_count",
    "timeout_count",
    "connection_error_count",
    "rate_limit_count",
    "fallback_frame_count",
    "fallback_frame_rate",
    "llm_error_cooldown_frames",
    "llm_error_cooldown_activations",
    "llm_error_cooldown_skipped_frames",
    "rag_evidence_debounce_frames",
    "rag_evidence_change_raw_count",
    "rag_evidence_change_confirmed_count",
    "rag_evidence_change_debounced_frames",
    "grounding_refresh_debounce_frames",
    "grounding_refresh_raw_count",
    "grounding_refresh_confirmed_count",
    "grounding_refresh_debounced_frames",
    "max_llm_calls",
    "max_planning_calls",
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
    "llm_budget_exhausted",
    "planning_budget_exhausted",
    "llm_budget_exhausted_frames",
    "planning_budget_exhausted_frames",
    "precision",
    "recall",
    "f1",
    "accuracy",
    "avg_underreaction_rate",
    "avg_overreaction_rate",
    "avg_reaction_success_rate",
    "avg_reaction_delay_frames",
    "avg_decision_flip_rate",
    "avg_planning_precision",
    "avg_decision_intervention_rate",
    "avg_unnecessary_intervention_rate",
    "avg_missed_intervention_rate",
    "avg_offline_profile_utility",
    "profile_changed_parameter_count",
    "profile_parameter_delta_l1",
    "llm_call_rate",
    "reactive_llm_call_rate",
    "overall_reactive_llm_call_rate",
    "reactive_llm_attempt_rate",
    "evaluation_reactive_llm_attempt_rate",
    "planning_call_rate",
    "evaluation_planning_call_rate",
    "non_llm_frame_rate",
    "reactive_total_tokens",
    "planning_total_tokens",
    "reactive_latency_ms_p50",
    "reactive_latency_ms_p95",
    "planning_latency_ms_p50",
    "planning_latency_ms_p95",
    "adaptation_frames",
    "adaptation_reactive_attempts",
    "adaptation_reactive_total_tokens",
    "adaptation_reactive_latency_ms_p95",
    "adaptation_planning_attempts",
    "adaptation_planning_total_tokens",
    "adaptation_rag_calls",
    "adaptation_rag_latency_ms_p95",
    "evaluation_frames",
    "evaluation_reactive_attempts",
    "evaluation_reactive_total_tokens",
    "evaluation_reactive_latency_ms_p95",
    "evaluation_planning_attempts",
    "evaluation_planning_total_tokens",
    "evaluation_rag_calls",
    "evaluation_rag_latency_ms_p95",
    "retrieval_coverage",
    "evidence_usage_rate",
    "grounded_decision_rate",
    "hallucinated_citation_rate",
    "raw_invalid_citation_attempt_rate",
    "output_invalid_citation_frame_rate",
    "citation_precision",
    "avg_evidence_per_frame",
]


def _get_nested(summary: dict, *path: str):
    obj = summary
    for key in path:
        if not isinstance(obj, dict):
            return None
        obj = obj.get(key)
    return obj


def _mean(values):
    values = [
        float(value)
        for value in values
        if value not in (None, "")
    ]
    return sum(values) / len(values) if values else None


def _is_formal_performance_row(row: dict) -> bool:
    enabled = str(
        row.get("profile_protocol_enabled", "")
    ).strip().lower() in {"1", "true"}
    eligible = str(
        row.get("profile_formal_inference_eligible", "")
    ).strip().lower()
    return not (enabled and eligible in {"0", "false"})


def _row_from_status(status: dict) -> dict | None:
    run_dir = status.get("run_dir")
    if not run_dir:
        return None

    summary_path = Path(run_dir) / "summary.json"
    if not summary_path.exists():
        return None

    summary = load_json(summary_path)
    job = status.get("job", {}) or {}
    extra_args = job.get("extra_args", {}) or {}
    validation = validate_run_dir(run_dir, job)
    execution_valid = bool(
        validation.get("execution_valid", validation.get("valid", False))
    )
    if not execution_valid:
        return None
    decision_metrics = _decision_episode_metrics(Path(run_dir))

    return {
        "job_id": status.get("job_id"),
        "valid": execution_valid,
        "execution_valid": execution_valid,
        "quality_gate_pass": validation.get("quality_gate_pass"),
        "quality_failures": "; ".join(validation.get("quality_failures", [])),
        "quality_observations": "; ".join(
            validation.get("quality_observations", [])
        ),
        "run_dir": run_dir,
        "dataset": job.get("dataset", summary.get("dataset")),
        "mode": job.get("mode", summary.get("mode", "")),
        "profile_name": job.get("profile_name", summary.get("profile_name")),
        "use_profile_learner": extra_args.get(
            "use_profile_learner",
            (summary.get("ablation") or {}).get(
                "use_profile_learner", ""
            ),
        ),
        "profile_protocol_enabled": extra_args.get(
            "profile_protocol_enabled",
            _get_nested(summary, "profile_protocol", "enabled"),
        ),
        "profile_protocol_pilot_limited": _get_nested(
            summary, "profile_protocol", "pilot_limited"
        ),
        "profile_formal_inference_eligible": _get_nested(
            summary,
            "profile_protocol",
            "formal_inference_eligible",
        ),
        "profile_adaptation_episodes": extra_args.get(
            "profile_adaptation_episodes",
            _get_nested(
                summary,
                "profile_protocol",
                "adaptation_episodes_requested",
            ),
        ),
        "profile_adaptation_pool_episodes": extra_args.get(
            "profile_adaptation_pool_episodes",
            _get_nested(
                summary,
                "profile_protocol",
                "adaptation_pool_episodes_requested",
            ),
        ),
        "profile_adaptation_allocation": extra_args.get(
            "profile_adaptation_allocation",
            _get_nested(
                summary,
                "profile_protocol",
                "adaptation_allocation",
            ),
        ),
        "adaptation_census_path": _get_nested(
            summary, "profile_protocol", "adaptation_census_path"
        ),
        "rag_variant": job.get("rag_variant", ""),
        "rag_mode": job.get("rag_mode", ""),
        "use_retriever": job.get("use_retriever", ""),
        "require_grounded_decision": job.get("require_grounded_decision", ""),
        "planning_variant": job.get("planning_variant", ""),
        "use_planning_thread": job.get("use_planning_thread", ""),
        "llm_policy_variant": job.get("llm_policy_variant", ""),
        "llm_policy": job.get("llm_policy", ""),
        "limit": job.get("limit", ""),
        "start_index": extra_args.get("start_index", summary.get("row_start_index", "")),
        "end_index": extra_args.get("end_index", summary.get("row_end_index", "")),
        "shard_id": extra_args.get("shard_id", summary.get("shard_id", "")),
        "num_shards": extra_args.get("num_shards", summary.get("num_shards", "")),
        "frame_selection": extra_args.get("frame_selection", summary.get("frame_selection", "")),
        "critical_top_k": extra_args.get("critical_top_k", summary.get("critical_top_k", "")),
        "candidate_frames": summary.get("candidate_frames", ""),
        "selected_frames": summary.get("selected_frames", ""),
        "overall_total_events": summary.get("total_events"),
        "overall_total_frames": summary.get("total_frames"),
        "total_events": (
            _get_nested(summary, "profile_protocol", "evaluation_events")
            if _get_nested(summary, "profile_protocol", "enabled")
            else summary.get("total_events")
        ),
        "total_frames": (
            _get_nested(summary, "profile_protocol", "evaluation_frames")
            if _get_nested(summary, "profile_protocol", "enabled")
            else summary.get("total_frames")
        ),
        "reactive_frames": (
            _get_nested(summary, "profile_protocol", "evaluation_frames")
            if _get_nested(summary, "profile_protocol", "enabled")
            else summary.get("reactive_frames")
        ),
        "overall_llm_calls": summary.get("llm_calls"),
        "llm_calls": (
            _get_nested(
                summary, "phase_costs", "evaluation", "reactive_successes"
            )
            if _get_nested(summary, "profile_protocol", "enabled")
            else summary.get("llm_calls")
        ),
        "llm_attempts": summary.get("llm_attempts", summary.get("llm_calls")),
        "llm_cache_hits": summary.get("llm_cache_hits", 0),
        "llm_cache_misses": summary.get("llm_cache_misses", 0),
        "llm_cache_hit_rate": _get_nested(
            summary, "token_time_efficiency", "llm_cache_hit_rate"
        ),
        "non_llm_frames": summary.get("non_llm_frames"),
        "overall_planning_calls": summary.get("planning_calls"),
        "planning_calls": (
            _get_nested(
                summary, "phase_costs", "evaluation", "planning_attempts"
            )
            if _get_nested(summary, "profile_protocol", "enabled")
            else summary.get("planning_calls")
        ),
        "planning_llm_attempts": summary.get(
            "planning_llm_attempts", summary.get("planning_calls")
        ),
        "planning_reuse_frames": summary.get("planning_reuse_frames", 0),
        "planning_reuse_rate": _get_nested(
            summary, "token_time_efficiency", "planning_reuse_rate"
        ),
        "phase_transition_count": summary.get("phase_transition_count", 0),
        "risk_phase_transition_rate": _get_nested(
            summary, "token_time_efficiency", "risk_phase_transition_rate"
        ),
        "planning_cache_hits": summary.get("planning_cache_hits", 0),
        "planning_cache_misses": summary.get("planning_cache_misses", 0),
        "planning_cache_hit_rate": _get_nested(
            summary, "token_time_efficiency", "planning_cache_hit_rate"
        ),
        "rag_cache_hits": summary.get("rag_cache_hits", 0),
        "rag_cache_misses": summary.get("rag_cache_misses", 0),
        "rag_cache_hit_rate": _get_nested(
            summary, "token_time_efficiency", "rag_cache_hit_rate"
        ),
        "llm_error_count": summary.get("llm_error_count", 0),
        "timeout_count": summary.get("timeout_count", 0),
        "connection_error_count": summary.get("connection_error_count", 0),
        "rate_limit_count": summary.get("rate_limit_count", 0),
        "fallback_frame_count": summary.get("fallback_frame_count", 0),
        "fallback_frame_rate": summary.get("fallback_frame_rate", 0.0),
        "llm_error_cooldown_frames": summary.get("llm_error_cooldown_frames", 0),
        "llm_error_cooldown_activations": summary.get("llm_error_cooldown_activations", 0),
        "llm_error_cooldown_skipped_frames": summary.get("llm_error_cooldown_skipped_frames", 0),
        "rag_evidence_debounce_frames": summary.get("rag_evidence_debounce_frames", 0),
        "rag_evidence_change_raw_count": summary.get("rag_evidence_change_raw_count", 0),
        "rag_evidence_change_confirmed_count": summary.get("rag_evidence_change_confirmed_count", 0),
        "rag_evidence_change_debounced_frames": summary.get("rag_evidence_change_debounced_frames", 0),
        "grounding_refresh_debounce_frames": summary.get("grounding_refresh_debounce_frames", 0),
        "grounding_refresh_raw_count": summary.get("grounding_refresh_raw_count", 0),
        "grounding_refresh_confirmed_count": summary.get("grounding_refresh_confirmed_count", 0),
        "grounding_refresh_debounced_frames": summary.get("grounding_refresh_debounced_frames", 0),
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
        "reactive_request_budget_exhausted": summary.get(
            "reactive_request_budget_exhausted", False
        ),
        "reactive_token_budget_exhausted": summary.get(
            "reactive_token_budget_exhausted", False
        ),
        "planning_request_budget_exhausted": summary.get(
            "planning_request_budget_exhausted", False
        ),
        "planning_token_budget_exhausted": summary.get(
            "planning_token_budget_exhausted", False
        ),
        "reactive_token_overshoot": summary.get(
            "reactive_token_overshoot", 0
        ),
        "planning_token_overshoot": summary.get(
            "planning_token_overshoot", 0
        ),
        "llm_budget_exhausted": summary.get("llm_budget_exhausted", False),
        "planning_budget_exhausted": summary.get("planning_budget_exhausted", False),
        "llm_budget_exhausted_frames": summary.get("llm_budget_exhausted_frames", 0),
        "planning_budget_exhausted_frames": summary.get("planning_budget_exhausted_frames", 0),
        "precision": summary.get("precision"),
        "recall": summary.get("recall"),
        "f1": summary.get("f1"),
        "accuracy": summary.get("accuracy"),
        "avg_underreaction_rate": _get_nested(summary, "global_alignment", "avg_underreaction_rate"),
        "avg_overreaction_rate": _get_nested(summary, "global_alignment", "avg_overreaction_rate"),
        "avg_reaction_success_rate": _get_nested(
            summary, "global_behavior", "avg_reaction_success_rate"
        ),
        "avg_reaction_delay_frames": _get_nested(summary, "global_behavior", "avg_reaction_delay_frames"),
        "avg_decision_flip_rate": _get_nested(summary, "global_behavior", "avg_decision_flip_rate"),
        "avg_planning_precision": _get_nested(
            summary, "global_planning", "avg_planning_precision"
        ),
        "avg_decision_intervention_rate": _mean(
            item.get("decision_intervention_rate")
            for item in decision_metrics.values()
        ),
        "avg_unnecessary_intervention_rate": _mean(
            item.get("unnecessary_intervention_rate")
            for item in decision_metrics.values()
        ),
        "avg_missed_intervention_rate": _mean(
            item.get("missed_intervention_rate")
            for item in decision_metrics.values()
        ),
        "avg_offline_profile_utility": _mean(
            item.get("offline_profile_utility")
            for item in decision_metrics.values()
        ),
        "profile_changed_parameter_count": _get_nested(
            summary,
            "global_profile_adaptation",
            "changed_parameter_count",
        ),
        "profile_parameter_delta_l1": _get_nested(
            summary,
            "global_profile_adaptation",
            "parameter_delta_l1",
        ),
        "llm_call_rate": summary.get("llm_call_rate"),
        "reactive_llm_call_rate": (
            _get_nested(
                summary,
                "token_time_efficiency",
                "evaluation_reactive_llm_call_rate",
            )
            if _get_nested(summary, "profile_protocol", "enabled")
            else _get_nested(
                summary,
                "token_time_efficiency",
                "reactive_llm_call_rate",
            )
        ),
        "overall_reactive_llm_call_rate": _get_nested(
            summary, "token_time_efficiency", "reactive_llm_call_rate"
        ),
        "reactive_llm_attempt_rate": _get_nested(
            summary, "token_time_efficiency", "reactive_llm_attempt_rate"
        ),
        "evaluation_reactive_llm_attempt_rate": _get_nested(
            summary,
            "token_time_efficiency",
            "evaluation_reactive_llm_attempt_rate",
        ),
        "planning_call_rate": _get_nested(summary, "token_time_efficiency", "planning_call_rate"),
        "evaluation_planning_call_rate": _get_nested(
            summary,
            "token_time_efficiency",
            "evaluation_planning_call_rate",
        ),
        "non_llm_frame_rate": _get_nested(summary, "token_time_efficiency", "non_llm_frame_rate"),
        "reactive_total_tokens": _get_nested(
            summary, "token_time_efficiency", "reactive_total_tokens"
        ),
        "planning_total_tokens": _get_nested(
            summary, "token_time_efficiency", "planning_total_tokens"
        ),
        "reactive_latency_ms_p50": _get_nested(
            summary, "token_time_efficiency", "reactive_latency_ms_p50"
        ),
        "reactive_latency_ms_p95": _get_nested(
            summary, "token_time_efficiency", "reactive_latency_ms_p95"
        ),
        "planning_latency_ms_p50": _get_nested(
            summary, "token_time_efficiency", "planning_latency_ms_p50"
        ),
        "planning_latency_ms_p95": _get_nested(
            summary, "token_time_efficiency", "planning_latency_ms_p95"
        ),
        "adaptation_frames": _get_nested(
            summary, "phase_costs", "adaptation", "frames"
        ),
        "adaptation_reactive_attempts": _get_nested(
            summary, "phase_costs", "adaptation", "reactive_attempts"
        ),
        "adaptation_reactive_total_tokens": _get_nested(
            summary, "phase_costs", "adaptation", "reactive_total_tokens"
        ),
        "adaptation_reactive_latency_ms_p95": _get_nested(
            summary, "phase_costs", "adaptation", "reactive_latency_ms_p95"
        ),
        "adaptation_planning_attempts": _get_nested(
            summary, "phase_costs", "adaptation", "planning_attempts"
        ),
        "adaptation_planning_total_tokens": _get_nested(
            summary, "phase_costs", "adaptation", "planning_total_tokens"
        ),
        "adaptation_rag_calls": _get_nested(
            summary, "phase_costs", "adaptation", "rag_calls"
        ),
        "adaptation_rag_latency_ms_p95": _get_nested(
            summary, "phase_costs", "adaptation", "rag_latency_ms_p95"
        ),
        "evaluation_frames": _get_nested(
            summary, "phase_costs", "evaluation", "frames"
        ),
        "evaluation_reactive_attempts": _get_nested(
            summary, "phase_costs", "evaluation", "reactive_attempts"
        ),
        "evaluation_reactive_total_tokens": _get_nested(
            summary, "phase_costs", "evaluation", "reactive_total_tokens"
        ),
        "evaluation_reactive_latency_ms_p95": _get_nested(
            summary, "phase_costs", "evaluation", "reactive_latency_ms_p95"
        ),
        "evaluation_planning_attempts": _get_nested(
            summary, "phase_costs", "evaluation", "planning_attempts"
        ),
        "evaluation_planning_total_tokens": _get_nested(
            summary, "phase_costs", "evaluation", "planning_total_tokens"
        ),
        "evaluation_rag_calls": _get_nested(
            summary, "phase_costs", "evaluation", "rag_calls"
        ),
        "evaluation_rag_latency_ms_p95": _get_nested(
            summary, "phase_costs", "evaluation", "rag_latency_ms_p95"
        ),
        "retrieval_coverage": _get_nested(summary, "global_rag", "retrieval_coverage"),
        "evidence_usage_rate": _get_nested(summary, "global_rag", "evidence_usage_rate"),
        "grounded_decision_rate": _get_nested(summary, "global_rag", "grounded_decision_rate"),
        "hallucinated_citation_rate": _get_nested(summary, "global_rag", "hallucinated_citation_rate"),
        "raw_invalid_citation_attempt_rate": _get_nested(
            summary, "global_rag", "raw_invalid_citation_attempt_rate"
        ),
        "output_invalid_citation_frame_rate": _get_nested(
            summary, "global_rag", "output_invalid_citation_frame_rate"
        ),
        "citation_precision": _get_nested(
            summary, "global_rag", "citation_precision"
        ),
        "avg_evidence_per_frame": _get_nested(summary, "global_rag", "avg_evidence_per_frame"),
    }


def aggregate_experiment(experiment_dir: str | Path) -> list[dict]:
    experiment_dir = Path(experiment_dir)
    validate_experiment_dir(experiment_dir)
    completion = matrix_completion_status(experiment_dir)

    usable_statuses = latest_usable_completed_statuses(experiment_dir)
    rows = []
    for status in usable_statuses:
        row = _row_from_status(status)
        if row is not None:
            rows.append(row)
    formal_rows = [
        row for row in rows if _is_formal_performance_row(row)
    ]

    write_csv(experiment_dir / "aggregate_summary.csv", rows, fieldnames=AGGREGATE_FIELDS)
    rag_table = make_rag_ablation_table(formal_rows, experiment_dir)
    cross_dataset_table = make_cross_dataset_table(
        formal_rows, experiment_dir
    )
    profile_table = make_profile_adaptation_table(
        formal_rows, experiment_dir
    )
    profile_learning_table = make_profile_learning_ablation_table(
        formal_rows, experiment_dir
    )
    budget_match_audit = make_budget_match_audit(
        formal_rows, experiment_dir
    )
    mode_table = make_mode_comparison_table(
        formal_rows, experiment_dir
    )
    significance_table = make_significance_tables(
        formal_rows, experiment_dir
    )
    weighted_rows = []
    if completion.get("primary_matrix_ready", False):
        try:
            weighted_rows, _ = build_weighted_estimates(experiment_dir)
        except FileNotFoundError as exc:
            print(f"[INFO] Design-weighted estimates unavailable: {exc}")
    else:
        clear_weighted_outputs(experiment_dir)
        print(
            "[WARN] Primary matrix is not ready; weighted tables and "
            "significance claims are suppressed. "
            f"completion={completion}"
        )
    weighted_primary_table = make_weighted_primary_table(
        weighted_rows, experiment_dir
    )
    profile_adaptation_budget_curve = (
        make_profile_adaptation_budget_curve(
            formal_rows, weighted_rows, experiment_dir
        )
    )
    profile_adaptation_budget_significance = (
        make_profile_adaptation_budget_significance(
            weighted_rows,
            experiment_dir,
            observation_rows=_read_csv_rows(
                experiment_dir / "weighted_episode_observations.csv"
            ),
        )
        if completion.get("primary_matrix_ready", False)
        else make_profile_adaptation_budget_significance(
            [], experiment_dir
        )
    )
    weighted_significance_table = make_weighted_significance_tables(
        weighted_rows,
        experiment_dir,
        observation_csv=(
            experiment_dir / "weighted_episode_observations.csv"
        ),
    )
    profile_learning_significance = (
        make_profile_learning_significance_tables(
            weighted_rows,
            experiment_dir,
            observation_csv=(
                experiment_dir / "weighted_episode_observations.csv"
            ),
        )
    )
    rag_evidence_summary, rag_top_evidence = make_rag_evidence_tables(
        formal_rows, experiment_dir
    )
    write_report(
        experiment_dir,
        rows,
        rag_table,
        cross_dataset_table=cross_dataset_table,
        profile_table=profile_table,
        profile_learning_table=profile_learning_table,
        profile_adaptation_budget_curve=profile_adaptation_budget_curve,
        profile_adaptation_budget_significance=(
            profile_adaptation_budget_significance
        ),
        mode_table=mode_table,
        significance_table=significance_table,
        weighted_primary_table=weighted_primary_table,
        weighted_significance_table=weighted_significance_table,
        rag_evidence_summary=rag_evidence_summary,
        rag_top_evidence=rag_top_evidence,
        matrix_completion=completion,
        budget_match_audit=budget_match_audit,
        profile_learning_significance=profile_learning_significance,
    )
    try:
        from .paper_figure_plotter import build_paper_figures

        build_paper_figures(experiment_dir)
    except Exception as exc:
        print(f"[WARN] Paper figure generation failed: {exc}")
    write_analysis_provenance(experiment_dir, usable_statuses)
    return rows


def main():
    parser = argparse.ArgumentParser(description="Aggregate ResponsiveGPT experiment matrix outputs.")
    parser.add_argument("--experiment_dir", required=True)
    args = parser.parse_args()
    rows = aggregate_experiment(args.experiment_dir)
    print(f"Aggregated {len(rows)} completed runs.")


if __name__ == "__main__":
    main()
