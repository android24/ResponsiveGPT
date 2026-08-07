from pathlib import Path


def _fmt(value):
    if isinstance(value, float):
        return f"{value:.4f}"
    return "" if value is None else str(value)


def write_report(
    output_dir: str | Path,
    aggregate_rows: list[dict],
    rag_table: list[dict],
    *,
    cross_dataset_table: list[dict] | None = None,
    profile_table: list[dict] | None = None,
    profile_learning_table: list[dict] | None = None,
    profile_adaptation_budget_curve: list[dict] | None = None,
    profile_adaptation_budget_significance: list[dict] | None = None,
    mode_table: list[dict] | None = None,
    significance_table: list[dict] | None = None,
    weighted_primary_table: list[dict] | None = None,
    weighted_significance_table: list[dict] | None = None,
    planning_significance_table: list[dict] | None = None,
    memory_budget_significance_table: list[dict] | None = None,
    rag_evidence_summary: list[dict] | None = None,
    rag_top_evidence: list[dict] | None = None,
    matrix_completion: dict | None = None,
    budget_match_audit: list[dict] | None = None,
    profile_learning_significance: list[dict] | None = None,
) -> Path:
    output_dir = Path(output_dir)
    report_path = output_dir / "report.md"

    lines = [
        "# ResponsiveGPT MVP Experiment Report",
        "",
        f"Total completed runs: {len(aggregate_rows)}",
        "",
        (
            "Primary-result status: READY"
            if (matrix_completion or {}).get("primary_matrix_ready")
            else "Primary-result status: INCOMPLETE - no primary claims"
        ),
        "",
        "## Unweighted Descriptive RAG Ablation",
        "",
    ]

    if rag_table:
        headers = [
            "dataset",
            "treatment_cell",
            "rag_variant",
            "runs",
            "events",
            "frames",
            "f1",
            "underreaction",
            "llm_rate",
            "retrieval",
            "grounded",
            "hallucination",
        ]
        lines.append("| " + " | ".join(headers) + " |")
        lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
        for row in rag_table:
            values = [
                row.get("dataset"),
                "/".join(str(value) for value in (
                    row.get("mode"),
                    row.get("profile_name"),
                    row.get("use_profile_learner"),
                    row.get("planning_variant"),
                    row.get("llm_policy_family"),
                    row.get("profile_adaptation_episodes"),
                )),
                row.get("rag_variant"),
                row.get("num_runs"),
                row.get("total_events"),
                row.get("total_frames"),
                row.get("f1"),
                row.get("avg_underreaction_rate"),
                row.get("reactive_llm_call_rate"),
                row.get("retrieval_coverage"),
                row.get("grounded_decision_rate"),
                row.get("hallucinated_citation_rate"),
            ]
            lines.append("| " + " | ".join(_fmt(v) for v in values) + " |")
    else:
        lines.append("No completed runs were available for aggregation.")

    if weighted_primary_table:
        lines.extend([
            "",
            "## Design-Weighted Primary Results",
            "",
            "| dataset | treatment_cell | rag | planning | metric | mean | ci_low | ci_high | coverage | completeness | N/A | censored | policy |",
            "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
        ])
        for row in weighted_primary_table[:36]:
            values = [
                row.get("dataset"),
                "/".join(str(value) for value in (
                    row.get("mode"),
                    row.get("profile_name"),
                    row.get("use_profile_learner"),
                    row.get("profile_adaptation_episodes"),
                    row.get("llm_policy_variant"),
                )),
                row.get("rag_variant"),
                row.get("planning_variant"),
                row.get("metric"),
                row.get("weighted_mean"),
                row.get("ci95_low"),
                row.get("ci95_high"),
                row.get("population_coverage"),
                row.get("metric_completeness"),
                row.get("not_applicable_rows"),
                row.get("censored_rows"),
                row.get("missingness_policy"),
            ]
            lines.append("| " + " | ".join(_fmt(v) for v in values) + " |")

    if cross_dataset_table:
        lines.extend([
            "",
            "## Cross Dataset Table",
            "",
            "| dataset | treatment_cell | rag_variant | runs | f1 | underreaction | grounded | hallucination |",
            "| --- | --- | --- | --- | --- | --- | --- | --- |",
        ])
        for row in cross_dataset_table[:24]:
            values = [
                row.get("dataset"),
                "/".join(str(value) for value in (
                    row.get("mode"),
                    row.get("profile_name"),
                    row.get("use_profile_learner"),
                    row.get("planning_variant"),
                    row.get("llm_policy_family"),
                    row.get("profile_adaptation_episodes"),
                )),
                row.get("rag_variant"),
                row.get("num_runs"),
                row.get("f1"),
                row.get("avg_underreaction_rate"),
                row.get("grounded_decision_rate"),
                row.get("hallucinated_citation_rate"),
            ]
            lines.append("| " + " | ".join(_fmt(v) for v in values) + " |")

    if profile_adaptation_budget_curve:
        lines.extend([
            "",
            "## Profile Adaptation Budget Curve",
            "",
            "| dataset | profile | treatment cell | episodes | runs | underreaction | utility | adaptation_tokens | evaluation_tokens |",
            "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
        ])
        for row in profile_adaptation_budget_curve:
            adaptation_tokens = (
                float(row.get("adaptation_reactive_total_tokens") or 0)
                + float(row.get("adaptation_planning_total_tokens") or 0)
            )
            evaluation_tokens = (
                float(row.get("evaluation_reactive_total_tokens") or 0)
                + float(row.get("evaluation_planning_total_tokens") or 0)
            )
            values = [
                row.get("dataset"),
                row.get("profile_name"),
                "/".join(str(value) for value in (
                    row.get("mode"),
                    row.get("use_profile_learner"),
                    row.get("rag_variant"),
                    row.get("planning_variant"),
                    row.get("llm_policy_family"),
                )),
                row.get("profile_adaptation_episodes"),
                row.get("num_runs"),
                row.get("avg_underreaction_rate"),
                row.get("avg_offline_profile_utility"),
                adaptation_tokens,
                evaluation_tokens,
            ]
            lines.append("| " + " | ".join(_fmt(v) for v in values) + " |")

    if profile_adaptation_budget_significance:
        lines.extend([
            "",
            "## Profile Adaptation Budget Significance",
            "",
            "| dataset | profile | treatment cell | treatment | metric | pairs | delta | ci_low | ci_high | holm_p |",
            "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
        ])
        for row in profile_adaptation_budget_significance:
            values = [
                row.get("dataset"),
                row.get("profile_name"),
                "/".join(str(value) for value in (
                    row.get("mode"),
                    row.get("use_profile_learner"),
                    row.get("rag_variant"),
                    row.get("planning_variant"),
                    row.get("llm_policy_family"),
                )),
                row.get("treatment_variant"),
                row.get("metric"),
                row.get("num_pairs"),
                row.get("mean_delta"),
                row.get("bootstrap_ci_low"),
                row.get("bootstrap_ci_high"),
                row.get("wilcoxon_p_holm"),
            ]
            lines.append("| " + " | ".join(_fmt(v) for v in values) + " |")

    if mode_table:
        lines.extend([
            "",
            "## Batch Vs Episode Mode",
            "",
            "| dataset | mode | treatment_cell | rag | runs | events | selected_frames | f1 | underreaction | llm_rate | grounded |",
            "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
        ])
        for row in mode_table[:24]:
            values = [
                row.get("dataset"),
                row.get("mode"),
                "/".join(str(value) for value in (
                    row.get("profile_name"),
                    row.get("use_profile_learner"),
                    row.get("planning_variant"),
                    row.get("llm_policy_family"),
                    row.get("profile_adaptation_episodes"),
                )),
                row.get("rag_variant"),
                row.get("num_runs"),
                row.get("total_events"),
                row.get("selected_frames"),
                row.get("f1"),
                row.get("avg_underreaction_rate"),
                row.get("reactive_llm_call_rate"),
                row.get("grounded_decision_rate"),
            ]
            lines.append("| " + " | ".join(_fmt(v) for v in values) + " |")

    if profile_learning_table:
        lines.extend([
            "",
            "## Profile Learner Ablation",
            "",
            "| dataset | treatment_cell | learner | runs | underreaction | overreaction | delta_l1 | changed_params |",
            "| --- | --- | --- | --- | --- | --- | --- | --- |",
        ])
        for row in profile_learning_table[:24]:
            values = [
                row.get("dataset"),
                "/".join(str(value) for value in (
                    row.get("mode"),
                    row.get("profile_name"),
                    row.get("rag_variant"),
                    row.get("planning_variant"),
                    row.get("llm_policy_family"),
                    row.get("profile_adaptation_episodes"),
                )),
                row.get("use_profile_learner"),
                row.get("num_runs"),
                row.get("avg_underreaction_rate"),
                row.get("avg_overreaction_rate"),
                row.get("profile_parameter_delta_l1"),
                row.get("profile_changed_parameter_count"),
            ]
            lines.append("| " + " | ".join(_fmt(v) for v in values) + " |")

    if budget_match_audit:
        passed = sum(
            bool(row.get("actual_usage_matched_5pct"))
            for row in budget_match_audit
        )
        lines.extend([
            "",
            "## Budget Matching Audit",
            "",
            (
                f"Actual usage matched within 5% for {passed}/"
                f"{len(budget_match_audit)} comparison cells."
            ),
            "",
        ])

    if profile_learning_significance:
        lines.extend([
            "",
            "## Design-Weighted Profile Learning Effects",
            "",
            "| metric | inference | valid | seeds | pairs | coverage | mean_delta | effect | ci_low | ci_high | holm_p |",
            "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
        ])
        for row in profile_learning_significance:
            values = [
                row.get("metric"),
                row.get("primary_inference"),
                row.get("inference_valid"),
                row.get("num_clusters"),
                row.get("num_pairs"),
                row.get("paired_weight_coverage"),
                row.get("mean_delta"),
                row.get("paired_standardized_effect"),
                row.get("bootstrap_ci_low"),
                row.get("bootstrap_ci_high"),
                row.get("wilcoxon_p_holm"),
            ]
            lines.append("| " + " | ".join(_fmt(v) for v in values) + " |")

    if significance_table:
        lines.extend([
            "",
            "## Unweighted Exploratory Significance Vs No RAG",
            "",
            "| treatment | metric | pairs | mean_delta | ci_low | ci_high | sign_p | wilcoxon_p |",
            "| --- | --- | --- | --- | --- | --- | --- | --- |",
        ])
        for row in significance_table[:24]:
            values = [
                row.get("treatment_variant"),
                row.get("metric"),
                row.get("num_pairs"),
                row.get("mean_delta"),
                row.get("bootstrap_ci_low"),
                row.get("bootstrap_ci_high"),
                row.get("sign_test_p"),
                row.get("wilcoxon_p"),
            ]
            lines.append("| " + " | ".join(_fmt(v) for v in values) + " |")

    if weighted_significance_table:
        lines.extend([
            "",
            "## Design-Weighted Inference Vs No RAG",
            "",
            "| treatment | metric | valid | matched/expected | weight_coverage | mean_delta | ci_low | ci_high | auxiliary_sign_p | auxiliary_wilcoxon_p |",
            "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
        ])
        for row in weighted_significance_table[:24]:
            values = [
                row.get("treatment_variant"),
                row.get("metric"),
                row.get("inference_valid"),
                (
                    f"{row.get('matched_clusters')}/"
                    f"{row.get('expected_clusters')}"
                ),
                row.get("paired_weight_coverage"),
                row.get("mean_delta"),
                row.get("bootstrap_ci_low"),
                row.get("bootstrap_ci_high"),
                row.get("sign_test_p"),
                row.get("wilcoxon_p"),
            ]
            lines.append("| " + " | ".join(_fmt(v) for v in values) + " |")

    if planning_significance_table:
        lines.extend([
            "",
            "## Fast-Slow Planning Inference Vs Planning Off",
            "",
            "| treatment | metric | valid | matched/expected | weight_coverage | mean_delta | ci_low | ci_high | auxiliary_sign_p | auxiliary_wilcoxon_p |",
            "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
        ])
        for row in planning_significance_table[:36]:
            values = [
                row.get("treatment_variant"),
                row.get("metric"),
                row.get("inference_valid"),
                (
                    f"{row.get('matched_clusters')}/"
                    f"{row.get('expected_clusters')}"
                ),
                row.get("paired_weight_coverage"),
                row.get("mean_delta"),
                row.get("bootstrap_ci_low"),
                row.get("bootstrap_ci_high"),
                row.get("sign_test_p"),
                row.get("wilcoxon_p"),
            ]
            lines.append("| " + " | ".join(_fmt(v) for v in values) + " |")

    if memory_budget_significance_table:
        lines.extend([
            "",
            "## Case Memory And Budget Governor Inference Vs Baseline",
            "",
            "| treatment | metric | valid | matched/expected | weight_coverage | mean_delta | ci_low | ci_high | auxiliary_sign_p | auxiliary_wilcoxon_p |",
            "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
        ])
        for row in memory_budget_significance_table[:36]:
            values = [
                row.get("treatment_variant"),
                row.get("metric"),
                row.get("inference_valid"),
                (
                    f"{row.get('matched_clusters')}/"
                    f"{row.get('expected_clusters')}"
                ),
                row.get("paired_weight_coverage"),
                row.get("mean_delta"),
                row.get("bootstrap_ci_low"),
                row.get("bootstrap_ci_high"),
                row.get("sign_test_p"),
                row.get("wilcoxon_p"),
            ]
            lines.append("| " + " | ".join(_fmt(v) for v in values) + " |")

    if rag_evidence_summary:
        lines.extend([
            "",
            "## RAG Evidence Audit",
            "",
            "| dataset | rag_variant | frames | law | case | scenario | core | grounded |",
            "| --- | --- | --- | --- | --- | --- | --- | --- |",
        ])
        for row in rag_evidence_summary[:24]:
            values = [
                row.get("dataset"),
                row.get("rag_variant"),
                row.get("frames"),
                row.get("law_coverage"),
                row.get("case_coverage"),
                row.get("scenario_coverage"),
                row.get("core_law_case_scenario_coverage"),
                row.get("grounded_rate"),
            ]
            lines.append("| " + " | ".join(_fmt(v) for v in values) + " |")

    lines.extend([
        "",
        "## Output Files",
        "",
        "- `aggregate_summary.csv`",
        "- `rag_ablation_table.csv`",
        "- `cross_dataset_table.csv`",
        "- `profile_adaptation_table.csv`",
        "- `profile_learning_ablation_table.csv`",
        "- `profile_learning_weighted_effects.csv`",
        "- `budget_match_audit.csv`",
        "- `mode_comparison_table.csv`",
        "- `significance_vs_no_rag.csv`",
        "- `paper_primary_weighted_table.csv`",
        "- `weighted_significance_vs_no_rag.csv`",
        "- `planning_weighted_effects.csv`",
        "- `memory_budget_weighted_effects.csv`",
        "- `weighted_metric_summary.csv`",
        "- `weighted_stratum_metric_summary.csv`",
        "- `rag_evidence_summary.csv`",
        "- `rag_top_evidence.csv`",
        "- `rag_evidence_examples.md`",
        "- `validation_summary.csv`",
        "",
    ])

    report_path.write_text("\n".join(lines), encoding="utf-8")
    return report_path
