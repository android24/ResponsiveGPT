import argparse
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .io_utils import ensure_dir, load_json, write_csv


FIGURE_ROLES = {
    "auto",
    "main_matrix",
    "planning_ablation",
    "dense_sparse_calibration",
    "mode_comparison",
    "memory_budget_ablation",
    "profile_adaptation_budget",
    "profile_learning_ablation",
    "rag_budget_matched",
    "final_showcase",
    "generic",
}


def _read_csv(path: str | Path) -> list[dict]:
    path = Path(path)
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as stream:
        return [dict(row) for row in csv.DictReader(stream)]


def _to_float(value):
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _mean(values) -> float | None:
    clean = [
        value
        for value in (_to_float(item) for item in values)
        if value is not None
    ]
    if not clean:
        return None
    return sum(clean) / len(clean)


def _truthy(value) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def _safe_name(value) -> str:
    text = str(value or "unknown").strip()
    out = []
    for char in text:
        out.append(char if char.isalnum() or char in {"-", "_"} else "_")
    return "".join(out).strip("_") or "unknown"


def _experiment_name(experiment_dir: Path) -> str:
    identity_path = experiment_dir / "experiment_identity.json"
    if identity_path.exists():
        try:
            return str(load_json(identity_path).get("experiment_name") or "")
        except Exception:
            return experiment_dir.name
    snapshot_path = experiment_dir / "config.snapshot.json"
    if snapshot_path.exists():
        try:
            return str(load_json(snapshot_path).get("name") or "")
        except Exception:
            return experiment_dir.name
    return experiment_dir.name


def infer_figure_role(experiment_dir: str | Path, rows: list[dict] | None = None) -> str:
    experiment_dir = Path(experiment_dir)
    name = _experiment_name(experiment_dir).lower()
    if "main_sampled" in name or "main_matrix" in name:
        return "main_matrix"
    if "planning_ablation" in name:
        return "planning_ablation"
    if "dense_sparse" in name or "calibration" in name:
        return "dense_sparse_calibration"
    if "mode_comparison" in name:
        return "mode_comparison"
    if "case_memory_budget" in name or "memory_budget" in name:
        return "memory_budget_ablation"
    if "profile_adaptation_budget" in name or "budget_curve" in name:
        return "profile_adaptation_budget"
    if "profile_learning" in name:
        return "profile_learning_ablation"
    if "rag_budget" in name:
        return "rag_budget_matched"
    if "showcase" in name or "fullframe" in name:
        return "final_showcase"

    rows = rows or []
    planning_variants = {row.get("planning_variant") for row in rows}
    rag_variants = {row.get("rag_variant") for row in rows}
    profiles = {row.get("profile_name") for row in rows}
    llm_variants = {row.get("llm_policy_variant") for row in rows}
    if {"dense_all", "sparse_critical"}.issubset(llm_variants):
        return "dense_sparse_calibration"
    if "planning_off" in planning_variants and len(planning_variants) >= 2:
        return "planning_ablation"
    if {"no_rag", "naive_rag", "full_rag_grounded"}.issubset(rag_variants):
        return "main_matrix" if len(profiles) >= 3 else "rag_budget_matched"
    return "generic"


def _group_mean(rows: list[dict], keys: list[str], metric: str) -> list[dict]:
    grouped = defaultdict(list)
    for row in rows:
        grouped[tuple(row.get(key, "") for key in keys)].append(row)
    out = []
    for key, items in sorted(grouped.items()):
        result = {field: value for field, value in zip(keys, key)}
        result[metric] = _mean(item.get(metric) for item in items)
        result["num_rows"] = len(items)
        out.append(result)
    return out


def _dedupe_rows(rows: list[dict], keys: list[str]) -> list[dict]:
    seen = set()
    out = []
    for row in rows:
        key = tuple(row.get(field, "") for field in keys)
        if key in seen:
            continue
        seen.add(key)
        out.append(row)
    return out


def _metric_rows_from_weighted(rows: list[dict], metric: str) -> list[dict]:
    return [
        {
            **row,
            metric: row.get("weighted_mean"),
            f"{metric}_low": row.get("ci95_low"),
            f"{metric}_high": row.get("ci95_high"),
        }
        for row in rows
        if row.get("metric") == metric
        and _to_float(row.get("weighted_mean")) is not None
    ]


def _plot_grouped_bars_with_ci(
    rows: list[dict],
    *,
    category: str,
    series: str,
    metric: str,
    low_metric: str,
    high_metric: str,
    title: str,
    ylabel: str,
    output_path: Path,
) -> Path | None:
    rows = [row for row in rows if _to_float(row.get(metric)) is not None]
    if not rows:
        return None
    categories = sorted({str(row.get(category, "")) for row in rows})
    series_values = sorted({str(row.get(series, "")) for row in rows})
    if not categories or not series_values:
        return None

    width = 0.8 / max(1, len(series_values))
    x = list(range(len(categories)))
    plt.figure(figsize=(max(7.4, len(categories) * 1.45), 4.9))
    for idx, label in enumerate(series_values):
        values = []
        yerr_low = []
        yerr_high = []
        for cat in categories:
            matched = [
                row for row in rows
                if str(row.get(category, "")) == cat
                and str(row.get(series, "")) == label
            ]
            value = _mean(row.get(metric) for row in matched)
            low = _mean(row.get(low_metric) for row in matched)
            high = _mean(row.get(high_metric) for row in matched)
            value = value if value is not None else 0.0
            values.append(value)
            yerr_low.append(max(0.0, value - low) if low is not None else 0.0)
            yerr_high.append(max(0.0, high - value) if high is not None else 0.0)
        offsets = [
            item + (idx - (len(series_values) - 1) / 2) * width
            for item in x
        ]
        plt.bar(
            offsets,
            values,
            width=width,
            label=label,
            yerr=[yerr_low, yerr_high],
            capsize=3,
            error_kw={"linewidth": 0.9},
        )
    plt.xticks(x, categories, rotation=20, ha="right")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend(fontsize=8)
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=220)
    plt.close()
    return output_path


def _plot_grouped_bars(
    rows: list[dict],
    *,
    category: str,
    series: str,
    metric: str,
    title: str,
    ylabel: str,
    output_path: Path,
) -> Path | None:
    rows = [row for row in rows if _to_float(row.get(metric)) is not None]
    if not rows:
        return None
    categories = sorted({str(row.get(category, "")) for row in rows})
    series_values = sorted({str(row.get(series, "")) for row in rows})
    if not categories or not series_values:
        return None

    width = 0.8 / max(1, len(series_values))
    x = list(range(len(categories)))
    plt.figure(figsize=(max(7, len(categories) * 1.35), 4.8))
    for idx, label in enumerate(series_values):
        values = []
        for cat in categories:
            matched = [
                row for row in rows
                if str(row.get(category, "")) == cat and str(row.get(series, "")) == label
            ]
            values.append(_mean(row.get(metric) for row in matched) or 0.0)
        offsets = [item + (idx - (len(series_values) - 1) / 2) * width for item in x]
        plt.bar(offsets, values, width=width, label=label)
    plt.xticks(x, categories, rotation=20, ha="right")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend(fontsize=8)
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=220)
    plt.close()
    return output_path


def _plot_stacked_metric_bars(
    rows: list[dict],
    *,
    category: str,
    metrics: list[tuple[str, str]],
    title: str,
    ylabel: str,
    output_path: Path,
) -> Path | None:
    if not rows or not metrics:
        return None
    labels = sorted({str(row.get(category, "")) for row in rows})
    if not labels:
        return None
    bottoms = [0.0 for _ in labels]
    plt.figure(figsize=(max(7.2, len(labels) * 1.3), 4.8))
    wrote = False
    for metric, metric_label in metrics:
        values = []
        for label in labels:
            matched = [
                row for row in rows
                if str(row.get(category, "")) == label
            ]
            values.append(_mean(row.get(metric) for row in matched) or 0.0)
        if any(value != 0 for value in values):
            wrote = True
        plt.bar(labels, values, bottom=bottoms, label=metric_label)
        bottoms = [left + value for left, value in zip(bottoms, values)]
    if not wrote:
        plt.close()
        return None
    plt.xticks(rotation=20, ha="right")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend(fontsize=8)
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=220)
    plt.close()
    return output_path


def _plot_effect_forest(
    rows: list[dict],
    *,
    title: str,
    output_path: Path,
    max_rows: int = 24,
) -> Path | None:
    points = [
        row for row in rows
        if _truthy(row.get("inference_valid", "true"))
        and _to_float(row.get("mean_delta")) is not None
        and _to_float(row.get("bootstrap_ci_low")) is not None
        and _to_float(row.get("bootstrap_ci_high")) is not None
    ]
    if not points:
        return None
    points = sorted(
        points,
        key=lambda row: (
            str(row.get("metric", "")),
            str(row.get("dataset", "")),
            str(row.get("treatment_variant", "")),
        ),
    )[:max_rows]
    labels = [
        " / ".join(
            item for item in [
                str(row.get("dataset") or "all"),
                str(row.get("metric") or ""),
                str(row.get("treatment_variant") or row.get("rag_variant") or ""),
            ]
            if item
        )
        for row in points
    ]
    y = list(range(len(points)))
    values = [_to_float(row.get("mean_delta")) or 0.0 for row in points]
    lows = [_to_float(row.get("bootstrap_ci_low")) or value for row, value in zip(points, values)]
    highs = [_to_float(row.get("bootstrap_ci_high")) or value for row, value in zip(points, values)]
    xerr = [
        [max(0.0, value - low) for value, low in zip(values, lows)],
        [max(0.0, high - value) for value, high in zip(values, highs)],
    ]
    plt.figure(figsize=(8.6, max(4.8, 0.34 * len(points) + 1.8)))
    plt.errorbar(values, y, xerr=xerr, fmt="o", capsize=3, linewidth=1.0)
    plt.axvline(0.0, color="black", linewidth=0.8, alpha=0.7)
    plt.yticks(y, labels, fontsize=7)
    plt.xlabel("Treatment - baseline delta")
    plt.title(title)
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=220)
    plt.close()
    return output_path


def _plot_simple_bars(
    rows: list[dict],
    *,
    category: str,
    metric: str,
    title: str,
    ylabel: str,
    output_path: Path,
) -> Path | None:
    grouped = _group_mean(rows, [category], metric)
    grouped = [row for row in grouped if _to_float(row.get(metric)) is not None]
    if not grouped:
        return None
    labels = [str(row.get(category, "")) for row in grouped]
    values = [_to_float(row.get(metric)) or 0.0 for row in grouped]
    plt.figure(figsize=(max(7, len(labels) * 1.3), 4.8))
    plt.bar(labels, values)
    plt.xticks(rotation=20, ha="right")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=220)
    plt.close()
    return output_path


def _plot_planning_mechanism(
    rows: list[dict],
    *,
    output_path: Path,
) -> Path | None:
    metrics = [
        ("planning_call_rate", "planning calls"),
        ("planning_reuse_rate", "reuse"),
        ("avg_planning_precision", "precision"),
        ("avg_underreaction_rate", "underreaction"),
    ]
    rows = [
        row for row in rows
        if row.get("planning_variant")
        and any(_to_float(row.get(metric)) is not None for metric, _ in metrics)
    ]
    if not rows:
        return None
    grouped = defaultdict(lambda: defaultdict(list))
    for row in rows:
        label = str(row.get("planning_variant") or "unknown")
        for metric, _ in metrics:
            value = _to_float(row.get(metric))
            if value is not None:
                grouped[label][metric].append(value)
    labels = sorted(grouped)
    x = list(range(len(labels)))
    width = 0.2
    plt.figure(figsize=(max(8.0, len(labels) * 1.25), 4.9))
    for idx, (metric, metric_label) in enumerate(metrics):
        values = [
            _mean(grouped[label].get(metric, [])) or 0.0
            for label in labels
        ]
        offsets = [item + (idx - 1.5) * width for item in x]
        plt.bar(offsets, values, width=width, label=metric_label)
    plt.xticks(x, labels, rotation=20, ha="right")
    plt.ylabel("Rate")
    plt.title("Fast-Slow Mechanism: Planning Use, Reuse, Consistency, Underreaction")
    plt.legend(fontsize=8)
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=220)
    plt.close()
    return output_path


def _plot_stratum_coverage(
    rows: list[dict],
    *,
    output_path: Path,
) -> Path | None:
    rows = _dedupe_rows(
        [
            row for row in rows
            if _to_float(row.get("population_rows")) is not None
            and _to_float(row.get("sample_rows")) is not None
        ],
        [
            "dataset",
            "risk_stratum",
            "event_type",
            "dataset_risk_label",
            "vru_present",
        ],
    )
    if not rows:
        return None
    grouped = defaultdict(lambda: {"population": 0.0, "sample": 0.0})
    for row in rows:
        key = (str(row.get("dataset") or ""), str(row.get("risk_stratum") or "unknown"))
        grouped[key]["population"] += _to_float(row.get("population_rows")) or 0.0
        grouped[key]["sample"] += _to_float(row.get("sample_rows")) or 0.0
    labels = [
        f"{dataset}:{stratum}"
        for dataset, stratum in sorted(grouped)
    ]
    if not labels:
        return None
    population = [grouped[key]["population"] for key in sorted(grouped)]
    sample = [grouped[key]["sample"] for key in sorted(grouped)]
    x = list(range(len(labels)))
    width = 0.38
    plt.figure(figsize=(max(8.0, len(labels) * 0.72), 5.0))
    plt.bar([item - width / 2 for item in x], population, width=width, label="full pool")
    plt.bar([item + width / 2 for item in x], sample, width=width, label="weighted sample")
    plt.xticks(x, labels, rotation=45, ha="right", fontsize=7)
    plt.ylabel("Episodes")
    plt.title("Coverage: Full Pool Strata Versus Weighted Evaluation Sample")
    plt.legend(fontsize=8)
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=220)
    plt.close()
    return output_path


def _plot_scatter(
    rows: list[dict],
    *,
    x_metric: str,
    y_metric: str,
    label_field: str,
    title: str,
    xlabel: str,
    ylabel: str,
    output_path: Path,
) -> Path | None:
    points = [
        row for row in rows
        if _to_float(row.get(x_metric)) is not None and _to_float(row.get(y_metric)) is not None
    ]
    if not points:
        return None
    plt.figure(figsize=(7.2, 5.2))
    for row in points:
        x_value = _to_float(row.get(x_metric)) or 0.0
        y_value = _to_float(row.get(y_metric)) or 0.0
        label = str(row.get(label_field, ""))
        plt.scatter(x_value, y_value, s=42)
        if label:
            plt.annotate(label, (x_value, y_value), fontsize=7, alpha=0.8)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=220)
    plt.close()
    return output_path


def _plot_line_by_group(
    rows: list[dict],
    *,
    x_field: str,
    y_field: str,
    group_field: str,
    title: str,
    xlabel: str,
    ylabel: str,
    output_path: Path,
) -> Path | None:
    grouped = defaultdict(list)
    for row in rows:
        x_value = _to_float(row.get(x_field))
        y_value = _to_float(row.get(y_field))
        if x_value is None or y_value is None:
            continue
        grouped[str(row.get(group_field, ""))].append((x_value, y_value))
    if not grouped:
        return None
    plt.figure(figsize=(7.6, 4.8))
    for label, points in sorted(grouped.items()):
        points = sorted(points)
        plt.plot(
            [item[0] for item in points],
            [item[1] for item in points],
            marker="o",
            label=label or "all",
        )
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend(fontsize=8)
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=220)
    plt.close()
    return output_path


def _record(manifest: list[dict], role: str, key: str, path: Path | None, source: str) -> None:
    if path is None:
        manifest.append({
            "figure_role": role,
            "figure_key": key,
            "path": "",
            "source": source,
            "status": "skipped_no_data",
        })
        return
    manifest.append({
        "figure_role": role,
        "figure_key": key,
        "path": str(path),
        "source": source,
        "status": "written",
    })


def _build_main_matrix_figures(experiment_dir: Path, out_dir: Path, manifest: list[dict]) -> None:
    rows = _read_csv(experiment_dir / "aggregate_summary.csv")
    weighted = (
        _read_csv(experiment_dir / "paper_primary_weighted_table.csv")
        or _read_csv(experiment_dir / "weighted_metric_summary.csv")
    )
    weighted_strata = _read_csv(experiment_dir / "weighted_stratum_metric_summary.csv")
    weighted_effects = _read_csv(experiment_dir / "weighted_significance_vs_no_rag.csv")
    rag_evidence = _read_csv(experiment_dir / "rag_evidence_summary.csv")
    source_rows = weighted if weighted else rows
    if weighted:
        weighted_f1_rows = [
            {
                "dataset": row.get("dataset"),
                "profile_name": row.get("profile_name"),
                "rag_variant": row.get("rag_variant"),
                "f1": row.get("weighted_mean"),
            }
            for row in weighted
            if row.get("metric") == "f1"
        ]
        f1_rows = weighted_f1_rows or rows
    else:
        f1_rows = rows

    _record(
        manifest,
        "main_matrix",
        "rag_f1_by_dataset",
        _plot_grouped_bars(
            f1_rows,
            category="dataset",
            series="rag_variant",
            metric="f1",
            title="Main Matrix: F1 By Dataset And RAG Variant",
            ylabel="F1",
            output_path=out_dir / "main_matrix_rag_f1_by_dataset.png",
        ),
        "weighted_metric_summary.csv" if weighted else "aggregate_summary.csv",
    )
    _record(
        manifest,
        "main_matrix",
        "profile_f1_by_dataset",
        _plot_grouped_bars(
            f1_rows,
            category="dataset",
            series="profile_name",
            metric="f1",
            title="Main Matrix: F1 By Dataset And Driver Profile",
            ylabel="F1",
            output_path=out_dir / "main_matrix_profile_f1_by_dataset.png",
        ),
        "weighted_metric_summary.csv" if weighted else "aggregate_summary.csv",
    )
    _record(
        manifest,
        "main_matrix",
        "performance_token_pareto",
        _plot_scatter(
            rows,
            x_metric="reactive_total_tokens",
            y_metric="f1",
            label_field="rag_variant",
            title="Main Matrix: Performance Versus Reactive Tokens",
            xlabel="Reactive tokens",
            ylabel="F1",
            output_path=out_dir / "main_matrix_f1_vs_reactive_tokens.png",
        ),
        "aggregate_summary.csv",
    )
    _record(
        manifest,
        "main_matrix",
        "rag_grounding",
        _plot_grouped_bars(
            rag_evidence or rows,
            category="dataset",
            series="rag_variant",
            metric=(
                "grounded_rate"
                if rag_evidence else "grounded_decision_rate"
            ),
            title="Main Matrix: Grounded Decision Rate",
            ylabel="Grounded decision rate",
            output_path=out_dir / "main_matrix_grounded_decision_rate.png",
        ),
        "rag_evidence_summary.csv" if rag_evidence else "aggregate_summary.csv",
    )
    for metric, ylabel in [
        ("underreaction_rate", "Underreaction rate"),
        ("rag_grounded_decision_rate", "Grounded decision rate"),
        ("reaction_delay_frames", "Reaction delay frames"),
    ]:
        metric_rows = _metric_rows_from_weighted(weighted, metric)
        _record(
            manifest,
            "main_matrix",
            f"weighted_{metric}_ci",
            _plot_grouped_bars_with_ci(
                metric_rows,
                category="dataset",
                series="rag_variant",
                metric=metric,
                low_metric=f"{metric}_low",
                high_metric=f"{metric}_high",
                title=f"Design-Weighted {ylabel} With 95% CI",
                ylabel=ylabel,
                output_path=out_dir / f"main_matrix_weighted_{metric}_ci.png",
            ),
            "paper_primary_weighted_table.csv"
            if (experiment_dir / "paper_primary_weighted_table.csv").exists()
            else "weighted_metric_summary.csv",
        )
    _record(
        manifest,
        "main_matrix",
        "weighted_effect_forest",
        _plot_effect_forest(
            [
                row for row in weighted_effects
                if row.get("metric") in {
                    "underreaction_rate",
                    "rag_grounded_decision_rate",
                    "rag_output_invalid_citation_frame_rate",
                    "reaction_success_rate",
                    "missed_intervention_rate",
                    "safety_action_appropriateness",
                }
            ],
            title="Design-Weighted Effects Versus No-RAG Baseline",
            output_path=out_dir / "main_matrix_weighted_effect_forest.png",
        ),
        "weighted_significance_vs_no_rag.csv",
    )
    _record(
        manifest,
        "main_matrix",
        "rag_evidence_composition",
        _plot_stacked_metric_bars(
            rag_evidence,
            category="dataset",
            metrics=[
                ("law_coverage", "law"),
                ("case_coverage", "case"),
                ("scenario_coverage", "scenario"),
            ],
            title="RAG Evidence Composition: Law, Case, Scenario Coverage",
            ylabel="Frame coverage, stacked by evidence type",
            output_path=out_dir / "main_matrix_rag_evidence_composition.png",
        ),
        "rag_evidence_summary.csv",
    )
    _record(
        manifest,
        "main_matrix",
        "stratum_coverage",
        _plot_stratum_coverage(
            weighted_strata,
            output_path=out_dir / "main_matrix_stratum_coverage.png",
        ),
        "weighted_stratum_metric_summary.csv",
    )
    if not source_rows:
        _record(manifest, "main_matrix", "no_source_rows", None, "aggregate_summary.csv")


def _build_planning_ablation_figures(experiment_dir: Path, out_dir: Path, manifest: list[dict]) -> None:
    rows = _read_csv(experiment_dir / "aggregate_summary.csv")
    weighted = (
        _read_csv(experiment_dir / "paper_primary_weighted_table.csv")
        or _read_csv(experiment_dir / "weighted_metric_summary.csv")
    )
    weighted_effects = (
        _read_csv(experiment_dir / "planning_weighted_effects.csv")
        or _read_csv(experiment_dir / "profile_adaptation_budget_significance.csv")
        or _read_csv(experiment_dir / "weighted_significance_vs_no_rag.csv")
        or _read_csv(experiment_dir / "significance_vs_no_rag.csv")
    )
    _record(
        manifest,
        "planning_ablation",
        "planning_f1_by_dataset",
        _plot_grouped_bars(
            rows,
            category="dataset",
            series="planning_variant",
            metric="f1",
            title="Fast-Slow Ablation: F1 By Planning Variant",
            ylabel="F1",
            output_path=out_dir / "planning_ablation_f1_by_dataset.png",
        ),
        "aggregate_summary.csv",
    )
    _record(
        manifest,
        "planning_ablation",
        "planning_underreaction_by_dataset",
        _plot_grouped_bars(
            rows,
            category="dataset",
            series="planning_variant",
            metric="avg_underreaction_rate",
            title="Fast-Slow Ablation: Underreaction By Planning Variant",
            ylabel="Underreaction rate",
            output_path=out_dir / "planning_ablation_underreaction_by_dataset.png",
        ),
        "aggregate_summary.csv",
    )
    _record(
        manifest,
        "planning_ablation",
        "planning_token_cost",
        _plot_simple_bars(
            rows,
            category="planning_variant",
            metric="reactive_total_tokens",
            title="Fast-Slow Ablation: Reactive Token Cost",
            ylabel="Reactive tokens",
            output_path=out_dir / "planning_ablation_reactive_tokens.png",
        ),
        "aggregate_summary.csv",
    )
    _record(
        manifest,
        "planning_ablation",
        "planning_reuse_rate",
        _plot_simple_bars(
            rows,
            category="planning_variant",
            metric="planning_reuse_rate",
            title="Fast-Slow Ablation: Slow-Plan Reuse Rate",
            ylabel="Planning reuse rate",
            output_path=out_dir / "planning_ablation_reuse_rate.png",
        ),
        "aggregate_summary.csv",
    )
    _record(
        manifest,
        "planning_ablation",
        "planning_mechanism",
        _plot_planning_mechanism(
            rows,
            output_path=out_dir / "planning_ablation_fast_slow_mechanism.png",
        ),
        "aggregate_summary.csv",
    )
    for metric, ylabel in [
        ("underreaction_rate", "Underreaction rate"),
        ("planning_miss_rate", "Planning miss rate"),
        ("planning_reactive_consistency", "Planning-reactive consistency"),
    ]:
        metric_rows = _metric_rows_from_weighted(weighted, metric)
        _record(
            manifest,
            "planning_ablation",
            f"weighted_{metric}_ci",
            _plot_grouped_bars_with_ci(
                metric_rows,
                category="dataset",
                series="planning_variant",
                metric=metric,
                low_metric=f"{metric}_low",
                high_metric=f"{metric}_high",
                title=f"Fast-Slow Ablation: Design-Weighted {ylabel}",
                ylabel=ylabel,
                output_path=out_dir / f"planning_ablation_weighted_{metric}_ci.png",
            ),
            "paper_primary_weighted_table.csv"
            if (experiment_dir / "paper_primary_weighted_table.csv").exists()
            else "weighted_metric_summary.csv",
        )
    _record(
        manifest,
        "planning_ablation",
        "planning_effect_forest",
        _plot_effect_forest(
            [
                row for row in weighted_effects
                if row.get("metric") in {
                    "underreaction_rate",
                    "overreaction_rate",
                    "reaction_success_rate",
                    "rag_grounded_decision_rate",
                    "safety_action_appropriateness",
                }
            ],
            title="Fast-Slow Ablation: Planning Effects Versus Planning Off",
            output_path=out_dir / "planning_ablation_effect_forest.png",
        ),
        "planning_weighted_effects.csv",
    )


def _build_dense_sparse_figures(experiment_dir: Path, out_dir: Path, manifest: list[dict]) -> None:
    rows = _read_csv(experiment_dir / "dense_sparse_calibration_summary.csv")
    _record(
        manifest,
        "dense_sparse_calibration",
        "violation_agreement",
        _plot_grouped_bars(
            rows,
            category="dataset",
            series="planning_variant",
            metric="violation_agreement_rate",
            title="Dense-Sparse Calibration: Violation Agreement",
            ylabel="Agreement rate",
            output_path=out_dir / "dense_sparse_violation_agreement.png",
        ),
        "dense_sparse_calibration_summary.csv",
    )
    _record(
        manifest,
        "dense_sparse_calibration",
        "frame_reduction",
        _plot_grouped_bars(
            rows,
            category="dataset",
            series="planning_variant",
            metric="avg_frame_reduction_rate",
            title="Dense-Sparse Calibration: Frame Reduction",
            ylabel="Frame reduction rate",
            output_path=out_dir / "dense_sparse_frame_reduction.png",
        ),
        "dense_sparse_calibration_summary.csv",
    )
    _record(
        manifest,
        "dense_sparse_calibration",
        "alignment_delta",
        _plot_grouped_bars(
            rows,
            category="dataset",
            series="planning_variant",
            metric="avg_abs_alignment_accuracy_delta",
            title="Dense-Sparse Calibration: Alignment Delta",
            ylabel="Average absolute delta",
            output_path=out_dir / "dense_sparse_alignment_delta.png",
        ),
        "dense_sparse_calibration_summary.csv",
    )
    _record(
        manifest,
        "dense_sparse_calibration",
        "tradeoff_frame_reduction_alignment",
        _plot_scatter(
            rows,
            x_metric="avg_frame_reduction_rate",
            y_metric="avg_abs_alignment_accuracy_delta",
            label_field="dataset",
            title="Dense-Sparse Calibration: Cost Reduction Versus Alignment Drift",
            xlabel="Frame reduction rate",
            ylabel="Absolute alignment accuracy delta",
            output_path=out_dir / "dense_sparse_tradeoff_frame_reduction_alignment.png",
        ),
        "dense_sparse_calibration_summary.csv",
    )


def _build_mode_figures(experiment_dir: Path, out_dir: Path, manifest: list[dict]) -> None:
    rows = _read_csv(experiment_dir / "mode_comparison_table.csv")
    source = "mode_comparison_table.csv"
    if not rows:
        rows = _read_csv(experiment_dir / "aggregate_summary.csv")
        source = "aggregate_summary.csv"
    _record(
        manifest,
        "mode_comparison",
        "mode_f1_by_dataset",
        _plot_grouped_bars(
            rows,
            category="dataset",
            series="mode",
            metric="f1",
            title="Batch Versus Episode: F1",
            ylabel="F1",
            output_path=out_dir / "mode_comparison_f1_by_dataset.png",
        ),
        source,
    )
    _record(
        manifest,
        "mode_comparison",
        "mode_underreaction_by_dataset",
        _plot_grouped_bars(
            rows,
            category="dataset",
            series="mode",
            metric="avg_underreaction_rate",
            title="Batch Versus Episode: Underreaction",
            ylabel="Underreaction rate",
            output_path=out_dir / "mode_comparison_underreaction_by_dataset.png",
        ),
        source,
    )


def _build_memory_budget_figures(experiment_dir: Path, out_dir: Path, manifest: list[dict]) -> None:
    rows = _read_csv(experiment_dir / "aggregate_summary.csv")
    effect_rows = _read_csv(experiment_dir / "memory_budget_weighted_effects.csv")
    effect_metrics = {
        "underreaction_rate",
        "reaction_success_rate",
        "missed_intervention_rate",
        "safety_action_appropriateness",
        "offline_profile_utility",
        "rag_grounded_decision_rate",
    }
    effect_rows = [
        row for row in effect_rows
        if row.get("metric") in effect_metrics
    ]
    _record(
        manifest,
        "memory_budget_ablation",
        "variant_f1",
        _plot_simple_bars(
            rows,
            category="llm_policy_variant",
            metric="f1",
            title="Case Memory And Budget Governor: F1",
            ylabel="F1",
            output_path=out_dir / "memory_budget_f1.png",
        ),
        "aggregate_summary.csv",
    )
    _record(
        manifest,
        "memory_budget_ablation",
        "variant_llm_attempts",
        _plot_simple_bars(
            rows,
            category="llm_policy_variant",
            metric="llm_attempts",
            title="Case Memory And Budget Governor: LLM Attempts",
            ylabel="LLM attempts",
            output_path=out_dir / "memory_budget_llm_attempts.png",
        ),
        "aggregate_summary.csv",
    )
    _record(
        manifest,
        "memory_budget_ablation",
        "variant_tokens",
        _plot_simple_bars(
            rows,
            category="llm_policy_variant",
            metric="reactive_total_tokens",
            title="Case Memory And Budget Governor: Reactive Tokens",
            ylabel="Reactive tokens",
            output_path=out_dir / "memory_budget_reactive_tokens.png",
        ),
        "aggregate_summary.csv",
    )
    _record(
        manifest,
        "memory_budget_ablation",
        "quality_cost_tradeoff",
        _plot_scatter(
            rows,
            x_metric="reactive_total_tokens",
            y_metric="f1",
            label_field="llm_policy_variant",
            title="Case Memory And Budget Governor: Performance Versus Tokens",
            xlabel="Reactive tokens",
            ylabel="F1",
            output_path=out_dir / "memory_budget_f1_vs_reactive_tokens.png",
        ),
        "aggregate_summary.csv",
    )
    _record(
        manifest,
        "memory_budget_ablation",
        "memory_budget_effect_forest",
        _plot_effect_forest(
            effect_rows,
            title="Case Memory And Budget Governor: Weighted Effects",
            output_path=out_dir / "memory_budget_effect_forest.png",
            max_rows=30,
        ),
        "memory_budget_weighted_effects.csv",
    )


def _build_profile_budget_figures(experiment_dir: Path, out_dir: Path, manifest: list[dict]) -> None:
    rows = _read_csv(experiment_dir / "profile_adaptation_budget_curve.csv")
    if not rows:
        rows = _read_csv(experiment_dir / "aggregate_summary.csv")
    _record(
        manifest,
        "profile_adaptation_budget",
        "budget_underreaction_curve",
        _plot_line_by_group(
            rows,
            x_field="profile_adaptation_episodes",
            y_field="avg_underreaction_rate",
            group_field="dataset",
            title="Profile Adaptation Budget: Underreaction",
            xlabel="Adaptation episodes",
            ylabel="Underreaction rate",
            output_path=out_dir / "profile_budget_underreaction_curve.png",
        ),
        "profile_adaptation_budget_curve.csv",
    )
    _record(
        manifest,
        "profile_adaptation_budget",
        "budget_delta_curve",
        _plot_line_by_group(
            rows,
            x_field="profile_adaptation_episodes",
            y_field="profile_parameter_delta_l1",
            group_field="dataset",
            title="Profile Adaptation Budget: Profile Movement",
            xlabel="Adaptation episodes",
            ylabel="Profile parameter delta L1",
            output_path=out_dir / "profile_budget_delta_curve.png",
        ),
        "profile_adaptation_budget_curve.csv",
    )


def _build_rag_budget_figures(experiment_dir: Path, out_dir: Path, manifest: list[dict]) -> None:
    rows = _read_csv(experiment_dir / "aggregate_summary.csv")
    budget = _read_csv(experiment_dir / "budget_match_audit.csv")
    _record(
        manifest,
        "rag_budget_matched",
        "rag_f1_by_dataset",
        _plot_grouped_bars(
            rows,
            category="dataset",
            series="rag_variant",
            metric="f1",
            title="Budget-Matched RAG: F1",
            ylabel="F1",
            output_path=out_dir / "rag_budget_f1_by_dataset.png",
        ),
        "aggregate_summary.csv",
    )
    _record(
        manifest,
        "rag_budget_matched",
        "reactive_token_ratio",
        _plot_simple_bars(
            budget,
            category="dataset",
            metric="reactive_token_ratio",
            title="Budget-Matched RAG: Reactive Token Ratio",
            ylabel="Max/min token ratio",
            output_path=out_dir / "rag_budget_reactive_token_ratio.png",
        ),
        "budget_match_audit.csv",
    )


def _build_profile_learning_figures(experiment_dir: Path, out_dir: Path, manifest: list[dict]) -> None:
    rows = _read_csv(experiment_dir / "profile_learning_ablation_table.csv")
    if not rows:
        rows = _read_csv(experiment_dir / "aggregate_summary.csv")
    _record(
        manifest,
        "profile_learning_ablation",
        "learner_f1_by_dataset",
        _plot_grouped_bars(
            rows,
            category="dataset",
            series="use_profile_learner",
            metric="f1",
            title="Profile Learning Ablation: F1",
            ylabel="F1",
            output_path=out_dir / "profile_learning_f1_by_dataset.png",
        ),
        "profile_learning_ablation_table.csv",
    )
    _record(
        manifest,
        "profile_learning_ablation",
        "profile_delta",
        _plot_simple_bars(
            rows,
            category="profile_name",
            metric="profile_parameter_delta_l1",
            title="Profile Learning Ablation: Profile Parameter Movement",
            ylabel="Profile parameter delta L1",
            output_path=out_dir / "profile_learning_delta_l1.png",
        ),
        "profile_learning_ablation_table.csv",
    )


def _build_generic_figures(experiment_dir: Path, out_dir: Path, manifest: list[dict], role: str) -> None:
    rows = _read_csv(experiment_dir / "aggregate_summary.csv")
    _record(
        manifest,
        role,
        "f1_by_dataset",
        _plot_simple_bars(
            rows,
            category="dataset",
            metric="f1",
            title="Experiment: F1 By Dataset",
            ylabel="F1",
            output_path=out_dir / f"{_safe_name(role)}_f1_by_dataset.png",
        ),
        "aggregate_summary.csv",
    )
    _record(
        manifest,
        role,
        "reactive_tokens_by_dataset",
        _plot_simple_bars(
            rows,
            category="dataset",
            metric="reactive_total_tokens",
            title="Experiment: Reactive Tokens By Dataset",
            ylabel="Reactive tokens",
            output_path=out_dir / f"{_safe_name(role)}_reactive_tokens_by_dataset.png",
        ),
        "aggregate_summary.csv",
    )


def build_paper_figures(
    experiment_dir: str | Path,
    *,
    figure_role: str = "auto",
    output_dir: str | Path | None = None,
) -> list[dict]:
    if figure_role not in FIGURE_ROLES:
        raise ValueError(f"Unsupported figure_role={figure_role}")
    experiment_dir = Path(experiment_dir)
    rows = _read_csv(experiment_dir / "aggregate_summary.csv")
    role = infer_figure_role(experiment_dir, rows) if figure_role == "auto" else figure_role
    out_root = Path(output_dir) if output_dir else experiment_dir / "paper_figures"
    out_dir = ensure_dir(out_root / role)
    manifest: list[dict] = []

    if role == "main_matrix":
        _build_main_matrix_figures(experiment_dir, out_dir, manifest)
    elif role == "planning_ablation":
        _build_planning_ablation_figures(experiment_dir, out_dir, manifest)
    elif role == "dense_sparse_calibration":
        _build_dense_sparse_figures(experiment_dir, out_dir, manifest)
    elif role == "mode_comparison":
        _build_mode_figures(experiment_dir, out_dir, manifest)
    elif role == "memory_budget_ablation":
        _build_memory_budget_figures(experiment_dir, out_dir, manifest)
    elif role == "profile_adaptation_budget":
        _build_profile_budget_figures(experiment_dir, out_dir, manifest)
    elif role == "profile_learning_ablation":
        _build_profile_learning_figures(experiment_dir, out_dir, manifest)
    elif role == "rag_budget_matched":
        _build_rag_budget_figures(experiment_dir, out_dir, manifest)
    elif role == "final_showcase":
        _build_generic_figures(experiment_dir, out_dir, manifest, role)
    else:
        _build_generic_figures(experiment_dir, out_dir, manifest, role)

    manifest_path = out_root / "paper_figures_manifest.csv"
    existing = _read_csv(manifest_path)
    keep = [
        row for row in existing
        if row.get("figure_role") != role
    ]
    write_csv(
        manifest_path,
        keep + manifest,
        fieldnames=["figure_role", "figure_key", "path", "source", "status"],
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate paper-facing figures from ResponsiveGPT experiment tables."
    )
    parser.add_argument("--experiment_dir", required=True)
    parser.add_argument(
        "--figure_role",
        default="auto",
        choices=sorted(FIGURE_ROLES),
    )
    parser.add_argument("--output_dir", default="")
    args = parser.parse_args()
    manifest = build_paper_figures(
        args.experiment_dir,
        figure_role=args.figure_role,
        output_dir=args.output_dir or None,
    )
    written = [row for row in manifest if row.get("status") == "written"]
    print(
        f"Generated {len(written)}/{len(manifest)} paper figures "
        f"for role={args.figure_role}."
    )


if __name__ == "__main__":
    main()
