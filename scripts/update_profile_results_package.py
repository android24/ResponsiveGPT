"""Add completed profile-learning evidence to the paper results package."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


PROFILE_FILES = (
    "matrix_completion.json",
    "profile_learning_ablation_table.csv",
    "profile_learning_weighted_effects.csv",
    "profile_adaptation_budget_curve.csv",
    "profile_adaptation_budget_significance.csv",
    "profile_learning_episode_sensitivity.csv",
    "analysis_provenance.json",
)
START = "<!-- PROFILE_LEARNING_UPDATE_START -->"
END = "<!-- PROFILE_LEARNING_UPDATE_END -->"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def number(value: str | float, digits: int = 4) -> str:
    if value in (None, ""):
        return ""
    return f"{float(value):.{digits}f}"


def effect_status(row: dict[str, str]) -> str:
    low = float(row["bootstrap_ci_low"])
    high = float(row["bootstrap_ci_high"])
    if row["direction"] == "lower_is_better":
        if high < 0:
            return "improved"
        if low > 0:
            return "degraded"
    else:
        if low > 0:
            return "improved"
        if high < 0:
            return "degraded"
    return "inconclusive"


def effect_rows(profile_dir: Path) -> list[dict]:
    rows = read_csv(profile_dir / "profile_learning_weighted_effects.csv")
    output = []
    for row in rows:
        if row.get("inference_valid") != "True":
            continue
        base = float(row["baseline_mean"])
        delta = float(row["mean_delta"])
        relative = delta / abs(base) if base else None
        output.append({
            "metric": row["metric"],
            "direction": row["direction"],
            "fixed_profile_mean": number(row["baseline_mean"], 6),
            "adaptive_profile_mean": number(row["treatment_mean"], 6),
            "mean_delta": number(row["mean_delta"], 6),
            "relative_delta": number(relative, 4) if relative is not None else "",
            "bootstrap_ci95": (
                f"[{number(row['bootstrap_ci_low'], 6)}, "
                f"{number(row['bootstrap_ci_high'], 6)}]"
            ),
            "improved_seeds": row["improved_pairs"],
            "degraded_seeds": row["degraded_pairs"],
            "ties": row["ties"],
            "holm_sign_p": number(row["sign_test_p_holm"], 6),
            "holm_wilcoxon_p": number(row["wilcoxon_p_holm"], 6),
            "evidence_status": effect_status(row),
            "primary_inference": row["primary_inference"],
            "num_order_seeds": row["num_clusters"],
        })
    return output


CELL_METRICS = (
    "avg_underreaction_rate",
    "avg_overreaction_rate",
    "avg_reaction_success_rate",
    "avg_reaction_delay_frames",
    "avg_decision_flip_rate",
    "avg_missed_intervention_rate",
    "avg_offline_profile_utility",
)


def cell_rows(profile_dir: Path) -> list[dict]:
    source = read_csv(profile_dir / "profile_learning_ablation_table.csv")
    groups: dict[tuple[str, str], dict[str, dict]] = {}
    for row in source:
        groups.setdefault(
            (row["dataset"], row["profile_name"]), {}
        )[row["use_profile_learner"]] = row
    output = []
    for (dataset, profile), pair in sorted(groups.items()):
        fixed, adaptive = pair.get("0"), pair.get("1")
        if fixed is None or adaptive is None:
            continue
        item = {
            "dataset": dataset,
            "profile": profile,
            "runs_per_variant": fixed["num_runs"],
            "estimator": fixed["estimator"],
        }
        for metric in CELL_METRICS:
            base = float(fixed[metric])
            treatment = float(adaptive[metric])
            label = metric.removeprefix("avg_")
            item[f"fixed_{label}"] = number(base, 6)
            item[f"adaptive_{label}"] = number(treatment, 6)
            item[f"delta_{label}"] = number(treatment - base, 6)
        item["changed_parameters"] = number(
            adaptive["profile_changed_parameter_count"], 2
        )
        item["parameter_delta_l1"] = number(
            adaptive["profile_parameter_delta_l1"], 4
        )
        output.append(item)
    return output


def cost_rows(profile_dir: Path) -> list[dict]:
    source = read_csv(profile_dir / "profile_adaptation_budget_curve.csv")
    output = []
    for row in source:
        output.append({
            "dataset": row["dataset"],
            "profile": row["profile_name"],
            "profile_learner": "adaptive" if row["use_profile_learner"] == "1" else "fixed",
            "num_runs": row["num_runs"],
            "adaptation_episodes": row["profile_adaptation_episodes"],
            "adaptation_frames": number(row["adaptation_frames"], 1),
            "adaptation_reactive_attempts": number(row["adaptation_reactive_attempts"], 1),
            "adaptation_reactive_tokens": number(row["adaptation_reactive_total_tokens"], 1),
            "adaptation_planning_attempts": number(row["adaptation_planning_attempts"], 1),
            "adaptation_planning_tokens": number(row["adaptation_planning_total_tokens"], 1),
            "evaluation_reactive_tokens": number(row["evaluation_reactive_total_tokens"], 1),
            "evaluation_planning_tokens": number(row["evaluation_planning_total_tokens"], 1),
            "underreaction_rate": number(row["avg_underreaction_rate"], 6),
            "overreaction_rate": number(row["avg_overreaction_rate"], 6),
            "offline_profile_utility": number(row["avg_offline_profile_utility"], 6),
            "changed_parameters": number(row["profile_changed_parameter_count"], 2),
            "parameter_delta_l1": number(row["profile_parameter_delta_l1"], 4),
            "estimator": row["estimator"],
        })
    return output


def inference_audit_rows(profile_dir: Path) -> list[dict]:
    rows = read_csv(profile_dir / "profile_adaptation_budget_significance.csv")
    grouped: dict[tuple[str, str, str], list[dict]] = {}
    for row in rows:
        grouped.setdefault(
            (row["dataset"], row["profile_name"], row["use_profile_learner"]), []
        ).append(row)
    output = []
    for (dataset, profile, learner), items in sorted(grouped.items()):
        valid = [row for row in items if row["inference_valid"] == "True"]
        output.append({
            "dataset": dataset,
            "profile": profile,
            "profile_learner": "adaptive" if learner == "1" else "fixed",
            "metrics_requested": len(items),
            "metrics_inference_valid": len(valid),
            "matched_clusters": max(int(row["matched_clusters"] or 0) for row in items),
            "inference_status": "valid" if len(valid) == len(items) else "suppressed",
            "reason": items[0]["auxiliary_test_note"],
        })
    return output


def markdown_table(rows: list[dict], fields: list[str]) -> str:
    if not rows:
        return "No rows."
    lines = [
        "| " + " | ".join(fields) + " |",
        "| " + " | ".join("---" for _ in fields) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(row.get(field, "")) for field in fields) + " |")
    return "\n".join(lines)


def report_section(
    effects: list[dict], cells: list[dict], costs: list[dict], completion: dict
) -> str:
    by_metric = {row["metric"]: row for row in effects}
    selected = [
        by_metric[name]
        for name in (
            "underreaction_rate",
            "decision_flip_rate",
            "reaction_delay_frames",
            "offline_profile_utility",
            "overreaction_rate",
            "unnecessary_intervention_rate",
        )
    ]
    fields = [
        "metric", "fixed_profile_mean", "adaptive_profile_mean",
        "mean_delta", "bootstrap_ci95", "holm_sign_p", "evidence_status",
    ]
    cost_by_variant = {}
    for variant in ("fixed", "adaptive"):
        selected_costs = [row for row in costs if row["profile_learner"] == variant]
        adaptation_tokens = sum(
            float(row["adaptation_reactive_tokens"])
            + float(row["adaptation_planning_tokens"])
            for row in selected_costs
        ) / len(selected_costs)
        evaluation_tokens = sum(
            float(row["evaluation_reactive_tokens"])
            + float(row["evaluation_planning_tokens"])
            for row in selected_costs
        ) / len(selected_costs)
        cost_by_variant[variant] = (adaptation_tokens, evaluation_tokens)
    adaptation_cost_delta = (
        cost_by_variant["adaptive"][0] / cost_by_variant["fixed"][0] - 1.0
    )
    evaluation_cost_delta = (
        cost_by_variant["adaptive"][1] / cost_by_variant["fixed"][1] - 1.0
    )
    best_underreaction = min(
        cells, key=lambda row: float(row["delta_underreaction_rate"])
    )
    best_utility = max(
        cells, key=lambda row: float(row["delta_offline_profile_utility"])
    )
    worst_utility = min(
        cells, key=lambda row: float(row["delta_offline_profile_utility"])
    )
    return f"""{START}
## Completed Profile-Learning Ablation

The profile-learning matrix is complete ({completion['usable_jobs']}/{completion['expected_jobs']} usable jobs; no missing, invalid, or quality-failed jobs). It covers highD, inD, and rounD; aggressive, balanced, and conservative profiles; fixed versus adaptive profiles; and ten complete episode-order seeds. Primary inference resamples complete order-seed learning paths and uses design weights.

{markdown_table(selected, fields)}

The strongest multiplicity-aware evidence is that adaptive profiles reduce underreaction and decision flipping: both improve in all 10 order seeds, with Holm-adjusted sign-test p=0.019531. Reaction delay also decreases, but its multiplicity-adjusted auxiliary tests do not cross 0.05. Overall offline profile utility is unchanged because its confidence interval crosses zero. The adaptive learner also increases overreaction, intervention, and unnecessary intervention rates; these safety-conservative tradeoffs prevent a claim of uniformly better personalization.

The defensible conclusion is therefore bounded: online profile learning changes the policy in a repeatable way and improves response stability and underreaction, but it does not deliver a universal utility gain under the current 10-episode adaptation protocol. Dataset/profile descriptive effects are in `table_17_profile_learning_by_dataset_profile.csv`; they are useful for heterogeneity analysis but should not replace the order-seed-cluster primary inference.

Descriptively, the largest underreaction reduction occurs for {best_underreaction['dataset']}/{best_underreaction['profile']} ({float(best_underreaction['delta_underreaction_rate']):+.6f}), and the largest utility increase occurs for {best_utility['dataset']}/{best_utility['profile']} ({float(best_utility['delta_offline_profile_utility']):+.6f}). The largest utility decrease occurs for {worst_utility['dataset']}/{worst_utility['profile']} ({float(worst_utility['delta_offline_profile_utility']):+.6f}). This heterogeneity is consistent with adaptation being context- and preference-dependent, but the individual cell contrasts are descriptive rather than separately powered confirmatory tests.

The adaptive variant did not obtain these changes by simply spending more tokens: averaged descriptively across the nine dataset/profile cells, adaptation-phase total tokens changed by {adaptation_cost_delta:+.1%} and evaluation-phase total tokens by {evaluation_cost_delta:+.1%} relative to fixed profiles. This cost comparison is descriptive and is reported in `table_18_profile_adaptation_cost.csv`; it is not a paired causal estimate.

The requested adaptation-budget comparison is not inferentially valid at the per-cell episode level because there are no exact budget-0/budget-10 episode pairs. Those tests are deliberately suppressed and documented in `table_19_profile_adaptation_inference_audit.csv`. This is a limitation, not a zero effect.
{END}"""


def replace_marked_section(path: Path, section: str) -> None:
    text = path.read_text(encoding="utf-8") if path.exists() else ""
    if START in text and END in text:
        before, rest = text.split(START, 1)
        _, after = rest.split(END, 1)
        text = before.rstrip() + "\n\n" + section + after
    else:
        text = text.rstrip() + "\n\n" + section + "\n"
    path.write_text(text, encoding="utf-8")


def update_inventory(package_dir: Path, profile_dir: Path, completion: dict) -> None:
    path = package_dir / "table_00_experiment_inventory.csv"
    rows = [row for row in read_csv(path) if row["experiment"] != "profile_learning"]
    aggregate_rows = len(read_csv(profile_dir / "aggregate_summary.csv"))
    rows.append({
        "experiment": "profile_learning",
        "path": "runs/experiments/paper_core_profile_learning_ablation",
        "primary_matrix_ready": str(completion["primary_matrix_ready"]),
        "missing_jobs": completion["missing_jobs"],
        "invalid_jobs": completion["invalid_jobs"],
        "quality_failed_jobs": completion["quality_failed_jobs"],
        "aggregate_rows": aggregate_rows,
        "datasets": "highd,ind,round",
        "modes": "episode",
        "profiles": "aggressive,balanced,conservative",
        "rag_variants": "full_rag_grounded",
        "planning_variants": "planning_adaptive_peek",
        "policy_variants": "fixed_profile,adaptive_profile",
        "figures_written": 0,
        "figures_skipped": 0,
    })
    fields = list(rows[0])
    write_csv(path, rows, fields)


def update_claims(package_dir: Path, effects: list[dict]) -> None:
    path = package_dir / "table_15_key_claim_numbers.csv"
    rows = [row for row in read_csv(path) if not row["source"].startswith("table_16_")]
    lookup = {row["metric"]: row for row in effects}
    claims = (
        ("Adaptive profiles reduce underreaction", "underreaction_rate"),
        ("Adaptive profiles reduce decision flipping", "decision_flip_rate"),
        ("Adaptive profiles do not improve overall profile utility", "offline_profile_utility"),
        ("Adaptive profiles increase overreaction as a safety-conservative tradeoff", "overreaction_rate"),
    )
    for claim, metric in claims:
        row = lookup[metric]
        rows.append({
            "claim": claim,
            "metric": metric,
            "delta": row["mean_delta"],
            "ci95": row["bootstrap_ci95"],
            "source": "table_16_profile_learning_overall_effects.csv",
        })
    write_csv(path, rows, list(rows[0]))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--profile-dir",
        default="nas_runs/experiments/paper_core_profile_learning_ablation",
    )
    parser.add_argument(
        "--package-dir",
        default="nas_runs/experiments/paper_results_package",
    )
    args = parser.parse_args()
    profile_dir = Path(args.profile_dir)
    package_dir = Path(args.package_dir)
    completion = json.loads((profile_dir / "matrix_completion.json").read_text())
    if not completion.get("primary_matrix_ready") or completion.get("usable_jobs") != 180:
        raise SystemExit(f"Profile matrix is not publication-ready: {completion}")

    effects = effect_rows(profile_dir)
    cells = cell_rows(profile_dir)
    costs = cost_rows(profile_dir)
    audit = inference_audit_rows(profile_dir)
    write_csv(
        package_dir / "table_16_profile_learning_overall_effects.csv",
        effects,
        list(effects[0]),
    )
    write_csv(
        package_dir / "table_17_profile_learning_by_dataset_profile.csv",
        cells,
        list(cells[0]),
    )
    write_csv(
        package_dir / "table_18_profile_adaptation_cost.csv",
        costs,
        list(costs[0]),
    )
    write_csv(
        package_dir / "table_19_profile_adaptation_inference_audit.csv",
        audit,
        list(audit[0]),
    )

    section = report_section(effects, cells, costs, completion)
    replace_marked_section(package_dir / "paper_results_report.md", section)
    replace_marked_section(package_dir / "paper_results_visual_report.md", section)
    (package_dir / "profile_learning_results_report.md").write_text(
        "# Profile-Learning Results Audit\n\n" + section + "\n",
        encoding="utf-8",
    )
    update_inventory(package_dir, profile_dir, completion)
    update_claims(package_dir, effects)

    generated = [
        package_dir / "table_16_profile_learning_overall_effects.csv",
        package_dir / "table_17_profile_learning_by_dataset_profile.csv",
        package_dir / "table_18_profile_adaptation_cost.csv",
        package_dir / "table_19_profile_adaptation_inference_audit.csv",
        package_dir / "profile_learning_results_report.md",
        package_dir / "table_00_experiment_inventory.csv",
        package_dir / "table_15_key_claim_numbers.csv",
        package_dir / "paper_results_report.md",
        package_dir / "paper_results_visual_report.md",
    ]
    provenance = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_experiment": str(profile_dir),
        "source_analysis_fingerprint": json.loads(
            (profile_dir / "analysis_provenance.json").read_text()
        )["analysis_fingerprint"],
        "source_sha256": {
            name: sha256(profile_dir / name) for name in PROFILE_FILES
        },
        "generated_sha256": {
            path.name: sha256(path) for path in generated
        },
        "completion": completion,
    }
    (package_dir / "profile_learning_package_provenance.json").write_text(
        json.dumps(provenance, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"Updated {package_dir} with {len(effects)} profile-learning effects.")


if __name__ == "__main__":
    main()
