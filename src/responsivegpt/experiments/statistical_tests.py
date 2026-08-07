import argparse
import csv
import math
import random
import re
from pathlib import Path

from .io_utils import read_jsonl, write_csv


DEFAULT_BASELINE = "no_rag"
DEFAULT_METRICS = [
    "f1",
    "accuracy",
    "avg_underreaction_rate",
    "avg_overreaction_rate",
    "grounded_decision_rate",
    "hallucinated_citation_rate",
    "reactive_llm_call_rate",
]

LOWER_IS_BETTER = {
    "avg_underreaction_rate",
    "avg_overreaction_rate",
    "hallucinated_citation_rate",
    "reactive_llm_call_rate",
}

SIGNIFICANCE_FIELDS = [
    "estimator",
    "dataset",
    "mode",
    "profile_name",
    "use_profile_learner",
    "rag_variant",
    "planning_variant",
    "llm_policy_family",
    "baseline_variant",
    "treatment_variant",
    "metric",
    "metric_role",
    "direction",
    "inference_valid",
    "primary_inference",
    "num_pairs",
    "num_clusters",
    "expected_clusters",
    "matched_clusters",
    "baseline_weight_coverage",
    "treatment_weight_coverage",
    "paired_weight_coverage",
    "baseline_mean",
    "treatment_mean",
    "mean_delta",
    "median_delta",
    "paired_standardized_effect",
    "bootstrap_ci_low",
    "bootstrap_ci_high",
    "improved_pairs",
    "degraded_pairs",
    "ties",
    "sign_test_p",
    "sign_test_p_holm",
    "wilcoxon_p",
    "wilcoxon_p_holm",
    "auxiliary_test_note",
    "pair_key",
]


def _llm_policy_family(value) -> str:
    text = str(value or "")
    normalized = re.sub(
        r"(?:_?(?:order_)?seed_?)\d+$",
        "",
        text,
    ).rstrip("_")
    return normalized or "seed_family"


def _to_float(value):
    if value is None or value == "":
        return None
    try:
        return float(value)
    except Exception:
        return None


def _mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def _median(values: list[float]) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    mid = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[mid]
    return (ordered[mid - 1] + ordered[mid]) / 2


def _binom_two_sided_p(positive: int, negative: int) -> float | None:
    n = positive + negative
    if n == 0:
        return None
    k = min(positive, negative)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / (2 ** n)
    return min(1.0, 2.0 * tail)


def _average_ranks(values: list[float]) -> list[float]:
    indexed = sorted(enumerate(values), key=lambda x: x[1])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(indexed):
        j = i + 1
        while j < len(indexed) and indexed[j][1] == indexed[i][1]:
            j += 1
        avg_rank = (i + 1 + j) / 2.0
        for k in range(i, j):
            ranks[indexed[k][0]] = avg_rank
        i = j
    return ranks


def _wilcoxon_signed_rank_p(diffs: list[float]) -> float | None:
    nz = [d for d in diffs if d != 0]
    n = len(nz)
    if n == 0:
        return None

    abs_diffs = [abs(d) for d in nz]
    ranks = _average_ranks(abs_diffs)
    w_plus = sum(rank for rank, diff in zip(ranks, nz) if diff > 0)
    w_minus = sum(rank for rank, diff in zip(ranks, nz) if diff < 0)
    w = min(w_plus, w_minus)

    mean = n * (n + 1) / 4.0
    var = n * (n + 1) * (2 * n + 1) / 24.0
    if var <= 0:
        return None

    z = (w - mean + 0.5) / math.sqrt(var)
    return math.erfc(abs(z) / math.sqrt(2.0))


def _bootstrap_ci(diffs: list[float], *, seed: int = 20260606, rounds: int = 2000) -> tuple[float | None, float | None]:
    if not diffs:
        return None, None
    rng = random.Random(seed)
    means = []
    n = len(diffs)
    for _ in range(rounds):
        sample = [diffs[rng.randrange(n)] for _ in range(n)]
        means.append(sum(sample) / n)
    means.sort()
    lo_idx = int(0.025 * (rounds - 1))
    hi_idx = int(0.975 * (rounds - 1))
    return means[lo_idx], means[hi_idx]


def _standardized_paired_effect(diffs: list[float]) -> float | None:
    if len(diffs) < 2:
        return None
    variance = sum(
        (value - _mean(diffs)) ** 2 for value in diffs
    ) / (len(diffs) - 1)
    if variance <= 0:
        return 0.0
    return _mean(diffs) / math.sqrt(variance)


def _paired_observation_rows(
    observation_rows: list[dict],
    *,
    variant_field: str,
    baseline_variant: str,
    treatment_variant: str,
    metric: str,
    lower_is_better: bool,
    cell_fields: tuple[str, ...],
    estimator: str,
    metric_role: str = "primary",
    normalize_variant=None,
    seed: int = 20260625,
    rounds: int = 2000,
    min_weight_coverage: float = 0.95,
) -> dict | None:
    accum = {}
    candidate_rows = 0
    for row in observation_rows:
        if str(row.get("metric")) != metric:
            continue
        raw_variant = row.get(variant_field)
        variant = (
            normalize_variant(raw_variant)
            if normalize_variant is not None
            else str(raw_variant or "")
        )
        if variant not in {baseline_variant, treatment_variant}:
            continue
        candidate_rows += 1
        value = _to_float(row.get("value"))
        weight = _to_float(row.get("design_weight"))
        cluster = str(row.get("recording_cluster") or "")
        raw_event_index = row.get("event_index")
        event_index = (
            ""
            if raw_event_index is None
            else str(raw_event_index).strip()
        )
        if (
            value is None
            or weight is None
            or weight <= 0
            or not cluster
            or not event_index
        ):
            continue
        cell = tuple(row.get(field) for field in cell_fields)
        key = (variant, cell, cluster, event_index)
        state = accum.setdefault(key, [0.0, 0.0])
        state[0] += value * weight
        state[1] += weight

    baseline_keys = {
        (cell, cluster, event_index)
        for variant, cell, cluster, event_index in accum
        if variant == baseline_variant
    }
    treatment_keys = {
        (cell, cluster, event_index)
        for variant, cell, cluster, event_index in accum
        if variant == treatment_variant
    }
    expected_keys = baseline_keys | treatment_keys
    matched_keys = baseline_keys & treatment_keys
    if not expected_keys:
        if not candidate_rows:
            return None
        return {
            "estimator": estimator,
            "baseline_variant": baseline_variant,
            "treatment_variant": treatment_variant,
            "metric": metric,
            "metric_role": metric_role,
            "direction": (
                "lower_is_better"
                if lower_is_better else "higher_is_better"
            ),
            "inference_valid": False,
            "primary_inference": "episode_paired_cluster_bootstrap_ci",
            "num_pairs": 0,
            "num_clusters": 0,
            "expected_clusters": 0,
            "matched_clusters": 0,
            "baseline_weight_coverage": 0.0,
            "treatment_weight_coverage": 0.0,
            "paired_weight_coverage": 0.0,
            "auxiliary_test_note": (
                "Episode identifiers are missing; exact paired inference "
                "is suppressed. Rebuild weighted observations."
            ),
            "pair_key": "",
        }

    baseline_total_weight = sum(
        accum[(baseline_variant, cell, cluster, event_index)][1]
        for cell, cluster, event_index in baseline_keys
    )
    treatment_total_weight = sum(
        accum[(treatment_variant, cell, cluster, event_index)][1]
        for cell, cluster, event_index in treatment_keys
    )
    baseline_matched_weight = sum(
        accum[(baseline_variant, cell, cluster, event_index)][1]
        for cell, cluster, event_index in matched_keys
    )
    treatment_matched_weight = sum(
        accum[(treatment_variant, cell, cluster, event_index)][1]
        for cell, cluster, event_index in matched_keys
    )
    baseline_coverage = (
        baseline_matched_weight / baseline_total_weight
        if baseline_total_weight else 0.0
    )
    treatment_coverage = (
        treatment_matched_weight / treatment_total_weight
        if treatment_total_weight else 0.0
    )
    paired_coverage = min(baseline_coverage, treatment_coverage)
    inference_valid = bool(matched_keys) and (
        paired_coverage >= min_weight_coverage
    )

    episode_pairs = []
    cells = set()
    for variant, cell, cluster, event_index in sorted(accum):
        if variant != baseline_variant:
            continue
        treatment = accum.get(
            (treatment_variant, cell, cluster, event_index)
        )
        if treatment is None:
            continue
        baseline = accum[(variant, cell, cluster, event_index)]
        b = baseline[0] / baseline[1]
        t = treatment[0] / treatment[1]
        episode_pairs.append((
            cell,
            cluster,
            event_index,
            b,
            t,
            baseline[0],
            baseline[1],
            treatment[0],
            treatment[1],
        ))
        cells.add(cell)
    expected_clusters = {
        (cell, cluster)
        for cell, cluster, _ in expected_keys
    }
    matched_clusters = {
        (cell, cluster)
        for cell, cluster, _ in matched_keys
    }
    if not episode_pairs:
        return {
            "estimator": estimator,
            "baseline_variant": baseline_variant,
            "treatment_variant": treatment_variant,
            "metric": metric,
            "metric_role": metric_role,
            "direction": (
                "lower_is_better"
                if lower_is_better else "higher_is_better"
            ),
            "inference_valid": False,
            "primary_inference": "episode_paired_cluster_bootstrap_ci",
            "num_pairs": 0,
            "num_clusters": 0,
            "expected_clusters": len(expected_clusters),
            "matched_clusters": 0,
            "baseline_weight_coverage": baseline_coverage,
            "treatment_weight_coverage": treatment_coverage,
            "paired_weight_coverage": paired_coverage,
            "auxiliary_test_note": (
                "No exactly matched episodes; inference suppressed."
            ),
            "pair_key": "",
        }

    cluster_values = {}
    for (
        cell,
        cluster,
        _,
        _,
        _,
        baseline_num,
        baseline_den,
        treatment_num,
        treatment_den,
    ) in episode_pairs:
        state = cluster_values.setdefault(
            (cell, cluster), [0.0, 0.0, 0.0, 0.0]
        )
        state[0] += baseline_num
        state[1] += baseline_den
        state[2] += treatment_num
        state[3] += treatment_den

    def cell_average(index: int, cluster_weights=None):
        by_cell = {}
        for (
            cell,
            cluster,
        ), (
            baseline_num,
            baseline_den,
            treatment_num,
            treatment_den,
        ) in cluster_values.items():
            numerator = (baseline_num, treatment_num)[index]
            denominator = (baseline_den, treatment_den)[index]
            weight = (
                cluster_weights.get((cell, cluster), 1.0)
                if cluster_weights else 1.0
            )
            state = by_cell.setdefault(cell, [0.0, 0.0])
            state[0] += numerator * weight
            state[1] += denominator * weight
        return _mean([
            total / weight
            for total, weight in by_cell.values()
            if weight > 0
        ])

    baseline_mean = cell_average(0)
    treatment_mean = cell_average(1)
    point_delta = treatment_mean - baseline_mean
    rng = random.Random(seed)
    boot = []
    cluster_ids = sorted({
        (row[0], row[1])
        for row in episode_pairs
    })
    if inference_valid:
        for _ in range(rounds):
            weights = {
                cluster_id: rng.expovariate(1.0)
                for cluster_id in cluster_ids
            }
            boot.append(
                cell_average(1, weights) - cell_average(0, weights)
            )
        boot.sort()
        ci_low = boot[int(0.025 * (rounds - 1))]
        ci_high = boot[int(0.975 * (rounds - 1))]
    else:
        ci_low = ci_high = None
    diffs = [
        treatment_num / treatment_den - baseline_num / baseline_den
        for baseline_num, baseline_den, treatment_num, treatment_den
        in cluster_values.values()
        if baseline_den > 0 and treatment_den > 0
    ]
    positive = sum(value > 0 for value in diffs)
    negative = sum(value < 0 for value in diffs)
    ties = sum(value == 0 for value in diffs)
    return {
        "estimator": estimator,
        "baseline_variant": baseline_variant,
        "treatment_variant": treatment_variant,
        "metric": metric,
        "metric_role": metric_role,
        "direction": (
            "lower_is_better" if lower_is_better else "higher_is_better"
        ),
        "inference_valid": inference_valid,
        "primary_inference": "episode_paired_cluster_bootstrap_ci",
        "num_pairs": len(episode_pairs),
        "num_clusters": len(cluster_values),
        "expected_clusters": len(expected_clusters),
        "matched_clusters": len(matched_clusters),
        "baseline_weight_coverage": baseline_coverage,
        "treatment_weight_coverage": treatment_coverage,
        "paired_weight_coverage": paired_coverage,
        "baseline_mean": baseline_mean,
        "treatment_mean": treatment_mean,
        "mean_delta": point_delta,
        "median_delta": _median(diffs),
        "paired_standardized_effect": (
            _standardized_paired_effect(diffs)
            if inference_valid else None
        ),
        "bootstrap_ci_low": ci_low,
        "bootstrap_ci_high": ci_high,
        "improved_pairs": negative if lower_is_better else positive,
        "degraded_pairs": positive if lower_is_better else negative,
        "ties": ties,
        "sign_test_p": (
            _binom_two_sided_p(positive, negative)
            if inference_valid else None
        ),
        "sign_test_p_holm": None,
        "wilcoxon_p": (
            _wilcoxon_signed_rank_p(diffs)
            if inference_valid else None
        ),
        "wilcoxon_p_holm": None,
        "auxiliary_test_note": (
            "Episodes are paired exactly before recording-cluster "
            "aggregation. Sign and Wilcoxon tests are unweighted "
            "cluster-level sensitivity analyses; the cluster-bootstrap "
            "CI is primary."
        ),
        "pair_key": ";".join(
            "/".join(str(item) for item in cell)
            for cell in sorted(cells)
        ),
    }


def _paired_cluster_rows(
    observation_rows: list[dict],
    *,
    baseline_variant: str,
    treatment_variant: str,
    metric: str,
    lower_is_better: bool,
    seed: int = 20260625,
    rounds: int = 2000,
    min_weight_coverage: float = 0.95,
) -> dict | None:
    return _paired_observation_rows(
        observation_rows,
        variant_field="rag_variant",
        baseline_variant=baseline_variant,
        treatment_variant=treatment_variant,
        metric=metric,
        lower_is_better=lower_is_better,
        cell_fields=(
            "dataset",
            "mode",
            "profile_name",
            "use_profile_learner",
            "planning_variant",
            "llm_policy_variant",
            "profile_adaptation_episodes",
            "profile_adaptation_pool_episodes",
        ),
        estimator="paired_episode_recording_cluster_design_weighted",
        seed=seed,
        rounds=rounds,
        min_weight_coverage=min_weight_coverage,
    )


def _apply_holm(rows: list[dict], source: str, target: str) -> None:
    indexed = [
        (index, float(row[source]))
        for index, row in enumerate(rows)
        if row.get(source) is not None
    ]
    indexed.sort(key=lambda item: item[1])
    running = 0.0
    total = len(indexed)
    for rank, (index, p_value) in enumerate(indexed):
        adjusted = min(1.0, (total - rank) * p_value)
        running = max(running, adjusted)
        rows[index][target] = running
    for row in rows:
        row.setdefault(target, None)


def _read_aggregate_csv(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as f:
        return [dict(row) for row in csv.DictReader(f)]


def _pair_key(row: dict) -> tuple:
    return (
        row.get("dataset"),
        row.get("mode"),
        row.get("profile_name"),
        row.get("use_profile_learner"),
        row.get("planning_variant"),
        row.get("llm_policy_variant"),
        row.get("profile_adaptation_episodes"),
        row.get("frame_selection"),
        row.get("critical_top_k"),
    )


def _is_formal_performance_row(row: dict) -> bool:
    enabled = str(
        row.get("profile_protocol_enabled", "")
    ).strip().lower() in {"1", "true"}
    eligible = str(
        row.get("profile_formal_inference_eligible", "")
    ).strip().lower()
    return not (enabled and eligible in {"0", "false"})


def make_significance_tables(
    rows: list[dict],
    output_dir: str | Path,
    *,
    baseline_variant: str = DEFAULT_BASELINE,
    metrics: list[str] | None = None,
) -> list[dict]:
    output_dir = Path(output_dir)
    metrics = metrics or DEFAULT_METRICS

    by_variant_key = {}
    variants = set()
    for row in rows:
        if not _is_formal_performance_row(row):
            continue
        variant = str(row.get("rag_variant") or "")
        if not variant:
            continue
        variants.add(variant)
        by_variant_key[(variant, _pair_key(row))] = row

    out = []
    for treatment in sorted(variants):
        if treatment == baseline_variant:
            continue

        for metric in metrics:
            diffs = []
            baseline_values = []
            treatment_values = []
            pair_keys = []

            for (variant, key), baseline_row in by_variant_key.items():
                if variant != baseline_variant:
                    continue
                treatment_row = by_variant_key.get((treatment, key))
                if treatment_row is None:
                    continue

                b = _to_float(baseline_row.get(metric))
                t = _to_float(treatment_row.get(metric))
                if b is None or t is None:
                    continue
                baseline_values.append(b)
                treatment_values.append(t)
                diffs.append(t - b)
                pair_keys.append("/".join(str(x) for x in key))

            positive = sum(1 for d in diffs if d > 0)
            if not diffs:
                continue
            negative = sum(1 for d in diffs if d < 0)
            ties = sum(1 for d in diffs if d == 0)
            direction = "lower_is_better" if metric in LOWER_IS_BETTER else "higher_is_better"
            improved = negative if metric in LOWER_IS_BETTER else positive
            degraded = positive if metric in LOWER_IS_BETTER else negative
            ci_low, ci_high = _bootstrap_ci(diffs)

            out.append({
                "estimator": "unweighted_exploratory",
                "baseline_variant": baseline_variant,
                "treatment_variant": treatment,
                "metric": metric,
                "metric_role": "exploratory",
                "direction": direction,
                "inference_valid": False,
                "primary_inference": "none_exploratory_only",
                "num_pairs": len(diffs),
                "num_clusters": len(diffs),
                "baseline_mean": _mean(baseline_values),
                "treatment_mean": _mean(treatment_values),
                "mean_delta": _mean(diffs),
                "median_delta": _median(diffs),
                "paired_standardized_effect": (
                    _standardized_paired_effect(diffs)
                ),
                "bootstrap_ci_low": ci_low,
                "bootstrap_ci_high": ci_high,
                "improved_pairs": improved,
                "degraded_pairs": degraded,
                "ties": ties,
                "sign_test_p": _binom_two_sided_p(positive, negative),
                "sign_test_p_holm": None,
                "wilcoxon_p": _wilcoxon_signed_rank_p(diffs),
                "wilcoxon_p_holm": None,
                "auxiliary_test_note": (
                    "Unweighted exploratory cell-level comparison."
                ),
                "pair_key": ";".join(pair_keys),
            })

    _apply_holm(out, "sign_test_p", "sign_test_p_holm")
    _apply_holm(out, "wilcoxon_p", "wilcoxon_p_holm")
    write_csv(output_dir / "significance_vs_no_rag.csv", out, fieldnames=SIGNIFICANCE_FIELDS)
    return out


def make_weighted_significance_tables(
    weighted_rows: list[dict],
    output_dir: str | Path,
    *,
    baseline_variant: str = DEFAULT_BASELINE,
    metrics: list[str] | None = None,
    observation_csv: str | Path | None = None,
    observation_rows: list[dict] | None = None,
) -> list[dict]:
    output_dir = Path(output_dir)
    metrics = metrics or [
        "underreaction_rate",
        "overreaction_rate",
        "reaction_success_rate",
        "rag_grounded_decision_rate",
        "rag_output_invalid_citation_frame_rate",
        "planning_miss_rate",
        "unnecessary_intervention_rate",
        "missed_intervention_rate",
        "safety_action_appropriateness",
        "offline_profile_utility",
    ]
    lower_is_better = {
        "underreaction_rate",
        "overreaction_rate",
        "rag_output_invalid_citation_frame_rate",
        "planning_miss_rate",
        "unnecessary_intervention_rate",
        "missed_intervention_rate",
    }
    if observation_rows is None:
        observation_rows = (
            _read_aggregate_csv(Path(observation_csv))
            if observation_csv and Path(observation_csv).exists()
            else []
        )
    by_variant_key_metric = {}
    variants = set()
    for row in weighted_rows:
        valid = row.get("estimate_valid")
        if valid is not True and str(valid).lower() not in {"true", "1"}:
            continue
        variant = str(row.get("rag_variant") or "")
        metric = str(row.get("metric") or "")
        value = _to_float(row.get("weighted_mean"))
        if not variant or metric not in metrics or value is None:
            continue
        key = (
            row.get("dataset"),
            row.get("mode"),
            row.get("profile_name"),
            row.get("use_profile_learner"),
            row.get("planning_variant"),
            row.get("llm_policy_variant"),
            row.get("profile_adaptation_episodes"),
            row.get("profile_adaptation_pool_episodes"),
        )
        variants.add(variant)
        by_variant_key_metric[(variant, key, metric)] = value

    if baseline_variant not in variants or len(variants) < 2:
        write_csv(
            output_dir / "weighted_significance_vs_no_rag.csv",
            [],
            fieldnames=SIGNIFICANCE_FIELDS,
        )
        return []

    out = []
    for treatment in sorted(variants):
        if treatment == baseline_variant:
            continue
        for metric in metrics:
            if observation_rows:
                cluster_row = _paired_cluster_rows(
                    observation_rows,
                    baseline_variant=baseline_variant,
                    treatment_variant=treatment,
                    metric=metric,
                    lower_is_better=metric in lower_is_better,
                )
                if cluster_row is not None:
                    out.append(cluster_row)
                    continue
            diffs = []
            baseline_values = []
            treatment_values = []
            pair_keys = []
            for variant, key, row_metric in sorted(
                by_variant_key_metric
            ):
                if variant != baseline_variant or row_metric != metric:
                    continue
                treatment_value = by_variant_key_metric.get(
                    (treatment, key, metric)
                )
                if treatment_value is None:
                    continue
                baseline_value = by_variant_key_metric[
                    (variant, key, metric)
                ]
                baseline_values.append(baseline_value)
                treatment_values.append(treatment_value)
                diffs.append(treatment_value - baseline_value)
                pair_keys.append("/".join(str(x) for x in key))

            if not diffs:
                continue
            positive = sum(1 for value in diffs if value > 0)
            negative = sum(1 for value in diffs if value < 0)
            ties = sum(1 for value in diffs if value == 0)
            direction = (
                "lower_is_better"
                if metric in lower_is_better
                else "higher_is_better"
            )
            improved = negative if metric in lower_is_better else positive
            degraded = positive if metric in lower_is_better else negative
            ci_low, ci_high = _bootstrap_ci(diffs)
            out.append({
                "estimator": "design_weighted_cell_paired_auxiliary",
                "baseline_variant": baseline_variant,
                "treatment_variant": treatment,
                "metric": metric,
                "metric_role": "auxiliary",
                "direction": direction,
                "inference_valid": False,
                "primary_inference": "none_cell_level_auxiliary",
                "num_pairs": len(diffs),
                "num_clusters": len(diffs),
                "baseline_mean": _mean(baseline_values),
                "treatment_mean": _mean(treatment_values),
                "mean_delta": _mean(diffs),
                "median_delta": _median(diffs),
                "paired_standardized_effect": (
                    _standardized_paired_effect(diffs)
                ),
                "bootstrap_ci_low": ci_low,
                "bootstrap_ci_high": ci_high,
                "improved_pairs": improved,
                "degraded_pairs": degraded,
                "ties": ties,
                "sign_test_p": _binom_two_sided_p(positive, negative),
                "sign_test_p_holm": None,
                "wilcoxon_p": _wilcoxon_signed_rank_p(diffs),
                "wilcoxon_p_holm": None,
                "auxiliary_test_note": (
                    "Cell-level fallback only; recording observations "
                    "were unavailable."
                ),
                "pair_key": ";".join(pair_keys),
            })

    _apply_holm(out, "sign_test_p", "sign_test_p_holm")
    _apply_holm(out, "wilcoxon_p", "wilcoxon_p_holm")
    write_csv(
        output_dir / "weighted_significance_vs_no_rag.csv",
        out,
        fieldnames=SIGNIFICANCE_FIELDS,
    )
    return out


def make_planning_significance_tables(
    weighted_rows: list[dict],
    output_dir: str | Path,
    *,
    baseline_variant: str = "planning_off",
    metrics: list[str] | None = None,
    observation_csv: str | Path | None = None,
    observation_rows: list[dict] | None = None,
) -> list[dict]:
    output_dir = Path(output_dir)
    metrics = metrics or [
        "underreaction_rate",
        "overreaction_rate",
        "reaction_success_rate",
        "reaction_delay_frames",
        "decision_flip_rate",
        "rag_grounded_decision_rate",
        "rag_evidence_usage_rate",
        "unnecessary_intervention_rate",
        "missed_intervention_rate",
        "safety_action_appropriateness",
        "offline_profile_utility",
    ]
    lower_is_better = {
        "underreaction_rate",
        "overreaction_rate",
        "reaction_delay_frames",
        "decision_flip_rate",
        "unnecessary_intervention_rate",
        "missed_intervention_rate",
    }
    if observation_rows is None:
        observation_rows = (
            _read_aggregate_csv(Path(observation_csv))
            if observation_csv and Path(observation_csv).exists()
            else []
        )
    by_variant_key_metric = {}
    variants = set()
    for row in weighted_rows:
        valid = row.get("estimate_valid")
        if valid is not True and str(valid).lower() not in {"true", "1"}:
            continue
        variant = str(row.get("planning_variant") or "")
        metric = str(row.get("metric") or "")
        value = _to_float(row.get("weighted_mean"))
        if not variant or metric not in metrics or value is None:
            continue
        key = (
            row.get("dataset"),
            row.get("mode"),
            row.get("profile_name"),
            row.get("use_profile_learner"),
            row.get("rag_variant"),
            row.get("llm_policy_variant"),
            row.get("profile_adaptation_episodes"),
            row.get("profile_adaptation_pool_episodes"),
        )
        variants.add(variant)
        by_variant_key_metric[(variant, key, metric)] = value

    if baseline_variant not in variants or len(variants) < 2:
        write_csv(
            output_dir / "planning_weighted_effects.csv",
            [],
            fieldnames=SIGNIFICANCE_FIELDS,
        )
        return []

    out = []
    for treatment in sorted(variants):
        if treatment == baseline_variant:
            continue
        for metric in metrics:
            metric_role = (
                "secondary_conditional"
                if metric == "reaction_delay_frames"
                else "primary"
            )
            if observation_rows:
                cluster_row = _paired_observation_rows(
                    observation_rows,
                    variant_field="planning_variant",
                    baseline_variant=baseline_variant,
                    treatment_variant=treatment,
                    metric=metric,
                    lower_is_better=metric in lower_is_better,
                    cell_fields=(
                        "dataset",
                        "mode",
                        "profile_name",
                        "use_profile_learner",
                        "rag_variant",
                        "llm_policy_variant",
                        "profile_adaptation_episodes",
                        "profile_adaptation_pool_episodes",
                    ),
                    estimator=(
                        "planning_paired_episode_recording_cluster_"
                        "design_weighted"
                    ),
                    metric_role=metric_role,
                )
                if cluster_row is not None:
                    out.append(cluster_row)
                    continue

            diffs = []
            baseline_values = []
            treatment_values = []
            pair_keys = []
            for variant, key, row_metric in sorted(by_variant_key_metric):
                if variant != baseline_variant or row_metric != metric:
                    continue
                treatment_value = by_variant_key_metric.get(
                    (treatment, key, metric)
                )
                if treatment_value is None:
                    continue
                baseline_value = by_variant_key_metric[
                    (variant, key, metric)
                ]
                baseline_values.append(baseline_value)
                treatment_values.append(treatment_value)
                diffs.append(treatment_value - baseline_value)
                pair_keys.append("/".join(str(x) for x in key))

            if not diffs:
                continue
            positive = sum(1 for value in diffs if value > 0)
            negative = sum(1 for value in diffs if value < 0)
            ties = sum(1 for value in diffs if value == 0)
            lower = metric in lower_is_better
            ci_low, ci_high = _bootstrap_ci(diffs)
            out.append({
                "estimator": "planning_design_weighted_cell_paired_auxiliary",
                "baseline_variant": baseline_variant,
                "treatment_variant": treatment,
                "metric": metric,
                "metric_role": metric_role,
                "direction": (
                    "lower_is_better" if lower else "higher_is_better"
                ),
                "inference_valid": False,
                "primary_inference": "none_cell_level_auxiliary",
                "num_pairs": len(diffs),
                "num_clusters": len(diffs),
                "baseline_mean": _mean(baseline_values),
                "treatment_mean": _mean(treatment_values),
                "mean_delta": _mean(diffs),
                "median_delta": _median(diffs),
                "paired_standardized_effect": (
                    _standardized_paired_effect(diffs)
                ),
                "bootstrap_ci_low": ci_low,
                "bootstrap_ci_high": ci_high,
                "improved_pairs": negative if lower else positive,
                "degraded_pairs": positive if lower else negative,
                "ties": ties,
                "sign_test_p": _binom_two_sided_p(positive, negative),
                "sign_test_p_holm": None,
                "wilcoxon_p": _wilcoxon_signed_rank_p(diffs),
                "wilcoxon_p_holm": None,
                "auxiliary_test_note": (
                    "Cell-level fallback only; weighted episode "
                    "observations were unavailable."
                ),
                "pair_key": ";".join(pair_keys),
            })

    _apply_holm(out, "sign_test_p", "sign_test_p_holm")
    _apply_holm(out, "wilcoxon_p", "wilcoxon_p_holm")
    write_csv(
        output_dir / "planning_weighted_effects.csv",
        out,
        fieldnames=SIGNIFICANCE_FIELDS,
    )
    return out


def make_memory_budget_significance_tables(
    weighted_rows: list[dict],
    output_dir: str | Path,
    *,
    baseline_variant: str = "no_memory_no_governor",
    metrics: list[str] | None = None,
    observation_csv: str | Path | None = None,
    observation_rows: list[dict] | None = None,
) -> list[dict]:
    output_dir = Path(output_dir)
    metrics = metrics or [
        "underreaction_rate",
        "overreaction_rate",
        "reaction_success_rate",
        "decision_flip_rate",
        "rag_grounded_decision_rate",
        "rag_evidence_usage_rate",
        "unnecessary_intervention_rate",
        "missed_intervention_rate",
        "safety_action_appropriateness",
        "offline_profile_utility",
    ]
    lower_is_better = {
        "underreaction_rate",
        "overreaction_rate",
        "decision_flip_rate",
        "unnecessary_intervention_rate",
        "missed_intervention_rate",
    }
    if observation_rows is None:
        observation_rows = (
            _read_aggregate_csv(Path(observation_csv))
            if observation_csv and Path(observation_csv).exists()
            else []
        )
    by_variant_key_metric = {}
    variants = set()
    for row in weighted_rows:
        valid = row.get("estimate_valid")
        if valid is not True and str(valid).lower() not in {"true", "1"}:
            continue
        variant = str(row.get("llm_policy_variant") or "")
        metric = str(row.get("metric") or "")
        value = _to_float(row.get("weighted_mean"))
        if not variant or metric not in metrics or value is None:
            continue
        key = (
            row.get("dataset"),
            row.get("mode"),
            row.get("profile_name"),
            row.get("use_profile_learner"),
            row.get("rag_variant"),
            row.get("planning_variant"),
            row.get("profile_adaptation_episodes"),
            row.get("profile_adaptation_pool_episodes"),
        )
        variants.add(variant)
        by_variant_key_metric[(variant, key, metric)] = value

    if baseline_variant not in variants or len(variants) < 2:
        write_csv(
            output_dir / "memory_budget_weighted_effects.csv",
            [],
            fieldnames=SIGNIFICANCE_FIELDS,
        )
        return []

    out = []
    for treatment in sorted(variants):
        if treatment == baseline_variant:
            continue
        for metric in metrics:
            metric_role = (
                "secondary_conditional"
                if metric == "decision_flip_rate"
                else "primary"
            )
            if observation_rows:
                cluster_row = _paired_observation_rows(
                    observation_rows,
                    variant_field="llm_policy_variant",
                    baseline_variant=baseline_variant,
                    treatment_variant=treatment,
                    metric=metric,
                    lower_is_better=metric in lower_is_better,
                    cell_fields=(
                        "dataset",
                        "mode",
                        "profile_name",
                        "use_profile_learner",
                        "rag_variant",
                        "planning_variant",
                        "profile_adaptation_episodes",
                        "profile_adaptation_pool_episodes",
                    ),
                    estimator=(
                        "memory_budget_paired_episode_recording_cluster_"
                        "design_weighted"
                    ),
                    metric_role=metric_role,
                )
                if cluster_row is not None:
                    out.append(cluster_row)
                    continue

            diffs = []
            baseline_values = []
            treatment_values = []
            pair_keys = []
            for variant, key, row_metric in sorted(by_variant_key_metric):
                if variant != baseline_variant or row_metric != metric:
                    continue
                treatment_value = by_variant_key_metric.get(
                    (treatment, key, metric)
                )
                if treatment_value is None:
                    continue
                baseline_value = by_variant_key_metric[
                    (variant, key, metric)
                ]
                baseline_values.append(baseline_value)
                treatment_values.append(treatment_value)
                diffs.append(treatment_value - baseline_value)
                pair_keys.append("/".join(str(x) for x in key))

            if not diffs:
                continue
            positive = sum(1 for value in diffs if value > 0)
            negative = sum(1 for value in diffs if value < 0)
            ties = sum(1 for value in diffs if value == 0)
            lower = metric in lower_is_better
            ci_low, ci_high = _bootstrap_ci(diffs)
            out.append({
                "estimator": "memory_budget_design_weighted_cell_paired_auxiliary",
                "baseline_variant": baseline_variant,
                "treatment_variant": treatment,
                "metric": metric,
                "metric_role": metric_role,
                "direction": (
                    "lower_is_better" if lower else "higher_is_better"
                ),
                "inference_valid": False,
                "primary_inference": "none_cell_level_auxiliary",
                "num_pairs": len(diffs),
                "num_clusters": len(diffs),
                "baseline_mean": _mean(baseline_values),
                "treatment_mean": _mean(treatment_values),
                "mean_delta": _mean(diffs),
                "median_delta": _median(diffs),
                "paired_standardized_effect": (
                    _standardized_paired_effect(diffs)
                ),
                "bootstrap_ci_low": ci_low,
                "bootstrap_ci_high": ci_high,
                "improved_pairs": negative if lower else positive,
                "degraded_pairs": positive if lower else negative,
                "ties": ties,
                "sign_test_p": _binom_two_sided_p(positive, negative),
                "sign_test_p_holm": None,
                "wilcoxon_p": _wilcoxon_signed_rank_p(diffs),
                "wilcoxon_p_holm": None,
                "auxiliary_test_note": (
                    "Cell-level fallback only; weighted episode "
                    "observations were unavailable."
                ),
                "pair_key": ";".join(pair_keys),
            })

    _apply_holm(out, "sign_test_p", "sign_test_p_holm")
    _apply_holm(out, "wilcoxon_p", "wilcoxon_p_holm")
    write_csv(
        output_dir / "memory_budget_weighted_effects.csv",
        out,
        fieldnames=SIGNIFICANCE_FIELDS,
    )
    return out


def make_profile_learning_significance_tables(
    weighted_rows: list[dict],
    output_dir: str | Path,
    *,
    observation_csv: str | Path | None = None,
    observation_rows: list[dict] | None = None,
) -> list[dict]:
    output_dir = Path(output_dir)
    metrics = {
        "underreaction_rate",
        "overreaction_rate",
        "reaction_success_rate",
        "reaction_delay_frames",
        "decision_flip_rate",
        "decision_intervention_rate",
        "unnecessary_intervention_rate",
        "missed_intervention_rate",
        "safety_action_appropriateness",
        "offline_profile_utility",
    }
    lower_is_better = metrics - {
        "offline_profile_utility",
        "reaction_success_rate",
        "safety_action_appropriateness",
    }
    if observation_rows is None:
        observation_rows = (
            _read_aggregate_csv(Path(observation_csv))
            if observation_csv and Path(observation_csv).exists()
            else []
        )

    def normalize_learner(value):
        normalized = str(value).strip().lower()
        if normalized in {"true", "1", "on"}:
            return "on"
        if normalized in {"false", "0", "off"}:
            return "off"
        return ""

    values = {}
    for row in weighted_rows:
        valid = row.get("estimate_valid")
        if valid is not True and str(valid).lower() not in {"true", "1"}:
            continue
        metric = str(row.get("metric") or "")
        value = _to_float(row.get("weighted_mean"))
        learner = str(row.get("use_profile_learner", "")).lower()
        if metric not in metrics or value is None:
            continue
        if learner in {"true", "1"}:
            learner = "on"
        elif learner in {"false", "0"}:
            learner = "off"
        else:
            continue
        key = (
            row.get("dataset"),
            row.get("mode"),
            row.get("profile_name"),
            row.get("rag_variant"),
            row.get("planning_variant"),
            row.get("profile_adaptation_episodes"),
            row.get("profile_adaptation_pool_episodes"),
            row.get("llm_policy_variant"),
        )
        values[(learner, key, metric)] = value

    learners = {learner for learner, _, _ in values}
    if not {"off", "on"}.issubset(learners):
        write_csv(
            output_dir / "profile_learning_weighted_effects.csv",
            [],
            fieldnames=SIGNIFICANCE_FIELDS,
        )
        write_csv(
            output_dir / "profile_learning_episode_sensitivity.csv",
            [],
            fieldnames=SIGNIFICANCE_FIELDS,
        )
        return []

    out = []
    episode_sensitivity = []
    for metric in sorted(metrics):
        metric_role = (
            "secondary_conditional"
            if metric == "reaction_delay_frames"
            else "primary"
        )
        if observation_rows:
            cluster_row = _paired_observation_rows(
                observation_rows,
                variant_field="use_profile_learner",
                baseline_variant="off",
                treatment_variant="on",
                metric=metric,
                lower_is_better=metric in lower_is_better,
                cell_fields=(
                    "dataset",
                    "mode",
                    "profile_name",
                    "rag_variant",
                    "planning_variant",
                    "llm_policy_variant",
                    "profile_adaptation_episodes",
                    "profile_adaptation_pool_episodes",
                ),
                estimator=(
                    "profile_paired_episode_recording_cluster_"
                    "design_weighted"
                ),
                metric_role=metric_role,
                normalize_variant=normalize_learner,
            )
            if cluster_row is not None:
                cluster_row["inference_valid"] = False
                cluster_row["primary_inference"] = (
                    "episode_paired_sensitivity_only"
                )
                cluster_row["auxiliary_test_note"] = (
                    "Auxiliary episode-level sensitivity analysis. "
                    "Profile learning is path-dependent, so order-seed "
                    "cluster inference is primary."
                )
                episode_sensitivity.append(cluster_row)

        pairs = []
        baseline_keys = {
            key
            for learner, key, row_metric in values
            if learner == "off" and row_metric == metric
        }
        treatment_keys = {
            key
            for learner, key, row_metric in values
            if learner == "on" and row_metric == metric
        }
        matched_keys = baseline_keys & treatment_keys
        for learner, key, row_metric in sorted(values):
            if learner != "off" or row_metric != metric:
                continue
            adaptive = values.get(("on", key, metric))
            if adaptive is None:
                continue
            fixed = values[("off", key, metric)]
            seed = str(key[-1])
            cell = key[:-1]
            pairs.append((seed, cell, fixed, adaptive))
        if not pairs:
            continue

        def seed_weighted_means(seed_weights=None):
            baseline_by_cell = {}
            treatment_by_cell = {}
            for seed, cell, fixed, adaptive in pairs:
                weight = (
                    seed_weights.get(seed, 1.0)
                    if seed_weights else 1.0
                )
                b = baseline_by_cell.setdefault(cell, [0.0, 0.0])
                t = treatment_by_cell.setdefault(cell, [0.0, 0.0])
                b[0] += fixed * weight
                b[1] += weight
                t[0] += adaptive * weight
                t[1] += weight
            baseline_mean = _mean([
                numerator / denominator
                for numerator, denominator in baseline_by_cell.values()
                if denominator > 0
            ])
            treatment_mean = _mean([
                numerator / denominator
                for numerator, denominator in treatment_by_cell.values()
                if denominator > 0
            ])
            return baseline_mean, treatment_mean

        baseline_mean, treatment_mean = seed_weighted_means()
        mean_delta = treatment_mean - baseline_mean
        seed_diffs = {}
        for seed, _, fixed, adaptive in pairs:
            seed_diffs.setdefault(seed, []).append(adaptive - fixed)
        seed_diffs = {
            seed: _mean(diffs)
            for seed, diffs in seed_diffs.items()
        }
        seeds = sorted(seed_diffs)
        baseline_coverage = (
            len(matched_keys) / len(baseline_keys)
            if baseline_keys else 0.0
        )
        treatment_coverage = (
            len(matched_keys) / len(treatment_keys)
            if treatment_keys else 0.0
        )
        paired_coverage = min(
            baseline_coverage, treatment_coverage
        )
        expected_seeds = {
            str(key[-1]) for key in baseline_keys | treatment_keys
        }
        inference_valid = (
            len(seeds) >= 10
            and paired_coverage >= 0.95
        )
        if inference_valid:
            rng = random.Random(20260625)
            boot = []
            for _ in range(2000):
                weights = {
                    seed: rng.expovariate(1.0) for seed in seeds
                }
                b, t = seed_weighted_means(weights)
                boot.append(t - b)
            boot.sort()
            ci_low = boot[int(0.025 * (len(boot) - 1))]
            ci_high = boot[int(0.975 * (len(boot) - 1))]
        else:
            ci_low = ci_high = None
        diffs = list(seed_diffs.values())
        positive = sum(value > 0 for value in diffs)
        negative = sum(value < 0 for value in diffs)
        ties = sum(value == 0 for value in diffs)
        lower = metric in lower_is_better
        out.append({
            "estimator": "paired_order_seed_cluster_design_weighted",
            "baseline_variant": "profile_learner_off",
            "treatment_variant": "profile_learner_on",
            "metric": metric,
            "metric_role": metric_role,
            "direction": (
                "lower_is_better" if lower else "higher_is_better"
            ),
            "inference_valid": inference_valid,
            "primary_inference": "order_seed_cluster_bootstrap_ci",
            "num_pairs": len(pairs),
            "num_clusters": len(seeds),
            "expected_clusters": len(expected_seeds),
            "matched_clusters": len(seeds),
            "baseline_weight_coverage": baseline_coverage,
            "treatment_weight_coverage": treatment_coverage,
            "paired_weight_coverage": paired_coverage,
            "baseline_mean": baseline_mean,
            "treatment_mean": treatment_mean,
            "mean_delta": mean_delta,
            "median_delta": _median(diffs),
            "paired_standardized_effect": (
                _standardized_paired_effect(diffs)
                if inference_valid else None
            ),
            "bootstrap_ci_low": ci_low,
            "bootstrap_ci_high": ci_high,
            "improved_pairs": negative if lower else positive,
            "degraded_pairs": positive if lower else negative,
            "ties": ties,
            "sign_test_p": (
                _binom_two_sided_p(positive, negative)
                if inference_valid else None
            ),
            "sign_test_p_holm": None,
            "wilcoxon_p": (
                _wilcoxon_signed_rank_p(diffs)
                if inference_valid else None
            ),
            "wilcoxon_p_holm": None,
            "auxiliary_test_note": (
                "Reaction delay is conditional on observed response and "
                "must be interpreted jointly with reaction_success_rate."
                if metric == "reaction_delay_frames"
                else "Primary inference preserves complete episode-order "
                "learning paths and resamples order seeds."
            ),
            "pair_key": ";".join(seeds),
        })
    _apply_holm(out, "sign_test_p", "sign_test_p_holm")
    _apply_holm(out, "wilcoxon_p", "wilcoxon_p_holm")
    write_csv(
        output_dir / "profile_learning_weighted_effects.csv",
        out,
        fieldnames=SIGNIFICANCE_FIELDS,
    )
    write_csv(
        output_dir / "profile_learning_episode_sensitivity.csv",
        episode_sensitivity,
        fieldnames=SIGNIFICANCE_FIELDS,
    )
    return out


def _hierarchical_adaptation_budget_result(
    observation_rows: list[dict],
    *,
    dataset: str,
    profile: str,
    metric: str,
    budget: int,
    lower_is_better: bool,
    treatment_cell: dict | None = None,
    rounds: int = 2000,
) -> dict | None:
    treatment_cell = treatment_cell or {}
    accum = {}
    for row in observation_rows:
        if (
            str(row.get("dataset", "")) != dataset
            or str(row.get("profile_name", "")) != profile
            or str(row.get("metric", "")) != metric
        ):
            continue
        if any(
            (
                _llm_policy_family(row.get(field, ""))
                if field == "llm_policy_variant"
                else str(row.get(field, ""))
            )
            != str(expected)
            for field, expected in treatment_cell.items()
        ):
            continue
        row_budget = int(row.get("profile_adaptation_episodes") or 0)
        if row_budget not in {0, budget}:
            continue
        seed = str(row.get("llm_policy_variant", ""))
        cluster = str(row.get("recording_cluster", ""))
        raw_event_index = row.get("event_index")
        event_index = (
            ""
            if raw_event_index is None
            else str(raw_event_index).strip()
        )
        value = _to_float(row.get("value"))
        weight = _to_float(row.get("design_weight"))
        if (
            not seed
            or not cluster
            or not event_index
            or value is None
            or weight is None
            or weight <= 0
        ):
            continue
        key = (row_budget, seed, cluster, event_index)
        state = accum.setdefault(key, [0.0, 0.0])
        state[0] += value * weight
        state[1] += weight
    if not accum:
        return None

    seed_pairs = {}
    baseline_total_weight = 0.0
    treatment_total_weight = 0.0
    baseline_matched_weight = 0.0
    treatment_matched_weight = 0.0
    expected_clusters = set()
    matched_clusters = set()
    seeds = sorted({
        key[1] for key in accum
    })
    for seed in seeds:
        baseline_keys = {
            (cluster, event_index)
            for row_budget, row_seed, cluster, event_index in accum
            if row_budget == 0 and row_seed == seed
        }
        treatment_keys = {
            (cluster, event_index)
            for row_budget, row_seed, cluster, event_index in accum
            if row_budget == budget and row_seed == seed
        }
        matched = baseline_keys & treatment_keys
        baseline_total_weight += sum(
            accum[(0, seed, cluster, event_index)][1]
            for cluster, event_index in baseline_keys
        )
        treatment_total_weight += sum(
            accum[(budget, seed, cluster, event_index)][1]
            for cluster, event_index in treatment_keys
        )
        baseline_matched_weight += sum(
            accum[(0, seed, cluster, event_index)][1]
            for cluster, event_index in matched
        )
        treatment_matched_weight += sum(
            accum[(budget, seed, cluster, event_index)][1]
            for cluster, event_index in matched
        )
        expected_clusters.update(
            (seed, cluster)
            for cluster, _ in baseline_keys | treatment_keys
        )
        matched_clusters.update(
            (seed, cluster) for cluster, _ in matched
        )
        if matched:
            seed_pairs[seed] = [
                (
                    cluster,
                    event_index,
                    accum[(0, seed, cluster, event_index)],
                    accum[(budget, seed, cluster, event_index)],
                )
                for cluster, event_index in sorted(matched)
            ]

    matched_seeds = sorted(seed_pairs)
    if not matched_seeds:
        return {
            "estimator": (
                "paired_episode_two_stage_design_weighted_hierarchical"
            ),
            "dataset": dataset,
            "profile_name": profile,
            **{
                (
                    "llm_policy_family"
                    if key == "llm_policy_variant" else key
                ): value
                for key, value in treatment_cell.items()
            },
            "baseline_variant": "adapt_budget_0",
            "treatment_variant": f"adapt_budget_{budget}",
            "metric": metric,
            "metric_role": "primary",
            "direction": (
                "lower_is_better"
                if lower_is_better else "higher_is_better"
            ),
            "inference_valid": False,
            "primary_inference": (
                "hierarchical_order_seed_recording_cluster_bootstrap_ci"
            ),
            "num_pairs": 0,
            "num_clusters": 0,
            "expected_clusters": len(expected_clusters),
            "matched_clusters": 0,
            "baseline_weight_coverage": 0.0,
            "treatment_weight_coverage": 0.0,
            "paired_weight_coverage": 0.0,
            "baseline_mean": None,
            "treatment_mean": None,
            "mean_delta": None,
            "median_delta": None,
            "paired_standardized_effect": None,
            "bootstrap_ci_low": None,
            "bootstrap_ci_high": None,
            "improved_pairs": 0,
            "degraded_pairs": 0,
            "ties": 0,
            "sign_test_p": None,
            "sign_test_p_holm": None,
            "wilcoxon_p": None,
            "wilcoxon_p_holm": None,
            "auxiliary_test_note": (
                f"dataset={dataset}; profile={profile}; no exact episode "
                "pairs, so hierarchical inference is suppressed."
            ),
            "pair_key": "",
        }
    baseline_coverage = (
        baseline_matched_weight / baseline_total_weight
        if baseline_total_weight else 0.0
    )
    treatment_coverage = (
        treatment_matched_weight / treatment_total_weight
        if treatment_total_weight else 0.0
    )
    paired_coverage = min(
        baseline_coverage, treatment_coverage
    )
    inference_valid = (
        len(matched_seeds) >= 10 and paired_coverage >= 0.95
    )

    def seed_means(seed: str, cluster_weights=None):
        baseline_num = baseline_den = 0.0
        treatment_num = treatment_den = 0.0
        for cluster, _, baseline, treatment in seed_pairs[seed]:
            multiplier = (
                cluster_weights.get(cluster, 1.0)
                if cluster_weights else 1.0
            )
            baseline_num += baseline[0] * multiplier
            baseline_den += baseline[1] * multiplier
            treatment_num += treatment[0] * multiplier
            treatment_den += treatment[1] * multiplier
        return (
            baseline_num / baseline_den,
            treatment_num / treatment_den,
        )

    per_seed = {
        seed: seed_means(seed) for seed in matched_seeds
    }
    baseline_mean = _mean([value[0] for value in per_seed.values()])
    treatment_mean = _mean([value[1] for value in per_seed.values()])
    diffs = [
        per_seed[seed][1] - per_seed[seed][0]
        for seed in matched_seeds
    ]
    if inference_valid:
        rng = random.Random(
            f"hierarchical|{dataset}|{profile}|{metric}|{budget}"
        )
        bootstrap = []
        for _ in range(rounds):
            sampled_seeds = [
                matched_seeds[rng.randrange(len(matched_seeds))]
                for _ in matched_seeds
            ]
            all_clusters = sorted({
                pair[0]
                for seed in matched_seeds
                for pair in seed_pairs[seed]
            })
            cluster_weights = {
                cluster: rng.expovariate(1.0)
                for cluster in all_clusters
            }
            sampled_diffs = []
            for seed in sampled_seeds:
                baseline_value, treatment_value = seed_means(
                    seed, cluster_weights
                )
                sampled_diffs.append(
                    treatment_value - baseline_value
                )
            bootstrap.append(_mean(sampled_diffs))
        bootstrap.sort()
        ci_low = bootstrap[int(0.025 * (rounds - 1))]
        ci_high = bootstrap[int(0.975 * (rounds - 1))]
    else:
        ci_low = ci_high = None
    positive = sum(value > 0 for value in diffs)
    negative = sum(value < 0 for value in diffs)
    return {
        "estimator": (
            "paired_episode_two_stage_design_weighted_hierarchical"
        ),
        "dataset": dataset,
        "profile_name": profile,
        **{
            (
                "llm_policy_family"
                if key == "llm_policy_variant" else key
            ): value
            for key, value in treatment_cell.items()
        },
        "baseline_variant": "adapt_budget_0",
        "treatment_variant": f"adapt_budget_{budget}",
        "metric": metric,
        "metric_role": "primary",
        "direction": (
            "lower_is_better"
            if lower_is_better else "higher_is_better"
        ),
        "inference_valid": inference_valid,
        "primary_inference": (
            "hierarchical_order_seed_recording_cluster_bootstrap_ci"
        ),
        "num_pairs": sum(len(rows) for rows in seed_pairs.values()),
        "num_clusters": len(matched_clusters),
        "expected_clusters": len(expected_clusters),
        "matched_clusters": len(matched_clusters),
        "baseline_weight_coverage": baseline_coverage,
        "treatment_weight_coverage": treatment_coverage,
        "paired_weight_coverage": paired_coverage,
        "baseline_mean": baseline_mean,
        "treatment_mean": treatment_mean,
        "mean_delta": treatment_mean - baseline_mean,
        "median_delta": _median(diffs),
        "paired_standardized_effect": (
            _standardized_paired_effect(diffs)
            if inference_valid else None
        ),
        "bootstrap_ci_low": ci_low,
        "bootstrap_ci_high": ci_high,
        "improved_pairs": negative if lower_is_better else positive,
        "degraded_pairs": positive if lower_is_better else negative,
        "ties": sum(value == 0 for value in diffs),
        "sign_test_p": (
            _binom_two_sided_p(positive, negative)
            if inference_valid else None
        ),
        "sign_test_p_holm": None,
        "wilcoxon_p": (
            _wilcoxon_signed_rank_p(diffs)
            if inference_valid else None
        ),
        "wilcoxon_p_holm": None,
        "auxiliary_test_note": (
            f"dataset={dataset}; profile={profile}; exact episode pairs; "
            "outer order-seed resampling with one shared recording-cluster "
            "Bayesian-bootstrap weight vector per replicate."
        ),
        "pair_key": ";".join(matched_seeds),
    }


def make_profile_adaptation_budget_significance(
    weighted_rows: list[dict],
    output_dir: str | Path,
    observation_rows: list[dict] | None = None,
) -> list[dict]:
    metrics = {
        "underreaction_rate": True,
        "overreaction_rate": True,
        "reaction_success_rate": False,
        "offline_profile_utility": False,
    }
    grouped = {}
    for row in weighted_rows:
        if str(row.get("profile_protocol_enabled", "")).lower() not in {
            "1", "true"
        }:
            continue
        if str(row.get("estimate_valid", "")).lower() not in {
            "1", "true"
        } and row.get("estimate_valid") is not True:
            continue
        budget = int(row.get("profile_adaptation_episodes") or 0)
        seed = str(row.get("llm_policy_variant", ""))
        metric = str(row.get("metric", ""))
        if metric not in metrics:
            continue
        value = _to_float(row.get("weighted_mean"))
        if value is None:
            continue
        cell = (
            str(row.get("dataset", "")),
            str(row.get("mode", "")),
            str(row.get("profile_name", "")),
            str(row.get("use_profile_learner", "")),
            str(row.get("rag_variant", "")),
            str(row.get("planning_variant", "")),
            _llm_policy_family(row.get("llm_policy_variant", "")),
            metric,
            budget,
        )
        grouped.setdefault(cell, {})[seed] = value

    out = []
    cells = {
        key[:-1] for key in grouped
    }
    for cell in sorted(cells):
        (
            dataset,
            mode,
            profile,
            use_profile_learner,
            rag_variant,
            planning_variant,
            llm_policy_family,
            metric,
        ) = cell
        baseline = grouped.get(cell + (0,), {})
        for budget in sorted({
            key[-1]
            for key in grouped
            if key[:-1] == cell and key[-1] > 0
        }):
            treatment_cell = {
                "mode": mode,
                "use_profile_learner": use_profile_learner,
                "rag_variant": rag_variant,
                "planning_variant": planning_variant,
                "llm_policy_variant": llm_policy_family,
            }
            hierarchical = _hierarchical_adaptation_budget_result(
                observation_rows or [],
                dataset=dataset,
                profile=profile,
                metric=metric,
                budget=budget,
                lower_is_better=metrics[metric],
                treatment_cell=treatment_cell,
            )
            if hierarchical is not None:
                out.append(hierarchical)
                continue
            treatment = grouped[cell + (budget,)]
            seeds = sorted(set(baseline) & set(treatment))
            diffs = [
                treatment[seed] - baseline[seed] for seed in seeds
            ]
            inference_valid = len(seeds) >= 10
            if inference_valid:
                rng = random.Random(
                    f"adaptation|{dataset}|{profile}|{metric}|{budget}"
                )
                boot = []
                for _ in range(2000):
                    sample = [
                        diffs[rng.randrange(len(diffs))]
                        for _ in diffs
                    ]
                    boot.append(_mean(sample))
                boot.sort()
                ci_low = boot[int(0.025 * (len(boot) - 1))]
                ci_high = boot[int(0.975 * (len(boot) - 1))]
            else:
                ci_low = ci_high = None
            positive = sum(value > 0 for value in diffs)
            negative = sum(value < 0 for value in diffs)
            lower = metrics[metric]
            out.append({
                "estimator": (
                    "paired_order_seed_design_weighted_adaptation_budget"
                ),
                "dataset": dataset,
                "mode": mode,
                "profile_name": profile,
                "use_profile_learner": use_profile_learner,
                "rag_variant": rag_variant,
                "planning_variant": planning_variant,
                "llm_policy_family": llm_policy_family,
                "baseline_variant": "adapt_budget_0",
                "treatment_variant": f"adapt_budget_{budget}",
                "metric": metric,
                "metric_role": "primary",
                "direction": (
                    "lower_is_better" if lower else "higher_is_better"
                ),
                "inference_valid": inference_valid,
                "primary_inference": "paired_seed_bootstrap_ci",
                "num_pairs": len(seeds),
                "num_clusters": len(seeds),
                "expected_clusters": max(len(baseline), len(treatment)),
                "matched_clusters": len(seeds),
                "baseline_weight_coverage": (
                    len(seeds) / len(baseline) if baseline else 0.0
                ),
                "treatment_weight_coverage": (
                    len(seeds) / len(treatment) if treatment else 0.0
                ),
                "paired_weight_coverage": min(
                    len(seeds) / len(baseline) if baseline else 0.0,
                    len(seeds) / len(treatment) if treatment else 0.0,
                ),
                "baseline_mean": _mean([
                    baseline[seed] for seed in seeds
                ]),
                "treatment_mean": _mean([
                    treatment[seed] for seed in seeds
                ]),
                "mean_delta": _mean(diffs),
                "median_delta": _median(diffs),
                "paired_standardized_effect": (
                    _standardized_paired_effect(diffs)
                    if inference_valid else None
                ),
                "bootstrap_ci_low": ci_low,
                "bootstrap_ci_high": ci_high,
                "improved_pairs": negative if lower else positive,
                "degraded_pairs": positive if lower else negative,
                "ties": sum(value == 0 for value in diffs),
                "sign_test_p": (
                    _binom_two_sided_p(positive, negative)
                    if inference_valid else None
                ),
                "sign_test_p_holm": None,
                "wilcoxon_p": (
                    _wilcoxon_signed_rank_p(diffs)
                    if inference_valid else None
                ),
                "wilcoxon_p_holm": None,
                "auxiliary_test_note": (
                    f"dataset={dataset}; profile={profile}; common fixed "
                    "evaluation set required."
                ),
                "pair_key": ";".join(seeds),
            })
    _apply_holm(out, "sign_test_p", "sign_test_p_holm")
    _apply_holm(out, "wilcoxon_p", "wilcoxon_p_holm")
    write_csv(
        Path(output_dir) / "profile_adaptation_budget_significance.csv",
        out,
        fieldnames=SIGNIFICANCE_FIELDS,
    )
    return out


def main():
    parser = argparse.ArgumentParser(description="Build paired significance tables for an experiment matrix.")
    parser.add_argument("--experiment_dir", required=True)
    parser.add_argument("--baseline", default=DEFAULT_BASELINE)
    args = parser.parse_args()

    experiment_dir = Path(args.experiment_dir)
    rows = _read_aggregate_csv(experiment_dir / "aggregate_summary.csv")
    if not rows and (experiment_dir / "job_status.jsonl").exists():
        from .aggregate_runs import aggregate_experiment

        rows = aggregate_experiment(experiment_dir)

    out = make_significance_tables(rows, experiment_dir, baseline_variant=args.baseline)
    print(f"Wrote {len(out)} significance rows to {experiment_dir / 'significance_vs_no_rag.csv'}")


if __name__ == "__main__":
    main()
