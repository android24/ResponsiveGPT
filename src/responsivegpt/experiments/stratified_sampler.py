import argparse
import csv
import json
import math
import random
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

from .config_loader import load_config
from .dataset_registry import resolve_dataset_config


DEFAULT_CONFIG = "src/responsivegpt/experiments/configs/paper_responsivegpt_main_token_saver.json"
DEFAULT_CENSUS = "data/full_pool_census/cornercase_v1/full_pool_episode_census.csv"


def _read_csv(path: str | Path) -> list[dict]:
    with Path(path).open("r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _write_csv(path: Path, rows: list[dict], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = fieldnames or (list(rows[0].keys()) if rows else [])
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _parse_targets(raw: str, default_target: int) -> dict[str, int]:
    targets = {}
    if not raw:
        return targets
    for item in raw.split(","):
        item = item.strip()
        if not item:
            continue
        if "=" not in item:
            targets[item] = default_target
            continue
        name, value = item.split("=", 1)
        targets[name.strip()] = int(value.strip())
    return targets


def _stable_dataset_seed(seed: int, dataset: str) -> int:
    return seed + sum(ord(c) for c in dataset) * 997


def _to_float(value, default=None):
    if value is None or value == "":
        return default
    try:
        out = float(value)
    except Exception:
        return default
    if math.isnan(out) or math.isinf(out):
        return default
    return out


def _group_key(row: dict) -> tuple[str, str, str, str]:
    return (
        row.get("risk_stratum", "unknown"),
        row.get("event_type", "unknown"),
        str(row.get("dataset_risk_label", "")),
        str(row.get("vru_present", "")),
    )


def _score_std(rows: list[dict]) -> float:
    values = [
        _to_float(row.get("deterministic_risk_score"))
        for row in rows
    ]
    values = [value for value in values if value is not None]
    if len(values) <= 1:
        return 0.0
    mean = sum(values) / len(values)
    variance = sum((value - mean) ** 2 for value in values) / (len(values) - 1)
    return math.sqrt(max(0.0, variance))


def _allocation_weights(groups: dict[tuple, list[dict]], method: str) -> dict[tuple, float]:
    method = str(method or "neyman").lower()
    weights = {}
    for key, rows in groups.items():
        population = len(rows)
        if method == "proportional":
            weight = float(population)
        elif method == "neyman":
            # Neyman allocation: n_h = n * N_h * S_h / sum_h(N_h * S_h).
            # The floor keeps rare but low-variance corner-case strata visible.
            std = max(_score_std(rows), 0.05)
            weight = float(population) * std
        else:
            raise ValueError("--allocation must be one of: neyman / proportional")
        weights[key] = weight
    return weights


def _allocate_quotas(groups: dict[tuple, list[dict]], target: int, method: str = "neyman") -> dict[tuple, int]:
    total = sum(len(rows) for rows in groups.values())
    if total <= target:
        return {key: len(rows) for key, rows in groups.items()}

    weights = _allocation_weights(groups, method)
    total_weight = sum(weights.values())
    if total_weight <= 0:
        weights = _allocation_weights(groups, "proportional")
        total_weight = sum(weights.values())

    sorted_groups = sorted(groups.items(), key=lambda item: (-weights.get(item[0], 0.0), -len(item[1]), item[0]))
    if target < len(sorted_groups):
        selected_keys = {key for key, _ in sorted_groups[:target]}
        return {
            key: (1 if key in selected_keys else 0)
            for key in groups
        }

    quotas = {}
    for key, rows in sorted_groups:
        raw_quota = round(target * weights.get(key, 0.0) / total_weight)
        quotas[key] = max(1, min(len(rows), raw_quota))

    while sum(quotas.values()) > target:
        candidates = [
            key for key, quota in quotas.items()
            if quota > 1
        ]
        if not candidates:
            break
        key = max(candidates, key=lambda k: (quotas[k], -weights.get(k, 0.0), len(groups[k])))
        quotas[key] -= 1

    while sum(quotas.values()) < target:
        candidates = [
            key for key, quota in quotas.items()
            if quota < len(groups[key])
        ]
        if not candidates:
            break
        key = max(candidates, key=lambda k: (weights.get(k, 0.0), len(groups[k]) - quotas[k], len(groups[k])))
        quotas[key] += 1

    return quotas


def _sample_census_rows(
    census_rows: list[dict],
    target: int,
    seed: int,
    dataset: str,
    allocation: str,
) -> tuple[list[dict], list[dict]]:
    groups = defaultdict(list)
    for row in census_rows:
        groups[_group_key(row)].append(row)

    quotas = _allocate_quotas(groups, target, method=allocation)
    weights = _allocation_weights(groups, allocation)
    rng = random.Random(_stable_dataset_seed(seed, dataset))
    selected = []
    quota_rows = []
    for key, rows in sorted(groups.items(), key=lambda item: item[0]):
        quota = quotas.get(key, 0)
        if quota <= 0:
            continue
        if quota >= len(rows):
            chosen = list(rows)
        else:
            chosen = rng.sample(rows, quota)
        selected.extend(chosen)
        risk_stratum, event_type, risk_label, vru_present = key
        quota_rows.append({
            "dataset": dataset,
            "risk_stratum": risk_stratum,
            "event_type": event_type,
            "dataset_risk_label": risk_label,
            "vru_present": vru_present,
            "population_rows": len(rows),
            "risk_score_std": round(_score_std(rows), 8),
            "allocation_weight": round(weights.get(key, 0.0), 8),
            "sample_quota": quota,
        })

    selected.sort(key=lambda row: int(row["row_index"]))
    return selected, quota_rows


def build_samples(
    config_path: str,
    census_csv: str,
    out_dir: str,
    target_per_dataset: int,
    targets: dict[str, int],
    seed: int,
    allocation: str,
) -> dict:
    config = load_config(config_path)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    census = _read_csv(census_csv)
    by_dataset = defaultdict(list)
    for row in census:
        by_dataset[row["dataset"]].append(row)

    manifest = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "config_path": str(config_path),
        "census_csv": str(census_csv),
        "out_dir": str(out),
        "seed": seed,
        "target_per_dataset": target_per_dataset,
        "targets": targets,
        "allocation": allocation,
        "allocation_formula": (
            "neyman: n_h = n * N_h * S_h / sum(N_h * S_h), "
            "where S_h is std(deterministic_risk_score) with a 0.05 floor"
        ),
        "outputs": {},
        "datasets": {},
    }

    combined_sample_rows = []
    strata_summary_rows = []
    allocation_summary_rows = []

    for dataset in sorted(by_dataset):
        target = int(targets.get(dataset, target_per_dataset))
        selected_census, quota_rows = _sample_census_rows(
            by_dataset[dataset],
            target,
            seed,
            dataset,
            allocation=allocation,
        )
        allocation_summary_rows.extend(quota_rows)
        selected_indices = {int(row["row_index"]) for row in selected_census}

        dataset_cfg = resolve_dataset_config(config, dataset)
        original_rows = _read_csv(dataset_cfg["summary_csv"])
        original_fieldnames = list(original_rows[0].keys()) if original_rows else []
        sampled_original_rows = [
            row for idx, row in enumerate(original_rows)
            if idx in selected_indices
        ]

        sample_path = out / f"{dataset}_core_sample_seed{seed}.csv"
        _write_csv(sample_path, sampled_original_rows, original_fieldnames)

        sample_census_path = out / f"{dataset}_core_sample_census_seed{seed}.csv"
        _write_csv(sample_census_path, selected_census)

        counts = Counter(_group_key(row) for row in selected_census)
        for key, count in sorted(counts.items()):
            risk_stratum, event_type, risk_label, vru_present = key
            strata_summary_rows.append({
                "dataset": dataset,
                "risk_stratum": risk_stratum,
                "event_type": event_type,
                "dataset_risk_label": risk_label,
                "vru_present": vru_present,
                "num_rows": count,
            })

        for row in selected_census:
            combined_sample_rows.append(row)

        manifest["outputs"][dataset] = {
            "summary_csv": str(sample_path),
            "sample_census_csv": str(sample_census_path),
        }
        manifest["datasets"][dataset] = {
            "source_summary_csv": dataset_cfg["summary_csv"],
            "source_rows": len(original_rows),
            "target_rows": target,
            "sampled_rows": len(sampled_original_rows),
            "unique_strata": len(counts),
        }

    combined_path = out / f"core_sample_census_seed{seed}.csv"
    _write_csv(combined_path, combined_sample_rows)
    strata_path = out / f"core_sample_strata_summary_seed{seed}.csv"
    _write_csv(strata_path, strata_summary_rows)
    allocation_path = out / f"core_sample_allocation_summary_seed{seed}.csv"
    _write_csv(allocation_path, allocation_summary_rows)

    manifest["outputs"]["combined_sample_census_csv"] = str(combined_path)
    manifest["outputs"]["sample_strata_summary_csv"] = str(strata_path)
    manifest["outputs"]["allocation_summary_csv"] = str(allocation_path)
    manifest_path = out / f"core_sample_manifest_seed{seed}.json"
    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    return manifest


def main():
    parser = argparse.ArgumentParser(description="Create stratified core paper samples from the full-pool census.")
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--census_csv", default=DEFAULT_CENSUS)
    parser.add_argument("--out_dir", default="data/eval_samples/core_v1")
    parser.add_argument("--target_per_dataset", type=int, default=300)
    parser.add_argument("--targets", default="", help="Optional comma list, e.g. highd=300,ind=300,round=300")
    parser.add_argument(
        "--allocation",
        default="neyman",
        choices=["neyman", "proportional"],
        help="Stratum budget allocation. Neyman uses population size times risk-score std.",
    )
    parser.add_argument("--seed", type=int, default=20260613)
    args = parser.parse_args()

    targets = _parse_targets(args.targets, args.target_per_dataset)
    manifest = build_samples(
        config_path=args.config,
        census_csv=args.census_csv,
        out_dir=args.out_dir,
        target_per_dataset=args.target_per_dataset,
        targets=targets,
        seed=args.seed,
        allocation=args.allocation,
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
