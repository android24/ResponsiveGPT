import argparse
import csv
import json
import random
from collections import defaultdict
from datetime import datetime
from pathlib import Path

from .config_loader import load_config
from .dataset_registry import resolve_dataset_config
from .io_utils import write_csv, write_json
from .stratified_sampler import _allocate_quotas, _group_key, _read_csv, _stable_dataset_seed


DEFAULT_CONFIG = "src/responsivegpt/experiments/configs/paper_responsivegpt_main_token_saver.json"
DEFAULT_CENSUS = "data/full_pool_census/cornercase_v1/full_pool_episode_census.csv"


def _sample_incremental(
    census_rows: list[dict],
    cumulative_target: int,
    seed: int,
    dataset: str,
    allocation: str,
    previous_selected: list[dict] | None = None,
) -> list[dict]:
    previous_selected = list(previous_selected or [])
    previous_indices = {
        int(row["row_index"]) for row in previous_selected
    }
    if len(previous_indices) != len(previous_selected):
        raise ValueError("previous_selected contains duplicate row_index values")

    groups = defaultdict(list)
    for row in census_rows:
        groups[_group_key(row)].append(row)

    target = min(max(0, cumulative_target), len(census_rows))
    if len(previous_selected) > target:
        raise ValueError(
            "cumulative_target cannot be smaller than the previous prefix"
        )

    quotas = _allocate_quotas(groups, target, method=allocation)
    rng = random.Random(_stable_dataset_seed(seed, dataset))
    selected = list(previous_selected)
    previous_counts = defaultdict(int)
    candidates = {}

    for key, rows in sorted(groups.items(), key=lambda item: item[0]):
        shuffled = list(rows)
        rng.shuffle(shuffled)
        candidates[key] = [
            row
            for row in shuffled
            if int(row["row_index"]) not in previous_indices
        ]

    for row in previous_selected:
        previous_counts[_group_key(row)] += 1

    additions_needed = target - len(selected)
    candidate_offsets = defaultdict(int)
    current_counts = defaultdict(int, previous_counts)
    while additions_needed > 0:
        eligible = [
            key
            for key in sorted(groups)
            if candidate_offsets[key] < len(candidates[key])
        ]
        if not eligible:
            break

        # Prefer the stratum furthest below its current Neyman target. Ties are
        # stable, so every later round appends to the exact prior sequence.
        key = min(
            eligible,
            key=lambda item: (
                -(quotas.get(item, 0) - current_counts[item]),
                item,
            ),
        )
        offset = candidate_offsets[key]
        selected.append(candidates[key][offset])
        candidate_offsets[key] += 1
        current_counts[key] += 1
        additions_needed -= 1

    if len(selected) != target:
        raise RuntimeError(
            f"Could only select {len(selected)} of {target} requested rows"
        )
    return selected


def _write_dataset_sample_from_census(config: dict, dataset: str, selected_census: list[dict], path: Path) -> None:
    dataset_cfg = resolve_dataset_config(config, dataset)
    original_rows = _read_csv(dataset_cfg["summary_csv"])
    original_fieldnames = list(original_rows[0].keys()) if original_rows else []
    original_by_index = {
        idx: row for idx, row in enumerate(original_rows)
    }
    sampled_original_rows = [
        original_by_index[int(row["row_index"])]
        for row in selected_census
    ]
    write_csv(path, sampled_original_rows, fieldnames=original_fieldnames)


def _allocation_audit(
    census_rows: list[dict],
    selected_rows: list[dict],
    target: int,
    allocation: str,
) -> dict:
    groups = defaultdict(list)
    for row in census_rows:
        groups[_group_key(row)].append(row)
    target_quotas = _allocate_quotas(
        groups, min(target, len(census_rows)), method=allocation
    )
    actual_counts = defaultdict(int)
    for row in selected_rows:
        actual_counts[_group_key(row)] += 1

    strata = []
    for key in sorted(groups):
        risk_stratum, event_type, risk_label, vru_present = key
        target_quota = int(target_quotas.get(key, 0))
        actual_count = int(actual_counts.get(key, 0))
        strata.append({
            "risk_stratum": risk_stratum,
            "event_type": event_type,
            "dataset_risk_label": risk_label,
            "vru_present": vru_present,
            "population_rows": len(groups[key]),
            "target_quota": target_quota,
            "actual_count": actual_count,
            "prefix_constraint_deviation": actual_count - target_quota,
        })
    return {
        "target_rows": min(target, len(census_rows)),
        "actual_rows": len(selected_rows),
        "max_abs_prefix_constraint_deviation": max(
            (
                abs(item["prefix_constraint_deviation"])
                for item in strata
            ),
            default=0,
        ),
        "strata": strata,
    }


def _config_for_round(base_config: dict, name: str, sample_dir: Path) -> dict:
    config = dict(base_config)
    config["name"] = name
    config.pop("dataset_registry_path", None)
    config.pop("dataset_registry", None)
    config["datasets"] = {
        "highd": {
            "summary_csv": str(sample_dir / "highd.csv"),
            "sequence_root": "data/highD/clips_multi_fixed_window",
        },
        "ind": {
            "summary_csv": str(sample_dir / "ind.csv"),
            "sequence_root": "data/inD/output_ind_risk_v4",
        },
        "round": {
            "summary_csv": str(sample_dir / "round.csv"),
            "sequence_root": "data/rounD/output_high_risk",
        },
    }
    return config


def build_sequential_plan(
    *,
    base_config_path: str,
    census_csv: str,
    out_dir: str,
    rounds: int,
    batch_per_dataset: int,
    seed: int,
    allocation: str,
) -> dict:
    base_config = load_config(base_config_path)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    census = _read_csv(census_csv)
    by_dataset = defaultdict(list)
    for row in census:
        by_dataset[row["dataset"]].append(row)

    manifest = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "base_config_path": base_config_path,
        "census_csv": census_csv,
        "rounds": rounds,
        "batch_per_dataset": batch_per_dataset,
        "seed": seed,
        "allocation": allocation,
        "rounds_detail": [],
    }

    previous_selected = defaultdict(list)
    for round_id in range(1, rounds + 1):
        cumulative_target = round_id * batch_per_dataset
        round_dir = out / f"round_{round_id:02d}"
        sample_dir = round_dir / "samples"
        sample_dir.mkdir(parents=True, exist_ok=True)

        round_detail = {
            "round_id": round_id,
            "cumulative_target_per_dataset": cumulative_target,
            "batch_target_per_dataset": batch_per_dataset,
            "samples": {},
        }

        for dataset in sorted(by_dataset):
            selected = _sample_incremental(
                by_dataset[dataset],
                cumulative_target,
                seed=seed,
                dataset=dataset,
                allocation=allocation,
                previous_selected=previous_selected[dataset],
            )
            previous_count = len(previous_selected[dataset])
            new_batch = selected[previous_count:]
            previous_selected[dataset] = list(selected)

            cumulative_census_path = sample_dir / f"{dataset}_cumulative_census.csv"
            batch_census_path = sample_dir / f"{dataset}_batch_census.csv"
            summary_path = sample_dir / f"{dataset}.csv"

            write_csv(cumulative_census_path, selected)
            write_csv(batch_census_path, new_batch)
            _write_dataset_sample_from_census(base_config, dataset, selected, summary_path)

            round_detail["samples"][dataset] = {
                "summary_csv": str(summary_path),
                "cumulative_census_csv": str(cumulative_census_path),
                "batch_census_csv": str(batch_census_path),
                "cumulative_rows": len(selected),
                "new_batch_rows": len(new_batch),
                "allocation_audit": _allocation_audit(
                    by_dataset[dataset],
                    selected,
                    cumulative_target,
                    allocation,
                ),
            }

        config_name = f"paper_sequential_round_{round_id:02d}"
        round_config = _config_for_round(base_config, config_name, sample_dir)
        config_path = round_dir / f"{config_name}.json"
        write_json(config_path, round_config)
        round_detail["config_path"] = str(config_path)
        manifest["rounds_detail"].append(round_detail)

    write_json(out / "sequential_plan_manifest.json", manifest)
    return manifest


def main():
    parser = argparse.ArgumentParser(description="Build sequential Neyman-stratified experiment rounds.")
    parser.add_argument("--base_config", default=DEFAULT_CONFIG)
    parser.add_argument("--census_csv", default=DEFAULT_CENSUS)
    parser.add_argument("--out_dir", default="data/eval_samples/sequential_v1")
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--batch_per_dataset", type=int, default=100)
    parser.add_argument("--allocation", default="neyman", choices=["neyman", "proportional"])
    parser.add_argument("--seed", type=int, default=20260613)
    args = parser.parse_args()

    manifest = build_sequential_plan(
        base_config_path=args.base_config,
        census_csv=args.census_csv,
        out_dir=args.out_dir,
        rounds=args.rounds,
        batch_per_dataset=args.batch_per_dataset,
        seed=args.seed,
        allocation=args.allocation,
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
