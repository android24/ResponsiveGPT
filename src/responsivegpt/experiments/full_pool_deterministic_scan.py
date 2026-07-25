import argparse
import csv
import json
from collections import Counter
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

from .config_loader import load_config
from .dataset_registry import resolve_dataset_config
from .io_utils import write_csv, write_json
from responsivegpt.evaluation.safety_metrics import (
    aggregate_episode_safety_metrics,
    compute_frame_safety_metrics,
    thresholds_for_dataset,
)
from responsivegpt.interface.adapters.adapter_factory import build_event_adapter, build_sequence_adapter
from responsivegpt.interface.runner_core import derive_dataset_risk_label


DEFAULT_CONFIG = "src/responsivegpt/experiments/configs/paper_fullpool_census_base.json"

FIELDS = [
    "dataset",
    "row_index",
    "recording_id",
    "event_type",
    "sequence_path",
    "dataset_risk_label",
    "num_frames",
    "min_ttc_s",
    "avg_ttc_s",
    "min_thw_s",
    "avg_thw_s",
    "max_drac_mps2",
    "min_dcpa_m",
    "min_future_distance_m",
    "unsafe_ttc_ratio",
    "unsafe_thw_ratio",
    "unsafe_drac_ratio",
    "unsafe_dcpa_ratio",
    "unsafe_future_distance_ratio",
    "physical_risk_exposure",
    "max_physical_risk_index",
    "has_critical_ttc",
    "has_critical_drac",
    "has_critical_spatial_risk",
    "scan_status",
    "missing_key",
]


def _risk_args():
    return SimpleNamespace(ttc_threshold=3.0, distance_threshold=2.0, drac_threshold=8.0)


def _recording_id(metadata: dict):
    return metadata.get("recordingId") or metadata.get("recording_prefix") or metadata.get("prefix")


def _event_type(dataset: str, metadata: dict) -> str:
    if dataset == "ind":
        return f"{metadata.get('class_1')}_{metadata.get('class_2')}"
    return str(metadata.get("eventType") or metadata.get("pair_type") or "unknown")


def _row_from_episode(dataset: str, row_index: int, metadata: dict, sequence_path: str, risk_label: bool, episode_safety) -> dict:
    values = asdict(episode_safety)
    return {
        "dataset": dataset,
        "row_index": row_index,
        "recording_id": _recording_id(metadata),
        "event_type": _event_type(dataset, metadata),
        "sequence_path": sequence_path,
        "dataset_risk_label": int(bool(risk_label)),
        "num_frames": values.get("num_frames"),
        "min_ttc_s": values.get("min_ttc_s"),
        "avg_ttc_s": values.get("avg_ttc_s"),
        "min_thw_s": values.get("min_thw_s"),
        "avg_thw_s": values.get("avg_thw_s"),
        "max_drac_mps2": values.get("max_drac_mps2"),
        "min_dcpa_m": values.get("min_dcpa_m"),
        "min_future_distance_m": values.get("min_future_distance_m"),
        "unsafe_ttc_ratio": values.get("unsafe_ttc_ratio"),
        "unsafe_thw_ratio": values.get("unsafe_thw_ratio"),
        "unsafe_drac_ratio": values.get("unsafe_drac_ratio"),
        "unsafe_dcpa_ratio": values.get("unsafe_dcpa_ratio"),
        "unsafe_future_distance_ratio": values.get("unsafe_future_distance_ratio"),
        "physical_risk_exposure": values.get("physical_risk_exposure"),
        "max_physical_risk_index": values.get("max_physical_risk_index"),
        "has_critical_ttc": int(bool(values.get("has_critical_ttc"))),
        "has_critical_drac": int(bool(values.get("has_critical_drac"))),
        "has_critical_spatial_risk": int(bool(values.get("has_critical_spatial_risk"))),
        "scan_status": "ok",
        "missing_key": "",
    }


def scan_dataset(config: dict, dataset: str, *, limit: int = 0, start_index: int = 0, end_index: int = -1) -> list[dict]:
    dataset_cfg = resolve_dataset_config(config, dataset)
    event_adapter = build_event_adapter(dataset, dataset_cfg["summary_csv"])
    args = SimpleNamespace(
        dataset=dataset,
        sequence_root=dataset_cfg["sequence_root"],
        ttc_threshold=3.0,
        distance_threshold=2.0,
        drac_threshold=8.0,
    )
    thresholds = thresholds_for_dataset(dataset)
    rows = []
    processed = 0

    for row_index, row in enumerate(event_adapter.iter_rows()):
        if row_index < start_index:
            continue
        if end_index >= 0 and row_index >= end_index:
            break
        if limit > 0 and processed >= limit:
            break

        metadata = event_adapter.row_metadata(row)
        risk_label = derive_dataset_risk_label(dataset, row, _risk_args())
        seq_adapter, sequence_path, missing_key = build_sequence_adapter(dataset, metadata, args)
        if seq_adapter is None:
            rows.append({
                "dataset": dataset,
                "row_index": row_index,
                "recording_id": _recording_id(metadata),
                "event_type": _event_type(dataset, metadata),
                "sequence_path": sequence_path,
                "dataset_risk_label": int(bool(risk_label)),
                "scan_status": "missing_sequence",
                "missing_key": missing_key or "",
            })
            processed += 1
            continue

        scenes = list(seq_adapter.iter_scenes())
        if not scenes:
            rows.append({
                "dataset": dataset,
                "row_index": row_index,
                "recording_id": _recording_id(metadata),
                "event_type": _event_type(dataset, metadata),
                "sequence_path": sequence_path,
                "dataset_risk_label": int(bool(risk_label)),
                "scan_status": "empty_sequence",
                "missing_key": "",
            })
            processed += 1
            continue

        frame_safety = [compute_frame_safety_metrics(scene, thresholds) for scene in scenes]
        episode_safety = aggregate_episode_safety_metrics(frame_safety)
        rows.append(_row_from_episode(dataset, row_index, metadata, sequence_path, risk_label, episode_safety))
        processed += 1

    return rows


def build_scan(config_path: str, datasets: list[str], out_dir: str, *, limit: int = 0, start_index: int = 0, end_index: int = -1) -> dict:
    config = load_config(config_path)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    all_rows = []
    summary_rows = []
    for dataset in datasets:
        rows = scan_dataset(config, dataset, limit=limit, start_index=start_index, end_index=end_index)
        all_rows.extend(rows)
        counts = Counter(row.get("scan_status", "") for row in rows)
        summary_rows.append({
            "dataset": dataset,
            "num_rows": len(rows),
            "ok_rows": counts.get("ok", 0),
            "missing_sequence_rows": counts.get("missing_sequence", 0),
            "empty_sequence_rows": counts.get("empty_sequence", 0),
        })

    scan_path = out / "full_pool_deterministic_scan.csv"
    summary_path = out / "full_pool_deterministic_scan_summary.csv"
    write_csv(scan_path, all_rows, fieldnames=FIELDS)
    write_csv(summary_path, summary_rows)

    manifest = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "config_path": str(config_path),
        "datasets": datasets,
        "limit": limit,
        "start_index": start_index,
        "end_index": end_index,
        "outputs": {
            "scan_csv": str(scan_path),
            "summary_csv": str(summary_path),
        },
        "description": "Full-window deterministic scan over fixed-duration multi-agent clips/scenes. No LLM calls.",
    }
    write_json(out / "full_pool_deterministic_scan_manifest.json", manifest)
    return manifest


def main():
    parser = argparse.ArgumentParser(description="Run a deterministic full-window safety scan over the episode pool.")
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--datasets", default="highd,ind,round")
    parser.add_argument("--out_dir", default="data/full_pool_deterministic_scan/cornercase_v1")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--start_index", type=int, default=0)
    parser.add_argument("--end_index", type=int, default=-1)
    args = parser.parse_args()

    datasets = [item.strip() for item in args.datasets.split(",") if item.strip()]
    manifest = build_scan(
        args.config,
        datasets,
        args.out_dir,
        limit=args.limit,
        start_index=args.start_index,
        end_index=args.end_index,
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
