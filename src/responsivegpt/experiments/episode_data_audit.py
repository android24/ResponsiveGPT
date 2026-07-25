import argparse
import csv
import json
from pathlib import Path
from types import SimpleNamespace

from .config_loader import load_config
from .dataset_registry import resolve_dataset_config
from .io_utils import ensure_dir, write_csv, write_json
from ..interface.adapters.adapter_factory import build_event_adapter, build_sequence_adapter


DEFAULT_CONFIG = "src/responsivegpt/experiments/configs/paper_fullpool_census_base.json"
DEFAULT_OUT_DIR = "data/episode_audit"


def _dataset_names(config: dict, requested: str | None) -> list[str]:
    if requested:
        return [x.strip() for x in requested.split(",") if x.strip()]

    matrix_datasets = (config.get("matrix") or {}).get("datasets")
    if matrix_datasets:
        return list(matrix_datasets)

    datasets = config.get("datasets", {})
    if isinstance(datasets, list):
        return list(datasets)
    return list(datasets.keys())


def _read_csv_rows(path: str | Path) -> tuple[list[dict], list[str]]:
    with Path(path).open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        return rows, list(reader.fieldnames or [])


def _write_available_summary(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    output_fields = list(fieldnames)
    if "_resolved_sequence_path" not in output_fields:
        output_fields.append("_resolved_sequence_path")
    write_csv(path, rows, fieldnames=output_fields)


def _to_int(value, default=None):
    if value is None or value == "":
        return default
    try:
        return int(float(value))
    except Exception:
        return default


def _safe_ratio(num, den):
    if not den:
        return None
    return num / den


def _mean(values):
    values = [v for v in values if isinstance(v, (int, float))]
    return sum(values) / len(values) if values else None


def _raw_sequence_stats(path: str | Path) -> dict:
    rows, _ = _read_csv_rows(path)
    frame_counts = {}
    track_ids = set()
    ego_rows = 0
    primary_rows = 0

    for row in rows:
        frame = _to_int(row.get("frame"))
        if frame is not None:
            frame_counts[frame] = frame_counts.get(frame, 0) + 1

        track_id = row.get("trackId") or row.get("vehicleId")
        if track_id not in (None, ""):
            track_ids.add(str(track_id))

        if str(row.get("is_ego", "")).strip().lower() in {"1", "true", "yes"}:
            ego_rows += 1
        if str(row.get("is_primary_actor", "")).strip().lower() in {"1", "true", "yes"}:
            primary_rows += 1

    frames = sorted(frame_counts)
    return {
        "raw_rows": len(rows),
        "raw_frame_count": len(frames),
        "raw_frame_min": frames[0] if frames else None,
        "raw_frame_max": frames[-1] if frames else None,
        "raw_avg_objects_per_frame": _mean(list(frame_counts.values())),
        "raw_max_objects_per_frame": max(frame_counts.values()) if frame_counts else None,
        "raw_unique_track_count": len(track_ids),
        "raw_ego_rows": ego_rows,
        "raw_primary_actor_rows": primary_rows,
    }


def _event_window(metadata: dict) -> tuple[int | None, int | None]:
    start_keys = ("startFrame", "start_frame")
    end_keys = ("endFrame", "end_frame")

    event_start = None
    event_end = None
    for key in start_keys:
        if key in metadata:
            event_start = _to_int(metadata.get(key))
            break
    for key in end_keys:
        if key in metadata:
            event_end = _to_int(metadata.get(key))
            break
    return event_start, event_end


def _sample_indices(total: int, sample_size: int) -> set[int]:
    if sample_size <= 0 or total <= sample_size:
        return set(range(total))
    if sample_size == 1:
        return {0}
    return {
        round(i * (total - 1) / (sample_size - 1))
        for i in range(sample_size)
    }


def _coverage_record(dataset: str, row_index: int, metadata: dict, sequence_path: str, sequence_adapter) -> dict:
    raw = _raw_sequence_stats(sequence_path)
    scenes = list(sequence_adapter.iter_scenes())
    usable_frames = [
        s.frame_index
        for s in scenes
        if s.frame_index is not None
    ]
    usable_min = min(usable_frames) if usable_frames else None
    usable_max = max(usable_frames) if usable_frames else None
    event_start, event_end = _event_window(metadata)

    return {
        "dataset": dataset,
        "row_index": row_index,
        "sequence_path": sequence_path,
        "event_start_frame": event_start,
        "event_end_frame": event_end,
        "raw_frame_count": raw["raw_frame_count"],
        "raw_frame_min": raw["raw_frame_min"],
        "raw_frame_max": raw["raw_frame_max"],
        "raw_pre_event_frames": (
            event_start - raw["raw_frame_min"]
            if event_start is not None and raw["raw_frame_min"] is not None
            else None
        ),
        "raw_post_event_frames": (
            raw["raw_frame_max"] - event_end
            if event_end is not None and raw["raw_frame_max"] is not None
            else None
        ),
        "usable_frame_count": len(scenes),
        "usable_frame_min": usable_min,
        "usable_frame_max": usable_max,
        "usable_pre_event_frames": (
            event_start - usable_min
            if event_start is not None and usable_min is not None
            else None
        ),
        "usable_post_event_frames": (
            usable_max - event_end
            if event_end is not None and usable_max is not None
            else None
        ),
        "usable_to_raw_frame_ratio": _safe_ratio(len(scenes), raw["raw_frame_count"]),
        "raw_rows": raw["raw_rows"],
        "raw_avg_objects_per_frame": raw["raw_avg_objects_per_frame"],
        "raw_max_objects_per_frame": raw["raw_max_objects_per_frame"],
        "raw_unique_track_count": raw["raw_unique_track_count"],
        "raw_ego_rows": raw["raw_ego_rows"],
        "raw_primary_actor_rows": raw["raw_primary_actor_rows"],
        "event_type": metadata.get("eventType") or metadata.get("pair_type"),
        "scene_file": metadata.get("scene_file"),
        "clip_file": metadata.get("clip_file"),
        "clipPath": metadata.get("clipPath"),
    }


def audit_dataset(
    *,
    dataset: str,
    summary_csv: str,
    sequence_root: str,
    out_dir: Path,
    coverage_sample: int,
    write_available: bool,
) -> dict:
    rows, fieldnames = _read_csv_rows(summary_csv)
    event_adapter = build_event_adapter(dataset, summary_csv)
    args = SimpleNamespace(sequence_root=sequence_root)

    available_rows = []
    missing_examples = []
    coverage_rows = []
    empty_sequences = 0
    available_records = []

    for row_index, row in enumerate(rows):
        metadata = event_adapter.row_metadata(row)
        sequence_adapter, resolved_path, missing_key = build_sequence_adapter(dataset, metadata, args)

        if missing_key:
            if len(missing_examples) < 50:
                missing_examples.append({
                    "dataset": dataset,
                    "row_index": row_index,
                    "missing_key": missing_key,
                    "resolved_path": resolved_path,
                    "sequence_ref": (
                        metadata.get("clipPath")
                        or metadata.get("clip_file")
                        or metadata.get("scene_file")
                    ),
                })
            continue

        available_row = dict(row)
        available_row["_resolved_sequence_path"] = resolved_path
        available_rows.append(available_row)
        available_records.append((row_index, metadata, resolved_path, sequence_adapter))

    sample_indices = _sample_indices(len(available_records), coverage_sample)
    for available_index, (row_index, metadata, resolved_path, sequence_adapter) in enumerate(available_records):
        if available_index not in sample_indices:
            continue
        record = _coverage_record(dataset, row_index, metadata, resolved_path, sequence_adapter)
        if record["usable_frame_count"] <= 0:
            empty_sequences += 1
        coverage_rows.append(record)

    if write_available:
        _write_available_summary(
            out_dir / f"{dataset}_episode_available_summary.csv",
            available_rows,
            fieldnames,
        )

    missing_count = len(rows) - len(available_rows)
    summary = {
        "dataset": dataset,
        "summary_csv": summary_csv,
        "sequence_root": sequence_root,
        "total_rows": len(rows),
        "available_rows": len(available_rows),
        "missing_sequences": missing_count,
        "availability_rate": _safe_ratio(len(available_rows), len(rows)),
        "coverage_sample_size": len(coverage_rows),
        "coverage_empty_sequences": empty_sequences,
        "avg_raw_frame_count": _mean([r["raw_frame_count"] for r in coverage_rows]),
        "avg_usable_frame_count": _mean([r["usable_frame_count"] for r in coverage_rows]),
        "avg_usable_to_raw_frame_ratio": _mean([r["usable_to_raw_frame_ratio"] for r in coverage_rows]),
        "avg_raw_pre_event_frames": _mean([r["raw_pre_event_frames"] for r in coverage_rows]),
        "avg_raw_post_event_frames": _mean([r["raw_post_event_frames"] for r in coverage_rows]),
        "avg_usable_pre_event_frames": _mean([r["usable_pre_event_frames"] for r in coverage_rows]),
        "avg_usable_post_event_frames": _mean([r["usable_post_event_frames"] for r in coverage_rows]),
        "avg_raw_objects_per_frame": _mean([r["raw_avg_objects_per_frame"] for r in coverage_rows]),
        "avg_raw_unique_track_count": _mean([r["raw_unique_track_count"] for r in coverage_rows]),
    }

    return {
        "summary": summary,
        "missing_examples": missing_examples,
        "coverage_rows": coverage_rows,
    }


def run_audit(
    *,
    config_path: str,
    datasets: str | None,
    out_dir: str,
    coverage_sample: int,
    write_available: bool,
) -> list[dict]:
    config = load_config(config_path)
    out_path = ensure_dir(out_dir)

    summaries = []
    missing_examples = []
    coverage_rows = []

    for dataset in _dataset_names(config, datasets):
        dataset_cfg = resolve_dataset_config(config, dataset)
        result = audit_dataset(
            dataset=dataset,
            summary_csv=dataset_cfg["summary_csv"],
            sequence_root=dataset_cfg["sequence_root"],
            out_dir=out_path,
            coverage_sample=coverage_sample,
            write_available=write_available,
        )
        summaries.append(result["summary"])
        missing_examples.extend(result["missing_examples"])
        coverage_rows.extend(result["coverage_rows"])

    write_csv(out_path / "episode_availability_summary.csv", summaries)
    write_csv(out_path / "episode_missing_examples.csv", missing_examples)
    write_csv(out_path / "episode_window_coverage_sample.csv", coverage_rows)
    write_json(out_path / "episode_audit_manifest.json", {
        "config_path": config_path,
        "datasets": _dataset_names(config, datasets),
        "coverage_sample": coverage_sample,
        "write_available": write_available,
        "outputs": {
            "availability_summary": str(out_path / "episode_availability_summary.csv"),
            "missing_examples": str(out_path / "episode_missing_examples.csv"),
            "coverage_sample": str(out_path / "episode_window_coverage_sample.csv"),
        },
    })
    return summaries


def main():
    parser = argparse.ArgumentParser(description="Audit episode-level summary-to-clip coverage.")
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--datasets", default=None, help="Comma-separated dataset names. Defaults to config datasets.")
    parser.add_argument("--out_dir", default=DEFAULT_OUT_DIR)
    parser.add_argument("--coverage_sample", type=int, default=100)
    parser.add_argument("--write_available", type=int, default=1)
    args = parser.parse_args()

    summaries = run_audit(
        config_path=args.config,
        datasets=args.datasets,
        out_dir=args.out_dir,
        coverage_sample=args.coverage_sample,
        write_available=bool(args.write_available),
    )
    print(json.dumps(summaries, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
