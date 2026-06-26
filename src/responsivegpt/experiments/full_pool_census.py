import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

from .config_loader import load_config
from .dataset_registry import resolve_dataset_config
from responsivegpt.interface.adapters.adapter_factory import build_event_adapter
from responsivegpt.interface.runner_core import derive_dataset_risk_label


DEFAULT_CONFIG = "src/responsivegpt/experiments/configs/paper_responsivegpt_main_token_saver.json"


def _to_float(value):
    if value is None or value == "":
        return None
    try:
        out = float(value)
    except Exception:
        return None
    if math.isnan(out) or math.isinf(out):
        return None
    return out


def _metric(metadata: dict, *names):
    for name in names:
        value = _to_float(metadata.get(name))
        if value is not None:
            return value
    return None


def _low_is_risky(value, danger: float, safe: float):
    if value is None:
        return None
    if value <= danger:
        return 1.0
    if value >= safe:
        return 0.0
    return (safe - value) / (safe - danger)


def _high_is_risky(value, safe: float, danger: float):
    if value is None:
        return None
    if value >= danger:
        return 1.0
    if value <= safe:
        return 0.0
    return (value - safe) / (danger - safe)


def _duration_bin(duration_s):
    if duration_s is None:
        return "unknown"
    if duration_s < 5:
        return "short_lt5s"
    if duration_s <= 12:
        return "fixed_window_5_12s"
    return "long_gt12s"


def _score_bin(score):
    if score >= 0.75:
        return "critical"
    if score >= 0.55:
        return "high"
    if score >= 0.35:
        return "medium"
    if score >= 0.20:
        return "borderline"
    return "low_candidate"


def _is_vru(metadata: dict) -> bool:
    text = " ".join(
        str(metadata.get(k, ""))
        for k in ("pair_type", "egoClass", "otherClass", "class_1", "class_2", "eventType")
    ).lower()
    return any(token in text for token in ("pedestrian", "bicycle", "cyclist"))


def _event_type(dataset: str, metadata: dict) -> str:
    if dataset == "highd":
        return str(metadata.get("eventType") or "unknown")
    if dataset == "ind":
        c1 = metadata.get("class_1") or "unknown"
        c2 = metadata.get("class_2") or "unknown"
        return f"{c1}_{c2}"
    if dataset == "round":
        return str(metadata.get("pair_type") or "unknown")
    return "unknown"


def _recording_id(metadata: dict):
    return metadata.get("recordingId") or metadata.get("recording_prefix") or metadata.get("prefix")


def _sequence_ref(metadata: dict) -> str:
    return str(metadata.get("clipPath") or metadata.get("scene_file") or metadata.get("clip_file") or "")


def _duration_s(dataset: str, metadata: dict):
    duration = _metric(metadata, "duration_s", "clip_duration_sec")
    if duration is not None:
        return duration
    duration_frames = _metric(metadata, "duration_frames", "clip_num_frames")
    fps = _metric(metadata, "fps")
    if duration_frames is not None and fps:
        return duration_frames / fps
    return None


def _risk_score(dataset: str, metadata: dict) -> tuple[float, dict]:
    min_ttc = _metric(metadata, "minTTC", "min_ttc")
    min_thw = _metric(metadata, "minTHW")
    min_distance = _metric(metadata, "minDHW", "min_center_distance", "min_distance")
    max_drac = _metric(metadata, "max_drac", "maxAbsDecel")
    min_dcpa = _metric(metadata, "min_dcpa")
    min_future = _metric(metadata, "min_future_rect_dist")
    scene_score = _metric(metadata, "scene_score", "risk_score_peak", "max_frame_score")
    rel_speed = _metric(metadata, "minRelSpeed", "max_rel_speed")

    components = {
        "ttc_score": _low_is_risky(min_ttc, danger=1.5, safe=5.0),
        "thw_score": _low_is_risky(min_thw, danger=0.5, safe=2.0),
        "distance_score": _low_is_risky(min_distance, danger=2.0, safe=8.0),
        "drac_score": _high_is_risky(max_drac, safe=3.0, danger=8.0),
        "dcpa_score": _low_is_risky(min_dcpa, danger=1.5, safe=6.0),
        "future_distance_score": _low_is_risky(min_future, danger=2.0, safe=8.0),
        "scene_score": _high_is_risky(scene_score, safe=0.25, danger=0.85),
        "rel_speed_score": _high_is_risky(abs(rel_speed) if rel_speed is not None else None, safe=2.0, danger=10.0),
    }

    dataset_weights = {
        "highd": ["ttc_score", "thw_score", "distance_score", "drac_score", "rel_speed_score"],
        "ind": ["ttc_score", "distance_score", "drac_score", "future_distance_score", "scene_score"],
        "round": ["ttc_score", "distance_score", "dcpa_score", "scene_score", "rel_speed_score"],
    }
    active = [components[name] for name in dataset_weights.get(dataset, []) if components.get(name) is not None]
    if not active:
        return 0.0, components

    max_component = max(active)
    avg_component = sum(active) / len(active)
    vru_bump = 0.05 if _is_vru(metadata) else 0.0
    score = min(1.0, 0.65 * max_component + 0.35 * avg_component + vru_bump)
    return round(score, 6), components


def _risk_args():
    return SimpleNamespace(ttc_threshold=3.0, distance_threshold=2.0, drac_threshold=8.0)


def build_census(config_path: str, datasets: list[str], out_dir: str) -> Path:
    config = load_config(config_path)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    rows = []
    dataset_counter = Counter()
    strata_counter = Counter()

    for dataset in datasets:
        dataset_cfg = resolve_dataset_config(config, dataset)
        adapter = build_event_adapter(dataset, dataset_cfg["summary_csv"])

        for row_index, row in enumerate(adapter.iter_rows()):
            metadata = adapter.row_metadata(row)
            score, components = _risk_score(dataset, metadata)
            risk_label = derive_dataset_risk_label(dataset, row, _risk_args())
            duration_s = _duration_s(dataset, metadata)
            event_type = _event_type(dataset, metadata)
            stratum = _score_bin(score)
            vru_present = _is_vru(metadata)

            record = {
                "dataset": dataset,
                "row_index": row_index,
                "recording_id": _recording_id(metadata),
                "event_type": event_type,
                "dataset_risk_label": int(bool(risk_label)),
                "risk_stratum": stratum,
                "deterministic_risk_score": score,
                "duration_s": "" if duration_s is None else round(duration_s, 6),
                "duration_bin": _duration_bin(duration_s),
                "vru_present": int(vru_present),
                "sequence_ref": _sequence_ref(metadata),
                "start_frame": metadata.get("startFrame") or metadata.get("start_frame"),
                "end_frame": metadata.get("endFrame") or metadata.get("end_frame"),
                "min_ttc_s": _metric(metadata, "minTTC", "min_ttc"),
                "min_thw_s": _metric(metadata, "minTHW"),
                "min_distance_m": _metric(metadata, "minDHW", "min_center_distance", "min_distance"),
                "max_drac_mps2": _metric(metadata, "max_drac", "maxAbsDecel"),
                "min_dcpa_m": _metric(metadata, "min_dcpa"),
                "min_future_distance_m": _metric(metadata, "min_future_rect_dist"),
                "scenario_score": _metric(metadata, "scene_score", "risk_score_peak", "max_frame_score"),
                "component_scores_json": json.dumps(components, ensure_ascii=False, sort_keys=True),
                "source_summary_csv": dataset_cfg["summary_csv"],
                "sequence_root": dataset_cfg["sequence_root"],
            }
            rows.append(record)
            dataset_counter[dataset] += 1
            strata_counter[(dataset, stratum, event_type, int(bool(risk_label)), int(vru_present))] += 1

    census_path = out / "full_pool_episode_census.csv"
    fieldnames = list(rows[0].keys()) if rows else []
    with census_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    dataset_summary_path = out / "full_pool_dataset_summary.csv"
    with dataset_summary_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["dataset", "num_rows"])
        writer.writeheader()
        for dataset, count in sorted(dataset_counter.items()):
            writer.writerow({"dataset": dataset, "num_rows": count})

    strata_summary_path = out / "full_pool_strata_summary.csv"
    with strata_summary_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["dataset", "risk_stratum", "event_type", "dataset_risk_label", "vru_present", "num_rows"],
        )
        writer.writeheader()
        for key, count in sorted(strata_counter.items()):
            dataset, stratum, event_type, risk_label, vru_present = key
            writer.writerow({
                "dataset": dataset,
                "risk_stratum": stratum,
                "event_type": event_type,
                "dataset_risk_label": risk_label,
                "vru_present": vru_present,
                "num_rows": count,
            })

    manifest = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "config_path": str(config_path),
        "datasets": datasets,
        "total_rows": len(rows),
        "outputs": {
            "census_csv": str(census_path),
            "dataset_summary_csv": str(dataset_summary_path),
            "strata_summary_csv": str(strata_summary_path),
        },
        "risk_score_definition": {
            "purpose": "Deterministic full-pool triage within high-risk strong-interaction candidates.",
            "strata": {
                "critical": "score >= 0.75",
                "high": "0.55 <= score < 0.75",
                "medium": "0.35 <= score < 0.55",
                "borderline": "0.20 <= score < 0.35",
                "low_candidate": "score < 0.20",
            },
        },
    }
    with (out / "full_pool_census_manifest.json").open("w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    return census_path


def main():
    parser = argparse.ArgumentParser(description="Build a deterministic census of the full high-risk candidate pool.")
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--datasets", default="highd,ind,round")
    parser.add_argument("--out_dir", default="data/full_pool_census/cornercase_v1")
    args = parser.parse_args()

    datasets = [item.strip() for item in args.datasets.split(",") if item.strip()]
    census_path = build_census(args.config, datasets, args.out_dir)
    print(f"Full-pool census saved to: {census_path}")


if __name__ == "__main__":
    main()
