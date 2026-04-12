import csv
from typing import Iterator, Dict, Any
from ...domain.models import SceneState


def _to_float(v, default=None):
    if v is None or v == "":
        return default
    try:
        return float(v)
    except Exception:
        return default


def _to_int(v, default=None):
    if v is None or v == "":
        return default
    try:
        return int(float(v))
    except Exception:
        return default


class InDEventAdapter:
    """
    把 inD 风险事件 summary 每一行映射成一个事件级 SceneState。
    """

    def __init__(self, csv_path: str):
        self.csv_path = csv_path

    def iter_rows(self) -> Iterator[dict]:
        with open(self.csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                yield row

    def row_to_scene(self, row: Dict[str, Any]) -> SceneState:
        cls1 = str(row.get("class_1", "")).lower()
        cls2 = str(row.get("class_2", "")).lower()
        vrus_present = ("pedestrian" in [cls1, cls2]) or ("bicycle" in [cls1, cls2]) or ("cyclist" in [cls1, cls2])

        duration_frames = _to_float(row.get("duration_frames"), 0.0)
        fps = _to_float(row.get("fps"), 25.0)
        duration_s = (duration_frames / fps) if fps else None

        return SceneState(
            scene_type="inD",
            ego_speed_mps=_to_float(row.get("ego_mean_speed"), 0.0),
            headway_m=_to_float(row.get("min_center_distance"), 0.0),
            lane_change=False,

            # inD 是交叉口场景
            dist_to_intersection_m=0.0,
            traffic_light="unknown",
            vrus_present=vrus_present,

            lead_speed_mps=None,
            rel_speed_mps=None,

            ego_x=_to_float(row.get("ego_start_x")),
            ego_y=_to_float(row.get("ego_start_y")),
            other_x=None,
            other_y=None,

            event_type=f"{row.get('class_1')}_{row.get('class_2')}",
            frame_index=None,
            duration_s=duration_s,

            min_ttc_raw=_to_float(row.get("min_ttc")),
            min_thw_raw=None,
            min_dhw_raw=_to_float(row.get("min_center_distance")),
        )

    def row_metadata(self, row: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "recording_prefix": _to_int(row.get("recording_prefix")),
            "recordingId": _to_int(row.get("recordingId")),
            "location_id": _to_int(row.get("location_id")),
            "trackId_1": _to_int(row.get("trackId_1")),
            "trackId_2": _to_int(row.get("trackId_2")),
            "class_1": row.get("class_1"),
            "class_2": row.get("class_2"),
            "ego_idx": _to_int(row.get("ego_idx")),
            "ego_track_id": _to_int(row.get("ego_track_id")),
            "start_frame": _to_int(row.get("start_frame")),
            "end_frame": _to_int(row.get("end_frame")),
            "duration_frames": _to_int(row.get("duration_frames")),
            "fps": _to_float(row.get("fps")),
            "min_center_distance": _to_float(row.get("min_center_distance")),
            "min_ttc": _to_float(row.get("min_ttc")),
            "max_drac": _to_float(row.get("max_drac")),
            "min_pet_like": _to_float(row.get("min_pet_like")),
            "min_future_rect_dist": _to_float(row.get("min_future_rect_dist")),
            "max_frame_score": _to_float(row.get("max_frame_score")),
            "mean_frame_score": _to_float(row.get("mean_frame_score")),
            "scene_score": _to_float(row.get("scene_score")),
            "conflict_x": _to_float(row.get("conflict_x")),
            "conflict_y": _to_float(row.get("conflict_y")),
            "ego_mean_speed": _to_float(row.get("ego_mean_speed")),
            "ego_max_speed": _to_float(row.get("ego_max_speed")),
            "ego_min_speed": _to_float(row.get("ego_min_speed")),
            "ego_mean_acc": _to_float(row.get("ego_mean_acc")),
            "ego_max_acc": _to_float(row.get("ego_max_acc")),
            "ego_mean_heading": _to_float(row.get("ego_mean_heading")),
            "ego_start_x": _to_float(row.get("ego_start_x")),
            "ego_start_y": _to_float(row.get("ego_start_y")),
            "ego_end_x": _to_float(row.get("ego_end_x")),
            "ego_end_y": _to_float(row.get("ego_end_y")),
            "scene_file": row.get("scene_file"),
        }