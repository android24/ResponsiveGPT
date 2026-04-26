import csv
from typing import Iterator, Dict, Any
from ...domain.models import SceneState
from .base_event_adapter import BaseEventAdapter


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


class RoundEventAdapter(BaseEventAdapter):
    """
    把 rounD 高风险事件 summary 的每一行映射成一个事件级 SceneState。
    """

    def __init__(self, csv_path: str):
        self.csv_path = csv_path

    def iter_rows(self) -> Iterator[dict]:
        with open(self.csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                yield row

    def row_to_scene(self, row: Dict[str, Any]) -> SceneState:
        pair_type = row.get("pair_type", "")
        pair_type_l = str(pair_type).lower()
        vrus_present = (
            "cyclist" in pair_type_l
            or "pedestrian" in pair_type_l
            or "bicycle" in pair_type_l
        )

        ego_speed = _to_float(row.get("ego_speed_at_peak_mps"), 0.0)
        rel_speed = _to_float(row.get("other_rel_speed_to_ego_at_peak"))

        return SceneState(
            scene_type="rounD",
            ego_speed_mps=ego_speed,
            headway_m=_to_float(row.get("min_distance"), 0.0),
            lane_change=False,

            # 环岛里不依赖红绿灯
            dist_to_intersection_m=0.0,
            traffic_light="none",
            vrus_present=vrus_present,

            lead_speed_mps=None,
            rel_speed_mps=rel_speed,

            ego_x=None,
            ego_y=None,
            other_x=None,
            other_y=None,

            # batch/event-level 没有二维方向时，只给 ego 速度近似
            ego_vx=ego_speed,
            ego_vy=0.0,
            other_vx=None,
            other_vy=None,

            event_type=row.get("pair_type"),
            frame_index=None,
            duration_s=_to_float(row.get("clip_duration_sec")),

            min_ttc_raw=_to_float(row.get("min_ttc")),
            min_thw_raw=None,
            min_dhw_raw=_to_float(row.get("min_distance")),
        )

    def row_metadata(self, row: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "dataset": "rounD",
            "prefix": _to_int(row.get("prefix")),
            "recordingId": _to_int(row.get("recordingId")),
            "trackId_1": _to_int(row.get("trackId_1")),
            "trackId_2": _to_int(row.get("trackId_2")),
            "egoTrackId": _to_int(row.get("egoTrackId")),
            "otherTrackId": _to_int(row.get("otherTrackId")),
            "event_id": row.get("event_id"),
            "pair_type": row.get("pair_type"),
            "egoClass": row.get("egoClass"),
            "otherClass": row.get("otherClass"),
            "start_frame": _to_int(row.get("start_frame")),
            "end_frame": _to_int(row.get("end_frame")),
            "peak_frame": _to_int(row.get("peak_frame")),
            "start_time_sec": _to_float(row.get("start_time_sec")),
            "end_time_sec": _to_float(row.get("end_time_sec")),
            "peak_time_sec": _to_float(row.get("peak_time_sec")),
            "clip_file": row.get("clip_file"),
            "clip_start_frame": _to_int(row.get("clip_start_frame")),
            "clip_end_frame": _to_int(row.get("clip_end_frame")),
            "clip_num_frames": _to_int(row.get("clip_num_frames")),
            "clip_duration_sec": _to_float(row.get("clip_duration_sec")),
            "min_distance": _to_float(row.get("min_distance")),
            "min_ttc": _to_float(row.get("min_ttc")),
            "min_dcpa": _to_float(row.get("min_dcpa")),
            "max_rel_speed": _to_float(row.get("max_rel_speed")),
            "risk_score_peak": _to_float(row.get("risk_score_peak")),
            "ego_speed_at_peak_mps": _to_float(row.get("ego_speed_at_peak_mps")),
            "ego_mean_speed_in_clip_mps": _to_float(row.get("ego_mean_speed_in_clip_mps")),
            "ego_min_speed_in_clip_mps": _to_float(row.get("ego_min_speed_in_clip_mps")),
            "ego_max_speed_in_clip_mps": _to_float(row.get("ego_max_speed_in_clip_mps")),
            "other_distance_to_ego_at_peak": _to_float(row.get("other_distance_to_ego_at_peak")),
            "other_rel_speed_to_ego_at_peak": _to_float(row.get("other_rel_speed_to_ego_at_peak")),
            "num_context_objects": _to_int(row.get("num_context_objects")),
            "num_primary_objects": _to_int(row.get("num_primary_objects")),
        }
    
    def derive_risk_label(self, row: Dict[str, Any]) -> bool:
        min_ttc = _to_float(row.get("min_ttc"))
        min_distance = _to_float(row.get("min_distance"))

        if min_ttc is not None and min_ttc < 3.0:
            return True
        if min_distance is not None and min_distance < 2.0:
            return True
        return False

    def get_sequence_ref(self, metadata: dict) -> str | None:
        return metadata.get("clip_file")