import csv
import os
from typing import Dict, Any, Iterator, List, Optional

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


def _to_bool(v):
    if isinstance(v, bool):
        return v
    return str(v).strip().lower() in ["true", "1", "yes"]


class RoundClipSequenceAdapter:
    """
    读取单个 rounD clip CSV，并按 frame 逐帧产出 SceneState。
    每帧选择一个最关键交互对象：
      1) 优先 primary actor 且非 ego
      2) 否则选 distance_to_ego 最小的非 ego 对象
    """

    def __init__(self, clip_csv_path: str):
        self.clip_csv_path = clip_csv_path

    def _load_rows(self) -> List[dict]:
        with open(self.clip_csv_path, "r", encoding="utf-8") as f:
            return list(csv.DictReader(f))

    def _group_by_frame(self, rows: List[dict]) -> Dict[int, List[dict]]:
        out: Dict[int, List[dict]] = {}
        for row in rows:
            frame = _to_int(row.get("frame"))
            if frame is None:
                continue
            out.setdefault(frame, []).append(row)
        return out

    def _pick_target_row(self, frame_rows: List[dict]) -> Optional[dict]:
        non_ego = [r for r in frame_rows if not _to_bool(r.get("is_ego"))]
        if not non_ego:
            return None

        primary = [
            r for r in non_ego
            if _to_bool(r.get("is_primary_actor"))
        ]
        if primary:
            primary.sort(key=lambda r: _to_float(r.get("distance_to_ego"), 1e18))
            return primary[0]

        non_ego.sort(key=lambda r: _to_float(r.get("distance_to_ego"), 1e18))
        return non_ego[0]

    def iter_scenes(self) -> Iterator[SceneState]:
        rows = self._load_rows()
        grouped = self._group_by_frame(rows)

        for frame in sorted(grouped.keys()):
            frame_rows = grouped[frame]
            target = self._pick_target_row(frame_rows)
            if target is None:
                continue

            pair_type = target.get("pair_type", "")
            vrus_present = ("pedestrian" in pair_type) or ("cyclist" in pair_type)

            yield SceneState(
                scene_type="rounD",
                ego_speed_mps=_to_float(target.get("ego_speed_mps"), 0.0),
                headway_m=_to_float(target.get("distance_to_ego"), 0.0),
                lane_change=False,

                # 环岛场景，不依赖红绿灯
                dist_to_intersection_m=0.0,
                traffic_light="none",
                vrus_present=vrus_present,

                lead_speed_mps=None,
                rel_speed_mps=_to_float(target.get("rel_speed_to_ego")),

                ego_x=_to_float(target.get("ego_xCenter")),
                ego_y=_to_float(target.get("ego_yCenter")),
                other_x=_to_float(target.get("xCenter")),
                other_y=_to_float(target.get("yCenter")),

                event_type=pair_type,
                frame_index=_to_int(target.get("frame")),
                duration_s=None,

                min_ttc_raw=_to_float(target.get("event_min_ttc")),
                min_thw_raw=None,
                min_dhw_raw=_to_float(target.get("event_min_distance")),
            )

    def clip_metadata(self) -> dict:
        rows = self._load_rows()
        if not rows:
            return {}

        r = rows[0]
        return {
            "clip_csv_path": self.clip_csv_path,
            "recordingId": _to_int(r.get("recordingId")),
            "event_id": r.get("event_id"),
            "egoTrackId": _to_int(r.get("egoTrackId")),
            "pair_type": r.get("pair_type"),
            "event_peak_frame": _to_int(r.get("event_peak_frame")),
            "event_start_frame": _to_int(r.get("event_start_frame")),
            "event_end_frame": _to_int(r.get("event_end_frame")),
            "event_min_ttc": _to_float(r.get("event_min_ttc")),
            "event_min_dcpa": _to_float(r.get("event_min_dcpa")),
            "event_min_distance": _to_float(r.get("event_min_distance")),
            "risk_score_peak": _to_float(r.get("risk_score_peak")),
        }