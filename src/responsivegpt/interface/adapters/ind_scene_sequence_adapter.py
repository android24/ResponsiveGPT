import csv
import math
from typing import Dict, Iterator, List, Optional

from ...domain.models import SceneState
from .base_sequence_adapter import BaseSequenceAdapter


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

def _first_float(row: dict, keys: List[str], default=None):
    for k in keys:
        val = _to_float(row.get(k), None)
        if val is not None:
            return val
    return default

def _speed_norm(vx: float, vy: float) -> float:
    vx = vx or 0.0
    vy = vy or 0.0
    return math.sqrt(vx * vx + vy * vy)


def _dist(x1: float, y1: float, x2: float, y2: float) -> float:
    dx = x1 - x2
    dy = y1 - y2
    return math.sqrt(dx * dx + dy * dy)


class InDSceneSequenceAdapter(BaseSequenceAdapter):
    """
    读取单个 inD scene CSV，并按 frame 逐帧产出 SceneState。
    每帧选择一个最关键交互对象：
      1) 优先 primary actor 且非 ego
      2) 否则选距离 ego 最近的非 ego 对象

    支持两类字段格式：
      A. xCenter / yCenter / xVelocity / yVelocity
      B. x / y / vx / vy + ego_x / ego_y / ego_vx / ego_vy
    """

    def __init__(self, scene_csv_path: str):
        self.scene_csv_path = scene_csv_path

    def _load_rows(self) -> List[dict]:
        with open(self.scene_csv_path, "r", encoding="utf-8") as f:
            return list(csv.DictReader(f))

    def _group_by_frame(self, rows: List[dict]) -> Dict[int, List[dict]]:
        out: Dict[int, List[dict]] = {}
        for row in rows:
            frame = _to_int(row.get("frame"))
            if frame is None:
                continue
            out.setdefault(frame, []).append(row)
        return out

    def _find_ego_row(self, frame_rows: List[dict]) -> Optional[dict]:
        for r in frame_rows:
            if _to_bool(r.get("is_ego")):
                return r
        return None

    def _pick_target_row(self, frame_rows: List[dict], ego_row: dict) -> Optional[dict]:
        non_ego = [r for r in frame_rows if not _to_bool(r.get("is_ego"))]
        if not non_ego:
            return None

        ego_x = _first_float(ego_row, ["xCenter", "x", "ego_x"], 0.0)
        ego_y = _first_float(ego_row, ["yCenter", "y", "ego_y"], 0.0)

        def row_distance(r):
            tx = _first_float(r, ["xCenter", "x"], 0.0)
            ty = _first_float(r, ["yCenter", "y"], 0.0)
            return _dist(ego_x, ego_y, tx, ty)

        primary = [r for r in non_ego if _to_bool(r.get("is_primary_actor"))]
        if primary:
            primary.sort(key=row_distance)
            return primary[0]

        non_ego.sort(key=row_distance)
        return non_ego[0]

    def iter_scenes(self) -> Iterator[SceneState]:
        rows = self._load_rows()
        grouped = self._group_by_frame(rows)

        for frame in sorted(grouped.keys()):
            frame_rows = grouped[frame]
            ego_row = self._find_ego_row(frame_rows)
            if ego_row is None:
                continue

            target = self._pick_target_row(frame_rows, ego_row)
            if target is None:
                continue

            ego_x = _first_float(ego_row, ["xCenter", "x", "ego_x"], 0.0)
            ego_y = _first_float(ego_row, ["yCenter", "y", "ego_y"], 0.0)
            ego_vx = _first_float(ego_row, ["xVelocity", "vx", "ego_vx"], 0.0)
            ego_vy = _first_float(ego_row, ["yVelocity", "vy", "ego_vy"], 0.0)

            tgt_x = _first_float(target, ["xCenter", "x"], 0.0)
            tgt_y = _first_float(target, ["yCenter", "y"], 0.0)
            tgt_vx = _first_float(target, ["xVelocity", "vx"], 0.0)
            tgt_vy = _first_float(target, ["yVelocity", "vy"], 0.0)

            headway = _first_float(target, ["distance_to_ego", "distance"], None)
            if headway is None:
                headway = _dist(ego_x, ego_y, tgt_x, tgt_y)

            rel_speed = _first_float(target, ["rel_speed_to_ego", "rel_speed"], None)
            if rel_speed is None:
                rel_speed = _speed_norm(ego_vx - tgt_vx, ego_vy - tgt_vy)

            ego_cls = str(ego_row.get("class", "")).lower()
            tgt_cls = str(target.get("class", "")).lower()
            vrus_present = (
                "pedestrian" in tgt_cls
                or "bicycle" in tgt_cls
                or "cyclist" in tgt_cls
            )

            raw_ttc = _first_float(target, ["ttc", "event_min_ttc"], None)
            raw_dist = _first_float(target, ["distance_to_ego", "distance", "event_min_distance"], headway)

            yield SceneState(
                scene_type="inD",
                ego_speed_mps=_speed_norm(ego_vx, ego_vy),
                headway_m=headway,
                lane_change=False,

                dist_to_intersection_m=0.0,
                traffic_light="unknown",
                vrus_present=vrus_present,

                lead_speed_mps=_speed_norm(tgt_vx, tgt_vy),
                rel_speed_mps=rel_speed,

                ego_x=ego_x,
                ego_y=ego_y,
                other_x=tgt_x,
                other_y=tgt_y,

                # 关键：二维速度分量
                ego_vx=ego_vx,
                ego_vy=ego_vy,
                other_vx=tgt_vx,
                other_vy=tgt_vy,

                event_type=f"{ego_cls}_{tgt_cls}",
                frame_index=frame,
                duration_s=None,

                min_ttc_raw=raw_ttc,
                min_thw_raw=None,
                min_dhw_raw=raw_dist,
            )

    def sequence_metadata(self) -> dict:
        rows = self._load_rows()
        if not rows:
            return {}

        first = rows[0]
        frames = sorted({_to_int(r.get("frame")) for r in rows if _to_int(r.get("frame")) is not None})

        ego_track_ids = sorted({
            _to_int(r.get("trackId"))
            for r in rows
            if _to_bool(r.get("is_ego")) and _to_int(r.get("trackId")) is not None
        })

        primary_track_ids = sorted({
            _to_int(r.get("trackId"))
            for r in rows
            if _to_bool(r.get("is_primary_actor")) and _to_int(r.get("trackId")) is not None
        })

        return {
            "scene_csv_path": self.scene_csv_path,
            "recordingId": _to_int(first.get("recordingId")),
            "start_frame": frames[0] if frames else None,
            "end_frame": frames[-1] if frames else None,
            "num_frames": len(frames),
            "ego_track_ids": ego_track_ids,
            "primary_track_ids": primary_track_ids,
        }