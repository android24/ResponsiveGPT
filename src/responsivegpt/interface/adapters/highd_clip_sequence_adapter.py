import csv
import math
from typing import Dict, List, Iterator, Optional, Tuple
from ...domain.models import SceneState
from .base_sequence_adapter import BaseSequenceAdapter


def _to_float(v, default=None):
    if v is None or v == "":
        return default
    try:
        x = float(v)
        if math.isnan(x) or math.isinf(x):
            return default
        return x
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

def _speed_norm(vx: Optional[float], vy: Optional[float]) -> Optional[float]:
    if vx is None and vy is None:
        return None
    vx = vx or 0.0
    vy = vy or 0.0
    return math.sqrt(vx * vx + vy * vy)

def _euclidean_distance(x1, y1, x2, y2):
    if None in [x1, y1, x2, y2]:
        return None
    return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

class HighDClipSequenceAdapter(BaseSequenceAdapter):
    """
    读取单个 highD clip CSV，并逐帧生成 SceneState。

    highD clip 是多车多行格式：
      同一 frame 下通常至少有 ego / other 两行。
    因此不能逐行 yield，必须：
      frame -> pick ego/other -> merge -> SceneState

    支持两类字段：
      1. 完整版：vx/vy 或 ego_vx/ego_vy/other_vx/other_vy
      2. 简化版：xVelocity，vy 近似为 0
    """

    def __init__(self, clip_csv_path: str):
        self.clip_csv_path = clip_csv_path

    def _load_rows(self) -> List[dict]:
        with open(self.clip_csv_path, "r", encoding="utf-8-sig") as f:
            return list(csv.DictReader(f))

    def _group_by_frame(self, rows: List[dict]) -> Dict[int, List[dict]]:
        out = {}
        for r in rows:
            frame = _to_int(r.get("frame"))
            if frame is None:
                continue
            out.setdefault(frame, []).append(r)
        return out
    
    def _is_target_pair(self, row: dict) -> bool:
        # 如果没有 isTargetPair 字段，则默认目标 ego/other 仍然可用
        if "isTargetPair" not in row:
            return True
        return _to_bool(row.get("isTargetPair"))

    def _pick_ego_other(self, frame_rows: List[dict]) -> Tuple[Optional[dict], Optional[dict]]:
        """
        优先从 target pair 中选择 role=ego / role=other。
        这是 highD clip 最稳的逻辑。
        """
        target_rows = [r for r in frame_rows if self._is_target_pair(r)]
        if not target_rows:
            target_rows = frame_rows

        ego = None
        other = None

        for r in target_rows:
            role = str(r.get("role", "")).strip().lower()
            if role == "ego":
                ego = r
            elif role == "other":
                other = r

        # fallback 1：如果 role 不完整，尝试找 egoTrack / egoId 标记
        if ego is None:
            for r in target_rows:
                role = str(r.get("role", "")).strip().lower()
                if "ego" in role:
                    ego = r
                    break

        if other is None:
            for r in target_rows:
                if r is ego:
                    continue
                role = str(r.get("role", "")).strip().lower()
                if "other" in role or role != "ego":
                    other = r
                    break

        return ego, other
    
    def _extract_xy(self, row: dict):
        """
        兼容字段：
        - x, y
        - xCenter, yCenter
        """
        x = _first_float(row, ["x", "xCenter"])
        y = _first_float(row, ["y", "yCenter"], 0.0)
        return x, y
    
    def _extract_velocity(self, row: dict, role_prefix: str = ""):
        """
        兼容多种 highD clip 字段格式：
        - vx, vy
        - xVelocity, yVelocity
        - ego_vx, ego_vy / other_vx, other_vy
        - ego_xVelocity, ego_yVelocity / other_xVelocity, other_yVelocity

        如果只有 xVelocity，则 vy = 0.0。
        highD 高速场景中这是合理近似。
        """
        prefixed_vx_keys = []
        prefixed_vy_keys = []

        if role_prefix:
            prefixed_vx_keys = [
                f"{role_prefix}_vx",
                f"{role_prefix}Vx",
                f"{role_prefix}_xVelocity",
                f"{role_prefix}XVelocity",
            ]
            prefixed_vy_keys = [
                f"{role_prefix}_vy",
                f"{role_prefix}Vy",
                f"{role_prefix}_yVelocity",
                f"{role_prefix}YVelocity",
            ]

        vx = _first_float(row, prefixed_vx_keys + ["vx", "xVelocity"])
        vy = _first_float(row, prefixed_vy_keys + ["vy", "yVelocity"], 0.0)

        if vx is None:
            return None, None

        if vy is None:
            vy = 0.0

        return vx, vy
    
    def _closing_speed(self, ego_x, other_x, ego_vx, other_vx):
        """
        对 highD，主要沿 x 方向。
        closing speed > 0 表示 ego 正在接近 other。
        需要考虑 other 在前还是在后：
          - other_x > ego_x: 前车，closing = ego_vx - other_vx
          - other_x < ego_x: 后车，closing = other_vx - ego_vx
        """
        if None in [ego_x, other_x, ego_vx, other_vx]:
            return None

        if other_x >= ego_x:
            return ego_vx - other_vx
        return other_vx - ego_vx
    
    def validate_schema(self):
        from .schema_validation import require_fields, require_any_group

        require_fields(
            self.clip_csv_path,
            ["frame", "role", "x", "y"],
            context="highD clip",
        )

        require_any_group(
            self.clip_csv_path,
            [
                ["xVelocity"],
                ["vx"],
                ["ego_vx", "other_vx"],
            ],
            context="highD velocity fields",
        )

    def iter_scenes(self) -> Iterator[SceneState]:
        rows = self._load_rows()
        grouped = self._group_by_frame(rows)

        for frame in sorted(grouped.keys()):
            frame_rows = grouped[frame]
            ego, other = self._pick_ego_other(frame_rows)

            if ego is None or other is None:
                continue

            ego_x, ego_y = self._extract_xy(ego)
            other_x, other_y = self._extract_xy(other)

            if ego_x is None or other_x is None:
                continue

            ego_vx, ego_vy = self._extract_velocity(ego, role_prefix="ego")
            other_vx, other_vy = self._extract_velocity(other, role_prefix="other")

            # 如果速度字段仍然拿不到，最后尝试从 speed / xVelocity 中兜底
            if ego_vx is None:
                ego_vx = _first_float(ego, ["speed", "xVelocity"], 0.0)
                ego_vy = 0.0

            if other_vx is None:
                other_vx = _first_float(other, ["speed", "xVelocity"], 0.0)
                other_vy = 0.0

            ego_speed = _speed_norm(ego_vx, ego_vy)
            other_speed = _speed_norm(other_vx, other_vy)

            # highD 纵向 headway 优先使用 dhw；没有再用 x 差
            dhw = _first_float(other, ["dhw", "DHW"])
            if dhw is None:
                dhw = _first_float(ego, ["dhw", "DHW"])

            headway = dhw
            if headway is None:
                headway = abs(other_x - ego_x)

            # 欧式距离作为空间指标补充，但 SceneState 当前用 headway_m 承载主距离
            euclidean_distance = _euclidean_distance(ego_x, ego_y, other_x, other_y)

            rel_v = self._closing_speed(ego_x, other_x, ego_vx, other_vx)

            # 如果 clip 自带 ttc/thw/dhw，保留到 raw 字段中
            raw_ttc = _first_float(other, ["ttc", "TTC"])
            if raw_ttc is None:
                raw_ttc = _first_float(ego, ["ttc", "TTC"])

            raw_thw = _first_float(other, ["thw", "THW"])
            if raw_thw is None:
                raw_thw = _first_float(ego, ["thw", "THW"])

            raw_dhw = _first_float(other, ["dhw", "DHW"])
            if raw_dhw is None:
                raw_dhw = _first_float(ego, ["dhw", "DHW"], headway)

            event_type = ego.get("eventType") or other.get("eventType")
            lane_change = _to_bool(ego.get("isLaneChangeEvent") or other.get("isLaneChangeEvent"))
            lane_change_direction = ego.get("laneChangeDirection") or other.get("laneChangeDirection")

            yield SceneState(
                scene_type="highD",
                ego_speed_mps=ego_speed or 0.0,
                headway_m=headway or 0.0,
                lane_change=lane_change,

                dist_to_intersection_m=9999.0,
                traffic_light="none",
                vrus_present=False,

                lead_speed_mps=other_speed,
                rel_speed_mps=rel_v,

                ego_x=ego_x,
                ego_y=ego_y,
                other_x=other_x,
                other_y=other_y,

                ego_vx=ego_vx,
                ego_vy=ego_vy,
                other_vx=other_vx,
                other_vy=other_vy,

                event_type=event_type,
                frame_index=frame,
                duration_s=None,

                min_ttc_raw=raw_ttc,
                min_thw_raw=raw_thw,
                min_dhw_raw=raw_dhw if raw_dhw is not None else euclidean_distance,

                lane_change_direction=lane_change_direction,
            )

    def sequence_metadata(self) -> dict:
        rows = self._load_rows()
        if not rows:
            return {"clip_csv_path": self.clip_csv_path}

        frames = sorted({
            _to_int(r.get("frame"))
            for r in rows
            if _to_int(r.get("frame")) is not None
        })

        target_rows = [r for r in rows if self._is_target_pair(r)]

        return {
            "clip_csv_path": self.clip_csv_path,
            "start_frame": frames[0] if frames else None,
            "end_frame": frames[-1] if frames else None,
            "num_frames": len(frames),
            "num_rows": len(rows),
            "num_target_rows": len(target_rows),
        }