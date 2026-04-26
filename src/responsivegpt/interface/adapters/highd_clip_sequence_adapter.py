import csv
from typing import Dict, List, Iterator, Optional
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


class HighDClipSequenceAdapter(BaseSequenceAdapter):
    """
    读取单个 highD clip CSV，并逐帧生成 SceneState。
    依赖 role 字段识别 ego / other。
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

    def _pick_ego_other(self, frame_rows: List[dict]) -> tuple[Optional[dict], Optional[dict]]:
        ego = None
        other = None

        # 优先按 role
        for r in frame_rows:
            role = str(r.get("role", "")).lower()
            if role == "ego":
                ego = r
            elif role == "other":
                other = r

        # 如果 role 不稳定，再用 isTargetPair + 非 ego 补救
        if ego is None:
            for r in frame_rows:
                role = str(r.get("role", "")).lower()
                if role == "ego":
                    ego = r
                    break

        if other is None and ego is not None:
            for r in frame_rows:
                if r is ego:
                    continue
                if _to_bool(r.get("isTargetPair")):
                    other = r
                    break

        # 最后兜底：取非 ego 的第一个 target pair
        if other is None:
            for r in frame_rows:
                role = str(r.get("role", "")).lower()
                if role != "ego":
                    other = r
                    break

        return ego, other

    def iter_scenes(self) -> Iterator[SceneState]:
        rows = self._load_rows()
        grouped = self._group_by_frame(rows)

        for frame in sorted(grouped.keys()):
            frame_rows = grouped[frame]
            ego, other = self._pick_ego_other(frame_rows)
            if ego is None or other is None:
                continue

            ego_x = _to_float(ego.get("x"))
            ego_y = _to_float(ego.get("y"))
            other_x = _to_float(other.get("x"))
            other_y = _to_float(other.get("y"))

            if ego_x is None or other_x is None:
                continue

            headway = abs(other_x - ego_x)

            ego_v = _to_float(ego.get("xVelocity"), 0.0)
            other_v = _to_float(other.get("xVelocity"), 0.0)
            rel_v = ego_v - other_v

            yield SceneState(
                scene_type="highD",
                ego_speed_mps=ego_v,
                headway_m=headway,
                lane_change=False,

                dist_to_intersection_m=9999.0,
                traffic_light="none",
                vrus_present=False,

                lead_speed_mps=other_v,
                rel_speed_mps=rel_v,

                ego_x=ego_x,
                ego_y=ego_y,
                other_x=other_x,
                other_y=other_y,

                event_type=ego.get("eventType") or other.get("eventType"),
                frame_index=frame,
                duration_s=None,

                min_ttc_raw=_to_float(other.get("ttc")),
                min_thw_raw=None,
                min_dhw_raw=headway,
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

        return {
            "clip_csv_path": self.clip_csv_path,
            "start_frame": frames[0] if frames else None,
            "end_frame": frames[-1] if frames else None,
            "num_frames": len(frames),
        }