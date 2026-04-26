# src/responsivegpt/interface/adapters/highd_event_adapter.py

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


def _to_bool(v):
    if isinstance(v, bool):
        return v
    return str(v).strip().lower() in ["true", "1", "yes"]


class HighDEventAdapter(BaseEventAdapter):
    def __init__(self, csv_path: str):
        self.csv_path = csv_path

    def iter_rows(self) -> Iterator[dict]:
        with open(self.csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                yield row

    def row_to_scene(self, row: Dict[str, Any]) -> SceneState:
        ego_speed = _to_float(row.get("egoSpeedMean"), 0.0)
        other_speed = _to_float(row.get("otherSpeedMean"))
        rel_speed = _to_float(row.get("relSpeedMean"))

        return SceneState(
            scene_type="highD",
            ego_speed_mps=ego_speed,
            headway_m=_to_float(row.get("minDHW"), 0.0),
            lane_change=_to_bool(row.get("isLaneChangeEvent")),
            lane_change_direction=row.get("laneChangeDirection"),

            dist_to_intersection_m=9999.0,
            traffic_light="none",
            vrus_present=False,

            lead_speed_mps=other_speed,
            rel_speed_mps=rel_speed,

            ego_x=_to_float(row.get("egoXStart")),
            ego_y=_to_float(row.get("egoYStart")),
            other_x=_to_float(row.get("otherXStart")),
            other_y=_to_float(row.get("otherYStart")),

            # 新增：event-level 近似速度分量
            ego_vx=ego_speed,
            ego_vy=0.0,
            other_vx=other_speed,
            other_vy=0.0,

            event_type=row.get("eventType"),
            frame_index=None,
            duration_s=_to_float(row.get("duration_s")),

            min_ttc_raw=_to_float(row.get("minTTC")),
            min_thw_raw=_to_float(row.get("minTHW")),
            min_dhw_raw=_to_float(row.get("minDHW")),
        )

    def row_metadata(self, row: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "dataset": "highD",
            "recordingId": _to_int(row.get("recordingId")),
            "egoId": _to_int(row.get("egoId")),
            "otherId": _to_int(row.get("otherId")),
            "locationId": _to_int(row.get("locationId")),
            "eventType": row.get("eventType"),
            "clipPath": row.get("clipPath"),
            "startFrame": _to_int(row.get("startFrame")),
            "endFrame": _to_int(row.get("endFrame")),
            "clipStartFrame": _to_int(row.get("clipStartFrame")),
            "clipEndFrame": _to_int(row.get("clipEndFrame")),
            "startTime_s": _to_float(row.get("startTime_s")),
            "endTime_s": _to_float(row.get("endTime_s")),
            "duration_s": _to_float(row.get("duration_s")),
            "isLaneChangeEvent": _to_bool(row.get("isLaneChangeEvent")),
            "laneChangeFrame": _to_int(row.get("laneChangeFrame")),
            "fromLane": _to_int(row.get("fromLane")),
            "toLane": _to_int(row.get("toLane")),
            "laneChangeDirection": row.get("laneChangeDirection"),
            "minTTC": _to_float(row.get("minTTC")),
            "minTHW": _to_float(row.get("minTHW")),
            "minDHW": _to_float(row.get("minDHW")),
            "maxAbsDecel": _to_float(row.get("maxAbsDecel")),
            "minRelSpeed": _to_float(row.get("minRelSpeed")),
            "relSpeedMean": _to_float(row.get("relSpeedMean")),
        }

    def derive_risk_label(self, row: Dict[str, Any]) -> bool:
        min_ttc = _to_float(row.get("minTTC"))
        min_thw = _to_float(row.get("minTHW"))

        if min_ttc is not None and min_ttc < 3.0:
            return True
        if min_thw is not None and min_thw < 0.5:
            return True
        return False

    def get_sequence_ref(self, metadata: dict) -> str | None:
        return metadata.get("clipPath")