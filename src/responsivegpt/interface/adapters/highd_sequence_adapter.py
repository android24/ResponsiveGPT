from typing import Iterator, Dict, Any, List
from ...domain.models import SceneState


def _to_float(v, default=None):
    if v is None or v == "":
        return default
    try:
        return float(v)
    except Exception:
        return default


def _to_bool(v):
    if isinstance(v, bool):
        return v
    return str(v).strip().lower() in ["true", "1", "yes"]


def _parse_series(s: str) -> List[float]:
    """
    默认假设 series 用 ';' 分隔。
    如果你导出的格式是 ',' 或 '|'，把这里改掉即可。
    """
    if not s:
        return []
    parts = str(s).strip().split(";")
    out = []
    for p in parts:
        p = p.strip()
        if not p:
            continue
        try:
            out.append(float(p))
        except Exception:
            pass
    return out


class HighDSequenceAdapter:
    """
    把一条强交互事件，展开为多帧 SceneState。
    适合逐帧运行 ResponsiveGPTService.step(...)。
    """

    def row_to_scenes(self, row: Dict[str, Any]) -> Iterator[SceneState]:
        ego_xs = _parse_series(row.get("egoXSeries", ""))
        ego_ys = _parse_series(row.get("egoYSeries", ""))
        ego_vs = _parse_series(row.get("egoSpeedSeries", ""))

        other_xs = _parse_series(row.get("otherXSeries", ""))
        other_ys = _parse_series(row.get("otherYSeries", ""))
        other_vs = _parse_series(row.get("otherSpeedSeries", ""))

        rel_vs = _parse_series(row.get("relSpeedSeries", ""))

        lengths = [
            len(ego_xs), len(ego_ys), len(ego_vs),
            len(other_xs), len(other_ys), len(other_vs),
            len(rel_vs),
        ]
        n = min(lengths) if lengths else 0
        if n <= 0:
            return

        lane_change = _to_bool(row.get("isLaneChangeEvent"))
        lane_change_direction = row.get("laneChangeDirection")
        event_type = row.get("eventType")
        duration_s = _to_float(row.get("duration_s"))
        min_ttc = _to_float(row.get("minTTC"))
        min_thw = _to_float(row.get("minTHW"))
        min_dhw = _to_float(row.get("minDHW"))

        for i in range(n):
            # highD 场景里，先用纵向 x 距离近似 headway
            headway = abs(other_xs[i] - ego_xs[i])

            yield SceneState(
                scene_type="highD",
                ego_speed_mps=ego_vs[i],
                headway_m=headway,
                lane_change=lane_change,

                dist_to_intersection_m=9999.0,
                traffic_light="none",
                vrus_present=False,

                lead_speed_mps=other_vs[i],
                rel_speed_mps=rel_vs[i],

                ego_x=ego_xs[i],
                ego_y=ego_ys[i],
                other_x=other_xs[i],
                other_y=other_ys[i],

                event_type=event_type,
                frame_index=i,
                duration_s=duration_s,

                min_ttc_raw=min_ttc,
                min_thw_raw=min_thw,
                min_dhw_raw=min_dhw,

                lane_change_direction=lane_change_direction,
            )
