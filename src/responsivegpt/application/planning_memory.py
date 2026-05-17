import json
from .planning_schema import validate_planning_output, default_planning_output


class PlanningMemory:
    """
    Reactive Thread 可以读取这里的 compact planning insight。
    注意：这里只保存压缩结论，不保存长 chain-of-thought。
    """

    def __init__(self):
        self.current = default_planning_output("init")
        self.last_update_frame = None
        self.history = []

    def update(self, planning_output: dict, frame_index: int):
        validated = validate_planning_output(planning_output)
        self.current = validated
        self.last_update_frame = frame_index
        self.history.append({
            "frame_index": frame_index,
            "planning": validated,
        })

    def get(self) -> dict:
        return self.current

    def to_reactive_hint(self, current_frame=None, max_chars: int = 1200) -> str:
        p = self.current or default_planning_output("empty")

        age = None
        if current_frame is not None and self.last_update_frame is not None:
            age = current_frame - self.last_update_frame

        hint = {
            "planning_age_frames": age,
            "risk_forecast": p.get("risk_forecast", {}),
            "recommended_strategy": p.get("recommended_strategy", {}),
            "reactive_guidance": p.get("reactive_guidance", {}),
            "confidence": p.get("confidence", 0.0),
            "validity": p.get("validity", {}),
        }

        text = json.dumps(hint, ensure_ascii=False)
        if len(text) > max_chars:
            text = text[:max_chars] + "..."
        return text

    def is_stale(self, current_frame: int) -> bool:
        p = self.current or {}
        validity = p.get("validity", {}) if isinstance(p, dict) else {}
        valid_for = validity.get("valid_for_frames", 10)

        if self.last_update_frame is None:
            return True
        return (current_frame - self.last_update_frame) > valid_for