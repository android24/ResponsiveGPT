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

        rf = p.get("risk_forecast", {})
        rs = p.get("recommended_strategy", {})
        rg = p.get("reactive_guidance", {})
        sc = p.get("staleness_control", {})

        hint = {
            "planning_age_frames": age,
            "risk_level": rf.get("risk_level"),
            "risk_trend": rf.get("risk_trend"),
            "main_risk_factors": rf.get("main_risk_factors", []),
            "metric_evidence": rf.get("metric_evidence", {}),
            "strategy": rs.get("strategy"),
            "priority": rs.get("priority"),
            "must_check": rg.get("must_check", []),
            "avoid_actions": rg.get("avoid_actions", []),
            "preferred_actions": rg.get("preferred_actions", []),
            "safety_constraints": rg.get("safety_constraints", []),
            "fast_rule_hint": rg.get("fast_rule_hint", ""),
            "valid_for_frames": sc.get("valid_for_frames"),
            "staleness_risk": sc.get("staleness_risk"),
            "confidence": p.get("confidence", 0.0),
        }

        text = json.dumps(hint, ensure_ascii=False)
        if len(text) > max_chars:
            text = text[:max_chars] + "..."
        return text

    def is_stale(self, current_frame: int) -> bool:
        p = self.current or {}
        staleness = (
            p.get("staleness_control", {})
            if isinstance(p, dict)
            else {}
        )
        valid_for = staleness.get("valid_for_frames", 10)
        try:
            valid_for = max(0, int(valid_for))
        except (TypeError, ValueError):
            valid_for = 10

        if self.last_update_frame is None:
            return True
        return (current_frame - self.last_update_frame) > valid_for
