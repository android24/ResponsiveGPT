from copy import deepcopy


def clamp(x, lo, hi):
    return max(lo, min(hi, x))


class LayeredProfileLearner:
    enabled = True

    """
    闭环学习器：
    - 输入：当前 profile + trigger list + profile_update proposal + decision
    - 输出：更新后的 profile
    - 特点：分层更新 global / longitudinal / lateral / interaction / temporal
    """

    def __init__(self, lr: float = 0.2):
        self.lr = lr

    def apply(self, profile: dict, triggers: list, profile_update: dict, decision: dict) -> dict:
        p = deepcopy(profile)

        self._ensure_structure(p)

        # profile_update is retained as an auditable proposal. Applying both
        # it and the originating trigger would double-count the same signal.
        for trig in triggers:
            if not trig.activated:
                continue
            self._apply_trigger(p, trig)

        # 3) 若 LLM 已给 tuning_suggestion，再做轻量融合
        tuning = decision.get("tuning_suggestion", {})
        self._apply_tuning_suggestion(p, tuning)

        # 4) 最终 guardrails
        self._normalize_and_guardrail(p)

        return p

    def _ensure_structure(self, p: dict):
        p.setdefault("driver_type", "unknown")
        p.setdefault("global", {})
        p.setdefault("longitudinal", {})
        p.setdefault("lateral", {})
        p.setdefault("interaction", {})
        p.setdefault("scenario_bias", {})
        p.setdefault("temporal", {})

        g = p["global"]
        g.setdefault("risk_sensitivity", 0.5)
        g.setdefault("safety_weight", 0.6)
        g.setdefault("efficiency_weight", 0.4)

        lg = p["longitudinal"]
        lg.setdefault("preferred_time_headway", 2.0)
        lg.setdefault("min_time_headway", 1.2)
        lg.setdefault("brake_aggressiveness", 0.5)
        lg.setdefault("acceleration_aggressiveness", 0.5)

        lat = p["lateral"]
        lat.setdefault("lane_change_aggressiveness", 0.5)
        lat.setdefault("min_gap_acceptance", 1.5)
        lat.setdefault("cut_in_tolerance", 0.5)

        inter = p["interaction"]
        inter.setdefault("vehicle_vehicle_assertiveness", 0.5)
        inter.setdefault("vehicle_cyclist_yield_bias", 0.7)
        inter.setdefault("vehicle_pedestrian_yield_bias", 0.9)

        tmp = p["temporal"]
        tmp.setdefault("risk_memory_decay", 0.9)
        tmp.setdefault("recent_violation_penalty", 0.2)

    def _smooth(self, old, new):
        return (1.0 - self.lr) * old + self.lr * new

    def _apply_profile_delta(self, p: dict, delta: dict):
        if not delta:
            return

        g = p["global"]
        lg = p["longitudinal"]
        lat = p["lateral"]
        inter = p["interaction"]

        if "risk_sensitivity_delta" in delta:
            g["risk_sensitivity"] += delta["risk_sensitivity_delta"]

        if "safety_weight_delta" in delta:
            g["safety_weight"] += delta["safety_weight_delta"]

        if "efficiency_weight_delta" in delta:
            g["efficiency_weight"] += delta["efficiency_weight_delta"]

        if "preferred_time_headway_delta" in delta:
            lg["preferred_time_headway"] += delta["preferred_time_headway_delta"]

        if "lane_change_aggressiveness_delta" in delta:
            lat["lane_change_aggressiveness"] += delta["lane_change_aggressiveness_delta"]

        if "vehicle_cyclist_yield_bias_delta" in delta:
            inter["vehicle_cyclist_yield_bias"] += delta["vehicle_cyclist_yield_bias_delta"]

        if "vehicle_vehicle_assertiveness_delta" in delta:
            inter["vehicle_vehicle_assertiveness"] += delta["vehicle_vehicle_assertiveness_delta"]

    def _apply_trigger(self, p: dict, trig):
        if trig.target_layer == "global":
            self._apply_global_trigger(p, trig)
        elif trig.target_layer == "longitudinal":
            self._apply_longitudinal_trigger(p, trig)
        elif trig.target_layer == "lateral":
            self._apply_lateral_trigger(p, trig)
        elif trig.target_layer == "interaction":
            self._apply_interaction_trigger(p, trig)
        elif trig.target_layer == "temporal":
            self._apply_temporal_trigger(p, trig)

    def _apply_global_trigger(self, p: dict, trig):
        g = p["global"]
        step = self._trigger_step(trig)

        if trig.trigger_type in ["risk_threshold", "compliance_violation", "human_feedback"]:
            g["risk_sensitivity"] += step

        if trig.action == "increase_safety_weight":
            g["safety_weight"] += step
            g["efficiency_weight"] -= step

        if trig.action == "decrease_efficiency_weight":
            g["efficiency_weight"] -= step
            g["safety_weight"] += step

        if trig.action == "increase_efficiency_weight":
            g["efficiency_weight"] += step
            g["safety_weight"] -= step

    def _apply_longitudinal_trigger(self, p: dict, trig):
        lg = p["longitudinal"]
        step = self._trigger_step(trig)

        if trig.trigger_type in ["persistent_high_risk", "risk_threshold", "compliance_violation"]:
            lg["preferred_time_headway"] += step
            lg["min_time_headway"] += 0.625 * step

        if trig.action == "slowdown_bias":
            lg["acceleration_aggressiveness"] -= step
            lg["brake_aggressiveness"] += step

    def _apply_lateral_trigger(self, p: dict, trig):
        lat = p["lateral"]
        step = self._trigger_step(trig)

        if trig.action == "freeze_lane_change":
            lat["lane_change_aggressiveness"] -= step
            lat["min_gap_acceptance"] += 0.8 * step

        if trig.trigger_type == "compliance_violation":
            lat["cut_in_tolerance"] -= step

    def _apply_interaction_trigger(self, p: dict, trig):
        inter = p["interaction"]
        step = self._trigger_step(trig)

        if trig.trigger_type == "vru_protection":
            inter["vehicle_cyclist_yield_bias"] += step
            inter["vehicle_vehicle_assertiveness"] -= 0.5 * step

        if trig.trigger_type == "human_feedback":
            inter["vehicle_cyclist_yield_bias"] += 0.5 * step

    def _apply_temporal_trigger(self, p: dict, trig):
        tmp = p["temporal"]
        if trig.trigger_type in ["persistent_high_risk", "compliance_violation"]:
            tmp["recent_violation_penalty"] += self._trigger_step(trig)

    def _trigger_step(self, trig) -> float:
        magnitude = min(abs(float(trig.action_value or 0.0)), 0.25)
        return self.lr * magnitude * max(0.0, float(trig.score))

    def _apply_tuning_suggestion(self, p: dict, tuning: dict):
        if not tuning:
            return

        g = p["global"]
        if "risk_sensitivity" in tuning:
            g["risk_sensitivity"] = self._smooth(g["risk_sensitivity"], tuning["risk_sensitivity"])
        if "safety_weight" in tuning:
            g["safety_weight"] = self._smooth(g["safety_weight"], tuning["safety_weight"])
        if "efficiency_weight" in tuning:
            g["efficiency_weight"] = self._smooth(g["efficiency_weight"], tuning["efficiency_weight"])

    def _normalize_and_guardrail(self, p: dict):
        g = p["global"]
        lg = p["longitudinal"]
        lat = p["lateral"]
        inter = p["interaction"]
        tmp = p["temporal"]

        # global
        g["risk_sensitivity"] = clamp(g["risk_sensitivity"], 0.0, 1.0)
        g["safety_weight"] = clamp(g["safety_weight"], 0.3, 1.0)
        g["efficiency_weight"] = clamp(g["efficiency_weight"], 0.0, 0.7)

        total = g["safety_weight"] + g["efficiency_weight"]
        if total > 0:
            g["safety_weight"] /= total
            g["efficiency_weight"] /= total

        # longitudinal
        lg["preferred_time_headway"] = clamp(lg["preferred_time_headway"], 1.0, 3.5)
        lg["min_time_headway"] = clamp(lg["min_time_headway"], 0.8, 2.5)
        lg["brake_aggressiveness"] = clamp(lg["brake_aggressiveness"], 0.0, 1.0)
        lg["acceleration_aggressiveness"] = clamp(lg["acceleration_aggressiveness"], 0.0, 1.0)

        # lateral
        lat["lane_change_aggressiveness"] = clamp(lat["lane_change_aggressiveness"], 0.0, 1.0)
        lat["min_gap_acceptance"] = clamp(lat["min_gap_acceptance"], 0.5, 3.0)
        lat["cut_in_tolerance"] = clamp(lat["cut_in_tolerance"], 0.0, 1.0)

        # interaction
        inter["vehicle_vehicle_assertiveness"] = clamp(inter["vehicle_vehicle_assertiveness"], 0.0, 1.0)
        inter["vehicle_cyclist_yield_bias"] = clamp(inter["vehicle_cyclist_yield_bias"], 0.0, 1.0)
        inter["vehicle_pedestrian_yield_bias"] = clamp(inter["vehicle_pedestrian_yield_bias"], 0.0, 1.0)

        # temporal
        tmp["risk_memory_decay"] = clamp(tmp["risk_memory_decay"], 0.0, 1.0)
        tmp["recent_violation_penalty"] = clamp(tmp["recent_violation_penalty"], 0.0, 1.0)
