from dataclasses import dataclass

@dataclass
class GuardrailState:
    freeze_lane_change: bool = False
    slowdown_bias: float = 0.0
    min_ttc_threshold: float = 0.0
    min_headway_threshold: float = 0.0

def default_guardrail_state() -> GuardrailState:
    return GuardrailState()

def apply_guardrail_action(state: GuardrailState, action: str, action_value: float) -> GuardrailState:
    if action == "freeze_lane_change":
        state.freeze_lane_change = True

    elif action == "slowdown_bias":
        state.slowdown_bias = max(state.slowdown_bias, action_value)

    elif action == "apply_guardrail":
        state.min_ttc_threshold = max(state.min_ttc_threshold, 3.0)
        state.min_headway_threshold = max(state.min_headway_threshold, 2.0)

    return state