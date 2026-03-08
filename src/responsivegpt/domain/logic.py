import json
from .models import DriverProfile, SceneState

def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))

def update_profile(profile: DriverProfile, driver_type: str, feedback: str) -> DriverProfile:
    dt = driver_type.strip().lower()
    fb = feedback.strip().lower()

    profile.driver_type = driver_type

    if dt in ["aggressive", "激进", "激进型"]:
        profile.risk_sensitivity = clamp(profile.risk_sensitivity - 0.2, 0.0, 1.0)
    elif dt in ["conservative", "保守", "保守型"]:
        profile.risk_sensitivity = clamp(profile.risk_sensitivity + 0.2, 0.0, 1.0)

    if any(k in fb for k in ["太危险", "不安全", "slow down", "safer", "保守", "谨慎", "危险"]):
        profile.risk_sensitivity = clamp(profile.risk_sensitivity + 0.1, 0.0, 1.0)
    if any(k in fb for k in ["太慢", "效率", "快点", "hurry", "faster", "aggressive", "慢"]):
        profile.risk_sensitivity = clamp(profile.risk_sensitivity - 0.1, 0.0, 1.0)

    rs = profile.risk_sensitivity
    profile.safety_weight = round(0.3 + 0.7 * rs, 3)
    profile.efficiency_weight = round(1.0 - profile.safety_weight, 3)
    return profile

def build_query(scene: SceneState) -> str:
    """
    Build a retrieval query string from SceneState.
    使用 getattr 防止字段扩展时出现 AttributeError。
    """
    return (
        f"scene_type={getattr(scene, 'scene_type', None)}; "
        f"ego_speed_mps={getattr(scene, 'ego_speed_mps', None)}; "
        f"headway_m={getattr(scene, 'headway_m', None)}; "
        f"lane_change={getattr(scene, 'lane_change', None)}; "
        f"dist_to_intersection_m={getattr(scene, 'dist_to_intersection_m', None)}; "
        f"traffic_light={getattr(scene, 'traffic_light', None)}; "
        f"vrus_present={getattr(scene, 'vrus_present', None)}; "
        f"rel_speed_mps={getattr(scene, 'rel_speed_mps', None)}; "
        f"event_type={getattr(scene, 'event_type', None)}; "
        f"lane_change_direction={getattr(scene, 'lane_change_direction', None)}; "
        f"frame_index={getattr(scene, 'frame_index', None)}"
    )

def make_prompts(profile: DriverProfile, scene: SceneState, feedback: str, rules, extra_context: str = ""):
    system = (
        "你是自动驾驶交互反馈系统 ResponsiveGPT 的 Safety+Policy 评审器。\n"
        "必须只输出合法 JSON，不要输出任何额外文本。\n"
        "JSON schema：\n"
        "{\n"
        '  "is_potential_violation": true/false,\n'
        '  "risk_level": "low|medium|high",\n'
        '  "warning": "string or empty",\n'
        '  "recommended_action": "string",\n'
        '  "rationale": "string",\n'
        '  "tuning_suggestion": {\n'
        '    "risk_sensitivity": number,\n'
        '    "safety_weight": number,\n'
        '    "efficiency_weight": number\n'
        "  }\n"
        "}\n"
    )

    rules_text = "\n".join([f"- ({r.id}, score={r.score:.3f}) {r.text}" for r in rules])

    user = (
        f"driver_profile={json.dumps(profile.__dict__, ensure_ascii=False)}\n"
        f"scene_state={json.dumps(scene.__dict__, ensure_ascii=False)}\n"
        f"human_feedback={feedback}\n"
    )

    if extra_context:
        user += f"extra_context={extra_context}\n"

    user += f"\nretrieved_rules:\n{rules_text}\n\n请根据 safety_weight/efficiency_weight 权衡安全与效率。"

    return system, user

def coerce_json(text: str) -> dict:
    text = text.strip()
    try:
        return json.loads(text)
    except Exception:
        l = text.find("{")
        r = text.rfind("}")
        if l != -1 and r != -1 and r > l:
            return json.loads(text[l:r+1])
    raise ValueError("LLM output is not valid JSON.")
