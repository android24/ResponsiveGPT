import json
import copy


def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def _truncate(text, max_chars: int = 1200) -> str:
    if text is None:
        return ""
    text = str(text)
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "...[truncated]"


def _json_dumps(obj, max_chars: int | None = None) -> str:
    try:
        text = json.dumps(obj, ensure_ascii=False)
    except Exception:
        text = json.dumps(str(obj), ensure_ascii=False)

    if max_chars is not None:
        return _truncate(text, max_chars=max_chars)
    return text


def _safe_obj_dict(obj):
    if obj is None:
        return {}
    if isinstance(obj, dict):
        return obj
    if hasattr(obj, "__dict__"):
        return dict(obj.__dict__)
    return {"value": str(obj)}


def update_profile(profile: dict, driver_type: str, human_feedback: str) -> dict:
    """
    dict 版本 profile 更新。
    建议返回 copy，避免在未 save 前污染原始 profile。
    """

    if not isinstance(profile, dict):
        raise TypeError(f"Expected dict profile, got {type(profile)}")

    profile = copy.deepcopy(profile)

    dt = (driver_type or "").strip().lower()
    fb = (human_feedback or "").strip().lower()

    profile.setdefault("driver_type", "均衡")
    profile.setdefault("global", {})

    g = profile["global"]
    g.setdefault("risk_sensitivity", 0.5)
    g.setdefault("safety_weight", 0.6)
    g.setdefault("efficiency_weight", 0.4)

    if dt:
        profile["driver_type"] = driver_type

    if dt in ["aggressive", "激进", "激进型"]:
        g["risk_sensitivity"] = clamp(g["risk_sensitivity"] - 0.2, 0.0, 1.0)

    elif dt in ["conservative", "保守", "保守型"]:
        g["risk_sensitivity"] = clamp(g["risk_sensitivity"] + 0.2, 0.0, 1.0)

    elif dt in ["balanced", "均衡"]:
        g["risk_sensitivity"] = clamp(
            0.8 * g["risk_sensitivity"] + 0.2 * 0.5,
            0.0,
            1.0,
        )

    if any(k in fb for k in ["危险", "不安全", "too dangerous", "unsafe"]):
        g["risk_sensitivity"] = clamp(g["risk_sensitivity"] + 0.1, 0.0, 1.0)

    if any(k in fb for k in ["太慢", "低效", "too slow", "效率"]):
        g["risk_sensitivity"] = clamp(g["risk_sensitivity"] - 0.1, 0.0, 1.0)

    rs = g["risk_sensitivity"]

    g["safety_weight"] = round(0.3 + 0.7 * rs, 3)
    g["efficiency_weight"] = round(1.0 - g["safety_weight"], 3)

    return profile


def _format_ranked_docs(
    title: str,
    docs,
    max_docs: int = 3,
    max_doc_chars: int = 500,
) -> str:
    """
    token-time abstraction 下，Reactive Thread 不能吃太长证据。
    所以每类 evidence 最多保留 max_docs 条，每条截断。
    """
    if not docs:
        return f"[{title}]\nNone\n"

    lines = [f"[{title}]"]

    for idx, d in enumerate(docs[:max_docs], start=1):
        doc_id = getattr(d.doc, "id", "")
        score = getattr(d, "final_score", 0.0)
        doc_title = getattr(d.doc, "title", "")
        doc_text = _truncate(getattr(d.doc, "text", ""), max_chars=max_doc_chars)

        lines.append(
            f"{idx}. id={doc_id}; score={score:.3f}; title={doc_title}; text={doc_text}"
        )

    lines.append("")
    return "\n".join(lines)


def _format_frame_safety(frame_safety) -> str:
    if frame_safety is None:
        return "None"

    fs = _safe_obj_dict(frame_safety)

    keep = {
        "frame_index": fs.get("frame_index"),
        "ttc_s": fs.get("ttc_s"),
        "thw_s": fs.get("thw_s"),
        "drac_mps2": fs.get("drac_mps2"),
        "dcpa_m": fs.get("dcpa_m"),
        "ttca_s": fs.get("ttca_s"),
        "predicted_ttc_s": fs.get("predicted_ttc_s"),
        "min_future_distance_m": fs.get("min_future_distance_m"),
        "physical_risk_index": fs.get("physical_risk_index"),
        "physical_risk_level": fs.get("physical_risk_level"),
        "unsafe_ttc": fs.get("unsafe_ttc"),
        "unsafe_drac": fs.get("unsafe_drac"),
        "unsafe_dcpa": fs.get("unsafe_dcpa"),
        "unsafe_future_distance": fs.get("unsafe_future_distance"),
    }

    return _json_dumps(keep, max_chars=1200)


def _format_planning_hint(
    planning_hint: str = "",
    planning_metadata: dict | None = None,
    max_chars: int = 1000,
) -> str:
    if not planning_hint:
        return ""

    planning_metadata = planning_metadata or {}

    age = planning_metadata.get("planning_age_frames")
    last_update = planning_metadata.get("last_update_frame")

    return f"""
[Planning Thread Insight]
The following insight comes from a slower long-horizon Planning Thread.

Planning metadata:
- planning_age_frames: {age}
- last_update_frame: {last_update}

Important constraints:
- Planning insight is advisory, not mandatory.
- Planning may be stale.
- Always prioritize the latest observation and current physical safety metrics.
- If Planning conflicts with current scene risk, follow current scene risk.
- Use Planning only as compact long-horizon guidance.

Planning insight:
{_truncate(planning_hint, max_chars=max_chars)}
"""


def make_evidence_prompts(
    profile: dict,
    scene,
    human_feedback: str,
    evidence,
    planning_hint: str = "",
    planning_metadata: dict | None = None,
    frame_safety=None,
    token_budget_class: str = "reactive_low",
) -> tuple[str, str]:
    """
    Reactive Thread prompt builder.

    设计原则：
    1. 当前 observation 永远优先；
    2. 当前 physical safety metrics 优先于 planning；
    3. Planning 只是 advisory memory；
    4. Reactive Thread 有严格 token-time budget；
    5. 输出 schema 保持稳定，避免下游解析崩。
    """

    system = (
        "你是自动驾驶交互反馈系统 ResponsiveGPT 的 Fast Reactive Thread。\n"
        "你负责在严格实时约束下，基于最新 observation、驾驶人偏好、物理安全指标、"
        "交通法规知识、风险案例知识和场景规则，快速判断当前帧是否存在潜在违规或高风险行为。\n\n"

        "你可能会收到来自慢速 Planning Thread 的 planning insight。"
        "该 insight 只能作为 advisory guidance，不能覆盖最新 observation 和当前物理安全指标。\n\n"

        "推理约束：\n"
        "1. 优先保证安全，其次考虑效率。\n"
        "2. 若当前物理安全指标显示高风险，应优先输出保守安全决策。\n"
        "3. 若 Planning insight 与当前 observation 冲突，以当前 observation 为准。\n"
        "4. 你处于低 token、低延迟 Reactive 模式，只需输出快速可行解，不要求全局最优。\n"
        "5. 必须只输出合法 JSON，不要输出任何额外文本。\n\n"

        "JSON schema:\n"
        "{\n"
        '  "is_potential_violation": true/false,\n'
        '  "risk_level": "low|medium|high",\n'
        '  "warning": "string or empty",\n'
        '  "recommended_action": "string",\n'
        '  "rationale": "string",\n'
        '  "evidence_ids": ["string", "..."],\n'
        '  "tuning_suggestion": {\n'
        '    "risk_sensitivity": number,\n'
        '    "safety_weight": number,\n'
        '    "efficiency_weight": number\n'
        "  }\n"
        "}\n\n"

        "额外要求：\n"
        "1. evidence_ids 只能从已提供证据中选择。\n"
        "2. 若涉及弱势交通参与者，应提高安全优先级。\n"
        "3. 如果证据不足，但物理指标高风险，也应标记为潜在风险。\n"
    )

    evidence_text = "\n".join([
        _format_ranked_docs("Relevant Laws", evidence.laws),
        _format_ranked_docs("Relevant Cases", evidence.cases),
        _format_ranked_docs("Scenario-specific Rules", evidence.scenarios),
    ])

    scene_dict = _safe_obj_dict(scene)
    planning_text = _format_planning_hint(
        planning_hint=planning_hint,
        planning_metadata=planning_metadata,
        max_chars=1000,
    )

    user = (
        f"[Token-Time Setting]\n"
        f"thread=Reactive; token_budget_class={token_budget_class}; "
        f"objective=fast_safe_feasible_decision\n\n"

        f"[Driver Profile]\n"
        f"{_json_dumps(profile, max_chars=2000)}\n\n"

        f"[Latest Scene State]\n"
        f"{_json_dumps(scene_dict, max_chars=2500)}\n\n"

        f"[Current Physical Safety Metrics]\n"
        f"{_format_frame_safety(frame_safety)}\n\n"

        f"[Human Feedback]\n"
        f"{human_feedback}\n\n"

        f"{evidence_text}\n"
    )

    if planning_text:
        user += f"{planning_text}\n"

    user += (
        "[Task]\n"
        "请基于最新场景、当前物理安全指标、驾驶人偏好、证据和可选 Planning insight，"
        "快速判断当前帧是否存在潜在违规或高风险行为，并输出结构化 JSON。\n"
        "注意：Planning insight 只提供长期倾向，不能覆盖当前帧安全指标。"
    )

    return system, user


def coerce_json(text: str) -> dict:
    text = text.strip()
    try:
        return json.loads(text)
    except Exception:
        l = text.find("{")
        r = text.rfind("}")
        if l != -1 and r != -1 and r > l:
            return json.loads(text[l:r + 1])
    raise ValueError("LLM output is not valid JSON.")