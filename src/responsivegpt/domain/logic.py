import json

def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def update_profile(profile: dict, driver_type: str, human_feedback: str) -> dict:
    """
    dict 版本 profile 更新（完全替代旧 DriverProfile 版本）
    """

    if not isinstance(profile, dict):
        raise TypeError(f"Expected dict profile, got {type(profile)}")

    dt = (driver_type or "").strip().lower()
    fb = (human_feedback or "").strip().lower()

    # ----------------------------
    # 1️⃣ 基础结构保证
    # ----------------------------
    profile.setdefault("driver_type", "均衡")
    profile.setdefault("global", {})

    g = profile["global"]
    g.setdefault("risk_sensitivity", 0.5)
    g.setdefault("safety_weight", 0.6)
    g.setdefault("efficiency_weight", 0.4)

    # ----------------------------
    # 2️⃣ driver_type 映射
    # ----------------------------
    if dt:
        profile["driver_type"] = driver_type

    if dt in ["aggressive", "激进", "激进型"]:
        g["risk_sensitivity"] = clamp(g["risk_sensitivity"] - 0.2, 0.0, 1.0)

    elif dt in ["conservative", "保守", "保守型"]:
        g["risk_sensitivity"] = clamp(g["risk_sensitivity"] + 0.2, 0.0, 1.0)

    elif dt in ["balanced", "均衡"]:
        # 可选：轻微回归中性
        g["risk_sensitivity"] = clamp(
            0.8 * g["risk_sensitivity"] + 0.2 * 0.5, 0.0, 1.0
        )

    # ----------------------------
    # 3️⃣ human feedback
    # ----------------------------
    if any(k in fb for k in ["危险", "不安全", "too dangerous", "unsafe"]):
        g["risk_sensitivity"] = clamp(g["risk_sensitivity"] + 0.1, 0.0, 1.0)

    if any(k in fb for k in ["太慢", "低效", "too slow", "效率"]):
        g["risk_sensitivity"] = clamp(g["risk_sensitivity"] - 0.1, 0.0, 1.0)

    # ----------------------------
    # 4️⃣ 权重重计算（核心）
    # ----------------------------
    rs = g["risk_sensitivity"]

    g["safety_weight"] = round(0.3 + 0.7 * rs, 3)
    g["efficiency_weight"] = round(1.0 - g["safety_weight"], 3)

    return profile


# ================================
# 工具函数
# ================================
def _format_ranked_docs(title: str, docs) -> str:
    if not docs:
        return f"[{title}]\nNone\n"

    lines = [f"[{title}]"]
    for idx, d in enumerate(docs, start=1):
        lines.append(
            f"{idx}. id={d.doc.id}; score={d.final_score:.3f}; title={d.doc.title}; text={d.doc.text}"
        )
    lines.append("")
    return "\n".join(lines)


# ================================
# ✅ 新版 Prompt 构建（支持 dict profile）
# ================================
def make_evidence_prompts(
    profile: dict,
    scene,
    human_feedback: str,
    evidence,
) -> tuple[str, str]:

    system = (
        "你是自动驾驶交互反馈系统 ResponsiveGPT 的 Safety and Regulation Agent。\n"
        "你必须结合当前场景、驾驶人偏好、交通法规知识、风险案例知识和场景规则，"
        "判断是否存在潜在违规或高风险行为，并给出可执行建议。\n"
        "必须只输出合法 JSON，不要输出任何额外文本。\n"
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
        "}\n"
        "要求：\n"
        "1. 结论必须与检索到的证据一致。\n"
        "2. 若涉及弱势交通参与者，应提高安全优先级。\n"
        "3. evidence_ids 只能从已提供证据中选择。\n"
    )

    evidence_text = "\n".join([
        _format_ranked_docs("Relevant Laws", evidence.laws),
        _format_ranked_docs("Relevant Cases", evidence.cases),
        _format_ranked_docs("Scenario-specific Rules", evidence.scenarios),
    ])

    # ✅ 关键修复点：不再用 __dict__
    user = (
        f"[Driver Profile]\n{json.dumps(profile, ensure_ascii=False)}\n\n"
        f"[Scene State]\n{json.dumps(scene.__dict__, ensure_ascii=False)}\n\n"
        f"[Human Feedback]\n{human_feedback}\n\n"
        f"{evidence_text}\n"
        "[Task]\n"
        "请基于上述证据判断当前场景是否存在潜在违规或高风险行为，"
        "并输出结构化 JSON。"
    )

    return system, user


# ================================
# JSON 修复（保留）
# ================================
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