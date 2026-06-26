import json
import copy
import re
from dataclasses import asdict, is_dataclass
from typing import Any


# =========================
# Basic utilities
# =========================

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
        text = json.dumps(obj, ensure_ascii=False, indent=2)
    except Exception:
        text = json.dumps(str(obj), ensure_ascii=False)

    if max_chars is not None:
        return _truncate(text, max_chars=max_chars)
    return text


def _safe_obj_dict(obj: Any) -> dict:
    if obj is None:
        return {}
    if isinstance(obj, dict):
        return obj
    if is_dataclass(obj):
        try:
            return asdict(obj)
        except Exception:
            pass
    if hasattr(obj, "__dict__"):
        return dict(obj.__dict__)
    return {"value": str(obj)}


def _safe_number(x, default: float) -> float:
    try:
        return float(x)
    except Exception:
        return default


# =========================
# Token-time budget config
# =========================

TOKEN_BUDGET_CONFIG = {
    "reactive_low": {
        "scene_chars": 1000,
        "profile_chars": 1000,
        "planning_chars": 800,
        "evidence_doc_chars": 360,
        "laws_k": 1,
        "cases_k": 1,
        "scenarios_k": 2,
        "rag_evidence_k": 4,
        "safety_chars": 800,
        "feedback_chars": 300,
    },
    "reactive_medium": {
        "scene_chars": 1500,
        "profile_chars": 1500,
        "planning_chars": 1200,
        "evidence_doc_chars": 500,
        "laws_k": 2,
        "cases_k": 2,
        "scenarios_k": 3,
        "rag_evidence_k": 6,
        "safety_chars": 1200,
        "feedback_chars": 500,
    },
    "reactive_high": {
        "scene_chars": 3600,
        "profile_chars": 2200,
        "planning_chars": 1600,
        "evidence_doc_chars": 900,
        "laws_k": 3,
        "cases_k": 3,
        "scenarios_k": 4,
        "rag_evidence_k": 8,
        "safety_chars": 1600,
        "feedback_chars": 1000,
    },
}


# =========================
# Profile update
# =========================

def update_profile(profile: dict, driver_type: str, human_feedback: str) -> dict:
    """
    dict 版本 profile 更新。
    返回 copy，避免在未 save 前污染原始 profile。
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

    aliases = {
        "aggressive": "aggressive",
        "激进": "aggressive",
        "激进型": "aggressive",
        "balanced": "balanced",
        "均衡": "balanced",
        "conservative": "conservative",
        "保守": "conservative",
        "保守型": "conservative",
    }
    current_type = aliases.get(
        str(profile.get("driver_type") or "").strip().lower()
    )
    requested_type = aliases.get(dt)
    weights_changed = False

    # Driver type selects the initial template. Reapplying a relative delta
    # every frame would force profiles to 0/1 independently of learning.
    if requested_type and requested_type != current_type:
        profile["driver_type"] = driver_type
        g["risk_sensitivity"] = {
            "aggressive": 0.2,
            "balanced": 0.5,
            "conservative": 0.8,
        }[requested_type]
        weights_changed = True

    if any(k in fb for k in ["危险", "不安全", "too dangerous", "unsafe"]):
        g["risk_sensitivity"] = clamp(g["risk_sensitivity"] + 0.1, 0.0, 1.0)
        weights_changed = True

    if any(k in fb for k in ["太慢", "低效", "too slow", "效率"]):
        g["risk_sensitivity"] = clamp(g["risk_sensitivity"] - 0.1, 0.0, 1.0)
        weights_changed = True

    if weights_changed:
        rs = g["risk_sensitivity"]
        g["safety_weight"] = round(0.3 + 0.7 * rs, 3)
        g["efficiency_weight"] = round(1.0 - g["safety_weight"], 3)

    return profile


# =========================
# Dataset / scenario / metric awareness
# =========================

def _infer_dataset(scene_dict: dict) -> str:
    raw = str(scene_dict.get("scene_type") or scene_dict.get("dataset") or "").lower()

    if "highd" in raw:
        return "highD"
    if "round" in raw or "roundd" in raw:
        return "rounD"
    if "ind" in raw:
        return "inD"

    return "unknown"


def _infer_scenario_type(scene_dict: dict) -> str:
    dataset = _infer_dataset(scene_dict).lower()
    event_type = str(scene_dict.get("event_type") or "").lower()
    vrus_present = bool(scene_dict.get("vrus_present", False))

    if dataset == "highd":
        if "cutin" in event_type or "cut-in" in event_type:
            return "highway_cut_in"
        if "following" in event_type:
            return "highway_car_following"
        if "lane" in event_type:
            return "highway_lane_change"
        return "highway_interaction"

    if dataset == "round":
        if vrus_present or "pedestrian" in event_type or "cyclist" in event_type or "bicycle" in event_type:
            return "roundabout_vru_interaction"
        return "roundabout_vehicle_interaction"

    if dataset == "ind":
        if vrus_present or "pedestrian" in event_type or "cyclist" in event_type or "bicycle" in event_type:
            return "intersection_vru_interaction"
        return "intersection_vehicle_interaction"

    return "generic_interaction"


def _metric_priority_for(dataset: str, scenario_type: str) -> dict:
    dataset_l = str(dataset or "").lower()
    scenario_l = str(scenario_type or "").lower()

    if dataset_l == "highd":
        if "cut_in" in scenario_l or "cutin" in scenario_l:
            return {
                "primary_metrics": ["TTC", "THW", "DRAC", "headway", "relative_speed"],
                "secondary_metrics": ["DCPA", "TCPA", "min_future_distance"],
                "risk_focus": "sudden headway reduction, fast closing, cut-in conflict",
                "preferred_actions": ["increase_headway", "decelerate", "avoid_lane_change"],
            }

        return {
            "primary_metrics": ["TTC", "THW", "DRAC", "headway", "relative_speed"],
            "secondary_metrics": ["DCPA", "TCPA", "min_future_distance"],
            "risk_focus": "longitudinal following risk and hard-braking demand",
            "preferred_actions": ["increase_headway", "decelerate", "maintain_speed"],
        }

    if dataset_l == "round":
        return {
            "primary_metrics": ["DCPA", "TCPA", "min_future_distance", "distance_to_ego"],
            "secondary_metrics": ["TTC", "DRAC", "relative_speed"],
            "risk_focus": "roundabout merging/crossing conflict and closest approach risk",
            "preferred_actions": ["yield", "decelerate", "monitor_conflict_point"],
        }

    if dataset_l == "ind":
        if "vru" in scenario_l:
            return {
                "primary_metrics": ["min_future_distance", "DCPA", "TCPA", "VRU_presence", "DRAC"],
                "secondary_metrics": ["TTC", "THW", "headway"],
                "risk_focus": "intersection VRU conflict and future spatial proximity",
                "preferred_actions": ["yield", "monitor_vru", "decelerate"],
            }

        return {
            "primary_metrics": ["DCPA", "TCPA", "min_future_distance", "DRAC"],
            "secondary_metrics": ["TTC", "THW", "headway"],
            "risk_focus": "intersection crossing conflict and required deceleration",
            "preferred_actions": ["yield", "decelerate", "monitor_crossing_vehicle"],
        }

    return {
        "primary_metrics": ["TTC", "DRAC", "DCPA", "min_future_distance"],
        "secondary_metrics": ["THW", "headway", "relative_speed"],
        "risk_focus": "generic near-term physical interaction risk",
        "preferred_actions": ["increase_headway", "decelerate", "keep_current"],
    }


# =========================
# Evidence formatting
# =========================

def _extract_doc_fields(d, fallback_id: str) -> dict:
    """
    Support multiple retriever return formats:
    - d.doc.id / d.doc.title / d.doc.text
    - d.final_score
    - plain dictionary
    """

    if isinstance(d, dict):
        return {
            "id": d.get("id") or d.get("doc_id") or d.get("source_id") or fallback_id,
            "score": d.get("score") or d.get("final_score"),
            "title": d.get("title") or d.get("name") or "",
            "text": d.get("text") or d.get("content") or d.get("body") or str(d),
        }

    doc = getattr(d, "doc", None)
    if doc is not None:
        return {
            "id": getattr(doc, "id", fallback_id),
            "score": getattr(d, "final_score", None),
            "title": getattr(doc, "title", ""),
            "text": getattr(doc, "text", ""),
        }

    dd = _safe_obj_dict(d)
    return {
        "id": dd.get("id") or dd.get("doc_id") or fallback_id,
        "score": dd.get("score") or dd.get("final_score"),
        "title": dd.get("title") or "",
        "text": dd.get("text") or dd.get("content") or str(d),
    }


def _format_ranked_docs(
    title: str,
    docs,
    top_k: int = 2,
    max_doc_chars: int = 600,
) -> str:
    """
    Keep evidence compact for token-time constrained Reactive reasoning.
    """

    docs = docs or []
    if not docs:
        return f"[{title}]\nNone\n"

    lines = [f"[{title}]"]

    for idx, d in enumerate(docs[:top_k], start=1):
        fields = _extract_doc_fields(d, fallback_id=f"{title}_{idx}")
        doc_id = fields["id"]
        score = fields["score"]
        doc_title = fields["title"]
        doc_text = _truncate(fields["text"], max_chars=max_doc_chars)

        if isinstance(score, (int, float)):
            score_text = f"{score:.3f}"
        else:
            score_text = str(score)

        lines.append(
            f"{idx}. id={doc_id}; score={score_text}; title={doc_title}; text={doc_text}"
        )

    lines.append("")
    return "\n".join(lines)

def _format_rag_evidence_pack(
    evidence_pack: dict | None,
    max_items: int = 4,
    max_doc_chars: int = 360,
) -> str:
    """
    Format RAG v1 evidence pack into a compact prompt block.

    Expected evidence_pack:
    {
        "items": [
            {
                "evidence_id": "...",
                "doc_type": "law|scenario|safety|case|policy",
                "title": "...",
                "text": "...",
                "source": "...",
                "dataset_tags": [...],
                "scenario_tags": [...],
                "metric_tags": [...],
                "risk_tags": [...]
            }
        ]
    }
    """
    if not evidence_pack:
        return "No external RAG evidence provided."

    items = evidence_pack.get("items", [])
    if not items:
        return "No external RAG evidence retrieved."

    lines = []

    for i, item in enumerate(items[:max_items], 1):
        evidence_id = (
            item.get("evidence_id")
            or item.get("chunk_id")
            or item.get("doc_id")
            or item.get("id")
            or f"E{i}"
        )

        doc_type = item.get("doc_type", "unknown")
        title = item.get("title", "")
        source = item.get("source", "")
        text = item.get("text", "")

        dataset_tags = item.get("dataset_tags", [])
        scenario_tags = item.get("scenario_tags", [])
        metric_tags = item.get("metric_tags", [])
        risk_tags = item.get("risk_tags", [])

        text = _truncate(str(text), max_doc_chars)

        lines.append(
            f"[E{i}]\n"
            f"evidence_id={evidence_id}\n"
            f"type={doc_type}\n"
            f"title={title}\n"
            f"source={source}\n"
            f"dataset_tags={dataset_tags}\n"
            f"scenario_tags={scenario_tags}\n"
            f"metric_tags={metric_tags}\n"
            f"risk_tags={risk_tags}\n"
            f"text={text}"
        )

    return "\n\n".join(lines)


def _extract_available_evidence_ids(evidence_pack: dict | None) -> list[str]:
    if not evidence_pack:
        return []

    ids = []
    for item in evidence_pack.get("items", []):
        evidence_id = (
            item.get("evidence_id")
            or item.get("chunk_id")
            or item.get("doc_id")
            or item.get("id")
        )
        if evidence_id is not None:
            ids.append(str(evidence_id))

    return ids


# =========================
# Safety formatting
# =========================

def _format_frame_safety(
    frame_safety,
    dataset: str = "unknown",
    scenario_type: str = "generic_interaction",
    metric_priority: dict | None = None,
    max_chars: int = 1200,
) -> str:
    if frame_safety is None:
        return "None"

    fs = _safe_obj_dict(frame_safety)
    metric_priority = metric_priority or {}

    keep = {
        "dataset": dataset,
        "scenario_type": scenario_type,
        "primary_metrics": metric_priority.get("primary_metrics", []),
        "secondary_metrics": metric_priority.get("secondary_metrics", []),
        "risk_focus": metric_priority.get("risk_focus", ""),
        "values": {
            "frame_index": fs.get("frame_index"),
            "distance_m": fs.get("distance_m"),
            "rel_speed_mps": fs.get("rel_speed_mps"),
            "ttc_s": fs.get("ttc_s"),
            "thw_s": fs.get("thw_s"),
            "drac_mps2": fs.get("drac_mps2"),
            "dcpa_m": fs.get("dcpa_m"),
            "ttca_s": fs.get("ttca_s"),
            "predicted_ttc_s": fs.get("predicted_ttc_s"),
            "min_future_distance_m": fs.get("min_future_distance_m"),
            "physical_risk_index": fs.get("physical_risk_index"),
            "physical_risk_level": fs.get("physical_risk_level"),
        },
        "flags": {
            "closing": fs.get("closing"),
            "unsafe_ttc": fs.get("unsafe_ttc"),
            "unsafe_thw": fs.get("unsafe_thw"),
            "unsafe_drac": fs.get("unsafe_drac"),
            "unsafe_dcpa": fs.get("unsafe_dcpa"),
            "unsafe_future_distance": fs.get("unsafe_future_distance"),
        },
        "reactive_instruction": (
            "If any unsafe flag is true or the physical_risk_level is high, "
            "prefer a conservative safety decision regardless of planning insight."
        ),
    }

    return _json_dumps(keep, max_chars=max_chars)


# =========================
# Planning hint formatting
# =========================

def _try_parse_json(text: str):
    if not text:
        return None

    text = str(text).strip()

    try:
        return json.loads(text)
    except Exception:
        pass

    cleaned = re.sub(r"^```(?:json)?", "", text.strip(), flags=re.IGNORECASE).strip()
    cleaned = re.sub(r"```$", "", cleaned).strip()

    try:
        return json.loads(cleaned)
    except Exception:
        pass

    l = cleaned.find("{")
    r = cleaned.rfind("}")
    if l != -1 and r != -1 and r > l:
        try:
            return json.loads(cleaned[l:r + 1])
        except Exception:
            return None

    return None


def _compact_planning_dict(p: dict) -> dict:
    """
    Keep only compact, actionable planning information for the Reactive Thread.
    """

    risk_forecast = p.get("risk_forecast", {}) if isinstance(p.get("risk_forecast"), dict) else {}
    strategy = p.get("recommended_strategy", {}) if isinstance(p.get("recommended_strategy"), dict) else {}
    guidance = p.get("reactive_guidance", {}) if isinstance(p.get("reactive_guidance"), dict) else {}
    staleness = p.get("staleness_control", {}) if isinstance(p.get("staleness_control"), dict) else {}

    return {
        "planning_schema_version": p.get("planning_schema_version"),
        "scenario_assessment": p.get("scenario_assessment", {}),
        "risk_level": risk_forecast.get("risk_level"),
        "risk_trend": risk_forecast.get("risk_trend"),
        "main_risk_factors": risk_forecast.get("main_risk_factors", []),
        "metric_evidence": risk_forecast.get("metric_evidence", {}),
        "strategy": strategy.get("strategy"),
        "strategy_priority": strategy.get("priority"),
        "must_check": guidance.get("must_check", []),
        "avoid_actions": guidance.get("avoid_actions", []),
        "preferred_actions": guidance.get("preferred_actions", []),
        "safety_constraints": guidance.get("safety_constraints", []),
        "fast_rule_hint": guidance.get("fast_rule_hint", ""),
        "valid_for_frames": staleness.get("valid_for_frames"),
        "staleness_risk": staleness.get("staleness_risk"),
        "confidence": p.get("confidence"),
    }


def _format_planning_hint(
    planning_hint: str = "",
    planning_metadata: dict | None = None,
    max_chars: int = 1000,
) -> str:
    if not planning_hint and not planning_metadata:
        return ""

    planning_metadata = planning_metadata or {}

    parsed = None
    if isinstance(planning_metadata.get("planning"), dict):
        parsed = planning_metadata["planning"]
    else:
        parsed = _try_parse_json(planning_hint)

    if isinstance(parsed, dict):
        compact = _compact_planning_dict(parsed)
        compact["planning_age_frames"] = planning_metadata.get("planning_age_frames")
        compact["last_update_frame"] = planning_metadata.get("last_update_frame")
        compact["priority_rule"] = (
            "Planning is advisory. Latest observation and current physical safety metrics have priority."
        )
        return _json_dumps(compact, max_chars=max_chars)

    text = str(planning_hint or "").strip()
    if not text:
        return ""

    fallback = {
        "planning_age_frames": planning_metadata.get("planning_age_frames"),
        "last_update_frame": planning_metadata.get("last_update_frame"),
        "planning_text": _truncate(text, max_chars=max_chars),
        "priority_rule": (
            "Planning is advisory. Latest observation and current physical safety metrics have priority."
        ),
    }

    return _json_dumps(fallback, max_chars=max_chars)


# =========================
# Reactive system prompt
# =========================

def _build_reactive_system_prompt(
    dataset: str,
    scenario_type: str,
    metric_priority: dict,
    token_budget_class: str,
) -> str:
    return (
        "You are the Fast Reactive Thread of ResponsiveGPT, an autonomous driving interaction feedback system.\n"
        "Your role is to make a fast, safe, and feasible frame-level decision under strict real-time constraints.\n"
        "You must use the latest observation, driver profile, physical safety metrics, traffic-law evidence, "
        "risk-case evidence, and scenario-specific rules.\n\n"

        "You may receive planning insight from a slower Planning Thread. "
        "This planning insight is advisory memory only. It must never override the latest observation "
        "or the current physical safety metrics.\n\n"

        f"Dataset: {dataset}\n"
        f"Scenario type: {scenario_type}\n"
        f"Reactive token budget: {token_budget_class}\n"
        f"Primary risk metrics: {metric_priority.get('primary_metrics', [])}\n"
        f"Secondary risk metrics: {metric_priority.get('secondary_metrics', [])}\n"
        f"Risk focus: {metric_priority.get('risk_focus', '')}\n"
        f"Preferred safe actions: {metric_priority.get('preferred_actions', [])}\n\n"

        "Reasoning constraints:\n"
        "1. The latest observation has the highest priority.\n"
        "2. Current physical safety metrics have higher priority than planning insight.\n"
        "3. If TTC, THW, DRAC, DCPA, TCPA, min_future_distance, or physical_risk_index indicates high risk, "
        "prefer a conservative safety decision.\n"
        "4. If pedestrians, cyclists, bicycles, or other vulnerable road users are involved, increase safety priority.\n"
        "5. You are operating under a low-token, low-latency Reactive mode. "
        "Return a fast feasible decision, not a globally optimal plan.\n"
        "6. Do not output long reasoning. Keep rationale short.\n"
        "7. Output valid JSON only. Do not output markdown or any extra text.\n\n"

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

        "Additional requirements:\n"
        "1. evidence_ids must only be selected from the provided evidence.\n"
        "2. If evidence is insufficient but the physical metrics indicate high risk, still mark it as potential risk.\n"
        "3. The rationale must be short and should prioritize key physical metrics and scenario mechanism.\n"
        "4. recommended_action must be executable, such as decelerate, yield, increase_headway, monitor_vru, or keep_current.\n"
    )


# =========================
# Main prompt builder
# =========================

def make_evidence_prompts(
    profile: dict,
    scene,
    human_feedback: str,
    evidence=None,
    evidence_pack: dict | None = None,
    planning_hint: str = "",
    planning_metadata: dict | None = None,
    frame_safety=None,
    token_budget_class: str = "reactive_low",
    require_grounded_decision: bool = False,
) -> tuple[str, str]:
    """
    Build prompts for the Fast Reactive Thread.

    RAG v1 design goals:
    1. Dataset-aware: adapt risk focus to highD, rounD, and inD.
    2. Scenario-aware: distinguish following, cut-in, roundabout, intersection, and VRU scenarios.
    3. Risk-metric-aware: prioritize TTC, DRAC, DCPA, TCPA, and minimum future distance when available.
    4. Token-time-aware: keep the prompt compact for low-latency reactive reasoning.
    5. Planning-aware: allow the Reactive Thread to use PlanningMemory, but never override current safety metrics.
    6. RAG-grounded: use evidence_pack and require evidence_id citation when evidence is used.
    7. Schema-stable: keep JSON output stable for downstream parsing and grounding validation.
    """

    cfg = TOKEN_BUDGET_CONFIG.get(
        token_budget_class,
        TOKEN_BUDGET_CONFIG["reactive_low"],
    )

    scene_dict = _safe_obj_dict(scene)
    dataset = _infer_dataset(scene_dict)
    scenario_type = _infer_scenario_type(scene_dict)
    metric_priority = _metric_priority_for(dataset, scenario_type)

    system = _build_reactive_system_prompt(
        dataset=dataset,
        scenario_type=scenario_type,
        metric_priority=metric_priority,
        token_budget_class=token_budget_class,
    )

    # ==================================================
    # 1. Legacy evidence formatting
    # ==================================================
    # 保留旧接口，避免历史代码直接崩。
    legacy_evidence_text = ""

    if evidence is not None:
        laws = getattr(evidence, "laws", [])
        cases = getattr(evidence, "cases", [])
        scenarios = getattr(evidence, "scenarios", [])

        legacy_evidence_text = "\n".join([
            _format_ranked_docs(
                title="Relevant Laws",
                docs=laws,
                top_k=cfg.get("laws_k", 2),
                max_doc_chars=cfg.get("evidence_doc_chars", 360),
            ),
            _format_ranked_docs(
                title="Relevant Cases",
                docs=cases,
                top_k=cfg.get("cases_k", 2),
                max_doc_chars=cfg.get("evidence_doc_chars", 360),
            ),
            _format_ranked_docs(
                title="Scenario-specific Rules",
                docs=scenarios,
                top_k=cfg.get("scenarios_k", 2),
                max_doc_chars=cfg.get("evidence_doc_chars", 360),
            ),
        ])

    # ==================================================
    # 2. RAG v1 evidence_pack formatting
    # ==================================================
    rag_evidence_text = _format_rag_evidence_pack(
        evidence_pack=evidence_pack,
        max_items=cfg.get("rag_evidence_k", 4),
        max_doc_chars=cfg.get("evidence_doc_chars", 360),
    )

    available_evidence_ids = _extract_available_evidence_ids(evidence_pack)

    # ==================================================
    # 3. Safety + Planning formatting
    # ==================================================
    safety_text = _format_frame_safety(
        frame_safety=frame_safety,
        dataset=dataset,
        scenario_type=scenario_type,
        metric_priority=metric_priority,
        max_chars=cfg.get("safety_chars", 1000),
    )

    planning_text = _format_planning_hint(
        planning_hint=planning_hint,
        planning_metadata=planning_metadata,
        max_chars=cfg.get("planning_chars", 1000),
    )

    user_parts = []

    # ==================================================
    # 4. Token-Time setting
    # ==================================================
    user_parts.append(
        "[Token-Time Setting]\n"
        f"thread=Reactive\n"
        f"token_budget_class={token_budget_class}\n"
        "objective=fast_safe_feasible_decision\n"
        "latency_priority=high\n"
        "optimality_requirement=feasible_not_global_optimal\n"
        "reasoning_style=compact_conclusion_only\n"
    )

    # ==================================================
    # 5. Dataset / scenario / metric context
    # ==================================================
    user_parts.append(
        "[Dataset-aware Context]\n"
        f"dataset={dataset}\n"
        f"scenario_type={scenario_type}\n"
        f"primary_metrics={metric_priority['primary_metrics']}\n"
        f"secondary_metrics={metric_priority['secondary_metrics']}\n"
        f"risk_focus={metric_priority['risk_focus']}\n"
        f"preferred_safe_actions={metric_priority['preferred_actions']}\n"
    )

    # ==================================================
    # 6. Driver profile
    # ==================================================
    user_parts.append(
        "[Driver Profile]\n"
        f"{_json_dumps(profile, max_chars=cfg.get('profile_chars', 1200))}\n"
    )

    # ==================================================
    # 7. Latest scene
    # ==================================================
    user_parts.append(
        "[Latest Scene State]\n"
        f"{_json_dumps(scene_dict, max_chars=cfg.get('scene_chars', 1200))}\n"
    )

    # ==================================================
    # 8. Current physical safety metrics
    # ==================================================
    user_parts.append(
        "[Current Physical Safety Metrics]\n"
        f"{safety_text}\n"
    )

    # ==================================================
    # 9. Human feedback
    # ==================================================
    user_parts.append(
        "[Human Feedback]\n"
        f"{_truncate(str(human_feedback or ''), cfg.get('feedback_chars', 500))}\n"
    )

    # ==================================================
    # 10. RAG v1 evidence pack
    # ==================================================
    user_parts.append(
        "[RAG Evidence Pack]\n"
        "The following evidence was retrieved by the scenario-aware RAG module.\n"
        "Use it only if it is relevant to the current scene and physical risk.\n\n"
        f"{rag_evidence_text}\n\n"
        f"Available evidence_ids={available_evidence_ids}\n\n"
        "Evidence usage rules:\n"
        "- If you use any evidence, cite its evidence_id in used_evidence_ids.\n"
        "- Do not invent evidence IDs.\n"
        "- If no evidence is useful, set used_evidence_ids=[] and evidence_support_level=\"none\".\n"
        "- If evidence is weak or only partially relevant, set evidence_support_level=\"weak\".\n"
        "- The latest scene and physical safety metrics override stale or irrelevant evidence.\n"
    )

    # ==================================================
    # 11. Legacy evidence block, optional
    # ==================================================
    # 如果你仍然希望兼容旧 laws/cases/scenarios，可保留。
    # 如果完全切到 RAG v1，也可以删掉这一段。
    if legacy_evidence_text.strip():
        user_parts.append(
            "[Legacy Evidence]\n"
            "The following legacy evidence is provided for backward compatibility.\n"
            "Prefer RAG Evidence Pack when evidence_id citation is required.\n\n"
            f"{legacy_evidence_text}\n"
        )

    # ==================================================
    # 12. Planning insight
    # ==================================================
    if planning_text.strip():
        user_parts.append(
            "[Advisory Planning Insight]\n"
            f"{planning_text}\n\n"
            "Planning usage rules:\n"
            "- Planning insight is advisory only.\n"
            "- Planning may be stale.\n"
            "- If planning conflicts with the latest scene or physical safety metrics, ignore planning.\n"
            "- Use planning only to choose among safe feasible decisions.\n"
        )

    # ==================================================
    # 13. Reactive decision task + strict schema
    # ==================================================
    grounding_rule = (
        "If relevant evidence exists and supports your decision, you should cite at least one valid evidence_id. "
        if require_grounded_decision
        else "Use evidence IDs only when the evidence actually supports the decision. "
    )

    user_parts.append(
        "[Reactive Decision Task]\n"
        "Use the latest scene, current physical safety metrics, driver profile, RAG evidence, "
        "and optional planning insight to quickly determine whether the current frame contains "
        "a potential traffic-rule violation or high-risk behavior.\n\n"

        "Decision priority:\n"
        "1. Latest observation.\n"
        "2. Current physical safety metrics.\n"
        "3. Dataset-specific and scenario-specific risk mechanism.\n"
        "4. RAG evidence.\n"
        "5. Planning insight.\n\n"

        "Reactive constraints:\n"
        "- You are the fast Reactive Thread.\n"
        "- Operate under strict real-time and token-budget constraints.\n"
        "- Prefer a safe feasible decision quickly.\n"
        "- The decision does not need to be globally optimal.\n"
        "- Keep the reason concise.\n\n"

        "RAG grounding requirement:\n"
        f"- {grounding_rule}\n"
        "- used_evidence_ids must only contain IDs listed in Available evidence_ids.\n"
        "- Never hallucinate or fabricate evidence IDs.\n\n"

        "Output requirements:\n"
        "- Return valid JSON only.\n"
        "- Do not output markdown.\n"
        "- Do not output text outside JSON.\n\n"

        "Required JSON schema:\n"
        "{\n"
        "  \"is_potential_violation\": true or false,\n"
        "  \"risk_level\": \"low | medium | high | unknown\",\n"
        "  \"recommended_action\": \"short action recommendation\",\n"
        "  \"warning\": \"short warning if needed\",\n"
        "  \"reason\": \"brief reason grounded in latest scene, safety metrics, and useful evidence\",\n"
        "  \"confidence\": 0.0,\n"
        "  \"used_evidence_ids\": [],\n"
        "  \"evidence_support_level\": \"strong | medium | weak | none\"\n"
        "}\n"
    )

    user = "\n\n".join(user_parts)

    return system, user


# =========================
# JSON parsing / validation
# =========================

def coerce_json(text: str) -> dict:
    """
    Parse JSON from LLM output.

    This function only parses JSON and does not repair the decision schema.
    PlanningService may also reuse this function for planning JSON.
    """

    obj = _try_parse_json(text)
    if isinstance(obj, dict):
        return obj

    raise ValueError("LLM output is not valid JSON.")


def validate_decision_json(obj: dict) -> dict:
    """
    Repair Reactive Thread decision JSON to keep downstream modules stable.

    This function is intended for Reactive decisions only.
    Do not use it for Planning Thread outputs.

    RAG v1 compatible:
    - preserves legacy evidence_ids
    - adds used_evidence_ids
    - adds evidence_support_level
    - supports reason/rationale alias
    """

    if not isinstance(obj, dict):
        obj = {}

    # ==================================================
    # 1. risk level
    # ==================================================
    risk_level = str(obj.get("risk_level", "low")).lower()
    if risk_level not in {"low", "medium", "high", "unknown"}:
        risk_level = "unknown"

    # ==================================================
    # 2. violation
    # ==================================================
    is_violation = obj.get("is_potential_violation")
    if not isinstance(is_violation, bool):
        is_violation = risk_level == "high"

    # ==================================================
    # 3. reason / rationale compatibility
    # ==================================================
    reason = obj.get("reason")
    if reason is None:
        reason = obj.get("rationale", "")

    rationale = obj.get("rationale")
    if rationale is None:
        rationale = reason

    # ==================================================
    # 4. evidence ids compatibility
    # ==================================================
    # New RAG v1 field
    used_evidence_ids = obj.get("used_evidence_ids")

    # Legacy field
    legacy_evidence_ids = obj.get("evidence_ids")

    if not isinstance(used_evidence_ids, list):
        if isinstance(legacy_evidence_ids, list):
            used_evidence_ids = legacy_evidence_ids
        else:
            used_evidence_ids = []

    used_evidence_ids = [str(x) for x in used_evidence_ids if x is not None]

    # 保留旧字段 evidence_ids，防止旧代码断掉
    evidence_ids = obj.get("evidence_ids")
    if not isinstance(evidence_ids, list):
        evidence_ids = used_evidence_ids

    evidence_ids = [str(x) for x in evidence_ids if x is not None]

    # ==================================================
    # 5. evidence support level
    # ==================================================
    evidence_support_level = str(
        obj.get("evidence_support_level", "")
    ).lower()

    if evidence_support_level not in {"strong", "medium", "weak", "none"}:
        evidence_support_level = "medium" if used_evidence_ids else "none"

    # ==================================================
    # 6. confidence
    # ==================================================
    confidence = _safe_number(obj.get("confidence"), 0.5)
    if confidence < 0.0:
        confidence = 0.0
    if confidence > 1.0:
        confidence = 1.0

    # ==================================================
    # 7. tuning suggestion
    # ==================================================
    tuning = obj.get("tuning_suggestion")
    if not isinstance(tuning, dict):
        tuning = {}
    repaired_tuning = {}
    for key in (
        "risk_sensitivity",
        "safety_weight",
        "efficiency_weight",
    ):
        if key not in tuning:
            continue
        value = _safe_number(tuning.get(key), None)
        if value is not None:
            repaired_tuning[key] = clamp(value, 0.0, 1.0)

    # ==================================================
    # 8. recommended action / warning
    # ==================================================
    warning = str(obj.get("warning", ""))
    recommended_action = str(obj.get("recommended_action", ""))

    # ==================================================
    # 9. return repaired decision
    # ==================================================
    repaired = {
        "is_potential_violation": is_violation,
        "risk_level": risk_level,
        "warning": warning,
        "recommended_action": recommended_action,

        # 新字段：RAG v1 / readable schema
        "reason": _truncate(str(reason), 600),
        "confidence": confidence,
        "used_evidence_ids": used_evidence_ids,
        "evidence_support_level": evidence_support_level,

        # 旧字段：保持兼容
        "rationale": _truncate(str(rationale), 600),
        "evidence_ids": evidence_ids,

        "tuning_suggestion": repaired_tuning,
    }

    # ==================================================
    # 10. preserve additional fields
    # ==================================================
    # 避免以后 prompt/schema 扩展后字段被吞掉
    for k, v in obj.items():
        if k not in repaired:
            repaired[k] = v

    return repaired
