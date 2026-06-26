import json


def summarize_scene(scene) -> dict:
    return {
        "frame": scene.frame_index,
        "scene_type": scene.scene_type,
        "event_type": scene.event_type,
        "ego_speed": scene.ego_speed_mps,
        "lead_speed": scene.lead_speed_mps,
        "rel_speed": scene.rel_speed_mps,
        "headway": scene.headway_m,
        "ego_xy": [scene.ego_x, scene.ego_y],
        "other_xy": [scene.other_x, scene.other_y],
        "ego_v": [getattr(scene, "ego_vx", None), getattr(scene, "ego_vy", None)],
        "other_v": [getattr(scene, "other_vx", None), getattr(scene, "other_vy", None)],
        "vrus_present": scene.vrus_present,
    }


def summarize_safety(frame_safety) -> dict:
    return {
        "frame": frame_safety.frame_index,
        "ttc": frame_safety.ttc_s,
        "thw": frame_safety.thw_s,
        "drac": frame_safety.drac_mps2,
        "dcpa": frame_safety.dcpa_m,
        "ttca": frame_safety.ttca_s,
        "predicted_ttc": frame_safety.predicted_ttc_s,
        "min_future_distance": frame_safety.min_future_distance_m,
        "risk_index": frame_safety.physical_risk_index,
        "risk_level": frame_safety.physical_risk_level,
    }


def summarize_decision(decision: dict) -> dict:
    return {
        "is_potential_violation": decision.get("is_potential_violation"),
        "risk_level": decision.get("risk_level"),
        "recommended_action": decision.get("recommended_action"),
        "source": decision.get("_source", "llm"),
    }


def compact_json(items, max_items=12, max_chars=3000) -> str:
    subset = items[-max_items:]
    text = json.dumps(subset, ensure_ascii=False)
    if len(text) > max_chars:
        return text[:max_chars] + "..."
    return text