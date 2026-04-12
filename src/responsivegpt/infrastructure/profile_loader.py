from copy import deepcopy


def _deep_update(base: dict, override: dict):
    """
    递归 merge dict（只覆盖 override 中出现的字段）
    """
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            _deep_update(base[k], v)
        else:
            base[k] = v
    return base


def materialize_profile_for_scene(profile: dict, scene) -> dict:
    """
    根据 scene_type / event_type / pair_type 生成当前帧使用的 profile

    优先级：
    base_profile
      -> scenario_bias
      -> scene_preferences（可选）
    """

    p = deepcopy(profile)

    scene_type = getattr(scene, "scene_type", None)
    event_type = getattr(scene, "event_type", None)

    # ----------------------------
    # 1️⃣ 应用 scenario_bias
    # ----------------------------
    scenario_bias = p.get("scenario_bias", {})

    if scene_type and scene_type in scenario_bias:
        bias = scenario_bias[scene_type]

        # 支持两种写法：
        # A: { "risk_sensitivity": ... }（global）
        # B: { "longitudinal": {...}, "interaction": {...} }

        for k, v in bias.items():
            if isinstance(v, dict):
                # 分层覆盖
                if k in p:
                    _deep_update(p[k], v)
            else:
                # 默认认为属于 global
                p.setdefault("global", {})
                p["global"][k] = v

    # ----------------------------
    # 2️⃣ 应用 scene_preferences（更细粒度）
    # ----------------------------
    scene_prefs = p.get("scene_preferences", {})

    if scene_type and scene_type in scene_prefs:
        type_block = scene_prefs[scene_type]

        # event_type 优先
        if event_type and event_type in type_block:
            _deep_update(p, type_block[event_type])

        # pair_type（如果存在）
        pair_type = getattr(scene, "pair_type", None)
        if pair_type and pair_type in type_block:
            _deep_update(p, type_block[pair_type])

    return p