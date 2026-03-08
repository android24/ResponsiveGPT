def highd_rules():
    return [
        {"id": "highd_001", "text": "高速场景中应避免无充分间隙的强行变道；变道前需确认后方来车速度与安全间隙。"},
        {"id": "highd_002", "text": "跟车距离不足或频繁急加减速会显著提高追尾风险；应优先保持安全车距。"},
        {"id": "highd_003", "text": "当最小TTC和最小车头时距持续下降时，应及时减速或中止激进操作。"},
        {"id": "highd_004", "text": "激进策略可关注效率，但不得以明显增大碰撞风险为代价。"},
    ]

def roundd_rules():
    return [
        {"id": "round_001", "text": "进入环岛前应让行已在环岛内通行车辆，避免强行切入。"},
        {"id": "round_002", "text": "环岛内车辆应保持稳定速度与轨迹，避免大角度急转或突然变向。"},
        {"id": "round_003", "text": "最小碰撞时间和最近点距离持续降低时，应及时减速或让行，不应继续激进抢行。"},
        {"id": "round_004", "text": "摩托车、自行车等小型交通参与者与汽车同时接近汇入或出口区域时，应优先考虑横向冲突风险。"},
    ]

def generic_rules():
    return [
        {"id": "generic_001", "text": "存在高风险迹象时，应优先采取保守、安全的规避操作。"},
        {"id": "generic_002", "text": "在不确定性较高的交互场景中，应通过减速、让行或保持稳定轨迹降低风险。"},
    ]

def rules_for_scene(scene_name: str):
    name = (scene_name or "").lower()
    if name == "highd":
        return highd_rules() + generic_rules()
    if name == "round":
        return roundd_rules() + generic_rules()
    if name == "roundd":
        return roundd_rules() + generic_rules()
    if name == "roundabout":
        return roundd_rules() + generic_rules()
    if name == "roundd":
        return roundd_rules() + generic_rules()
    if name == "roundd":
        return roundd_rules() + generic_rules()
    if name == "roundd":
        return roundd_rules() + generic_rules()
    return generic_rules()
