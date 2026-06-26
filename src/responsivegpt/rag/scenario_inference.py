def infer_scenario_type(
    *,
    dataset: str,
    event_type: str = "",
    pair_type: str = "",
    vrus_present: bool = False,
) -> str:
    dataset = (dataset or "").lower()
    event_type_l = str(event_type or "").lower()
    pair_type_l = str(pair_type or "").lower()

    if dataset == "highd":
        if "cutin" in event_type_l or "cut-in" in event_type_l:
            return "highway_cut_in"
        if "following" in event_type_l:
            return "highway_car_following"
        if "lane" in event_type_l:
            return "highway_lane_change"
        return "highway_interaction"

    if dataset == "round":
        if "cyclist" in pair_type_l or "pedestrian" in pair_type_l or vrus_present:
            return "roundabout_vru_interaction"
        return "roundabout_vehicle_interaction"

    if dataset == "ind":
        if "pedestrian" in pair_type_l or "bicycle" in pair_type_l or "cyclist" in pair_type_l or vrus_present:
            return "intersection_vru_interaction"
        return "intersection_vehicle_interaction"

    return "generic_interaction"


def dataset_metric_priorities(dataset: str, scenario_type: str) -> dict:
    dataset = (dataset or "").lower()

    if dataset == "highd":
        return {
            "primary_metrics": ["TTC", "THW", "DRAC", "headway", "relative_speed"],
            "secondary_metrics": ["DCPA", "TCPA", "min_future_distance"],
            "risk_focus": ["rear_end", "cut_in", "closing_speed", "headway_reduction"],
        }

    if dataset == "round":
        return {
            "primary_metrics": ["DCPA", "TCPA", "min_future_distance", "distance_to_ego"],
            "secondary_metrics": ["TTC", "DRAC", "relative_speed"],
            "risk_focus": ["merging_conflict", "yielding", "roundabout_crossing", "vru_conflict"],
        }

    if dataset == "ind":
        return {
            "primary_metrics": ["DCPA", "TCPA", "min_future_distance", "DRAC", "VRU"],
            "secondary_metrics": ["TTC", "THW", "headway"],
            "risk_focus": ["intersection_crossing", "vru_conflict", "future_collision_region"],
        }

    return {
        "primary_metrics": ["TTC", "DRAC", "DCPA", "min_future_distance"],
        "secondary_metrics": ["THW", "headway"],
        "risk_focus": ["generic_collision_risk"],
    }