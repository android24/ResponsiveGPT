def infer_scenario_type(dataset: str, event_type: str = "", pair_type: str = "", vrus_present: bool = False) -> str:
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


def dataset_planning_profile(dataset: str) -> dict:
    dataset = (dataset or "").lower()

    if dataset == "highd":
        return {
            "dataset_name": "highD",
            "environment": "highway",
            "dominant_interaction": "longitudinal following, cut-in, lane-change interaction",
            "primary_risk_metrics": ["TTC", "THW", "DRAC", "headway", "relative speed"],
            "secondary_risk_metrics": ["DCPA", "TCPA", "min_future_distance"],
            "planning_focus": [
                "detect fast closing behavior",
                "anticipate cut-in or following conflict",
                "maintain sufficient headway",
                "avoid aggressive acceleration under low TTC",
            ],
            "preferred_safe_strategies": [
                "increase_headway",
                "decelerate",
                "avoid_lane_change",
                "maintain_speed_when_safe",
            ],
        }

    if dataset == "round":
        return {
            "dataset_name": "rounD",
            "environment": "roundabout",
            "dominant_interaction": "merging, yielding, circulating vehicle interaction, VRU interaction",
            "primary_risk_metrics": ["DCPA", "TCPA", "min_future_distance", "distance_to_ego"],
            "secondary_risk_metrics": ["TTC", "DRAC", "relative speed"],
            "planning_focus": [
                "anticipate crossing or merging conflict",
                "identify closest point of approach",
                "yield to high-priority or vulnerable road users",
                "avoid entering conflict zone under high spatial risk",
            ],
            "preferred_safe_strategies": [
                "yield",
                "decelerate",
                "monitor_vru",
                "increase_headway",
            ],
        }

    if dataset == "ind":
        return {
            "dataset_name": "inD",
            "environment": "urban intersection",
            "dominant_interaction": "intersection crossing, vehicle-VRU interaction, multi-agent conflict",
            "primary_risk_metrics": ["DCPA", "TCPA", "min_future_distance", "DRAC", "VRU presence"],
            "secondary_risk_metrics": ["TTC", "THW", "headway"],
            "planning_focus": [
                "anticipate conflict at intersection crossing zone",
                "prioritize vulnerable road users",
                "avoid entering predicted collision region",
                "monitor future minimum distance and required deceleration",
            ],
            "preferred_safe_strategies": [
                "yield",
                "decelerate",
                "monitor_vru",
                "increase_headway",
            ],
        }

    return {
        "dataset_name": dataset or "unknown",
        "environment": "unknown",
        "dominant_interaction": "generic traffic interaction",
        "primary_risk_metrics": ["TTC", "DRAC", "DCPA", "min_future_distance"],
        "secondary_risk_metrics": ["THW", "headway", "relative speed"],
        "planning_focus": [
            "detect near-term risk",
            "maintain safe distance",
            "avoid conflict",
        ],
        "preferred_safe_strategies": [
            "keep_current",
            "increase_headway",
            "decelerate",
        ],
    }


def scenario_planning_guidance(scenario_type: str) -> dict:
    scenario_type = scenario_type or "generic_interaction"

    mapping = {
        "highway_car_following": {
            "key_questions": [
                "Is ego closing in on the lead vehicle?",
                "Is TTC or THW decreasing?",
                "Is DRAC increasing, indicating hard braking demand?",
            ],
            "avoid": ["aggressive acceleration", "reducing headway"],
            "prefer": ["increase_headway", "maintain_speed", "decelerate"],
        },
        "highway_cut_in": {
            "key_questions": [
                "Is another vehicle cutting into ego's path?",
                "Does the interaction create a sudden headway reduction?",
                "Is there a sharp increase in DRAC or decrease in TTC?",
            ],
            "avoid": ["accelerating into cut-in gap", "late reaction"],
            "prefer": ["decelerate", "increase_headway", "avoid_lane_change"],
        },
        "roundabout_vehicle_interaction": {
            "key_questions": [
                "Will ego and other reach the conflict zone simultaneously?",
                "Is DCPA small and TCPA near future?",
                "Should ego yield before entering the roundabout conflict area?",
            ],
            "avoid": ["entering conflict zone under high DCPA/TCPA risk"],
            "prefer": ["yield", "decelerate", "monitor_conflict_point"],
        },
        "roundabout_vru_interaction": {
            "key_questions": [
                "Is a cyclist or pedestrian close to ego's future path?",
                "Is min future distance decreasing?",
                "Should ego yield or slow down before conflict?",
            ],
            "avoid": ["forcing passage near VRU", "late braking"],
            "prefer": ["yield", "monitor_vru", "decelerate"],
        },
        "intersection_vehicle_interaction": {
            "key_questions": [
                "Do trajectories cross within the planning horizon?",
                "Is DCPA small and TCPA imminent?",
                "Is required deceleration increasing?",
            ],
            "avoid": ["entering predicted conflict zone", "assuming priority without evidence"],
            "prefer": ["yield", "decelerate", "monitor_crossing_vehicle"],
        },
        "intersection_vru_interaction": {
            "key_questions": [
                "Is a VRU present near the ego path?",
                "Is future minimum distance below threshold?",
                "Should ego yield even if TTC is not small?",
            ],
            "avoid": ["passing close to VRU", "prioritizing efficiency over safety"],
            "prefer": ["yield", "monitor_vru", "decelerate"],
        },
    }

    return mapping.get(scenario_type, {
        "key_questions": [
            "Is physical risk increasing?",
            "Which object is the main conflict partner?",
            "What conservative strategy should Reactive prefer?",
        ],
        "avoid": ["unsafe acceleration", "late reaction"],
        "prefer": ["increase_headway", "decelerate", "monitor"],
    })


def build_planning_context(dataset: str, event_type: str = "", pair_type: str = "", vrus_present: bool = False) -> dict:
    scenario_type = infer_scenario_type(
        dataset=dataset,
        event_type=event_type,
        pair_type=pair_type,
        vrus_present=vrus_present,
    )

    return {
        "dataset_profile": dataset_planning_profile(dataset),
        "scenario_type": scenario_type,
        "scenario_guidance": scenario_planning_guidance(scenario_type),
    }