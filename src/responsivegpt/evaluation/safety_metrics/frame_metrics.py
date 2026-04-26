from .base import FrameSafetyMetrics
from .thresholds import SafetyThresholds
from .future_risk import compute_predicted_ttc_and_mfd
from .spatial_metrics import compute_dcpa_tcpa
from .risk_index import compute_physical_risk_index, risk_level_from_index
from .utils import is_valid_number, euclidean_distance


def compute_ttc(distance_m, rel_speed_mps):
    """
    TTC = distance / closing_speed.
    rel_speed_mps > 0 表示 ego 正在接近 other。
    """
    if not is_valid_number(distance_m) or not is_valid_number(rel_speed_mps):
        return None
    if distance_m <= 0:
        return 0.0
    if rel_speed_mps <= 0:
        return None
    return distance_m / rel_speed_mps


def compute_thw(distance_m, ego_speed_mps):
    if not is_valid_number(distance_m) or not is_valid_number(ego_speed_mps):
        return None
    if ego_speed_mps <= 0:
        return None
    return distance_m / ego_speed_mps


def compute_drac(distance_m, rel_speed_mps):
    """
    DRAC = closing_speed^2 / (2 * distance)
    仅在 closing speed > 0 时有效。
    """
    if not is_valid_number(distance_m) or not is_valid_number(rel_speed_mps):
        return None
    if distance_m <= 0:
        return None
    if rel_speed_mps <= 0:
        return None
    return (rel_speed_mps ** 2) / (2.0 * distance_m)


def infer_distance(scene):
    if is_valid_number(scene.headway_m):
        return scene.headway_m

    if (
        is_valid_number(scene.ego_x) and
        is_valid_number(scene.ego_y) and
        is_valid_number(scene.other_x) and
        is_valid_number(scene.other_y)
    ):
        return euclidean_distance(scene.ego_x, scene.ego_y, scene.other_x, scene.other_y)

    return None


def infer_rel_speed(scene):
    if is_valid_number(scene.rel_speed_mps):
        return scene.rel_speed_mps

    if is_valid_number(scene.ego_speed_mps) and is_valid_number(scene.lead_speed_mps):
        return scene.ego_speed_mps - scene.lead_speed_mps

    return None


def compute_frame_safety_metrics(scene, thresholds: SafetyThresholds) -> FrameSafetyMetrics:
    distance_m = infer_distance(scene)
    rel_speed_mps = infer_rel_speed(scene)
    ego_speed_mps = scene.ego_speed_mps
    lead_speed_mps = scene.lead_speed_mps

    ttc_s = compute_ttc(distance_m, rel_speed_mps)
    thw_s = compute_thw(distance_m, ego_speed_mps)
    drac_mps2 = compute_drac(distance_m, rel_speed_mps)

    dcpa_m, ttca_s = compute_dcpa_tcpa(scene)
    predicted_ttc_s, min_future_distance_m = compute_predicted_ttc_and_mfd(
        scene=scene,
        horizon_s=thresholds.prediction_horizon_s,
        dt_s=thresholds.prediction_dt_s,
    )

    closing = is_valid_number(rel_speed_mps) and rel_speed_mps > 0

    unsafe_ttc = ttc_s is not None and ttc_s < thresholds.ttc_medium
    unsafe_thw = thw_s is not None and thw_s < thresholds.thw_medium
    unsafe_drac = drac_mps2 is not None and drac_mps2 > thresholds.drac_medium
    unsafe_dcpa = dcpa_m is not None and dcpa_m < thresholds.dcpa_medium
    unsafe_future_distance = (
        min_future_distance_m is not None and
        min_future_distance_m < thresholds.future_distance_medium
    )

    risk_index = compute_physical_risk_index(
        ttc_s=ttc_s,
        thw_s=thw_s,
        drac_mps2=drac_mps2,
        dcpa_m=dcpa_m,
        min_future_distance_m=min_future_distance_m,
        distance_m=distance_m,
    )

    return FrameSafetyMetrics(
        frame_index=scene.frame_index,
        distance_m=distance_m,
        rel_speed_mps=rel_speed_mps,
        ego_speed_mps=ego_speed_mps,
        lead_speed_mps=lead_speed_mps,

        ego_x=scene.ego_x,
        ego_y=scene.ego_y,
        other_x=scene.other_x,
        other_y=scene.other_y,
        ego_vx=getattr(scene, "ego_vx", None),
        ego_vy=getattr(scene, "ego_vy", None),
        other_vx=getattr(scene, "other_vx", None),
        other_vy=getattr(scene, "other_vy", None),

        ttc_s=ttc_s,
        thw_s=thw_s,
        drac_mps2=drac_mps2,

        ttca_s=ttca_s,
        dcpa_m=dcpa_m,
        predicted_ttc_s=predicted_ttc_s,
        min_future_distance_m=min_future_distance_m,

        closing=closing,
        unsafe_ttc=unsafe_ttc,
        unsafe_thw=unsafe_thw,
        unsafe_drac=unsafe_drac,
        unsafe_dcpa=unsafe_dcpa,
        unsafe_future_distance=unsafe_future_distance,

        physical_risk_index=risk_index,
        physical_risk_level=risk_level_from_index(risk_index),
    )