from .base import EpisodeSafetyMetrics
from .utils import safe_mean, safe_min, safe_max


def aggregate_episode_safety_metrics(frame_metrics_list):
    n = len(frame_metrics_list)

    if n == 0:
        return EpisodeSafetyMetrics(
            num_frames=0,
            min_ttc_s=None,
            avg_ttc_s=None,
            min_thw_s=None,
            avg_thw_s=None,
            max_drac_mps2=None,
            avg_drac_mps2=None,
            min_dcpa_m=None,
            avg_dcpa_m=None,
            min_predicted_ttc_s=None,
            min_future_distance_m=None,
            avg_future_distance_m=None,
            unsafe_ttc_ratio=0.0,
            unsafe_thw_ratio=0.0,
            unsafe_drac_ratio=0.0,
            unsafe_dcpa_ratio=0.0,
            unsafe_future_distance_ratio=0.0,
            avg_physical_risk_index=None,
            max_physical_risk_index=None,
            physical_risk_exposure=0.0,
            has_critical_ttc=False,
            has_critical_drac=False,
            has_critical_spatial_risk=False,
        )

    ttc_vals = [m.ttc_s for m in frame_metrics_list]
    thw_vals = [m.thw_s for m in frame_metrics_list]
    drac_vals = [m.drac_mps2 for m in frame_metrics_list]
    dcpa_vals = [m.dcpa_m for m in frame_metrics_list]
    pttc_vals = [m.predicted_ttc_s for m in frame_metrics_list]
    mfd_vals = [m.min_future_distance_m for m in frame_metrics_list]
    risk_vals = [m.physical_risk_index for m in frame_metrics_list]

    unsafe_ttc_ratio = sum(1 for m in frame_metrics_list if m.unsafe_ttc) / n
    unsafe_thw_ratio = sum(1 for m in frame_metrics_list if m.unsafe_thw) / n
    unsafe_drac_ratio = sum(1 for m in frame_metrics_list if m.unsafe_drac) / n
    unsafe_dcpa_ratio = sum(1 for m in frame_metrics_list if m.unsafe_dcpa) / n
    unsafe_future_distance_ratio = sum(1 for m in frame_metrics_list if m.unsafe_future_distance) / n

    valid_risk_vals = [v for v in risk_vals if v is not None]
    physical_risk_exposure = sum(valid_risk_vals) / n if valid_risk_vals else 0.0

    max_drac = safe_max(drac_vals)
    min_ttc = safe_min(ttc_vals)
    min_dcpa = safe_min(dcpa_vals)
    min_mfd = safe_min(mfd_vals)

    return EpisodeSafetyMetrics(
        num_frames=n,

        min_ttc_s=min_ttc,
        avg_ttc_s=safe_mean(ttc_vals),
        min_thw_s=safe_min(thw_vals),
        avg_thw_s=safe_mean(thw_vals),
        max_drac_mps2=max_drac,
        avg_drac_mps2=safe_mean(drac_vals),

        min_dcpa_m=min_dcpa,
        avg_dcpa_m=safe_mean(dcpa_vals),
        min_predicted_ttc_s=safe_min(pttc_vals),
        min_future_distance_m=min_mfd,
        avg_future_distance_m=safe_mean(mfd_vals),

        unsafe_ttc_ratio=unsafe_ttc_ratio,
        unsafe_thw_ratio=unsafe_thw_ratio,
        unsafe_drac_ratio=unsafe_drac_ratio,
        unsafe_dcpa_ratio=unsafe_dcpa_ratio,
        unsafe_future_distance_ratio=unsafe_future_distance_ratio,

        avg_physical_risk_index=safe_mean(risk_vals),
        max_physical_risk_index=safe_max(risk_vals),
        physical_risk_exposure=physical_risk_exposure,

        has_critical_ttc=min_ttc is not None and min_ttc < 1.5,
        has_critical_drac=max_drac is not None and max_drac > 6.0,
        has_critical_spatial_risk=(
            (min_dcpa is not None and min_dcpa < 1.5) or
            (min_mfd is not None and min_mfd < 1.5)
        ),
    )