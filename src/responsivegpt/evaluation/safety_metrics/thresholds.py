from dataclasses import dataclass


@dataclass(frozen=True)
class SafetyThresholds:
    # TTC thresholds: smaller is more dangerous
    ttc_low: float = 5.0
    ttc_medium: float = 3.0
    ttc_high: float = 1.5

    # THW thresholds: smaller is more dangerous
    thw_low: float = 2.0
    thw_medium: float = 1.0
    thw_high: float = 0.5

    # DRAC thresholds: larger is more dangerous
    drac_low: float = 2.0
    drac_medium: float = 4.0
    drac_high: float = 6.0

    # DCPA / distance thresholds: smaller is more dangerous
    dcpa_low: float = 5.0
    dcpa_medium: float = 3.0
    dcpa_high: float = 1.5

    future_distance_low: float = 5.0
    future_distance_medium: float = 3.0
    future_distance_high: float = 1.5

    # prediction horizon
    prediction_horizon_s: float = 5.0
    prediction_dt_s: float = 0.2


def thresholds_for_dataset(dataset: str) -> SafetyThresholds:
    """
    可根据数据集微调阈值，但默认保持统一。
    顶会论文里建议主实验使用统一阈值，附录做 dataset-specific sensitivity analysis。
    """
    dataset = (dataset or "").lower()

    if dataset == "highd":
        return SafetyThresholds(
            ttc_low=5.0,
            ttc_medium=3.0,
            ttc_high=1.5,
            thw_low=2.0,
            thw_medium=1.0,
            thw_high=0.5,
            drac_low=2.0,
            drac_medium=4.0,
            drac_high=6.0,
        )

    if dataset in ["round", "roundd", "roundD".lower(), "rounD".lower()]:
        return SafetyThresholds(
            ttc_low=5.0,
            ttc_medium=3.0,
            ttc_high=1.5,
            dcpa_low=5.0,
            dcpa_medium=3.0,
            dcpa_high=1.5,
            future_distance_low=5.0,
            future_distance_medium=3.0,
            future_distance_high=1.5,
        )

    if dataset == "ind":
        return SafetyThresholds(
            ttc_low=5.0,
            ttc_medium=3.0,
            ttc_high=1.5,
            dcpa_low=5.0,
            dcpa_medium=3.0,
            dcpa_high=1.5,
            future_distance_low=5.0,
            future_distance_medium=3.0,
            future_distance_high=1.5,
            drac_low=2.0,
            drac_medium=4.0,
            drac_high=6.0,
        )

    return SafetyThresholds()