from .utils import norm_inverse, norm_positive


def compute_physical_risk_index(
    ttc_s,
    thw_s,
    drac_mps2,
    dcpa_m,
    min_future_distance_m,
    distance_m,
):
    """
    Unified Physical Risk Index (UPRI), range roughly [0,1].

    设计思路：
    - TTC / THW / DCPA / FutureDistance / Distance 越小越危险
    - DRAC 越大越危险
    - 缺失指标不参与加权，自动按可用权重归一

    权重可在论文中说明为 fixed physically motivated weights。
    """
    components = []

    def add(weight, value):
        if value is not None:
            components.append((weight, value))

    # 1/TTC: TTC=1s 时接近最高风险
    add(0.25, norm_inverse(ttc_s, cap_inverse=1.0))

    # DRAC: 8 m/s² 近似高强度制动上限
    add(0.25, norm_positive(drac_mps2, cap=8.0))

    # 1/THW
    add(0.15, norm_inverse(thw_s, cap_inverse=1.0))

    # DCPA / future distance / distance
    add(0.15, norm_inverse(dcpa_m, cap_inverse=1.0))
    add(0.15, norm_inverse(min_future_distance_m, cap_inverse=1.0))
    add(0.05, norm_inverse(distance_m, cap_inverse=1.0))

    if not components:
        return None

    w_sum = sum(w for w, _ in components)
    if w_sum <= 0:
        return None

    return sum(w * v for w, v in components) / w_sum


def risk_level_from_index(x):
    if x is None:
        return "unknown"
    if x >= 0.65:
        return "high"
    if x >= 0.35:
        return "medium"
    return "low"