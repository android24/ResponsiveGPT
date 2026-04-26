import math
from typing import Optional, Iterable


def is_valid_number(x) -> bool:
    return isinstance(x, (int, float)) and not math.isnan(x) and not math.isinf(x)


def clean_float(x, default=None):
    try:
        v = float(x)
        if math.isnan(v) or math.isinf(v):
            return default
        return v
    except Exception:
        return default


def safe_mean(values: Iterable[Optional[float]]) -> Optional[float]:
    vals = [v for v in values if is_valid_number(v)]
    return sum(vals) / len(vals) if vals else None


def safe_min(values: Iterable[Optional[float]]) -> Optional[float]:
    vals = [v for v in values if is_valid_number(v)]
    return min(vals) if vals else None


def safe_max(values: Iterable[Optional[float]]) -> Optional[float]:
    vals = [v for v in values if is_valid_number(v)]
    return max(vals) if vals else None


def safe_var(values: Iterable[Optional[float]]) -> Optional[float]:
    vals = [v for v in values if is_valid_number(v)]
    if not vals:
        return None
    m = sum(vals) / len(vals)
    return sum((x - m) ** 2 for x in vals) / len(vals)


def norm_positive(x: Optional[float], cap: float) -> Optional[float]:
    """
    越大越危险的值，归一到 [0,1]
    """
    if not is_valid_number(x):
        return None
    if cap <= 0:
        return None
    return max(0.0, min(float(x), cap)) / cap


def norm_inverse(x: Optional[float], cap_inverse: float) -> Optional[float]:
    """
    越小越危险的正值，用 1/x 归一化。
    cap_inverse 表示 1/x 的封顶值。
    """
    if not is_valid_number(x):
        return None
    if x <= 0:
        return 1.0
    v = min(1.0 / x, cap_inverse)
    return v / cap_inverse if cap_inverse > 0 else None


def euclidean_distance(x1, y1, x2, y2) -> Optional[float]:
    vals = [x1, y1, x2, y2]
    if not all(is_valid_number(v) for v in vals):
        return None
    return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)