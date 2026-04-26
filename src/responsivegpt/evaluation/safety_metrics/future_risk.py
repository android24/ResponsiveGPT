from .utils import is_valid_number
import math


def compute_predicted_ttc_and_mfd(scene, horizon_s: float = 5.0, dt_s: float = 0.2):
    """
    基于二维匀速模型预测未来 horizon 内最小距离和 predicted TTC。

    predicted_ttc_s:
      第一次距离 <= collision_radius 时的时间。
      这里 collision_radius 可根据车辆大小扩展，目前用保守默认 2m。

    min_future_distance_m:
      horizon 内最小预测距离。

    如果缺少二维位置/速度分量，返回 (None, None)。
    """
    ego_x = scene.ego_x
    ego_y = scene.ego_y
    other_x = scene.other_x
    other_y = scene.other_y

    ego_vx = getattr(scene, "ego_vx", None)
    ego_vy = getattr(scene, "ego_vy", None)
    other_vx = getattr(scene, "other_vx", None)
    other_vy = getattr(scene, "other_vy", None)

    vals = [ego_x, ego_y, other_x, other_y, ego_vx, ego_vy, other_vx, other_vy]
    if not all(is_valid_number(v) for v in vals):
        return None, None

    collision_radius_m = 2.0

    t = 0.0
    min_dist = None
    predicted_ttc = None

    while t <= horizon_s + 1e-9:
        ex = ego_x + ego_vx * t
        ey = ego_y + ego_vy * t
        ox = other_x + other_vx * t
        oy = other_y + other_vy * t

        d = math.sqrt((ox - ex) ** 2 + (oy - ey) ** 2)

        if min_dist is None or d < min_dist:
            min_dist = d

        if predicted_ttc is None and d <= collision_radius_m:
            predicted_ttc = t

        t += dt_s

    return predicted_ttc, min_dist