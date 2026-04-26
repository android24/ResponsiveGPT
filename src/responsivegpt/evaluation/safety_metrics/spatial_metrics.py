from .utils import is_valid_number
import math


def compute_dcpa_tcpa(scene):
    """
    DCPA/TCPA:
    在二维匀速假设下计算最近接近距离和到最近接近点的时间。

    relative position:
      r = other - ego

    relative velocity:
      v = other_v - ego_v

    TCPA = - (r dot v) / ||v||^2
    DCPA = || r + v * TCPA ||

    若速度向量缺失，返回 (None, None)。
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

    rx = other_x - ego_x
    ry = other_y - ego_y
    vx = other_vx - ego_vx
    vy = other_vy - ego_vy

    vv = vx * vx + vy * vy
    if vv <= 1e-9:
        dcpa = math.sqrt(rx * rx + ry * ry)
        return dcpa, None

    tcpa = - (rx * vx + ry * vy) / vv

    # 过去的 CPA 不作为未来空间风险，但 DCPA 可以保留当前距离
    if tcpa < 0:
        dcpa = math.sqrt(rx * rx + ry * ry)
        return dcpa, 0.0

    cx = rx + vx * tcpa
    cy = ry + vy * tcpa
    dcpa = math.sqrt(cx * cx + cy * cy)

    return dcpa, tcpa