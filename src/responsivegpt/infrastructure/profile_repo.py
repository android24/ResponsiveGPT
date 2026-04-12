import json
import os
import shutil
from copy import deepcopy


def default_profile():
    return {
        "driver_type": "均衡",
        "global": {
            "risk_sensitivity": 0.5,
            "safety_weight": 0.6,
            "efficiency_weight": 0.4
        },
        "longitudinal": {
            "preferred_time_headway": 2.0,
            "min_time_headway": 1.2,
            "brake_aggressiveness": 0.5,
            "acceleration_aggressiveness": 0.5
        },
        "lateral": {
            "lane_change_aggressiveness": 0.5,
            "min_gap_acceptance": 1.5,
            "cut_in_tolerance": 0.5
        },
        "interaction": {
            "vehicle_vehicle_assertiveness": 0.5,
            "vehicle_cyclist_yield_bias": 0.7,
            "vehicle_pedestrian_yield_bias": 0.9
        },
        "scenario_bias": {},
        "temporal": {
            "risk_memory_decay": 0.9,
            "recent_violation_penalty": 0.2
        }
    }


class JsonProfileRepository:
    """
    支持：
    - template profile（data/profiles/*.json）
    - runtime profile（runs/.../runtime_profile.json）
    """

    def __init__(
        self,
        template_path: str,
        runtime_path: str = None,
        auto_init: bool = True,
    ):
        self.template_path = template_path
        self.runtime_path = runtime_path or template_path

        if auto_init:
            self._init_runtime_profile()

    def _init_runtime_profile(self):
        """
        如果 runtime 不存在，则从 template 复制
        """
        if not os.path.exists(self.runtime_path):
            os.makedirs(os.path.dirname(self.runtime_path), exist_ok=True)

            if os.path.exists(self.template_path):
                shutil.copy(self.template_path, self.runtime_path)
            else:
                with open(self.runtime_path, "w", encoding="utf-8") as f:
                    json.dump(default_profile(), f, ensure_ascii=False, indent=2)

    def load(self) -> dict:
        """
        加载 runtime profile
        """
        if not os.path.exists(self.runtime_path):
            return default_profile()

        with open(self.runtime_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        return self._normalize_profile(data)

    def save(self, profile: dict) -> None:
        """
        只写 runtime profile
        """
        os.makedirs(os.path.dirname(self.runtime_path), exist_ok=True)

        with open(self.runtime_path, "w", encoding="utf-8") as f:
            json.dump(profile, f, ensure_ascii=False, indent=2)

    def reset(self):
        """
        重新从 template 初始化 runtime（用于多次实验）
        """
        if os.path.exists(self.template_path):
            shutil.copy(self.template_path, self.runtime_path)

    def get_template(self) -> dict:
        """
        读取原始模板（不带学习）
        """
        if not os.path.exists(self.template_path):
            return default_profile()

        with open(self.template_path, "r", encoding="utf-8") as f:
            return self._normalize_profile(json.load(f))

    def _normalize_profile(self, data: dict) -> dict:
        """
        兼容旧格式（只有 risk_sensitivity 等）
        """
        if "global" in data:
            return data

        # 旧版转新版
        return {
            "driver_type": data.get("driver_type", "均衡"),
            "global": {
                "risk_sensitivity": float(data.get("risk_sensitivity", 0.5)),
                "safety_weight": float(data.get("safety_weight", 0.6)),
                "efficiency_weight": float(data.get("efficiency_weight", 0.4)),
            },
            "longitudinal": {},
            "lateral": {},
            "interaction": {},
            "scenario_bias": {},
            "temporal": {},
        }
