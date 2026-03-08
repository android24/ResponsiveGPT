import json, os
from ..domain.models import DriverProfile

class JsonProfileRepository:
    def __init__(self, path: str):
        self.path = path

    def load(self) -> DriverProfile:
        if not os.path.exists(self.path):
            return DriverProfile()
        with open(self.path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return DriverProfile(
            driver_type=data.get("driver_type", "unknown"),
            risk_sensitivity=float(data.get("risk_sensitivity", 0.5)),
            safety_weight=float(data.get("safety_weight", 0.65)),
            efficiency_weight=float(data.get("efficiency_weight", 0.35)),
        )

    def save(self, profile: DriverProfile) -> None:
        with open(self.path, "w", encoding="utf-8") as f:
            json.dump(profile.__dict__, f, ensure_ascii=False, indent=2)
