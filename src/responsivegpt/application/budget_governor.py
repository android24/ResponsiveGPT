from dataclasses import dataclass, field


def _ratio(used: float, limit: float) -> float:
    try:
        if not limit or float(limit) <= 0:
            return 0.0
        return max(0.0, float(used) / float(limit))
    except (TypeError, ValueError):
        return 0.0


@dataclass
class BudgetGovernor:
    enabled: bool = False
    warn_ratio: float = 0.80
    critical_ratio: float = 0.95
    max_wall_time_s: float = 0.0
    mode_counts: dict = field(default_factory=dict)
    adjustment_counts: dict = field(default_factory=dict)

    def govern(
        self,
        *,
        frame_pos: int,
        elapsed_s: float,
        reactive_budget: dict,
        planning_budget: dict,
        rag_top_k: int,
        llm_max_stale_frames: int,
        llm_risk_threshold: float,
        llm_risk_delta_threshold: float,
        planning_interval: int,
        planning_min_gap: int,
    ) -> dict:
        base = {
            "enabled": bool(self.enabled),
            "mode": "disabled",
            "reasons": [],
            "rag_top_k": max(1, int(rag_top_k)),
            "llm_max_stale_frames": max(1, int(llm_max_stale_frames)),
            "llm_risk_threshold": float(llm_risk_threshold),
            "llm_risk_delta_threshold": float(llm_risk_delta_threshold),
            "planning_interval": max(0, int(planning_interval)),
            "planning_min_gap": max(0, int(planning_min_gap)),
            "frame_pos": int(frame_pos),
        }
        if not self.enabled:
            return base

        pressure = self._pressure(
            elapsed_s=elapsed_s,
            reactive_budget=reactive_budget,
            planning_budget=planning_budget,
        )
        mode = "normal"
        if pressure["max_ratio"] >= float(self.critical_ratio):
            mode = "critical"
        elif pressure["max_ratio"] >= float(self.warn_ratio):
            mode = "conserve"

        result = dict(base)
        result["enabled"] = True
        result["mode"] = mode
        result["pressure"] = pressure
        result["reasons"] = pressure["reasons"]

        if mode == "conserve":
            result["rag_top_k"] = max(2, min(base["rag_top_k"], 6))
            result["llm_max_stale_frames"] = max(
                base["llm_max_stale_frames"], int(base["llm_max_stale_frames"] * 1.5)
            )
            result["llm_risk_threshold"] = min(
                0.85, base["llm_risk_threshold"] + 0.05
            )
            result["llm_risk_delta_threshold"] = min(
                0.50, base["llm_risk_delta_threshold"] + 0.05
            )
            result["planning_interval"] = max(
                base["planning_interval"], int(base["planning_interval"] * 1.5)
            )
            result["planning_min_gap"] = max(
                base["planning_min_gap"], int(base["planning_min_gap"] * 1.5)
            )
        elif mode == "critical":
            result["rag_top_k"] = max(1, min(base["rag_top_k"], 3))
            result["llm_max_stale_frames"] = max(
                base["llm_max_stale_frames"], int(base["llm_max_stale_frames"] * 2)
            )
            result["llm_risk_threshold"] = min(
                0.90, base["llm_risk_threshold"] + 0.12
            )
            result["llm_risk_delta_threshold"] = min(
                0.60, base["llm_risk_delta_threshold"] + 0.12
            )
            result["planning_interval"] = max(
                base["planning_interval"], base["planning_interval"] * 2
            )
            result["planning_min_gap"] = max(
                base["planning_min_gap"], base["planning_min_gap"] * 2
            )

        self._record(result, base)
        return result

    def stats(self) -> dict:
        return {
            "enabled": bool(self.enabled),
            "warn_ratio": float(self.warn_ratio),
            "critical_ratio": float(self.critical_ratio),
            "max_wall_time_s": float(self.max_wall_time_s),
            "mode_distribution": dict(self.mode_counts),
            "adjustment_counts": dict(self.adjustment_counts),
        }

    def _pressure(
        self,
        *,
        elapsed_s: float,
        reactive_budget: dict,
        planning_budget: dict,
    ) -> dict:
        ratios = {
            "reactive_attempts": _ratio(
                reactive_budget.get("attempts"),
                reactive_budget.get("max_attempts"),
            ),
            "reactive_tokens": _ratio(
                reactive_budget.get("total_tokens"),
                reactive_budget.get("max_tokens"),
            ),
            "planning_attempts": _ratio(
                planning_budget.get("attempts"),
                planning_budget.get("max_attempts"),
            ),
            "planning_tokens": _ratio(
                planning_budget.get("total_tokens"),
                planning_budget.get("max_tokens"),
            ),
            "wall_time": _ratio(elapsed_s, self.max_wall_time_s),
        }
        max_key, max_ratio = max(ratios.items(), key=lambda item: item[1])
        reasons = [
            f"{key}:{value:.3f}"
            for key, value in ratios.items()
            if value >= float(self.warn_ratio)
        ]
        return {
            "ratios": ratios,
            "max_key": max_key,
            "max_ratio": max_ratio,
            "reasons": reasons,
        }

    def _record(self, result: dict, base: dict) -> None:
        mode = result.get("mode", "unknown")
        self.mode_counts[mode] = int(self.mode_counts.get(mode, 0) or 0) + 1
        for key in (
            "rag_top_k",
            "llm_max_stale_frames",
            "llm_risk_threshold",
            "llm_risk_delta_threshold",
            "planning_interval",
            "planning_min_gap",
        ):
            if result.get(key) != base.get(key):
                self.adjustment_counts[key] = (
                    int(self.adjustment_counts.get(key, 0) or 0) + 1
                )
