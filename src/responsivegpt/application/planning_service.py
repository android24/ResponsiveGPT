from .planning_context import build_planning_context
from .planning_prompts import PLANNING_SYSTEM_PROMPT, PLANNING_USER_TEMPLATE
from .planning_schema import validate_planning_output, default_planning_output
from ..infrastructure.json_disk_cache import JsonDiskCache


class PlanningService:
    def __init__(
        self,
        chat_model,
        *,
        cache_dir: str | None = None,
        cache_enabled: bool = True,
    ):
        self.chat_model = chat_model
        self.cache = JsonDiskCache(cache_dir, enabled=cache_enabled)
        self.last_cache_hit = False

    def plan(
        self,
        *,
        dataset: str,
        driver_type: str,
        feedback: str,
        planning_interval: int,
        current_frame: int,
        time_horizon_s: float,
        recent_scene_summaries: str,
        recent_safety_summaries: str,
        recent_decision_summaries: str,
        current_safety_snapshot: str,
        event_type: str = "",
        pair_type: str = "",
        vrus_present: bool = False,
    ) -> dict:
        ctx = build_planning_context(
            dataset=dataset,
            event_type=event_type,
            pair_type=pair_type,
            vrus_present=vrus_present,
        )

        dataset_profile = ctx["dataset_profile"]
        scenario_guidance = ctx["scenario_guidance"]

        user = PLANNING_USER_TEMPLATE.format(
            dataset=dataset,
            scenario_type=ctx["scenario_type"],
            dataset_context=dataset_profile,
            scenario_context=scenario_guidance,
            primary_metrics=dataset_profile.get("primary_risk_metrics", []),
            secondary_metrics=dataset_profile.get("secondary_risk_metrics", []),
            driver_type=driver_type,
            feedback=feedback,
            planning_interval=planning_interval,
            current_frame=current_frame,
            time_horizon_s=time_horizon_s,
            recent_scene_summaries=recent_scene_summaries,
            recent_safety_summaries=recent_safety_summaries,
            recent_decision_summaries=recent_decision_summaries,
            current_safety_snapshot=current_safety_snapshot,
        )
        cache_key = JsonDiskCache.stable_hash({
            "cache_version": "planning_output_v1",
            "system": PLANNING_SYSTEM_PROMPT,
            "user": user,
        })
        self.last_cache_hit = False
        cached = self.cache.get(cache_key)
        if isinstance(cached, dict):
            self.last_cache_hit = True
            out = validate_planning_output(cached)
            out.setdefault("diagnostics", {})
            out["diagnostics"]["planning_cache_hit"] = True
            out["diagnostics"]["planning_cache_key"] = cache_key
            return out

        try:
            raw = self.chat_model.complete_json(PLANNING_SYSTEM_PROMPT, user)
            out = validate_planning_output(raw)
            out.setdefault("diagnostics", {})
            out["diagnostics"]["planning_cache_hit"] = False
            out["diagnostics"]["planning_cache_key"] = cache_key
            self.cache.set(
                cache_key,
                out,
                metadata={
                    "cache_version": "planning_output_v1",
                    "dataset": dataset,
                    "driver_type": driver_type,
                    "current_frame": current_frame,
                    "event_type": event_type,
                    "pair_type": pair_type,
                    "vrus_present": bool(vrus_present),
                },
            )
            return out
        except Exception as e:
            print(f"[WARN] PlanningService failed: {e}")
            return default_planning_output(str(e))

    def cache_stats(self) -> dict:
        return self.cache.stats()
