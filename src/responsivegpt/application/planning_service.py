from .planning_prompts import PLANNING_SYSTEM_PROMPT, PLANNING_USER_TEMPLATE
from .planning_schema import validate_planning_output, default_planning_output


class PlanningService:
    def __init__(self, chat_model):
        self.chat_model = chat_model

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
    ) -> dict:
        user = PLANNING_USER_TEMPLATE.format(
            dataset=dataset,
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

        try:
            raw = self.chat_model.complete_json(PLANNING_SYSTEM_PROMPT, user)
            return validate_planning_output(raw)
        except Exception as e:
            print(f"[WARN] PlanningService failed: {e}")
            return default_planning_output(str(e))