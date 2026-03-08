from ..domain.logic import update_profile, build_query, make_prompts
from ..domain.models import StepResult

class ResponsiveGPTService:
    def __init__(self, vectorstore, chat_model, profile_repo):
        self.vs = vectorstore
        self.llm = chat_model
        self.repo = profile_repo

    def step(self, scene, driver_type: str, feedback: str, extra_context: str = ""):
        profile = self.repo.load()
        profile = update_profile(profile, driver_type, feedback)
        self.repo.save(profile)

        query = build_query(scene)
        rules = self.vs.search(query, 4)

        system, user = make_prompts(profile, scene, feedback, rules, extra_context=extra_context)
        decision = self.llm.complete_json(system, user)

        return StepResult(profile=profile, rules=rules, decision=decision)
