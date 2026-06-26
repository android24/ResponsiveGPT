from dataclasses import dataclass, asdict
from typing import Any


@dataclass(frozen=True)
class ExperimentJob:
    job_id: str
    experiment_name: str
    dataset: str
    mode: str
    summary_csv: str
    sequence_root: str
    profile_name: str
    rag_variant: str
    rag_mode: str
    use_retriever: int
    require_grounded_decision: int
    planning_variant: str
    use_planning_thread: int
    planning_mode: str
    llm_policy_variant: str
    llm_policy: str
    llm_stride: int
    llm_risk_threshold: float
    limit: int
    model_role: str
    feedback: str
    tag: str
    extra_args: dict[str, Any]

    def to_dict(self) -> dict:
        return asdict(self)

