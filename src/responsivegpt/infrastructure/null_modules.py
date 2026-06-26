from dataclasses import dataclass, field
from typing import List, Dict, Any


# =========================
# Empty evidence object
# =========================

@dataclass
class EmptyEvidence:
    laws: List[Any] = field(default_factory=list)
    cases: List[Any] = field(default_factory=list)
    scenarios: List[Any] = field(default_factory=list)


# =========================
# Null Retriever
# =========================

class NullRetriever:
    """
    用于 no-RAG 消融实验
    保证 service.step() 仍然可以调用 retrieve()，
    但不会返回任何外部知识。
    """
    def retrieve(self, scene, profile):
        return EmptyEvidence()


# =========================
# Null Trigger Manager
# =========================

class NullTriggerManager:
    """
    用于 no-trigger 消融实验
    消融实验用的空 TriggerManager。

    保持与 TriggerManager.evaluate(...) 相同的调用接口，
    但不产生任何 trigger、guardrail 或 profile update。
    """

    def evaluate(self, *args, **kwargs):
        # 必须返回完整结构
        triggers = []
        guardrails = {}
        profile_update = {}

        return triggers, guardrails, profile_update

    # 兼容不同调用方式（保险）
    def check(self, *args, **kwargs):
        return []

    def detect(self, *args, **kwargs):
        return []

    def step(self, *args, **kwargs):
        return []


# =========================
# Null Trigger State Store
# =========================

class NullTriggerStateStore:
    """
    用于 no-trigger 消融实验
    完整模拟接口，但不做任何事情
    """

    def add(self, *args, **kwargs):
        return None

    def get(self, *args, **kwargs):
        return {}

    def update(self, *args, **kwargs):
        return None

    def reset(self, *args, **kwargs):
        return None

    def clear(self, *args, **kwargs):
        return None

    def step(self, *args, **kwargs):
        return None

    def tick_frame(self, *args, **kwargs):
        return None

    def tick_event(self, *args, **kwargs):
        return None

    def snapshot(self, *args, **kwargs):
        return {}

    def to_dict(self, *args, **kwargs):
        return {}


# =========================
# Null Profile Learner
# =========================

class NullProfileLearner:
    """
    用于消融实验：禁用 profile learning，但保持 service 接口不变。
    """
    enabled = False

    def update(self, profile, *args, **kwargs):
        return profile

    def step(self, profile, *args, **kwargs):
        return profile

    def learn(self, profile, *args, **kwargs):
        return profile

    def apply(self, profile, *args, **kwargs):
        return profile
