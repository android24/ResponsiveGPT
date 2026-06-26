from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass(frozen=True)
class KnowledgeDoc:
    id: str
    kb_type: str              # law / case / scenario / policy / safety
    title: str
    text: str

    scene_type: Optional[str] = None
    event_type: Optional[str] = None
    pair_type: Optional[str] = None

    source: Optional[str] = None
    priority: float = 1.0

    dataset_tags: list[str] = field(default_factory=list)
    scenario_tags: list[str] = field(default_factory=list)
    metric_tags: list[str] = field(default_factory=list)
    risk_tags: list[str] = field(default_factory=list)

    condition: dict[str, Any] = field(default_factory=dict)
    risk_mechanism: Optional[str] = None
    recommended_action: list[str] = field(default_factory=list)
    forbidden_action: list[str] = field(default_factory=list)
    severity: Optional[str] = None
    jurisdiction: Optional[str] = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class RankedKnowledgeDoc:
    doc: KnowledgeDoc
    semantic_score: float
    metadata_score: float
    priority_score: float
    final_score: float


@dataclass(frozen=True)
class EvidenceBundle:
    laws: list[RankedKnowledgeDoc]
    cases: list[RankedKnowledgeDoc]
    scenarios: list[RankedKnowledgeDoc]

    def all_ids(self) -> list[str]:
        ids = []
        for part in (self.laws, self.cases, self.scenarios):
            for x in part:
                ids.append(x.doc.id)
        return ids
