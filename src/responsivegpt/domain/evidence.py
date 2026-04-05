from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class KnowledgeDoc:
    id: str
    kb_type: str              # law / case / scenario
    title: str
    text: str

    scene_type: Optional[str] = None
    event_type: Optional[str] = None
    pair_type: Optional[str] = None

    source: Optional[str] = None
    priority: float = 1.0


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