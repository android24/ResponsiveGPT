from typing import Optional
from ..domain.evidence import KnowledgeDoc


class KnowledgeBase:
    def __init__(self, docs: list[KnowledgeDoc]):
        self.docs = docs

    def filter_docs(
        self,
        kb_type: Optional[str] = None,
        scene_type: Optional[str] = None,
        event_type: Optional[str] = None,
        pair_type: Optional[str] = None,
    ) -> list[KnowledgeDoc]:
        out = []
        for d in self.docs:
            if kb_type and d.kb_type != kb_type:
                continue

            if scene_type and d.scene_type not in [None, "all", scene_type]:
                continue

            if event_type and d.event_type not in [None, "all", event_type]:
                continue

            if pair_type and d.pair_type not in [None, "all", pair_type]:
                continue

            out.append(d)
        return out