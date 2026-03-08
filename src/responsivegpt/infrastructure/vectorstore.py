import math
from ..domain.models import RetrievedRule

def cosine(a: list[float], b: list[float]) -> float:
    dot = 0.0
    na = 0.0
    nb = 0.0
    for x, y in zip(a, b):
        dot += x * y
        na += x * x
        nb += y * y
    denom = math.sqrt(na) * math.sqrt(nb)
    return dot / denom if denom else 0.0

class SimpleVectorStore:
    def __init__(self, embedder, docs: list[dict]):
        self.embedder = embedder
        self.docs = docs
        self.embs: list[list[float]] = []

    def build(self) -> None:
        self.embs = [self.embedder.embed(d["text"]) for d in self.docs]

    def search(self, query: str, k: int):
        q = self.embedder.embed(query)
        scored = []
        for i, d in enumerate(self.docs):
            scored.append((cosine(q, self.embs[i]), d))
        scored.sort(key=lambda x: x[0], reverse=True)
        out = []
        for s, d in scored[:k]:
            out.append(RetrievedRule(id=d["id"], score=float(s), text=d["text"]))
        return out
