import requests

class OllamaEmbedder:
    def __init__(self, base_url: str, model: str, timeout_s: int = 30):
        self.url = base_url.rstrip("/") + "/api/embeddings"
        self.model = model
        self.timeout_s = timeout_s

    def embed(self, text: str) -> list[float]:
        r = requests.post(self.url, json={"model": self.model, "prompt": text}, timeout=self.timeout_s)
        r.raise_for_status()
        return r.json()["embedding"]
