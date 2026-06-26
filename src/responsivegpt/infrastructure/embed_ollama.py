import hashlib
import json
import math
import os
import time
from typing import Optional, Iterable

import requests


class EmbeddingError(RuntimeError):
    pass


class OllamaEmbedder:
    """
    Ollama 本地 embedding 封装。

    设计目标：
    1. 保持轻量：只依赖 requests + Python 标准库
    2. 支持内存缓存 / 可选磁盘缓存
    3. 支持重试，避免 Ollama 偶发失败导致实验中断
    4. 支持 L2 normalization，方便后续 cosine 相似度检索
    """

    def __init__(
        self,
        base_url: str,
        model: str,
        timeout_s: int = 30,
        max_retries: int = 2,
        retry_backoff_s: float = 0.5,
        normalize: bool = True,
        cache_dir: Optional[str] = None,
        max_chars: int = 12000,
    ):
        self.url = base_url.rstrip("/") + "/api/embeddings"
        self.model = model
        self.timeout_s = timeout_s
        self.max_retries = max_retries
        self.retry_backoff_s = retry_backoff_s
        self.normalize = normalize
        self.cache_dir = cache_dir
        self.max_chars = max_chars

        self.session = requests.Session()
        self._memory_cache: dict[str, list[float]] = {}

        if self.cache_dir:
            os.makedirs(self.cache_dir, exist_ok=True)

    def embed(self, text: str) -> list[float]:
        text = self._prepare_text(text)
        cache_key = self._cache_key(text)

        cached = self._get_from_cache(cache_key)
        if cached is not None:
            return list(cached)

        vec = self._request_embedding(text)

        if self.normalize:
            vec = self._l2_normalize(vec)

        self._save_to_cache(cache_key, vec)
        return list(vec)

    def embed_many(self, texts: Iterable[str]) -> list[list[float]]:
        """
        当前 Ollama embeddings 接口通常按单条 prompt 调用。
        这里保留批量接口，方便上层 KB 构建统一调用。
        """
        return [self.embed(t) for t in texts]

    def _prepare_text(self, text: str) -> str:
        if text is None:
            raise ValueError("Embedding text cannot be None.")

        text = str(text).strip()

        if not text:
            raise ValueError("Embedding text cannot be empty.")

        if self.max_chars and len(text) > self.max_chars:
            text = text[: self.max_chars]

        return text

    def _request_embedding(self, text: str) -> list[float]:
        payload = {
            "model": self.model,
            "prompt": text,
        }

        last_error = None

        for attempt in range(self.max_retries + 1):
            try:
                resp = self.session.post(
                    self.url,
                    json=payload,
                    timeout=(5, self.timeout_s),
                )
                resp.raise_for_status()

                data = resp.json()
                vec = data.get("embedding")

                if not isinstance(vec, list) or not vec:
                    raise EmbeddingError(
                        f"Invalid embedding response from Ollama: {data}"
                    )

                try:
                    vec = [float(x) for x in vec]
                except Exception as e:
                    raise EmbeddingError(
                        f"Embedding contains non-numeric values: {data}"
                    ) from e

                return vec

            except Exception as e:
                last_error = e

                if attempt < self.max_retries:
                    sleep_s = self.retry_backoff_s * (2 ** attempt)
                    time.sleep(sleep_s)
                    continue

        raise EmbeddingError(
            f"Failed to get embedding from Ollama after "
            f"{self.max_retries + 1} attempts. "
            f"url={self.url}, model={self.model}, error={last_error}"
        )

    def _cache_key(self, text: str) -> str:
        raw = f"{self.model}\n{text}".encode("utf-8")
        return hashlib.sha256(raw).hexdigest()

    def _get_from_cache(self, key: str) -> Optional[list[float]]:
        if key in self._memory_cache:
            return self._memory_cache[key]

        if not self.cache_dir:
            return None

        path = os.path.join(self.cache_dir, f"{key}.json")
        if not os.path.exists(path):
            return None

        try:
            with open(path, "r", encoding="utf-8") as f:
                vec = json.load(f)

            if isinstance(vec, list) and vec:
                vec = [float(x) for x in vec]
                self._memory_cache[key] = vec
                return vec

        except Exception:
            return None

        return None

    def _save_to_cache(self, key: str, vec: list[float]) -> None:
        self._memory_cache[key] = list(vec)

        if not self.cache_dir:
            return

        path = os.path.join(self.cache_dir, f"{key}.json")
        tmp_path = path + ".tmp"

        try:
            with open(tmp_path, "w", encoding="utf-8") as f:
                json.dump(vec, f)
            os.replace(tmp_path, path)
        except Exception:
            # 缓存失败不应影响主流程
            pass

    def _l2_normalize(self, vec: list[float]) -> list[float]:
        norm = math.sqrt(sum(x * x for x in vec))
        if norm <= 1e-12:
            return vec
        return [x / norm for x in vec]
