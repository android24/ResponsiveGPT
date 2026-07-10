import hashlib
import json
import os
from copy import deepcopy
from pathlib import Path
from typing import Any


class JsonDiskCache:
    """Small JSON-on-disk cache for deterministic experiment artifacts."""

    def __init__(self, cache_dir: str | None, *, enabled: bool = True):
        self.cache_dir = Path(cache_dir).expanduser() if cache_dir else None
        self.enabled = bool(enabled and self.cache_dir)
        self.hits = 0
        self.misses = 0
        self.writes = 0
        self.errors = 0
        if self.enabled:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def stable_hash(payload: Any) -> str:
        text = json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        )
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    def path_for_key(self, key: str) -> Path:
        if not self.cache_dir:
            raise RuntimeError("cache_dir is not configured")
        return self.cache_dir / f"{key}.json"

    def get(self, key: str) -> Any | None:
        if not self.enabled:
            return None
        path = self.path_for_key(key)
        if not path.exists():
            self.misses += 1
            return None
        try:
            with path.open("r", encoding="utf-8") as stream:
                obj = json.load(stream)
            self.hits += 1
            return deepcopy(obj.get("value"))
        except Exception:
            self.errors += 1
            self.misses += 1
            return None

    def set(self, key: str, value: Any, *, metadata: dict | None = None) -> None:
        if not self.enabled:
            return
        path = self.path_for_key(key)
        temp_path = path.with_suffix(".json.tmp")
        payload = {
            "key": key,
            "metadata": metadata or {},
            "value": value,
        }
        try:
            with temp_path.open("w", encoding="utf-8") as stream:
                json.dump(payload, stream, ensure_ascii=False, indent=2)
            os.replace(temp_path, path)
            self.writes += 1
        except Exception:
            self.errors += 1
            try:
                if temp_path.exists():
                    temp_path.unlink()
            except Exception:
                pass

    def stats(self) -> dict:
        return {
            "enabled": self.enabled,
            "cache_dir": str(self.cache_dir) if self.cache_dir else "",
            "hits": self.hits,
            "misses": self.misses,
            "writes": self.writes,
            "errors": self.errors,
        }
