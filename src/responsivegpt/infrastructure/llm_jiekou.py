from collections import defaultdict
from contextlib import contextmanager
from copy import deepcopy
import time

from openai import (
    OpenAI,
    APIConnectionError,
    APIError,
    APIStatusError,
    APITimeoutError,
    BadRequestError,
    RateLimitError,
)
from ..domain.logic import coerce_json
from .json_disk_cache import JsonDiskCache


RECOVERABLE_ERRORS = (
    APITimeoutError,
    APIConnectionError,
    APIStatusError,
    APIError,
    RateLimitError,
)


def _new_usage_stats():
    return {
        "attempts": 0,
        "successes": 0,
        "failures": 0,
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
        "cached_tokens": 0,
        "cache_hits": 0,
        "cache_misses": 0,
        "latencies_ms": [],
        "models": defaultdict(int),
    }


class LLMBudgetExceeded(RuntimeError):
    def __init__(self, context: str, budget_type: str, limit: int):
        super().__init__(
            f"LLM {budget_type} budget exhausted for {context}: {limit}"
        )
        self.context = context
        self.budget_type = budget_type
        self.limit = limit


class JiekouChatModel:
    """
    支持：
    1. 主模型
    2. 自动 fallback
    3. 不同模型参数兼容
    """

    def __init__(
        self,
        api_key: str,
        base_url: str,
        primary_model: str = "gpt-5.2",
        fallback_model: str | None = None,
        max_completion_tokens: int = 2048,
        timeout_s: float = 120.0,
        max_retries: int = 1,
        seed: int = 0,
        cache_dir: str | None = None,
        cache_enabled: bool = True,
    ):
        self.client = OpenAI(
            api_key=api_key,
            base_url=base_url,
            timeout=timeout_s,
            # Retries are explicit below so every network request is visible
            # to the experiment budget and telemetry.
            max_retries=0,
        )
        self.primary_model = primary_model
        self.fallback_model = fallback_model
        self.max_completion_tokens = max_completion_tokens
        self.timeout_s = timeout_s
        self.max_retries = max_retries
        self.seed = int(seed or 0)
        self.cache = JsonDiskCache(cache_dir, enabled=cache_enabled)
        self.last_cache_hit = False
        self.last_cache_key = ""
        self._usage_context = "unclassified"
        self._phase_usage_context = "unclassified"
        self._budgets = defaultdict(lambda: {
            "max_attempts": 0,
            "max_tokens": 0,
        })
        self._usage = defaultdict(_new_usage_stats)
        self._phase_usage = defaultdict(_new_usage_stats)

    def configure_budget(
        self,
        context: str,
        *,
        max_attempts: int = 0,
        max_tokens: int = 0,
    ) -> None:
        if max_attempts < 0 or max_tokens < 0:
            raise ValueError("LLM budgets must be >= 0")
        self._budgets[str(context)] = {
            "max_attempts": int(max_attempts),
            "max_tokens": int(max_tokens),
        }

    def budget_status(self, context: str) -> dict:
        context = str(context)
        stats = self._usage[context]
        budget = self._budgets[context]
        max_attempts = int(budget["max_attempts"])
        max_tokens = int(budget["max_tokens"])
        attempts_exhausted = (
            max_attempts > 0 and stats["attempts"] >= max_attempts
        )
        tokens_exhausted = (
            max_tokens > 0 and stats["total_tokens"] >= max_tokens
        )
        return {
            **budget,
            "attempts": int(stats["attempts"]),
            "total_tokens": int(stats["total_tokens"]),
            "attempts_exhausted": attempts_exhausted,
            "tokens_exhausted": tokens_exhausted,
            "exhausted": attempts_exhausted or tokens_exhausted,
            # Token usage is known only after a response; at most the final
            # admitted request can overshoot this cap.
            "token_overshoot": max(
                0, int(stats["total_tokens"]) - max_tokens
            ) if max_tokens > 0 else 0,
        }

    def budget_exhausted(self, context: str) -> bool:
        return bool(self.budget_status(context)["exhausted"])

    @contextmanager
    def usage_context(self, name: str):
        previous = self._usage_context
        self._usage_context = str(name or "unclassified")
        try:
            yield
        finally:
            self._usage_context = previous

    @contextmanager
    def phase_usage_context(self, name: str):
        previous = self._phase_usage_context
        self._phase_usage_context = str(name or "unclassified")
        try:
            yield
        finally:
            self._phase_usage_context = previous

    @staticmethod
    def _percentile(values: list[float], q: float) -> float:
        if not values:
            return 0.0
        ordered = sorted(values)
        index = min(len(ordered) - 1, int(round(q * (len(ordered) - 1))))
        return float(ordered[index])

    def usage_summary(self) -> dict:
        out = {}
        for context, stats in self._usage.items():
            latencies = list(stats["latencies_ms"])
            out[context] = {
                "attempts": stats["attempts"],
                "successes": stats["successes"],
                "failures": stats["failures"],
                "prompt_tokens": stats["prompt_tokens"],
                "completion_tokens": stats["completion_tokens"],
                "total_tokens": stats["total_tokens"],
                "cached_tokens": stats["cached_tokens"],
                "cache_hits": stats["cache_hits"],
                "cache_misses": stats["cache_misses"],
                "latency_ms_mean": (
                    sum(latencies) / len(latencies) if latencies else 0.0
                ),
                "latency_ms_p50": self._percentile(latencies, 0.50),
                "latency_ms_p95": self._percentile(latencies, 0.95),
                "models": dict(stats["models"]),
                "budget": self.budget_status(context),
            }
        return out

    def phase_usage_summary(self) -> dict:
        out = {}
        for context, stats in self._phase_usage.items():
            latencies = list(stats["latencies_ms"])
            out[context] = {
                "attempts": stats["attempts"],
                "successes": stats["successes"],
                "failures": stats["failures"],
                "prompt_tokens": stats["prompt_tokens"],
                "completion_tokens": stats["completion_tokens"],
                "total_tokens": stats["total_tokens"],
                "cached_tokens": stats["cached_tokens"],
                "cache_hits": stats["cache_hits"],
                "cache_misses": stats["cache_misses"],
                "latency_ms_mean": (
                    sum(latencies) / len(latencies)
                    if latencies else 0.0
                ),
                "latency_ms_p50": self._percentile(
                    latencies, 0.50
                ),
                "latency_ms_p95": self._percentile(
                    latencies, 0.95
                ),
                "models": dict(stats["models"]),
            }
        return out

    def export_usage_state(self) -> dict:
        def plain(mapping):
            return {
                key: {
                    **deepcopy(value),
                    "models": dict(value["models"]),
                }
                for key, value in mapping.items()
            }
        return {
            "usage": plain(self._usage),
            "phase_usage": plain(self._phase_usage),
        }

    def import_usage_state(self, state: dict) -> None:
        def restore(target, rows):
            target.clear()
            for key, value in (rows or {}).items():
                stats = _new_usage_stats()
                for field in (
                    "attempts", "successes", "failures",
                    "prompt_tokens", "completion_tokens",
                    "total_tokens", "cached_tokens",
                    "cache_hits", "cache_misses",
                ):
                    stats[field] = int(value.get(field, 0) or 0)
                stats["latencies_ms"] = [
                    float(item)
                    for item in value.get("latencies_ms", [])
                ]
                stats["models"].update(value.get("models", {}))
                target[str(key)] = stats
        restore(self._usage, state.get("usage", {}))
        restore(self._phase_usage, state.get("phase_usage", {}))

    def complete_json(self, system: str, user: str) -> dict:
        self.last_cache_hit = False
        self.last_cache_key = self._cache_key(system, user)
        context_stats = self._usage[self._usage_context]
        phase_stats = self._phase_usage[self._phase_usage_context]

        cached = self.cache.get(self.last_cache_key)
        if isinstance(cached, dict):
            context_stats["cache_hits"] += 1
            phase_stats["cache_hits"] += 1
            self.last_cache_hit = True
            return deepcopy(cached)

        context_stats["cache_misses"] += 1
        phase_stats["cache_misses"] += 1
        text = self._complete_with_fallback(system, user)

        try:
            obj = coerce_json(text)
            self.cache.set(
                self.last_cache_key,
                obj,
                metadata=self._cache_metadata(system, user, repaired=False),
            )
            return obj
        except Exception:
            repair_system = system + "\n如果你刚才输出不是合法 JSON，现在必须只输出合法 JSON。"
            repair_user = (
                "上一次输出解析失败。请只返回符合 schema 的合法 JSON，"
                "不要包含任何解释、markdown 或代码块标记。\n"
                f"原始输出：\n{text}"
            )

            repaired = self._complete_with_fallback(repair_system, repair_user)
            obj = coerce_json(repaired)
            self.cache.set(
                self.last_cache_key,
                obj,
                metadata=self._cache_metadata(system, user, repaired=True),
            )
            return obj

    def _cache_key(self, system: str, user: str) -> str:
        return JsonDiskCache.stable_hash({
            "cache_version": "llm_json_v1",
            "primary_model": self.primary_model,
            "fallback_model": self.fallback_model,
            "max_completion_tokens": self.max_completion_tokens,
            "seed": self.seed,
            "system": system,
            "user": user,
        })

    def _cache_metadata(
        self,
        system: str,
        user: str,
        *,
        repaired: bool,
    ) -> dict:
        return {
            "cache_version": "llm_json_v1",
            "primary_model": self.primary_model,
            "fallback_model": self.fallback_model,
            "max_completion_tokens": self.max_completion_tokens,
            "seed": self.seed,
            "usage_context": self._usage_context,
            "phase_usage_context": self._phase_usage_context,
            "system_chars": len(system or ""),
            "user_chars": len(user or ""),
            "repaired": bool(repaired),
        }

    def _complete_with_fallback(self, system: str, user: str) -> str:
        models = [self.primary_model]
        if self.fallback_model and self.fallback_model != self.primary_model:
            models.append(self.fallback_model)

        errors = []

        for i, model in enumerate(models):
            is_fallback = i > 0
            label = "fallback" if is_fallback else "primary"

            for attempt in range(self.max_retries + 1):
                try:
                    return self._complete(system, user, model)
                except LLMBudgetExceeded:
                    raise
                except BadRequestError as e:
                    errors.append(e)
                    break
                except RECOVERABLE_ERRORS as e:
                    errors.append(e)
                    if attempt < self.max_retries:
                        print(
                            f"[WARN] {label} model request failed: {model}. "
                            f"Retrying ({attempt + 1}/{self.max_retries}). "
                            f"Error: {e}"
                        )
                        continue
                    break

            if i + 1 < len(models):
                print(
                    f"[WARN] {label} model failed: {model}. "
                    f"Falling back to: {models[i + 1]}. "
                    f"Error: {errors[-1]}"
                )
                continue
            if errors:
                raise errors[-1]

        if errors:
            raise errors[-1]
        raise RuntimeError("No model configured for completion.")

    def _complete(self, system: str, user: str, model: str) -> str:
        kwargs = {
            "model": model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "max_completion_tokens": self.max_completion_tokens,
        }

        # GPT-5 路由上你已经实测到 temperature/top_p 之类会报错
        # 所以这里只给非 gpt-5 模型传 temperature
        if not model.startswith("gpt-5"):
            kwargs["temperature"] = 0.4
            if self.seed:
                kwargs["seed"] = self.seed

        context = self._usage_context
        stats = self._usage[context]
        phase_stats = self._phase_usage[self._phase_usage_context]
        budget = self._budgets[context]
        if budget["max_attempts"] > 0 and (
            stats["attempts"] >= budget["max_attempts"]
        ):
            raise LLMBudgetExceeded(
                context, "request", budget["max_attempts"]
            )
        if budget["max_tokens"] > 0 and (
            stats["total_tokens"] >= budget["max_tokens"]
        ):
            raise LLMBudgetExceeded(
                context, "token", budget["max_tokens"]
            )
        stats["attempts"] += 1
        stats["models"][model] += 1
        phase_stats["attempts"] += 1
        phase_stats["models"][model] += 1
        started = time.perf_counter()
        try:
            resp = self.client.chat.completions.create(**kwargs)
        except Exception:
            latency_ms = (time.perf_counter() - started) * 1000.0
            stats["failures"] += 1
            stats["latencies_ms"].append(latency_ms)
            phase_stats["failures"] += 1
            phase_stats["latencies_ms"].append(latency_ms)
            raise

        latency_ms = (time.perf_counter() - started) * 1000.0
        stats["successes"] += 1
        stats["latencies_ms"].append(latency_ms)
        phase_stats["successes"] += 1
        phase_stats["latencies_ms"].append(latency_ms)
        usage = getattr(resp, "usage", None)
        if usage is not None:
            prompt_tokens = int(
                getattr(usage, "prompt_tokens", 0) or 0
            )
            completion_tokens = int(
                getattr(usage, "completion_tokens", 0) or 0
            )
            total_tokens = int(
                getattr(usage, "total_tokens", 0) or 0
            )
            stats["prompt_tokens"] += prompt_tokens
            stats["completion_tokens"] += completion_tokens
            stats["total_tokens"] += total_tokens
            phase_stats["prompt_tokens"] += prompt_tokens
            phase_stats["completion_tokens"] += completion_tokens
            phase_stats["total_tokens"] += total_tokens
            details = getattr(usage, "prompt_tokens_details", None)
            cached_tokens = int(
                getattr(details, "cached_tokens", 0) or 0
            )
            stats["cached_tokens"] += cached_tokens
            phase_stats["cached_tokens"] += cached_tokens
        return resp.choices[0].message.content
