from .embed_ollama import OllamaEmbedder
from .llm_jiekou import JiekouChatModel
from .profile_repo import JsonProfileRepository

from .knowledge_base import KnowledgeBase
from .kb_seed import default_kb_docs
from .kb_json_loader import load_kb_json_dir, resolve_kb_dir
from .hybrid_retriever import HybridRetriever

from ..application.trigger_manager import TriggerManager
from ..application.layered_profile_learner import LayeredProfileLearner
from ..application.trigger_state import TriggerStateStore

from .null_modules import (
    NullRetriever,
    NullTriggerManager,
    NullProfileLearner,
    NullTriggerStateStore,
)


def build_retriever(env: dict, embedder, use_retriever: bool):
    if not use_retriever:
        return NullRetriever()

    kb_dir = resolve_kb_dir(env.get("KB_DIR"))
    if kb_dir:
        docs = load_kb_json_dir(kb_dir)
    else:
        docs = default_kb_docs()

    kb = KnowledgeBase(docs)
    return HybridRetriever(kb=kb, embedder=embedder)


def build_trigger_manager(use_trigger: bool):
    if not use_trigger:
        return NullTriggerManager()

    return TriggerManager(
        ttc_threshold=3.0,
        distance_threshold=2.0,
        persistent_risk_ratio_threshold=0.4,
        persistent_window=5,
    )


def build_trigger_state_store(use_trigger: bool):
    if not use_trigger:
        return NullTriggerStateStore()
    return TriggerStateStore()


def build_profile_learner(use_profile_learner: bool):
    if not use_profile_learner:
        return NullProfileLearner()
    return LayeredProfileLearner(lr=0.2)


def build_embedder(env: dict):
    return OllamaEmbedder(
        base_url=env.get("OLLAMA_BASE_URL", "http://localhost:11434"),
        model=env.get("OLLAMA_EMBED_MODEL", "nomic-embed-text"),
    )


def build_chat_model(env: dict, primary_model: str, fallback_model: str | None):
    return JiekouChatModel(
        api_key=env.get("JIEKOU_API_KEY", ""),
        base_url=env.get("JIEKOU_BASE_URL", "https://api.jiekou.ai/openai"),
        primary_model=primary_model,
        fallback_model=fallback_model,
        max_completion_tokens=int(env.get("LLM_MAX_COMPLETION_TOKENS", "2048")),
        timeout_s=float(env.get("LLM_TIMEOUT_S", "120")),
        max_retries=int(env.get("LLM_MAX_RETRIES", "1")),
    )


def build_profile_repo(template_profile_path: str, runtime_profile_path: str):
    return JsonProfileRepository(
        template_path=template_profile_path,
        runtime_path=runtime_profile_path,
        auto_init=True,
    )
