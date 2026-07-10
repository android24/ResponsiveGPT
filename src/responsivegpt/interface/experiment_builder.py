import os
import json
from pathlib import Path

from ..application.service import ResponsiveGPTService
from ..application.trigger_manager import TriggerManager
from ..application.layered_profile_learner import LayeredProfileLearner

from ..infrastructure.embed_ollama import OllamaEmbedder
from ..infrastructure.llm_jiekou import JiekouChatModel
from ..infrastructure.profile_repo import JsonProfileRepository
from ..infrastructure.knowledge_base import KnowledgeBase
from ..infrastructure.kb_seed import default_kb_docs
from ..infrastructure.kb_json_loader import load_kb_json_dir, resolve_kb_dir
from ..infrastructure.hybrid_retriever import HybridRetriever
from ..infrastructure.null_modules import NullRetriever
from ..infrastructure.null_modules import NullTriggerManager
from ..infrastructure.null_modules import NullProfileLearner
from ..infrastructure.account_config import load_private_env

from ..evaluation.run_logger import RunLogger


_RESUMABLE_RUN_ARTIFACTS = (
    "adapted_profile.json",
    "config.json",
    "decisions.jsonl",
    "episode_checkpoint.json.tmp",
    "episode_summary.jsonl",
    "final_profile.json",
    "frame_metrics.csv",
    "guardrail_trace.jsonl",
    "initial_profile.json",
    "planning_trace.jsonl",
    "profile_delta.jsonl",
    "profile_split_manifest.json",
    "profile_trace.jsonl",
    "rag_trace.jsonl",
    "runtime_profile.json",
    "summary.csv",
    "summary.json",
    "trigger_summary.csv",
    "trigger_trace.jsonl",
)


def _prepare_resume_run_dir(run_dir: str) -> None:
    """Reset an interrupted run that never committed its first episode."""
    path = Path(run_dir)
    checkpoint_path = path / "episode_checkpoint.json"
    if checkpoint_path.exists():
        return
    for name in _RESUMABLE_RUN_ARTIFACTS:
        artifact = path / name
        if artifact.is_file():
            artifact.unlink()


def load_env(path: str = ".env") -> dict:
    return load_private_env(path)


def resolve_profile_template_path(profiles_dir: str, profile_name: str) -> str:
    """
    对齐 highD / inD 已验证脚本：
    从项目根目录解析 profile 模板。
    同时保留相对路径 fallback，兼容 rounD 原写法。
    """
    current_file = Path(__file__).resolve()
    project_root = current_file.parents[3]

    candidate = project_root / profiles_dir / f"{profile_name}.json"
    if candidate.exists():
        return str(candidate)

    fallback = Path(profiles_dir) / f"{profile_name}.json"
    if fallback.exists():
        return str(fallback)

    raise FileNotFoundError(
        f"\n❌ Profile template not found:\n"
        f"  Tried: {candidate}\n"
        f"  Fallback: {fallback}\n"
    )


def select_model(env: dict, model_role: str):
    selected_model = env.get("PRIMARY_MODEL", "gpt-5.2")
    selected_fallback = env.get("FALLBACK_MODEL", "gpt-4.1")

    if model_role == "fallback":
        selected_model = env.get("FALLBACK_MODEL", "gpt-4.1")
        selected_fallback = None
    elif model_role == "cheap":
        selected_model = env.get("CHEAP_MODEL", "gpt-4o-mini")
        selected_fallback = None

    return selected_model, selected_fallback


def build_service(
    *,
    env: dict,
    template_profile_path: str,
    runtime_profile_path: str,
    model_role: str,
    use_trigger: bool,
    use_profile_learner: bool,
    use_retriever: bool,
    dataset: str,
    repeat_seed: int = 0,
    llm_cache_dir: str | None = None,
    llm_cache_enabled: bool = True,
) -> ResponsiveGPTService:
    embedder = OllamaEmbedder(
        base_url=env.get("OLLAMA_BASE_URL", "http://localhost:11434"),
        model=env.get("OLLAMA_EMBED_MODEL", "nomic-embed-text"),
    )

    if use_retriever:
        kb_dir = resolve_kb_dir(env.get("KB_DIR"))
        if kb_dir:
            docs = load_kb_json_dir(kb_dir)
        else:
            docs = default_kb_docs()

        kb = KnowledgeBase(docs)
        retriever = HybridRetriever(kb=kb, embedder=embedder)
    else:
        retriever = NullRetriever()

    selected_model, selected_fallback = select_model(env, model_role)

    llm = JiekouChatModel(
        api_key=env.get("JIEKOU_API_KEY", ""),
        base_url=env.get("JIEKOU_BASE_URL", "https://api.jiekou.ai/openai"),
        primary_model=selected_model,
        fallback_model=selected_fallback,
        max_completion_tokens=int(env.get("LLM_MAX_COMPLETION_TOKENS", "2048")),
        timeout_s=float(env.get("LLM_TIMEOUT_S", "120")),
        max_retries=int(env.get("LLM_MAX_RETRIES", "1")),
        seed=repeat_seed,
        cache_dir=llm_cache_dir,
        cache_enabled=llm_cache_enabled,
    )

    repo = JsonProfileRepository(
        template_path=template_profile_path,
        runtime_path=runtime_profile_path,
        auto_init=True,
    )

    if use_trigger:
        # 对齐已验证脚本：
        # highD / rounD distance_threshold=2.0
        # inD distance_threshold=2.5
        distance_threshold = 2.5 if dataset.lower() == "ind" else 2.0

        trigger_manager = TriggerManager(
            ttc_threshold=3.0,
            distance_threshold=distance_threshold,
            persistent_risk_ratio_threshold=0.4,
            persistent_window=5,
        )
        # Profile updates are already persisted in the runtime repository.
        # The legacy trigger store was write-only in this execution path and
        # accumulated TTL events without affecting decisions.
        trigger_state_store = None
    else:
        trigger_manager = NullTriggerManager()
        trigger_state_store = None

    profile_learner = LayeredProfileLearner(lr=0.2) if use_profile_learner else NullProfileLearner()

    return ResponsiveGPTService(
        retriever=retriever,
        chat_model=llm,
        profile_repo=repo,
        trigger_manager=trigger_manager,
        profile_learner=profile_learner,
        trigger_state_store=trigger_state_store,
    )


def build_experiment_context(args):
    env = load_private_env()
    if not env.get("JIEKOU_API_KEY"):
        raise RuntimeError(
            "Missing JIEKOU_API_KEY in .env or config/accounts.local.json"
        )

    resume_run_dir = str(
        getattr(args, "resume_run_dir", "") or ""
    )
    if resume_run_dir:
        _prepare_resume_run_dir(resume_run_dir)

    logger = RunLogger(
        runs_root="runs",
        tag=f"{args.tag}_{args.profile_name}",
        run_dir=resume_run_dir or None,
    )

    template_profile_path = resolve_profile_template_path(
        args.profiles_dir,
        args.profile_name,
    )

    runtime_profile_path = os.path.join(logger.run_dir, "runtime_profile.json")
    initial_profile_copy_path = os.path.join(logger.run_dir, "initial_profile.json")
    final_profile_path = os.path.join(logger.run_dir, "final_profile.json")

    with open(template_profile_path, "r", encoding="utf-8") as f:
        init_profile = json.load(f)

    if not os.path.exists(initial_profile_copy_path):
        with open(initial_profile_copy_path, "w", encoding="utf-8") as f:
            json.dump(init_profile, f, ensure_ascii=False, indent=2)

    effective_driver_type = args.driver_type or init_profile.get("driver_type", "均衡")

    service = build_service(
        env=env,
        template_profile_path=template_profile_path,
        runtime_profile_path=runtime_profile_path,
        model_role=args.model_role,
        use_trigger=bool(args.use_trigger),
        use_profile_learner=bool(args.use_profile_learner),
        use_retriever=bool(args.use_retriever),
        dataset=args.dataset,
        repeat_seed=int(getattr(args, "repeat_seed", 0) or 0),
        llm_cache_dir=str(getattr(args, "llm_cache_dir", "") or ""),
        llm_cache_enabled=not bool(
            getattr(args, "disable_llm_cache", False)
        ),
    )
    service.llm.configure_budget(
        "reactive",
        max_attempts=int(
            getattr(args, "max_reactive_api_attempts", 0) or 0
        ),
        max_tokens=int(getattr(args, "max_reactive_tokens", 0) or 0),
    )
    service.llm.configure_budget(
        "planning",
        max_attempts=int(
            getattr(args, "max_planning_api_attempts", 0) or 0
        ),
        max_tokens=int(getattr(args, "max_planning_tokens", 0) or 0),
    )

    logger.write_config({
        "dataset": args.dataset,
        "mode": args.mode,
        "summary_csv": args.summary_csv,
        "sequence_root": args.sequence_root,
        "profile_name": args.profile_name,
        "profiles_dir": args.profiles_dir,
        "template_profile_path": template_profile_path,
        "runtime_profile_path": runtime_profile_path,
        "initial_profile_copy_path": initial_profile_copy_path,
        "driver_type": effective_driver_type,
        "feedback": args.feedback,
        "limit": args.limit,
        "model_role": args.model_role,
        "experiment_fingerprint": getattr(args, "experiment_fingerprint", ""),
        "method_version": getattr(args, "method_version", ""),
        "repeat_seed": int(getattr(args, "repeat_seed", 0) or 0),
        "episode_order_seed": int(getattr(args, "episode_order_seed", 0) or 0),
        "max_reactive_api_attempts": int(
            getattr(args, "max_reactive_api_attempts", 0) or 0
        ),
        "max_reactive_tokens": int(
            getattr(args, "max_reactive_tokens", 0) or 0
        ),
        "max_planning_api_attempts": int(
            getattr(args, "max_planning_api_attempts", 0) or 0
        ),
        "max_planning_tokens": int(
            getattr(args, "max_planning_tokens", 0) or 0
        ),
        "cache": {
            "llm_cache_enabled": not bool(
                getattr(args, "disable_llm_cache", False)
            ),
            "llm_cache_dir": str(
                getattr(args, "llm_cache_dir", "") or ""
            ),
            "rag_cache_enabled": not bool(
                getattr(args, "disable_rag_cache", False)
            ),
            "rag_cache_dir": str(
                getattr(args, "rag_cache_dir", "") or ""
            ),
            "planning_cache_enabled": not bool(
                getattr(args, "disable_planning_cache", False)
            ),
            "planning_cache_dir": str(
                getattr(args, "planning_cache_dir", "") or ""
            ),
        },
        "ttc_threshold": getattr(args, "ttc_threshold", None),
        "distance_threshold": getattr(args, "distance_threshold", None),
        "drac_threshold": getattr(args, "drac_threshold", None),
        "history_window": args.history_window,
        "ablation": {
            "use_trigger": bool(args.use_trigger),
            "use_profile_learner": bool(args.use_profile_learner),
            "use_retriever": bool(args.use_retriever),
        },
        "safety_metrics": {
            "version": "full_v1",
            "prediction_horizon_s": 5.0,
            "prediction_dt_s": 0.2,
            "risk_index": "UPRI_v1",
        },
        "env": {
            "JIEKOU_BASE_URL": env.get("JIEKOU_BASE_URL", "https://api.jiekou.ai/openai"),
            "PRIMARY_MODEL": env.get("PRIMARY_MODEL"),
            "FALLBACK_MODEL": env.get("FALLBACK_MODEL"),
            "CHEAP_MODEL": env.get("CHEAP_MODEL"),
            "OLLAMA_BASE_URL": env.get("OLLAMA_BASE_URL", "http://localhost:11434"),
            "OLLAMA_EMBED_MODEL": env.get("OLLAMA_EMBED_MODEL", "nomic-embed-text"),
            "KB_DIR": env.get("KB_DIR", "src/responsivegpt/data/kb"),
            "KB_DIR_EFFECTIVE": resolve_kb_dir(env.get("KB_DIR")) or "default_kb_docs",
            "LLM_TIMEOUT_S": env.get("LLM_TIMEOUT_S", "120"),
            "LLM_MAX_RETRIES": env.get("LLM_MAX_RETRIES", "1"),
            "LLM_MAX_COMPLETION_TOKENS": env.get("LLM_MAX_COMPLETION_TOKENS", "2048"),
        },
        "llm_call_policy": {
            "policy": args.llm_policy,
            "stride": args.llm_stride,
            "risk_threshold": args.llm_risk_threshold,
            "max_stale_frames": getattr(args, "llm_max_stale_frames", None),
            "risk_delta_threshold": getattr(args, "llm_risk_delta_threshold", None),
            "reuse_last_decision": bool(args.reuse_last_decision),
        },
        "planning_thread": {
            "enabled": bool(args.use_planning_thread),
            "interval": args.planning_interval,
            "min_gap": args.planning_min_gap,
            "risk_threshold": args.planning_risk_threshold,
            "time_horizon_s": args.planning_time_horizon_s,
            "max_history": args.planning_max_history,
            "quality_horizon": args.planning_quality_horizon,
        },
        "token_time_abstraction": {
            "reactive_thread": "frame-level bounded reasoning",
            "planning_thread": "low-frequency long-horizon reasoning",
            "planning_hint_injection": True,
            "stale_planning_protection": True,
        },
        "rag": {
            "rag_version": "rag_v1_grounded",
            "use_retriever": bool(args.use_retriever),
            "rag_mode": getattr(args, "rag_mode", "full"),
            "rag_budget": getattr(args, "rag_budget", "reactive"),
            "rag_top_k": getattr(args, "rag_top_k", 12),
            "require_grounded_decision": bool(getattr(args, "require_grounded_decision", 0)),
        },
        "trace": {
            "detail": bool(getattr(args, "trace_detail", True)),
        },
    })

    return {
        "env": env,
        "logger": logger,
        "service": service,
        "effective_driver_type": effective_driver_type,
        "runtime_profile_path": runtime_profile_path,
        "final_profile_path": final_profile_path,
        "template_profile_path": template_profile_path,
    }
