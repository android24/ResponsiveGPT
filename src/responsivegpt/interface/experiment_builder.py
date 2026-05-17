import os
import json
from pathlib import Path

from ..application.service import ResponsiveGPTService
from ..application.trigger_manager import TriggerManager
from ..application.layered_profile_learner import LayeredProfileLearner
from ..application.trigger_state import TriggerStateStore

from ..infrastructure.embed_ollama import OllamaEmbedder
from ..infrastructure.llm_jiekou import JiekouChatModel
from ..infrastructure.profile_repo import JsonProfileRepository
from ..infrastructure.knowledge_base import KnowledgeBase
from ..infrastructure.kb_seed import default_kb_docs
from ..infrastructure.kb_json_loader import load_kb_json_dir
from ..infrastructure.hybrid_retriever import HybridRetriever
from ..infrastructure.null_modules import NullRetriever
from ..infrastructure.null_modules import NullTriggerManager
from ..infrastructure.null_modules import NullTriggerStateStore
from ..infrastructure.null_modules import NullTriggerManager
from ..infrastructure.null_modules import NullProfileLearner

from ..evaluation.run_logger import RunLogger


def load_env(path: str = ".env") -> dict:
    env = {}
    if not os.path.exists(path):
        return env

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            env[k.strip()] = v.strip()
    return env


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
) -> ResponsiveGPTService:
    embedder = OllamaEmbedder(
        base_url=env.get("OLLAMA_BASE_URL", "http://localhost:11434"),
        model=env.get("OLLAMA_EMBED_MODEL", "nomic-embed-text"),
    )

    if use_retriever:
        kb_dir = env.get("KB_DIR", "data/kb")
        if kb_dir and os.path.isdir(kb_dir):
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
        trigger_state_store = TriggerStateStore()
    else:
        trigger_manager = NullTriggerManager()
        trigger_state_store = NullTriggerStateStore()

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
    env = load_env(".env")
    if not env.get("JIEKOU_API_KEY"):
        raise RuntimeError("Missing JIEKOU_API_KEY in .env")

    logger = RunLogger(
        runs_root="runs",
        tag=f"{args.tag}_{args.profile_name}"
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
            "KB_DIR": env.get("KB_DIR", "data/kb"),
        },
        "llm_call_policy": {
            "policy": args.llm_policy,
            "stride": args.llm_stride,
            "risk_threshold": args.llm_risk_threshold,
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