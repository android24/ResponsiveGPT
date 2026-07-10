import argparse
import json
import os

from ..domain.models import SceneState
from ..application.service import ResponsiveGPTService
from ..infrastructure.embed_ollama import OllamaEmbedder
from ..infrastructure.llm_jiekou import JiekouChatModel
from ..infrastructure.profile_repo import JsonProfileRepository
from ..evaluation.run_logger import RunLogger
from ..evaluation.metrics import compute_step_metrics
from ..infrastructure.knowledge_base import KnowledgeBase
from ..infrastructure.kb_seed import default_kb_docs
from ..infrastructure.kb_json_loader import load_kb_json_dir
from ..infrastructure.hybrid_retriever import HybridRetriever
from ..evaluation.trigger_plotter import TriggerPlotter
from ..application.trigger_manager import TriggerManager
from ..application.layered_profile_learner import LayeredProfileLearner
from ..application.trigger_state import TriggerStateStore
from ..infrastructure.account_config import load_private_env

def load_env(path: str = ".env") -> dict:
    return load_private_env(path)

def demo_scene() -> SceneState:
    return SceneState(
        scene_type="custom",
        ego_speed_mps=13.5,
        headway_m=8.0,
        lane_change=True,
        dist_to_intersection_m=25.0,
        traffic_light="yellow",
        vrus_present=True,
        lead_speed_mps=10.0,
        rel_speed_mps=3.5,
        event_type="demo",
        frame_index=0,
    )

def build_service(env: dict) -> ResponsiveGPTService:
    embedder = OllamaEmbedder(
        base_url=env.get("OLLAMA_BASE_URL", "http://localhost:11434"),
        model=env.get("OLLAMA_EMBED_MODEL", "nomic-embed-text"),
    )

    kb_dir = env.get("KB_DIR", "data/kb")
    if kb_dir and os.path.isdir(kb_dir):
        docs = load_kb_json_dir(kb_dir)
    else:
        docs = default_kb_docs()

    kb = KnowledgeBase(docs)
    retriever = HybridRetriever(kb=kb, embedder=embedder)

    llm = JiekouChatModel(
        api_key=env.get("JIEKOU_API_KEY", ""),
        base_url=env.get("JIEKOU_BASE_URL", "https://api.jiekou.ai/openai"),
        primary_model=env.get("JIEKOU_MODEL", env.get("PRIMARY_MODEL", "gpt-5.2")),
        fallback_model=env.get("FALLBACK_MODEL"),
        max_completion_tokens=int(env.get("LLM_MAX_COMPLETION_TOKENS", "2048")),
        timeout_s=float(env.get("LLM_TIMEOUT_S", "120")),
        max_retries=int(env.get("LLM_MAX_RETRIES", "1")),
    )

    repo = JsonProfileRepository(env.get("PROFILE_PATH", "driver_profile.json"))

    trigger_manager = TriggerManager(
        ttc_threshold=float(env.get("TTC_THRESHOLD", 3.0)),
        distance_threshold=float(env.get("DISTANCE_THRESHOLD", 2.0)),
        persistent_risk_ratio_threshold=float(env.get("PERSISTENT_RISK_RATIO_THRESHOLD", 0.4)),
        persistent_window=int(env.get("PERSISTENT_WINDOW", 5)),
    )

    profile_learner = LayeredProfileLearner(
        lr=float(env.get("PROFILE_LEARNING_RATE", 0.2))
    )

    trigger_state_store = TriggerStateStore()

    return ResponsiveGPTService(
        retriever=retriever,
        chat_model=llm,
        profile_repo=repo,
        trigger_manager=trigger_manager,
        profile_learner=profile_learner,
        trigger_state_store=trigger_state_store,
    )

def main():
    parser = argparse.ArgumentParser(description="ResponsiveGPT demo")
    parser.add_argument("--demo", action="store_true")
    parser.add_argument("--driver_type", type=str, default="激进")
    parser.add_argument("--feedback", type=str, default="刚才变道有点冒险，但我还是希望不要太慢。")
    parser.add_argument("--tag", type=str, default="demo")
    args = parser.parse_args()

    env = load_private_env()
    api_key = env.get("JIEKOU_API_KEY", "")
    if not api_key:
        raise RuntimeError(
            "Missing JIEKOU_API_KEY. Put it into .env or "
            "config/accounts.local.json."
        )
    
    service = build_service(env)

    if not args.demo:
        raise RuntimeError("Use --demo for the minimal CLI.")

    scene = demo_scene()
    logger = RunLogger(runs_root="runs", tag=args.tag)
    logger.write_config({
        "mode": "demo",
        "driver_type": args.driver_type,
        "feedback": args.feedback,
    })

    result = service.step(
        scene=scene,
        driver_type=args.driver_type,
        feedback=args.feedback,
        recent_decisions=[],
    )
    m = compute_step_metrics(scene, result.decision)

    logger.append_decision({
        "scene": scene.__dict__,
        "profile": result.profile,
        "decision": result.decision,
        "metrics": {"ttc_s": m.ttc_s, "is_violation": m.is_violation},
        "triggers": [t.__dict__ for t in getattr(result, "triggers", [])],
        "guardrails": getattr(result, "guardrails", None).__dict__ if getattr(result, "guardrails", None) else None,
        "profile_update": getattr(result, "profile_update", {}),
    })

    print(f"Run saved to: {logger.run_dir}")
    print(json.dumps(result.decision, ensure_ascii=False, indent=2))

    # --------------------------------------------------
    # Trigger 可视化（论文图自动生成）
    # --------------------------------------------------
    try:
        plotter = TriggerPlotter(run_dir=logger.run_dir)
        fig_paths = plotter.plot_all()

        print("\n📊 Trigger plots generated:")
        for k, v in fig_paths.items():
            if v:
                print(f"{k}: {v}")

    except Exception as e:
        print("\n[WARN] Trigger plotting failed:", e)

if __name__ == "__main__":
    main()
