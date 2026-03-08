import argparse
import json
import os

from ..domain.models import SceneState
from ..application.service import ResponsiveGPTService
from ..infrastructure.embed_ollama import OllamaEmbedder
from ..infrastructure.vectorstore import SimpleVectorStore
from ..infrastructure.llm_jiekou import JiekouChatModel
from ..infrastructure.profile_repo import JsonProfileRepository
from ..infrastructure.rules import generic_rules
from ..evaluation.run_logger import RunLogger
from ..evaluation.metrics import compute_step_metrics

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

def main():
    parser = argparse.ArgumentParser(description="ResponsiveGPT demo")
    parser.add_argument("--demo", action="store_true")
    parser.add_argument("--driver_type", type=str, default="激进")
    parser.add_argument("--feedback", type=str, default="刚才变道有点冒险，但我还是希望不要太慢。")
    parser.add_argument("--tag", type=str, default="demo")
    args = parser.parse_args()

    env = load_env(".env")
    api_key = env.get("JIEKOU_API_KEY", "")
    if not api_key:
        raise RuntimeError("Missing JIEKOU_API_KEY. Put it into .env.")

    embedder = OllamaEmbedder(
        base_url=env.get("OLLAMA_BASE_URL", "http://localhost:11434"),
        model=env.get("OLLAMA_EMBED_MODEL", "nomic-embed-text"),
    )
    vs = SimpleVectorStore(embedder=embedder, docs=generic_rules())
    vs.build()

    llm = JiekouChatModel(
        api_key=api_key,
        base_url=env.get("JIEKOU_BASE_URL", "https://api.jiekou.ai/openai"),
        model=env.get("JIEKOU_MODEL", "gpt-5.2"),
    )
    repo = JsonProfileRepository(env.get("PROFILE_PATH", "driver_profile.json"))
    service = ResponsiveGPTService(vectorstore=vs, chat_model=llm, profile_repo=repo)

    if not args.demo:
        raise RuntimeError("Use --demo for the minimal CLI.")

    scene = demo_scene()
    logger = RunLogger(runs_root="runs", tag=args.tag)
    logger.write_config({
        "mode": "demo",
        "driver_type": args.driver_type,
        "feedback": args.feedback,
    })

    result = service.step(scene=scene, driver_type=args.driver_type, feedback=args.feedback)
    m = compute_step_metrics(scene, result.decision)

    logger.append_decision({
        "scene": scene.__dict__,
        "profile": result.profile.__dict__,
        "decision": result.decision,
        "metrics": {"ttc_s": m.ttc_s, "is_violation": m.is_violation},
    })

    print(f"Run saved to: {logger.run_dir}")
    print(json.dumps(result.decision, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()
