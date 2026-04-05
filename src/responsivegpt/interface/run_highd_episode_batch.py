import os
import json
import csv
import argparse
from collections import Counter

from ..application.service import ResponsiveGPTService
from ..infrastructure.embed_ollama import OllamaEmbedder
from ..infrastructure.llm_jiekou import JiekouChatModel
from ..infrastructure.profile_repo import JsonProfileRepository

from ..evaluation.run_logger import RunLogger
from ..evaluation.metrics import compute_step_metrics
from ..evaluation.classification import compute_confusion_and_scores

from .adapters.highd_event_adapter import HighDEventAdapter
from .adapters.highd_sequence_adapter import HighDSequenceAdapter

from ..infrastructure.knowledge_base import KnowledgeBase
from ..infrastructure.kb_seed import default_kb_docs
from ..infrastructure.kb_json_loader import load_kb_json_dir
from ..infrastructure.hybrid_retriever import HybridRetriever
from ..evaluation.trigger_plotter import TriggerPlotter

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


def derive_highd_risk_label(row: dict) -> bool:
    """
    基于 highD 原始事件指标构造弱标签。
    后续你可以改成论文正式阈值版本。
    """
    try:
        min_ttc = float(row["minTTC"]) if row.get("minTTC") else None
    except Exception:
        min_ttc = None

    try:
        min_thw = float(row["minTHW"]) if row.get("minTHW") else None
    except Exception:
        min_thw = None

    if min_ttc is not None and min_ttc < 3.0:
        return True
    if min_thw is not None and min_thw < 0.5:
        return True
    return False


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
        model=env.get("JIEKOU_MODEL", "gpt-5.2"),
    )

    repo = JsonProfileRepository(env.get("PROFILE_PATH", "driver_profile.json"))

    return ResponsiveGPTService(
        retriever=retriever,
        chat_model=llm,
        profile_repo=repo,
    )


def append_jsonl(path: str, obj: dict):
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def safe_profile_dict(profile):
    if hasattr(profile, "__dict__"):
        return dict(profile.__dict__)
    return {"value": str(profile)}


def main():
    parser = argparse.ArgumentParser(description="Run highD full sequence batch.")
    parser.add_argument("--csv_path", type=str, required=True, help="Path to highd_strong_interactions_full.csv")
    parser.add_argument("--tag", type=str, default="highd_episode")
    parser.add_argument("--driver_type", type=str, default="激进")
    parser.add_argument("--feedback", type=str, default="保持效率，但避免明显危险操作")
    parser.add_argument("--limit", type=int, default=0, help="0 means all")
    args = parser.parse_args()

    env = load_env(".env")
    if not env.get("JIEKOU_API_KEY"):
        raise RuntimeError("Missing JIEKOU_API_KEY in .env")

    service = build_service(env)
    event_adapter = HighDEventAdapter(args.csv_path)
    seq_adapter = HighDSequenceAdapter()

    logger = RunLogger(runs_root="runs", tag=args.tag)
    logger.write_config({
        "csv_path": args.csv_path,
        "driver_type": args.driver_type,
        "feedback": args.feedback,
        "limit": args.limit,
        "env": {
            "JIEKOU_BASE_URL": env.get("JIEKOU_BASE_URL", "https://api.jiekou.ai/openai"),
            "JIEKOU_MODEL": env.get("JIEKOU_MODEL", "gpt-5.2"),
            "OLLAMA_BASE_URL": env.get("OLLAMA_BASE_URL", "http://localhost:11434"),
            "OLLAMA_EMBED_MODEL": env.get("OLLAMA_EMBED_MODEL", "nomic-embed-text"),
            "PROFILE_PATH": env.get("PROFILE_PATH", "driver_profile.json"),
            "KB_DIR": env.get("KB_DIR", "data/kb"),
        }
    })

    frame_metrics_path = os.path.join(logger.run_dir, "frame_metrics.csv")
    episode_summary_path = os.path.join(logger.run_dir, "episode_summary.jsonl")

    # 每帧指标
    with open(frame_metrics_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "event_index",
            "recordingId",
            "egoId",
            "otherId",
            "eventType",
            "frame_index",
            "ttc_s",
            "is_violation",
            "trigger_count",
            "ego_speed_mps",
            "lead_speed_mps",
            "rel_speed_mps",
            "headway_m",
        ])

    summary = {
        "total_events": 0,
        "total_frames": 0,
        "dataset_risk_true": 0,
        "episode_llm_violation_true": 0,
        "episode_agreement": 0,
    }

    all_y_true = []
    all_y_pred = []

    # 全局 trigger 统计
    global_trigger_counter = Counter()

    for idx, row in enumerate(event_adapter.iter_rows()):
        if args.limit > 0 and idx >= args.limit:
            break

        metadata = event_adapter.row_metadata(row)
        event_scene = event_adapter.row_to_scene(row)
        dataset_risk = derive_highd_risk_label(row)

        scenes = list(seq_adapter.row_to_scenes(row))
        if not scenes:
            continue

        frame_records = []
        ttc_values = []
        violation_flags = []
        recent_decisions = []

        # 当前 episode trigger 统计
        episode_trigger_counter = Counter()

        for scene in scenes:
            result = service.step(
                scene=scene,
                driver_type=args.driver_type,
                feedback=args.feedback,
                recent_decisions=recent_decisions,
            )

            # 让 persistent trigger 生效
            recent_decisions.append(result.decision)
            recent_decisions = recent_decisions[-10:]

            m = compute_step_metrics(scene, result.decision)

            if m.ttc_s is not None:
                ttc_values.append(m.ttc_s)
            if m.is_violation is not None:
                violation_flags.append(bool(m.is_violation))

            # trigger 统计
            for trig in result.triggers:
                trig_type = trig.get("trigger_type", "unknown")
                episode_trigger_counter[trig_type] += 1
                global_trigger_counter[trig_type] += 1

            frame_record = {
                "metadata": metadata,
                "scene": scene.__dict__,
                "profile": safe_profile_dict(result.profile),
                "decision": result.decision,
                "triggers": result.triggers,
                "guardrails": result.guardrails,
                "profile_update": result.profile_update,
                "evidence": result.evidence,
                "step_metrics": {
                    "ttc_s": m.ttc_s,
                    "is_violation": m.is_violation,
                },
            }

            frame_records.append(frame_record)
            logger.append_decision(frame_record)

            with open(frame_metrics_path, "a", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow([
                    idx,
                    metadata.get("recordingId"),
                    metadata.get("egoId"),
                    metadata.get("otherId"),
                    metadata.get("eventType"),
                    scene.frame_index,
                    "" if m.ttc_s is None else round(m.ttc_s, 4),
                    "" if m.is_violation is None else int(m.is_violation),
                    len(result.triggers),
                    scene.ego_speed_mps,
                    scene.lead_speed_mps,
                    scene.rel_speed_mps,
                    scene.headway_m,
                ])

        episode_llm_violation = (sum(violation_flags) > 0) if violation_flags else False
        min_ttc_est = min(ttc_values) if ttc_values else None
        avg_ttc_est = (sum(ttc_values) / len(ttc_values)) if ttc_values else None
        violation_rate = (sum(1 for x in violation_flags if x) / len(violation_flags)) if violation_flags else None

        episode_summary = {
            "event_index": idx,
            "metadata": metadata,
            "event_scene": event_scene.__dict__,
            "dataset_risk_label": dataset_risk,
            "episode_num_frames": len(frame_records),
            "episode_llm_violation": episode_llm_violation,
            "episode_violation_rate": violation_rate,
            "episode_min_ttc_estimated": min_ttc_est,
            "episode_avg_ttc_estimated": avg_ttc_est,
            "event_minTTC_raw": event_scene.min_ttc_raw,
            "event_minTHW_raw": event_scene.min_thw_raw,
            "event_minDHW_raw": event_scene.min_dhw_raw,
            "episode_trigger_stats": dict(episode_trigger_counter),
            "episode_total_triggers": sum(episode_trigger_counter.values()),
        }
        append_jsonl(episode_summary_path, episode_summary)

        summary["total_events"] += 1
        summary["total_frames"] += len(frame_records)
        summary["dataset_risk_true"] += int(dataset_risk)
        summary["episode_llm_violation_true"] += int(episode_llm_violation)
        summary["episode_agreement"] += int(episode_llm_violation == dataset_risk)

        all_y_true.append(bool(dataset_risk))
        all_y_pred.append(bool(episode_llm_violation))

        print(
            f"[{idx}] eventType={metadata.get('eventType')} "
            f"frames={len(frame_records)} "
            f"dataset_risk={dataset_risk} "
            f"episode_llm_violation={episode_llm_violation} "
            f"triggers={sum(episode_trigger_counter.values())}"
        )

    summary["episode_agreement_rate"] = (
        summary["episode_agreement"] / summary["total_events"]
        if summary["total_events"] else 0.0
    )

    cls = compute_confusion_and_scores(all_y_true, all_y_pred)
    summary.update(cls)

    # 增加全局 trigger 分布
    summary["trigger_stats"] = dict(global_trigger_counter)
    summary["total_trigger_activations"] = sum(global_trigger_counter.values())

    with open(os.path.join(logger.run_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    summary_csv_path = os.path.join(logger.run_dir, "summary.csv")
    with open(summary_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["metric", "value"])
        writer.writerow(["total_events", summary["total_events"]])
        writer.writerow(["total_frames", summary["total_frames"]])
        writer.writerow(["dataset_risk_true", summary["dataset_risk_true"]])
        writer.writerow(["episode_llm_violation_true", summary["episode_llm_violation_true"]])
        writer.writerow(["episode_agreement", summary["episode_agreement"]])
        writer.writerow(["episode_agreement_rate", summary["episode_agreement_rate"]])
        writer.writerow(["tp", summary["confusion_matrix"]["tp"]])
        writer.writerow(["fp", summary["confusion_matrix"]["fp"]])
        writer.writerow(["fn", summary["confusion_matrix"]["fn"]])
        writer.writerow(["tn", summary["confusion_matrix"]["tn"]])
        writer.writerow(["precision", summary["precision"]])
        writer.writerow(["recall", summary["recall"]])
        writer.writerow(["f1", summary["f1"]])
        writer.writerow(["accuracy", summary["accuracy"]])
        writer.writerow(["total_trigger_activations", summary["total_trigger_activations"]])

        for trig_type, count in summary["trigger_stats"].items():
            writer.writerow([f"trigger_{trig_type}", count])

    print("\nRun saved to:", logger.run_dir)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    
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