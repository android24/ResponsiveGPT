import os
import json
import csv
import argparse

from ..application.service import ResponsiveGPTService
from ..infrastructure.embed_ollama import OllamaEmbedder
from ..infrastructure.llm_jiekou import JiekouChatModel
from ..infrastructure.profile_repo import JsonProfileRepository

from ..evaluation.run_logger import RunLogger
from ..evaluation.metrics import compute_step_metrics
from ..evaluation.classification import compute_confusion_and_scores
from ..evaluation.round_labels import derive_round_risk_label_from_summary_row

from .adapters.roundd_event_adapter import RoundEventAdapter
from .adapters.roundd_clip_sequence_adapter import RoundClipSequenceAdapter

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


def resolve_clip_path(clips_root: str, clip_file: str) -> str:
    """
    summary 里通常是 clips/18/18_event_xxx.csv
    这里统一解析成实际文件路径
    """
    clip_file = clip_file.replace("\\", "/").strip()

    if clip_file.startswith("clips/"):
        clip_file = clip_file[len("clips/"):]

    return os.path.join(clips_root, clip_file)


def _safe_trigger_type(trigger_item: dict) -> str:
    if not isinstance(trigger_item, dict):
        return "unknown"
    return str(trigger_item.get("trigger_type", "unknown"))


def _safe_len(x) -> int:
    try:
        return len(x)
    except Exception:
        return 0


def main():
    parser = argparse.ArgumentParser(description="Run rounD sequence batch with clips.")
    parser.add_argument("--summary_csv", type=str, required=True)
    parser.add_argument("--clips_root", type=str, required=True)
    parser.add_argument("--tag", type=str, default="round_episode")
    parser.add_argument("--driver_type", type=str, default="保守")
    parser.add_argument("--feedback", type=str, default="优先安全，避免与其他交通参与者发生近距离冲突")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--ttc_threshold", type=float, default=3.0)
    parser.add_argument("--distance_threshold", type=float, default=2.0)
    parser.add_argument("--history_window", type=int, default=10, help="用于 persistent trigger 的 recent_decisions 长度")
    args = parser.parse_args()

    env = load_env(".env")
    if not env.get("JIEKOU_API_KEY"):
        raise RuntimeError("Missing JIEKOU_API_KEY in .env")

    service = build_service(env)
    event_adapter = RoundEventAdapter(args.summary_csv)

    logger = RunLogger(runs_root="runs", tag=args.tag)
    logger.write_config({
        "summary_csv": args.summary_csv,
        "clips_root": args.clips_root,
        "driver_type": args.driver_type,
        "feedback": args.feedback,
        "limit": args.limit,
        "ttc_threshold": args.ttc_threshold,
        "distance_threshold": args.distance_threshold,
        "history_window": args.history_window,
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

    with open(frame_metrics_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "event_index",
            "recordingId",
            "event_id",
            "pair_type",
            "frame_index",
            "ttc_s",
            "is_violation",
            "ego_speed_mps",
            "rel_speed_mps",
            "headway_m",
            "vrus_present",
            "num_triggers",
            "num_rules",
            "num_law_evidence",
            "num_case_evidence",
            "num_scenario_evidence",
        ])

    summary = {
        "total_events": 0,
        "total_frames": 0,
        "dataset_risk_true": 0,
        "episode_llm_violation_true": 0,
        "episode_agreement": 0,
        "missing_clips": 0,
    }

    all_y_true = []
    all_y_pred = []

    global_trigger_stats = {}

    for idx, row in enumerate(event_adapter.iter_rows()):
        if args.limit > 0 and idx >= args.limit:
            break

        metadata = event_adapter.row_metadata(row)
        clip_file = metadata.get("clip_file")
        if not clip_file:
            summary["missing_clips"] += 1
            continue

        clip_path = resolve_clip_path(args.clips_root, clip_file)
        if not os.path.exists(clip_path):
            summary["missing_clips"] += 1
            print(f"[WARN] clip not found: {clip_path}")
            continue

        seq_adapter = RoundClipSequenceAdapter(clip_path)
        scenes = list(seq_adapter.iter_scenes())
        if not scenes:
            continue

        dataset_risk = derive_round_risk_label_from_summary_row(
            row,
            ttc_threshold=args.ttc_threshold,
            distance_threshold=args.distance_threshold,
        )

        ttc_values = []
        violation_flags = []
        recent_decisions = []

        episode_trigger_stats = {}
        episode_trigger_count = 0

        for scene in scenes:
            result = service.step(
                scene=scene,
                driver_type=args.driver_type,
                feedback=args.feedback,
                recent_decisions=recent_decisions,
            )

            m = compute_step_metrics(scene, result.decision)

            if m.ttc_s is not None:
                ttc_values.append(m.ttc_s)
            if m.is_violation is not None:
                violation_flags.append(bool(m.is_violation))

            # 更新 recent decision history，供 persistent trigger 使用
            recent_decisions.append(result.decision)
            if args.history_window > 0:
                recent_decisions = recent_decisions[-args.history_window:]

            # trigger 统计
            for trig in result.triggers:
                t_type = _safe_trigger_type(trig)
                episode_trigger_stats[t_type] = episode_trigger_stats.get(t_type, 0) + 1
                global_trigger_stats[t_type] = global_trigger_stats.get(t_type, 0) + 1
                episode_trigger_count += 1

            frame_record = {
                "event_index": idx,
                "metadata": metadata,
                "scene": scene.__dict__,
                "profile": result.profile.__dict__,
                "decision": result.decision,
                "triggers": result.triggers,
                "guardrails": result.guardrails,
                "profile_update": result.profile_update,
                "evidence": result.evidence,
                "rules": [
                    r.__dict__ if hasattr(r, "__dict__") else str(r)
                    for r in result.rules
                ],
                "step_metrics": {
                    "ttc_s": m.ttc_s,
                    "is_violation": m.is_violation,
                },
            }
            logger.append_decision(frame_record)

            with open(frame_metrics_path, "a", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow([
                    idx,
                    metadata.get("recordingId"),
                    metadata.get("event_id"),
                    metadata.get("pair_type"),
                    scene.frame_index,
                    "" if m.ttc_s is None else round(m.ttc_s, 4),
                    "" if m.is_violation is None else int(m.is_violation),
                    scene.ego_speed_mps,
                    scene.rel_speed_mps,
                    scene.headway_m,
                    int(scene.vrus_present),
                    _safe_len(result.triggers),
                    _safe_len(result.rules),
                    _safe_len(result.evidence.get("laws", [])),
                    _safe_len(result.evidence.get("cases", [])),
                    _safe_len(result.evidence.get("scenarios", [])),
                ])

        episode_llm_violation = (sum(violation_flags) > 0) if violation_flags else False
        min_ttc_est = min(ttc_values) if ttc_values else None
        avg_ttc_est = (sum(ttc_values) / len(ttc_values)) if ttc_values else None
        violation_rate = (
            sum(1 for x in violation_flags if x) / len(violation_flags)
            if violation_flags else None
        )

        episode_summary = {
            "event_index": idx,
            "metadata": metadata,
            "dataset_risk_label": dataset_risk,
            "episode_num_frames": len(scenes),
            "episode_llm_violation": episode_llm_violation,
            "episode_violation_rate": violation_rate,
            "episode_min_ttc_estimated": min_ttc_est,
            "episode_avg_ttc_estimated": avg_ttc_est,
            "event_min_ttc_raw": metadata.get("min_ttc"),
            "event_min_distance_raw": metadata.get("min_distance"),
            "trigger_count": episode_trigger_count,
            "trigger_distribution": episode_trigger_stats,
        }
        append_jsonl(episode_summary_path, episode_summary)

        summary["total_events"] += 1
        summary["total_frames"] += len(scenes)
        summary["dataset_risk_true"] += int(dataset_risk)
        summary["episode_llm_violation_true"] += int(episode_llm_violation)
        summary["episode_agreement"] += int(episode_llm_violation == dataset_risk)

        all_y_true.append(bool(dataset_risk))
        all_y_pred.append(bool(episode_llm_violation))

        print(
            f"[{idx}] event_id={metadata.get('event_id')} "
            f"frames={len(scenes)} "
            f"dataset_risk={dataset_risk} "
            f"episode_llm_violation={episode_llm_violation} "
            f"triggers={episode_trigger_count}"
        )

    summary["episode_agreement_rate"] = (
        summary["episode_agreement"] / summary["total_events"]
        if summary["total_events"] else 0.0
    )

    cls = compute_confusion_and_scores(all_y_true, all_y_pred)
    summary.update(cls)

    summary["trigger_distribution"] = global_trigger_stats
    summary["total_triggers"] = sum(global_trigger_stats.values())
    summary["avg_triggers_per_event"] = (
        summary["total_triggers"] / summary["total_events"]
        if summary["total_events"] else 0.0
    )
    summary["avg_triggers_per_frame"] = (
        summary["total_triggers"] / summary["total_frames"]
        if summary["total_frames"] else 0.0
    )

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
        writer.writerow(["missing_clips", summary["missing_clips"]])
        writer.writerow(["tp", summary["confusion_matrix"]["tp"]])
        writer.writerow(["fp", summary["confusion_matrix"]["fp"]])
        writer.writerow(["fn", summary["confusion_matrix"]["fn"]])
        writer.writerow(["tn", summary["confusion_matrix"]["tn"]])
        writer.writerow(["precision", summary["precision"]])
        writer.writerow(["recall", summary["recall"]])
        writer.writerow(["f1", summary["f1"]])
        writer.writerow(["accuracy", summary["accuracy"]])
        writer.writerow(["total_triggers", summary["total_triggers"]])
        writer.writerow(["avg_triggers_per_event", summary["avg_triggers_per_event"]])
        writer.writerow(["avg_triggers_per_frame", summary["avg_triggers_per_frame"]])

    trigger_csv_path = os.path.join(logger.run_dir, "trigger_summary.csv")
    with open(trigger_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["trigger_type", "count"])
        for trigger_type, count in sorted(global_trigger_stats.items(), key=lambda x: x[0]):
            writer.writerow([trigger_type, count])

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