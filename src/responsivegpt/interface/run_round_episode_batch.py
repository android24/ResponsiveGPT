import os
import json
import csv
import argparse

from ..application.service import ResponsiveGPTService
from ..infrastructure.embed_ollama import OllamaEmbedder
from ..infrastructure.vectorstore import SimpleVectorStore
from ..infrastructure.llm_jiekou import JiekouChatModel
from ..infrastructure.profile_repo import JsonProfileRepository
from ..infrastructure.rules import roundd_rules

from ..evaluation.run_logger import RunLogger
from ..evaluation.metrics import compute_step_metrics
from ..evaluation.classification import compute_confusion_and_scores
from ..evaluation.round_labels import derive_round_risk_label_from_summary_row

from .adapters.roundd_event_adapter import RoundEventAdapter
from .adapters.roundd_clip_sequence_adapter import RoundClipSequenceAdapter


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
    vs = SimpleVectorStore(embedder=embedder, docs=roundd_rules())
    vs.build()

    llm = JiekouChatModel(
        api_key=env.get("JIEKOU_API_KEY", ""),
        base_url=env.get("JIEKOU_BASE_URL", "https://api.jiekou.ai/openai"),
        model=env.get("JIEKOU_MODEL", "gpt-5.2"),
    )

    repo = JsonProfileRepository(env.get("PROFILE_PATH", "driver_profile.json"))
    return ResponsiveGPTService(vectorstore=vs, chat_model=llm, profile_repo=repo)


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

        for scene in scenes:
            result = service.step(
                scene=scene,
                driver_type=args.driver_type,
                feedback=args.feedback,
            )

            m = compute_step_metrics(scene, result.decision)

            if m.ttc_s is not None:
                ttc_values.append(m.ttc_s)
            if m.is_violation is not None:
                violation_flags.append(bool(m.is_violation))

            frame_record = {
                "event_index": idx,
                "metadata": metadata,
                "scene": scene.__dict__,
                "profile": result.profile.__dict__,
                "decision": result.decision,
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
                ])

        episode_llm_violation = (sum(violation_flags) > 0) if violation_flags else False
        min_ttc_est = min(ttc_values) if ttc_values else None
        avg_ttc_est = (sum(ttc_values) / len(ttc_values)) if ttc_values else None
        violation_rate = (sum(1 for x in violation_flags if x) / len(violation_flags)) if violation_flags else None

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
            f"episode_llm_violation={episode_llm_violation}"
        )

    summary["episode_agreement_rate"] = (
        summary["episode_agreement"] / summary["total_events"]
        if summary["total_events"] else 0.0
    )

    cls = compute_confusion_and_scores(all_y_true, all_y_pred)
    summary.update(cls)

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

    print("\nRun saved to:", logger.run_dir)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()