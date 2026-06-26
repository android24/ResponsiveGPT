import os
import json
import csv
import argparse
from collections import Counter
from pathlib import Path

from ...application.service import ResponsiveGPTService
from ...application.trigger_manager import TriggerManager
from ...application.layered_profile_learner import LayeredProfileLearner
from ...application.trigger_state import TriggerStateStore

from ...infrastructure.embed_ollama import OllamaEmbedder
from ...infrastructure.llm_jiekou import JiekouChatModel
from ...infrastructure.profile_repo import JsonProfileRepository

from ...evaluation.run_logger import RunLogger
from ...evaluation.metrics import compute_step_metrics
from ...evaluation.classification import compute_confusion_and_scores
from ...evaluation.ind_labels import derive_ind_risk_label
from ...evaluation.trigger_plotter import TriggerPlotter

from ..adapters.ind_event_adapter import InDEventAdapter
from ..adapters.ind_scene_sequence_adapter import InDSceneSequenceAdapter

from ...infrastructure.knowledge_base import KnowledgeBase
from ...infrastructure.kb_seed import default_kb_docs
from ...infrastructure.kb_json_loader import load_kb_json_dir
from ...infrastructure.hybrid_retriever import HybridRetriever


# ==================================================
# utils
# ==================================================

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


def append_jsonl(path: str, obj: dict):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def resolve_scene_path(scenes_root: str, scene_file: str) -> str:
    """
    summary 里的 scene_file 类似:
    output_ind_risk_v4/scenes/29_500_506_f26069_26196.csv

    这里统一映射为 scenes_root/<basename(scene_file)>
    """
    scene_file = str(scene_file or "").replace("\\", "/").strip()
    return os.path.join(scenes_root, os.path.basename(scene_file))


def resolve_profile_template_path(profiles_dir: str, profile_name: str) -> str:
    """
    与 highD 风格保持一致：从项目根目录解析 profile 模板
    """
    current_file = Path(__file__).resolve()
    project_root = current_file.parents[3]
    candidate = project_root / profiles_dir / f"{profile_name}.json"

    if not candidate.exists():
        raise FileNotFoundError(
            f"\n❌ Profile template not found:\n"
            f"  Tried: {candidate}\n\n"
            f"👉 当前建议检查：\n"
            f"1. profiles_dir 是否为: src/responsivegpt/data/profiles\n"
            f"2. profile_name 是否为: aggressive / balanced / conservative\n"
        )

    return str(candidate)


def safe_len(x) -> int:
    if x is None:
        return 0
    try:
        return len(x)
    except Exception:
        return 0


def safe_trigger_type(trigger_item) -> str:
    if isinstance(trigger_item, dict):
        return str(trigger_item.get("trigger_type", "unknown"))
    if hasattr(trigger_item, "trigger_type"):
        return str(trigger_item.trigger_type)
    return "unknown"


def safe_profile_dict(profile):
    if isinstance(profile, dict):
        return profile
    if hasattr(profile, "__dict__"):
        return dict(profile.__dict__)
    return {"value": str(profile)}


def safe_guardrail_dict(guardrails):
    if isinstance(guardrails, dict):
        return guardrails
    if hasattr(guardrails, "__dict__"):
        return dict(guardrails.__dict__)
    return {"value": str(guardrails)}


def safe_trigger_dict(trigger):
    if isinstance(trigger, dict):
        return trigger
    if hasattr(trigger, "__dict__"):
        return dict(trigger.__dict__)
    return {"value": str(trigger)}


def safe_doc_dict(doc):
    if isinstance(doc, dict):
        return doc
    if hasattr(doc, "__dict__"):
        return dict(doc.__dict__)
    return {"value": str(doc)}


# ==================================================
# build service
# ==================================================
def build_service(env: dict, template_profile_path: str, runtime_profile_path: str,
                  primary_model: str, fallback_model: str | None) -> ResponsiveGPTService:
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
        primary_model=primary_model,
        fallback_model=fallback_model,
        max_completion_tokens=int(env.get("LLM_MAX_COMPLETION_TOKENS", "2048")),
        timeout_s=float(env.get("LLM_TIMEOUT_S", "120")),
        max_retries=int(env.get("LLM_MAX_RETRIES", "1")),
    )

    repo = JsonProfileRepository(
        template_path=template_profile_path,
        runtime_path=runtime_profile_path,
        auto_init=True,
    )

    trigger_manager = TriggerManager(
        ttc_threshold=3.0,
        distance_threshold=2.5,
        persistent_risk_ratio_threshold=0.4,
        persistent_window=5,
    )
    profile_learner = LayeredProfileLearner(lr=0.2)
    trigger_state_store = TriggerStateStore()

    return ResponsiveGPTService(
        retriever=retriever,
        chat_model=llm,
        profile_repo=repo,
        trigger_manager=trigger_manager,
        profile_learner=profile_learner,
        trigger_state_store=trigger_state_store,
    )


# ==================================================
# main
# ==================================================
def main():
    parser = argparse.ArgumentParser(description="Run inD sequence batch with closed-loop trigger learning.")

    parser.add_argument("--summary_csv", type=str, required=True)
    parser.add_argument("--scenes_root", type=str, required=True)
    parser.add_argument("--model_role", type=str, default="primary",
                    choices=["primary", "fallback", "cheap"])

    parser.add_argument("--tag", type=str, default="ind_episode")
    parser.add_argument("--driver_type", type=str, default="", help="留空则使用 profile 模板中的 driver_type")
    parser.add_argument("--feedback", type=str, default="优先安全，避免在交叉口与其他交通参与者发生冲突")

    parser.add_argument(
        "--profile_name",
        type=str,
        default="conservative",
        choices=["aggressive", "balanced", "conservative"]
    )
    parser.add_argument(
        "--profiles_dir",
        type=str,
        default="src/responsivegpt/data/profiles"
    )

    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--ttc_threshold", type=float, default=3.0)
    parser.add_argument("--distance_threshold", type=float, default=2.5)
    parser.add_argument("--drac_threshold", type=float, default=8.0)
    parser.add_argument("--history_window", type=int, default=10, help="recent_decisions 长度")

    args = parser.parse_args()

    env = load_env(".env")
    if not env.get("JIEKOU_API_KEY"):
        raise RuntimeError("Missing JIEKOU_API_KEY in .env")

    logger = RunLogger(runs_root="runs", tag=f"{args.tag}_{args.profile_name}")

    # --------------------------------------------------
    # profile 初始化：模板 / 运行态 / 初始快照
    # --------------------------------------------------
    template_profile_path = resolve_profile_template_path(args.profiles_dir, args.profile_name)
    runtime_profile_path = os.path.join(logger.run_dir, "runtime_profile.json")
    initial_profile_copy_path = os.path.join(logger.run_dir, "initial_profile.json")
    final_profile_path = os.path.join(logger.run_dir, "final_profile.json")

    with open(template_profile_path, "r", encoding="utf-8") as f:
        init_profile = json.load(f)

    with open(initial_profile_copy_path, "w", encoding="utf-8") as f:
        json.dump(init_profile, f, ensure_ascii=False, indent=2)

    effective_driver_type = args.driver_type or init_profile.get("driver_type", "均衡")

    selected_model = env.get("PRIMARY_MODEL", "gpt-5.2")
    selected_fallback = env.get("FALLBACK_MODEL", "gpt-4.1")

    if args.model_role == "fallback":
        selected_model = env.get("FALLBACK_MODEL", "gpt-4.1")
        selected_fallback = None
    elif args.model_role == "cheap":
        selected_model = env.get("CHEAP_MODEL", "gpt-4o-mini")
        selected_fallback = None
    service = build_service(
        env=env,
        template_profile_path=template_profile_path,
        runtime_profile_path=runtime_profile_path,
        primary_model=selected_model,
        fallback_model=selected_fallback,
    )

    event_adapter = InDEventAdapter(args.summary_csv)

    logger.write_config({
        "summary_csv": args.summary_csv,
        "scenes_root": args.scenes_root,
        "profile_name": args.profile_name,
        "profiles_dir": args.profiles_dir,
        "template_profile_path": template_profile_path,
        "runtime_profile_path": runtime_profile_path,
        "initial_profile_copy_path": initial_profile_copy_path,
        "driver_type": effective_driver_type,
        "feedback": args.feedback,
        "limit": args.limit,
        "ttc_threshold": args.ttc_threshold,
        "distance_threshold": args.distance_threshold,
        "drac_threshold": args.drac_threshold,
        "history_window": args.history_window,
        "env": {
            "JIEKOU_BASE_URL": env.get("JIEKOU_BASE_URL", "https://api.jiekou.ai/openai"),
            "JIEKOU_MODEL": env.get("JIEKOU_MODEL", "gpt-5.2"),
            "OLLAMA_BASE_URL": env.get("OLLAMA_BASE_URL", "http://localhost:11434"),
            "OLLAMA_EMBED_MODEL": env.get("OLLAMA_EMBED_MODEL", "nomic-embed-text"),
            "KB_DIR": env.get("KB_DIR", "data/kb"),
        }
    })

    # --------------------------------------------------
    # 输出路径
    # --------------------------------------------------
    frame_metrics_path = os.path.join(logger.run_dir, "frame_metrics.csv")
    episode_summary_path = os.path.join(logger.run_dir, "episode_summary.jsonl")
    profile_trace_path = os.path.join(logger.run_dir, "profile_trace.jsonl")
    trigger_trace_path = os.path.join(logger.run_dir, "trigger_trace.jsonl")
    profile_delta_path = os.path.join(logger.run_dir, "profile_delta.jsonl")
    guardrail_trace_path = os.path.join(logger.run_dir, "guardrail_trace.jsonl")

    with open(frame_metrics_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "event_index",
            "recordingId",
            "location_id",
            "ego_track_id",
            "class_1",
            "class_2",
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
        "missing_scenes": 0,
        "profile_name": args.profile_name,
        "template_profile_path": template_profile_path,
    }

    all_y_true = []
    all_y_pred = []
    global_trigger_stats = Counter()

    # ==================================================
    # 主循环
    # ==================================================
    for idx, row in enumerate(event_adapter.iter_rows()):
        if args.limit > 0 and idx >= args.limit:
            break

        metadata = event_adapter.row_metadata(row)
        scene_file = metadata.get("scene_file")

        if not scene_file:
            summary["missing_scenes"] += 1
            continue

        scene_path = resolve_scene_path(args.scenes_root, scene_file)
        if not os.path.exists(scene_path):
            summary["missing_scenes"] += 1
            print(f"[WARN] scene not found: {scene_path}")
            continue

        seq_adapter = InDSceneSequenceAdapter(scene_path)
        scenes = list(seq_adapter.iter_scenes())
        if not scenes:
            continue

        dataset_risk = derive_ind_risk_label(
            row,
            ttc_threshold=args.ttc_threshold,
            distance_threshold=args.distance_threshold,
            drac_threshold=args.drac_threshold,
        )

        ttc_values = []
        violation_flags = []
        recent_decisions = []

        episode_trigger_stats = Counter()
        episode_trigger_count = 0

        for scene in scenes:
            result = service.step(
                scene=scene,
                driver_type=effective_driver_type,
                feedback=args.feedback,
                recent_decisions=recent_decisions,
            )

            m = compute_step_metrics(scene, result.decision)

            if m.ttc_s is not None:
                ttc_values.append(m.ttc_s)
            if m.is_violation is not None:
                violation_flags.append(bool(m.is_violation))

            recent_decisions.append(result.decision)
            if args.history_window > 0:
                recent_decisions = recent_decisions[-args.history_window:]

            trigger_dicts = [safe_trigger_dict(t) for t in getattr(result, "triggers", [])]
            guardrail_dict = safe_guardrail_dict(getattr(result, "guardrails", {}))
            profile_dict = safe_profile_dict(result.profile)
            evidence_dict = getattr(result, "evidence", {})
            rules_list = [safe_doc_dict(r) for r in getattr(result, "rules", [])]

            for trig in trigger_dicts:
                t_type = safe_trigger_type(trig)
                episode_trigger_stats[t_type] += 1
                global_trigger_stats[t_type] += 1
                episode_trigger_count += 1

            frame_record = {
                "event_index": idx,
                "metadata": metadata,
                "scene": scene.__dict__,
                "profile": profile_dict,
                "decision": result.decision,
                "triggers": trigger_dicts,
                "guardrails": guardrail_dict,
                "profile_update": getattr(result, "profile_update", {}),
                "evidence": evidence_dict,
                "rules": rules_list,
                "step_metrics": {
                    "ttc_s": m.ttc_s,
                    "is_violation": m.is_violation,
                },
            }
            logger.append_decision(frame_record)

            append_jsonl(profile_trace_path, {
                "event_index": idx,
                "frame_index": scene.frame_index,
                "profile": profile_dict,
            })

            append_jsonl(profile_delta_path, {
                "event_index": idx,
                "frame_index": scene.frame_index,
                "profile_update": getattr(result, "profile_update", {}),
            })

            append_jsonl(guardrail_trace_path, {
                "event_index": idx,
                "frame_index": scene.frame_index,
                "guardrails": guardrail_dict,
            })

            for trig in trigger_dicts:
                append_jsonl(trigger_trace_path, {
                    "event_index": idx,
                    "frame_index": scene.frame_index,
                    "trigger": trig,
                })

            with open(frame_metrics_path, "a", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow([
                    idx,
                    metadata.get("recordingId"),
                    metadata.get("location_id"),
                    metadata.get("ego_track_id"),
                    metadata.get("class_1"),
                    metadata.get("class_2"),
                    scene.frame_index,
                    "" if m.ttc_s is None else round(m.ttc_s, 4),
                    "" if m.is_violation is None else int(m.is_violation),
                    scene.ego_speed_mps,
                    scene.rel_speed_mps,
                    scene.headway_m,
                    int(scene.vrus_present),
                    safe_len(trigger_dicts),
                    safe_len(rules_list),
                    safe_len(evidence_dict.get("laws", [])),
                    safe_len(evidence_dict.get("cases", [])),
                    safe_len(evidence_dict.get("scenarios", [])),
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
            "event_min_distance_raw": metadata.get("min_center_distance"),
            "event_max_drac_raw": metadata.get("max_drac"),
            "trigger_count": episode_trigger_count,
            "trigger_distribution": dict(episode_trigger_stats),
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
            f"[{idx}] profile={args.profile_name} "
            f"recordingId={metadata.get('recordingId')} "
            f"ego={metadata.get('ego_track_id')} "
            f"frames={len(scenes)} "
            f"dataset_risk={dataset_risk} "
            f"episode_llm_violation={episode_llm_violation} "
            f"triggers={episode_trigger_count}"
        )

    # ==================================================
    # 汇总
    # ==================================================
    summary["episode_agreement_rate"] = (
        summary["episode_agreement"] / summary["total_events"]
        if summary["total_events"] else 0.0
    )

    cls = compute_confusion_and_scores(all_y_true, all_y_pred)
    summary.update(cls)

    summary["trigger_distribution"] = dict(global_trigger_stats)
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
        writer.writerow(["profile_name", args.profile_name])
        writer.writerow(["total_events", summary["total_events"]])
        writer.writerow(["total_frames", summary["total_frames"]])
        writer.writerow(["dataset_risk_true", summary["dataset_risk_true"]])
        writer.writerow(["episode_llm_violation_true", summary["episode_llm_violation_true"]])
        writer.writerow(["episode_agreement", summary["episode_agreement"]])
        writer.writerow(["episode_agreement_rate", summary["episode_agreement_rate"]])
        writer.writerow(["missing_scenes", summary["missing_scenes"]])
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

    if os.path.exists(runtime_profile_path):
        with open(runtime_profile_path, "r", encoding="utf-8") as f:
            final_profile = json.load(f)
        with open(final_profile_path, "w", encoding="utf-8") as f:
            json.dump(final_profile, f, ensure_ascii=False, indent=2)

    print("\nRun saved to:", logger.run_dir)
    print(json.dumps(summary, ensure_ascii=False, indent=2))

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