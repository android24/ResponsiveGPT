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

from ..adapters.highd_event_adapter import HighDEventAdapter
from ..adapters.highd_clip_sequence_adapter import HighDClipSequenceAdapter

from ...infrastructure.knowledge_base import KnowledgeBase
from ...infrastructure.kb_seed import default_kb_docs
from ...infrastructure.kb_json_loader import load_kb_json_dir
from ...infrastructure.hybrid_retriever import HybridRetriever
from ...evaluation.trigger_plotter import TriggerPlotter

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


def resolve_profile_template_path(profiles_dir: str, profile_name: str) -> str:
    """
    稳健解析 profile 模板路径。
    兼容文件位于：
    - src/responsivegpt/interface/
    - src/responsivegpt/interface/legacy/
    """

    current_file = Path(__file__).resolve()

    # 1. 如果 profiles_dir 本身就是绝对路径，直接使用
    p = Path(profiles_dir)
    if p.is_absolute():
        candidate = p / f"{profile_name}.json"
        if candidate.exists():
            return str(candidate)

    # 2. 从当前文件逐级向上寻找项目根目录
    # 项目根目录特征：下面存在 src/responsivegpt/data/profiles
    for parent in current_file.parents:
        candidate = parent / profiles_dir / f"{profile_name}.json"
        if candidate.exists():
            return str(candidate)

        candidate2 = parent / "src" / "responsivegpt" / "data" / "profiles" / f"{profile_name}.json"
        if candidate2.exists():
            return str(candidate2)

    raise FileNotFoundError(
        f"\n❌ Profile template not found:\n"
        f"  profile_name: {profile_name}\n"
        f"  profiles_dir: {profiles_dir}\n"
        f"  current_file: {current_file}\n\n"
        f"👉 建议检查：\n"
        f"1. profile_name 是否为 aggressive / balanced / conservative\n"
        f"2. profiles_dir 是否为 src/responsivegpt/data/profiles\n"
        f"3. 文件是否存在于 src/responsivegpt/data/profiles/{profile_name}.json\n"
    )

def resolve_highd_clip_path(clips_root: str, clip_file: str) -> str:
    clip_file = str(clip_file or "").replace("\\", "/").strip()

    if os.path.isabs(clip_file):
        return clip_file

    # 如果 clip_file 已经包含 clips_multi_fixed_window/xxx.csv
    p1 = os.path.join(clips_root, clip_file)
    if os.path.exists(p1):
        return p1

    # 如果 clips_root 已经是 clips_multi_fixed_window，而 clip_file 也带这个前缀
    parts = clip_file.split("/")
    if len(parts) > 1:
        p2 = os.path.join(clips_root, *parts[1:])
        if os.path.exists(p2):
            return p2

    # 兜底 basename
    return os.path.join(clips_root, os.path.basename(clip_file))

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

    # 关键：模板 + 运行态分离
    repo = JsonProfileRepository(
        template_path=template_profile_path,
        runtime_path=runtime_profile_path,
        auto_init=True,
    )

    trigger_manager = TriggerManager(
        ttc_threshold=3.0,
        distance_threshold=2.0,
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


def append_jsonl(path: str, obj: dict):
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")


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


def main():
    parser = argparse.ArgumentParser(description="Run highD full sequence batch with closed-loop trigger learning.")
    parser.add_argument("--csv_path", type=str, required=True, help="Path to highd_strong_interactions_summary.csv")
    parser.add_argument("--clips_root", type=str, required=True)
    parser.add_argument("--model_role", type=str, default="primary",
                    choices=["primary", "fallback", "cheap"])
    parser.add_argument("--tag", type=str, default="highd_episode")
    parser.add_argument("--driver_type", type=str, default="", help="留空则使用 profile 模板内的 driver_type")
    parser.add_argument("--feedback", type=str, default="保持效率，但避免明显危险操作")
    parser.add_argument("--limit", type=int, default=0, help="0 means all")

    # 新增：profile 体系
    parser.add_argument(
        "--profile_name",
        type=str,
        default="aggressive",
        choices=["aggressive", "balanced", "conservative"]
    )
    parser.add_argument(
        "--profiles_dir",
        type=str,
        default="src/responsivegpt/data/profiles"
    )

    args = parser.parse_args()

    env = load_env(".env")
    if not env.get("JIEKOU_API_KEY"):
        raise RuntimeError("Missing JIEKOU_API_KEY in .env")

    logger = RunLogger(runs_root="runs", tag=f"{args.tag}_{args.profile_name}")

    # profile 模板路径 + 运行态路径
    template_profile_path = resolve_profile_template_path(args.profiles_dir, args.profile_name)
    runtime_profile_path = os.path.join(logger.run_dir, "runtime_profile.json")
    initial_profile_copy_path = os.path.join(logger.run_dir, "initial_profile.json")
    final_profile_path = os.path.join(logger.run_dir, "final_profile.json")

    # 把模板拷一份到 runs 里，方便论文复现
    with open(template_profile_path, "r", encoding="utf-8") as f:
        initial_profile = json.load(f)
    with open(initial_profile_copy_path, "w", encoding="utf-8") as f:
        json.dump(initial_profile, f, ensure_ascii=False, indent=2)

    # 如果 CLI 没给 driver_type，就用模板里的
    effective_driver_type = args.driver_type or initial_profile.get("driver_type", "均衡")

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

    event_adapter = HighDEventAdapter(args.csv_path)

    logger.write_config({
        "csv_path": args.csv_path,
        "profile_name": args.profile_name,
        "profiles_dir": args.profiles_dir,
        "template_profile_path": template_profile_path,
        "runtime_profile_path": runtime_profile_path,
        "initial_profile_copy_path": initial_profile_copy_path,
        "driver_type": effective_driver_type,
        "feedback": args.feedback,
        "limit": args.limit,
        "env": {
            "JIEKOU_BASE_URL": env.get("JIEKOU_BASE_URL", "https://api.jiekou.ai/openai"),
            "JIEKOU_MODEL": env.get("JIEKOU_MODEL", "gpt-5.2"),
            "OLLAMA_BASE_URL": env.get("OLLAMA_BASE_URL", "http://localhost:11434"),
            "OLLAMA_EMBED_MODEL": env.get("OLLAMA_EMBED_MODEL", "nomic-embed-text"),
            "KB_DIR": env.get("KB_DIR", "data/kb"),
        }
    })

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
        "missing_clips": 0,
        "empty_sequences": 0,
        "profile_name": args.profile_name,
        "template_profile_path": template_profile_path,
    }

    all_y_true = []
    all_y_pred = []
    global_trigger_counter = Counter()

    for idx, row in enumerate(event_adapter.iter_rows()):
        if args.limit > 0 and idx >= args.limit:
            break

        metadata = event_adapter.row_metadata(row)
        event_scene = event_adapter.row_to_scene(row)
        dataset_risk = derive_highd_risk_label(row)

        clip_file = metadata.get("clipPath")
        if not clip_file:
            summary["missing_clips"] = summary.get("missing_clips", 0) + 1
            continue

        clip_path = resolve_highd_clip_path(args.clips_root, clip_file)
        if not os.path.exists(clip_path):
            summary["missing_clips"] = summary.get("missing_clips", 0) + 1
            print(f"[WARN] highD clip not found: {clip_path}")
            continue

        seq_adapter = HighDClipSequenceAdapter(clip_path)
        scenes = list(seq_adapter.iter_scenes())

        if not scenes:
            summary["empty_sequences"] = summary.get("empty_sequences", 0) + 1
            print(f"[WARN] highD empty clip sequence: {clip_path}")
            continue

        frame_records = []
        ttc_values = []
        violation_flags = []
        recent_decisions = []
        episode_trigger_counter = Counter()

        for scene in scenes:
            result = service.step(
                scene=scene,
                driver_type=effective_driver_type,
                feedback=args.feedback,
                recent_decisions=recent_decisions,
            )

            recent_decisions.append(result.decision)
            recent_decisions = recent_decisions[-10:]

            m = compute_step_metrics(scene, result.decision)

            if m.ttc_s is not None:
                ttc_values.append(m.ttc_s)
            if m.is_violation is not None:
                violation_flags.append(bool(m.is_violation))

            trigger_dicts = [safe_trigger_dict(t) for t in getattr(result, "triggers", [])]
            guardrail_dict = safe_guardrail_dict(getattr(result, "guardrails", {}))
            profile_dict = safe_profile_dict(result.profile)
            evidence_docs = [safe_doc_dict(d) for d in getattr(result, "rules", [])]

            for trig in trigger_dicts:
                trig_type = trig.get("trigger_type", "unknown")
                episode_trigger_counter[trig_type] += 1
                global_trigger_counter[trig_type] += 1

            frame_record = {
                "metadata": metadata,
                "scene": scene.__dict__,
                "profile": profile_dict,
                "decision": result.decision,
                "triggers": trigger_dicts,
                "trigger_count": len(trigger_dicts),
                "guardrails": guardrail_dict,
                "profile_update": getattr(result, "profile_update", {}),
                "evidence": evidence_docs,
                "step_metrics": {
                    "ttc_s": m.ttc_s,
                    "is_violation": m.is_violation,
                },
            }

            frame_records.append(frame_record)
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
                    metadata.get("egoId"),
                    metadata.get("otherId"),
                    metadata.get("eventType"),
                    scene.frame_index,
                    "" if m.ttc_s is None else round(m.ttc_s, 4),
                    "" if m.is_violation is None else int(m.is_violation),
                    len(trigger_dicts),
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
            f"[{idx}] profile={args.profile_name} "
            f"eventType={metadata.get('eventType')} "
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

    summary["trigger_stats"] = dict(global_trigger_counter)
    summary["total_trigger_activations"] = sum(global_trigger_counter.values())

    # 保存最终学习后的 runtime profile
    if os.path.exists(runtime_profile_path):
        with open(runtime_profile_path, "r", encoding="utf-8") as f:
            final_profile = json.load(f)
        with open(final_profile_path, "w", encoding="utf-8") as f:
            json.dump(final_profile, f, ensure_ascii=False, indent=2)

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