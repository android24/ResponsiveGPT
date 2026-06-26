import os
import json
import csv
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

from ...infrastructure.knowledge_base import KnowledgeBase
from ...infrastructure.kb_seed import default_kb_docs
from ...infrastructure.kb_json_loader import load_kb_json_dir
from ...infrastructure.hybrid_retriever import HybridRetriever


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
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")


from pathlib import Path
import os


def resolve_profile_template_path(profiles_dir: str, profile_name: str) -> str:
    """
    兼容 legacy 目录移动后的 profile 路径解析。
    支持：
    1. 绝对路径
    2. 相对项目根目录路径：src/responsivegpt/data/profiles
    3. 相对当前工作目录路径
    """

    filename = f"{profile_name}.json"

    # 1. profiles_dir 是绝对路径
    p = Path(profiles_dir)
    if p.is_absolute():
        candidate = p / filename
        if candidate.exists():
            return str(candidate)

    # 2. 相对当前工作目录
    candidate = Path.cwd() / profiles_dir / filename
    if candidate.exists():
        return str(candidate)

    # 3. 从当前文件向上寻找项目根目录：包含 pyproject.toml 或 src/responsivegpt
    current = Path(__file__).resolve()
    for parent in current.parents:
        candidate = parent / profiles_dir / filename
        if candidate.exists():
            return str(candidate)

        candidate = parent / "src" / "responsivegpt" / "data" / "profiles" / filename
        if candidate.exists():
            return str(candidate)

    raise FileNotFoundError(
        f"Profile template not found for profile_name={profile_name}\n"
        f"profiles_dir={profiles_dir}\n"
        f"cwd={Path.cwd()}\n"
        f"checked legacy-compatible paths."
    )

def select_models(env: dict, model_role: str) -> tuple[str, str | None]:
    primary_model = env.get("PRIMARY_MODEL", "gpt-5.2")
    fallback_model = env.get("FALLBACK_MODEL", "gpt-4.1")
    cheap_model = env.get("CHEAP_MODEL", "gpt-4o-mini")

    if model_role == "primary":
        return primary_model, fallback_model
    if model_role == "fallback":
        return fallback_model, None
    if model_role == "cheap":
        return cheap_model, None

    raise ValueError(f"Unknown model_role: {model_role}")


def build_service(
    env: dict,
    template_profile_path: str,
    runtime_profile_path: str,
    selected_model: str,
    selected_fallback: str | None,
) -> ResponsiveGPTService:
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

    # 这里假设你当前 llm_jiekou.py 已升级到支持 primary_model / fallback_model
    # 如果还没有，我后面给你兼容写法。
    llm = JiekouChatModel(
        api_key=env.get("JIEKOU_API_KEY", ""),
        base_url=env.get("JIEKOU_BASE_URL", "https://api.jiekou.ai/openai"),
        primary_model=selected_model,
        fallback_model=selected_fallback,
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


def run_batch(
    *,
    csv_path: str,
    model_role: str,
    tag: str,
    profile_name: str,
    profiles_dir: str,
    driver_type: str,
    feedback: str,
    limit: int,
    adapter,
    risk_label_fn,
    dataset_name: str,
    extra_config: dict | None = None,
):
    env = load_env(".env")
    if not env.get("JIEKOU_API_KEY"):
        raise RuntimeError("Missing JIEKOU_API_KEY in .env")

    selected_model, selected_fallback = select_models(env, model_role)

    logger = RunLogger(runs_root="runs", tag=f"{tag}_{profile_name}")

    template_profile_path = resolve_profile_template_path(profiles_dir, profile_name)
    initial_profile_copy_path = os.path.join(logger.run_dir, "initial_profile.json")

    with open(template_profile_path, "r", encoding="utf-8") as f:
        initial_profile = json.load(f)
    with open(initial_profile_copy_path, "w", encoding="utf-8") as f:
        json.dump(initial_profile, f, ensure_ascii=False, indent=2)

    effective_driver_type = driver_type or initial_profile.get("driver_type", "均衡")

    cfg = {
        "dataset_name": dataset_name,
        "csv_path": csv_path,
        "profile_name": profile_name,
        "profiles_dir": profiles_dir,
        "template_profile_path": template_profile_path,
        "initial_profile_copy_path": initial_profile_copy_path,
        "driver_type": effective_driver_type,
        "feedback": feedback,
        "limit": limit,
        "model_role": model_role,
        "selected_model": selected_model,
        "selected_fallback": selected_fallback,
        "env": {
            "JIEKOU_BASE_URL": env.get("JIEKOU_BASE_URL", "https://api.jiekou.ai/openai"),
            "OLLAMA_BASE_URL": env.get("OLLAMA_BASE_URL", "http://localhost:11434"),
            "OLLAMA_EMBED_MODEL": env.get("OLLAMA_EMBED_MODEL", "nomic-embed-text"),
            "KB_DIR": env.get("KB_DIR", "data/kb"),
        }
    }
    if extra_config:
        cfg.update(extra_config)
    logger.write_config(cfg)

    metrics_csv_path = os.path.join(logger.run_dir, "event_metrics.csv")
    summary_jsonl_path = os.path.join(logger.run_dir, "event_summary.jsonl")
    profile_trace_path = os.path.join(logger.run_dir, "profile_trace.jsonl")
    trigger_trace_path = os.path.join(logger.run_dir, "trigger_trace.jsonl")
    profile_delta_path = os.path.join(logger.run_dir, "profile_delta.jsonl")
    guardrail_trace_path = os.path.join(logger.run_dir, "guardrail_trace.jsonl")

    with open(metrics_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "event_index",
            "dataset_name",
            "recordingId",
            "event_type",
            "dataset_risk_label",
            "llm_violation",
            "ttc_s_estimated",
            "ego_speed_mps",
            "rel_speed_mps",
            "headway_m",
            "num_triggers",
            "num_rules",
            "num_law_evidence",
            "num_case_evidence",
            "num_scenario_evidence",
        ])

    summary = {
        "dataset_name": dataset_name,
        "total_events": 0,
        "dataset_risk_true": 0,
        "llm_violation_true": 0,
        "agreement": 0,
        "profile_name": profile_name,
        "template_profile_path": template_profile_path,
    }

    all_y_true = []
    all_y_pred = []
    global_trigger_count = 0
    global_trigger_distribution = {}

    for idx, row in enumerate(adapter.iter_rows()):
        if limit > 0 and idx >= limit:
            break

        metadata = adapter.row_metadata(row)
        scene = adapter.row_to_scene(row)

        # 关键修正：
        # 每个事件都重新创建 runtime_profile + service，确保 batch 真正 stateless
        runtime_profile_path = os.path.join(logger.run_dir, f"runtime_profile_event_{idx:06d}.json")
        final_profile_path = os.path.join(logger.run_dir, f"final_profile_event_{idx:06d}.json")

        service = build_service(
            env=env,
            template_profile_path=template_profile_path,
            runtime_profile_path=runtime_profile_path,
            selected_model=selected_model,
            selected_fallback=selected_fallback,
        )

        result = service.step(
            scene=scene,
            driver_type=effective_driver_type,
            feedback=feedback,
            recent_decisions=[],
        )

        m = compute_step_metrics(scene, result.decision)
        dataset_risk = bool(risk_label_fn(row))
        llm_risk = bool(result.decision.get("is_potential_violation", False))

        trigger_dicts = [safe_trigger_dict(t) for t in getattr(result, "triggers", [])]
        guardrail_dict = safe_guardrail_dict(getattr(result, "guardrails", {}))
        profile_dict = safe_profile_dict(result.profile)
        evidence_dict = getattr(result, "evidence", {})
        rules_list = [safe_doc_dict(r) for r in getattr(result, "rules", [])]

        for trig in trigger_dicts:
            trig_type = trig.get("trigger_type", "unknown")
            global_trigger_distribution[trig_type] = global_trigger_distribution.get(trig_type, 0) + 1
            global_trigger_count += 1

        summary["total_events"] += 1
        summary["dataset_risk_true"] += int(dataset_risk)
        summary["llm_violation_true"] += int(llm_risk)
        summary["agreement"] += int(dataset_risk == llm_risk)

        all_y_true.append(dataset_risk)
        all_y_pred.append(llm_risk)

        record = {
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
            "metrics": {
                "ttc_s": m.ttc_s,
                "is_violation": m.is_violation,
            },
            "dataset_risk_label": dataset_risk,
        }

        logger.append_decision(record)
        append_jsonl(summary_jsonl_path, record)
        append_jsonl(profile_trace_path, {
            "event_index": idx,
            "profile": profile_dict,
        })
        append_jsonl(profile_delta_path, {
            "event_index": idx,
            "profile_update": getattr(result, "profile_update", {}),
        })
        append_jsonl(guardrail_trace_path, {
            "event_index": idx,
            "guardrails": guardrail_dict,
        })
        for trig in trigger_dicts:
            append_jsonl(trigger_trace_path, {
                "event_index": idx,
                "trigger": trig,
            })

        with open(metrics_csv_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                idx,
                dataset_name,
                metadata.get("recordingId"),
                scene.event_type,
                int(dataset_risk),
                int(llm_risk),
                "" if m.ttc_s is None else round(m.ttc_s, 4),
                scene.ego_speed_mps,
                scene.rel_speed_mps,
                scene.headway_m,
                len(trigger_dicts),
                len(rules_list),
                len(evidence_dict.get("laws", [])) if isinstance(evidence_dict, dict) else 0,
                len(evidence_dict.get("cases", [])) if isinstance(evidence_dict, dict) else 0,
                len(evidence_dict.get("scenarios", [])) if isinstance(evidence_dict, dict) else 0,
            ])

        if os.path.exists(runtime_profile_path):
            with open(runtime_profile_path, "r", encoding="utf-8") as f:
                final_profile = json.load(f)
            with open(final_profile_path, "w", encoding="utf-8") as f:
                json.dump(final_profile, f, ensure_ascii=False, indent=2)

        print(
            f"[{idx}] dataset={dataset_name} "
            f"profile={profile_name} "
            f"dataset_risk={dataset_risk} "
            f"llm_violation={llm_risk} "
            f"triggers={len(trigger_dicts)}"
        )

    summary["agreement_rate"] = (
        summary["agreement"] / summary["total_events"]
        if summary["total_events"] else 0.0
    )

    cls = compute_confusion_and_scores(all_y_true, all_y_pred)
    summary.update(cls)
    summary["trigger_distribution"] = global_trigger_distribution
    summary["total_triggers"] = global_trigger_count

    with open(os.path.join(logger.run_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    summary_csv_path = os.path.join(logger.run_dir, "summary.csv")
    with open(summary_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["metric", "value"])
        writer.writerow(["dataset_name", dataset_name])
        writer.writerow(["profile_name", profile_name])
        writer.writerow(["model_role", model_role])
        writer.writerow(["selected_model", selected_model])
        writer.writerow(["selected_fallback", selected_fallback])
        writer.writerow(["total_events", summary["total_events"]])
        writer.writerow(["dataset_risk_true", summary["dataset_risk_true"]])
        writer.writerow(["llm_violation_true", summary["llm_violation_true"]])
        writer.writerow(["agreement", summary["agreement"]])
        writer.writerow(["agreement_rate", summary["agreement_rate"]])
        writer.writerow(["tp", summary["confusion_matrix"]["tp"]])
        writer.writerow(["fp", summary["confusion_matrix"]["fp"]])
        writer.writerow(["fn", summary["confusion_matrix"]["fn"]])
        writer.writerow(["tn", summary["confusion_matrix"]["tn"]])
        writer.writerow(["precision", summary["precision"]])
        writer.writerow(["recall", summary["recall"]])
        writer.writerow(["f1", summary["f1"]])
        writer.writerow(["accuracy", summary["accuracy"]])
        writer.writerow(["total_triggers", summary["total_triggers"]])

        for trig_type, count in summary["trigger_distribution"].items():
            writer.writerow([f"trigger_{trig_type}", count])

    print("\nRun saved to:", logger.run_dir)
    print(json.dumps(summary, ensure_ascii=False, indent=2))