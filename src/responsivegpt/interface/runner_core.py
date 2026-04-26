import os
import json
import csv
from collections import Counter

from ..evaluation.metrics import compute_step_metrics
from ..evaluation.classification import compute_confusion_and_scores
from ..evaluation.trigger_plotter import TriggerPlotter

from .adapters.adapter_factory import build_event_adapter, build_sequence_adapter
from ..evaluation.round_labels import derive_round_risk_label_from_summary_row
from ..evaluation.ind_labels import derive_ind_risk_label


def append_jsonl(path: str, obj: dict):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")


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


def safe_dict(obj):
    if isinstance(obj, dict):
        return obj
    if hasattr(obj, "__dict__"):
        return dict(obj.__dict__)
    return {"value": str(obj)}


def safe_list_dict(items):
    return [safe_dict(x) for x in items] if items else []


def derive_dataset_risk_label(dataset: str, row: dict, args) -> bool:
    dataset = dataset.lower()

    if dataset == "highd":
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

    if dataset == "round":
        return derive_round_risk_label_from_summary_row(
            row,
            ttc_threshold=args.ttc_threshold,
            distance_threshold=args.distance_threshold,
        )

    if dataset == "ind":
        return derive_ind_risk_label(
            row,
            ttc_threshold=args.ttc_threshold,
            distance_threshold=args.distance_threshold,
            drac_threshold=args.drac_threshold,
        )

    raise ValueError("dataset must be one of: highd / round / ind")


def init_summary(args, template_profile_path: str):
    return {
        "dataset": args.dataset,
        "mode": args.mode,
        "total_events": 0,
        "total_frames": 0,
        "dataset_risk_true": 0,
        "episode_llm_violation_true": 0,
        "episode_agreement": 0,
        "missing_files": 0,
        "missing_clips": 0,
        "missing_scenes": 0,
        "empty_sequences": 0,
        "rows_seen": 0,
        "profile_name": args.profile_name,
        "template_profile_path": template_profile_path,
        "ablation": {
            "use_trigger": bool(args.use_trigger),
            "use_profile_learner": bool(args.use_profile_learner),
            "use_retriever": bool(args.use_retriever),
            "history_window": args.history_window,
        },
    }


def run_interaction_experiment(args, ctx):
    logger = ctx["logger"]
    service = ctx["service"]
    effective_driver_type = ctx["effective_driver_type"]

    event_adapter = build_event_adapter(args.dataset, args.summary_csv)

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
            "event_id",
            "eventType",
            "pair_type",
            "location_id",
            "ego_id",
            "other_id",
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

    summary = init_summary(
        args,
        template_profile_path=ctx.get("template_profile_path", ""),
    )

    all_y_true = []
    all_y_pred = []
    global_trigger_stats = Counter()

    for idx, row in enumerate(event_adapter.iter_rows()):
        summary["rows_seen"] += 1

        if args.limit > 0 and idx >= args.limit:
            break

        metadata = event_adapter.row_metadata(row)

        if args.mode == "batch":
            scenes = [event_adapter.row_to_scene(row)]
            sequence_path = None
        else:
            seq_adapter, sequence_path, missing_key = build_sequence_adapter(
                dataset=args.dataset,
                metadata=metadata,
                args=args,
            )

            if seq_adapter is None:
                summary["missing_files"] += 1
                if missing_key:
                    summary[missing_key] += 1
                if summary["missing_files"] <= 5:
                    print(f"[WARN] sequence file missing: {sequence_path}")
                continue

            scenes = list(seq_adapter.iter_scenes())

        if not scenes:
            summary["empty_sequences"] += 1
            if summary["empty_sequences"] <= 5:
                print("[WARN] empty sequence:", {
                    "event_index": idx,
                    "dataset": args.dataset,
                    "metadata": metadata,
                    "sequence_path": sequence_path,
                })
            continue

        dataset_risk = derive_dataset_risk_label(args.dataset, row, args)

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
            if args.mode == "episode" and args.history_window > 0:
                recent_decisions = recent_decisions[-args.history_window:]
            elif args.mode == "batch":
                recent_decisions = []

            trigger_dicts = safe_list_dict(getattr(result, "triggers", []))
            guardrail_dict = safe_dict(getattr(result, "guardrails", {}))
            profile_dict = safe_dict(result.profile)
            evidence_dict = getattr(result, "evidence", {}) or {}
            rules_list = safe_list_dict(getattr(result, "rules", []))

            for trig in trigger_dicts:
                t_type = safe_trigger_type(trig)
                episode_trigger_stats[t_type] += 1
                global_trigger_stats[t_type] += 1
                episode_trigger_count += 1

            frame_record = {
                "event_index": idx,
                "metadata": metadata,
                "sequence_path": sequence_path,
                "scene": scene.__dict__,
                "profile": profile_dict,
                "decision": result.decision,
                "triggers": trigger_dicts,
                "trigger_count": len(trigger_dicts),
                "guardrails": guardrail_dict,
                "profile_update": getattr(result, "profile_update", {}),
                "evidence": evidence_dict,
                "rules": rules_list,
                "step_metrics": {
                    "ttc_s": m.ttc_s,
                    "is_violation": m.is_violation,
                },
                "dataset_risk_label": dataset_risk,
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
                    metadata.get("event_id"),
                    metadata.get("eventType"),
                    metadata.get("pair_type"),
                    metadata.get("location_id"),
                    metadata.get("egoId") or metadata.get("egoTrackId") or metadata.get("ego_track_id"),
                    metadata.get("otherId") or metadata.get("otherTrackId") or metadata.get("trackId_2"),
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
            "dataset": args.dataset,
            "mode": args.mode,
            "metadata": metadata,
            "sequence_path": sequence_path,
            "dataset_risk_label": dataset_risk,
            "episode_num_frames": len(scenes),
            "episode_llm_violation": episode_llm_violation,
            "episode_violation_rate": violation_rate,
            "episode_min_ttc_estimated": min_ttc_est,
            "episode_avg_ttc_estimated": avg_ttc_est,
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
            f"{args.dataset}/{args.mode} "
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

    with open(os.path.join(logger.run_dir, "summary.csv"), "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["metric", "value"])
        for k, v in summary.items():
            if isinstance(v, (dict, list)):
                writer.writerow([k, json.dumps(v, ensure_ascii=False)])
            else:
                writer.writerow([k, v])

    trigger_csv_path = os.path.join(logger.run_dir, "trigger_summary.csv")
    with open(trigger_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["trigger_type", "count"])
        for trigger_type, count in sorted(global_trigger_stats.items(), key=lambda x: x[0]):
            writer.writerow([trigger_type, count])

    try:
        plotter = TriggerPlotter(run_dir=logger.run_dir)
        plotter.plot_all()
    except Exception as e:
        print("[WARN] Trigger plotting failed:", e)

    return summary