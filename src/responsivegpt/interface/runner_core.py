import os
import json
import csv
from collections import Counter

from dataclasses import asdict

from ..evaluation.metrics import compute_step_metrics
from ..evaluation.classification import compute_confusion_and_scores
from ..evaluation.trigger_plotter import TriggerPlotter
from .llm_call_policy import should_call_llm, fallback_decision_from_physics

from ..evaluation.safety_metrics import (
    thresholds_for_dataset,
    compute_frame_safety_metrics,
    aggregate_episode_safety_metrics,
    compute_llm_physics_alignment,
    compute_behavior_safety_metrics,
)

from .adapters.adapter_factory import build_event_adapter, build_sequence_adapter
from ..evaluation.round_labels import derive_round_risk_label_from_summary_row
from ..evaluation.ind_labels import derive_ind_risk_label

from ..application.planning_memory import PlanningMemory
from ..application.planning_service import PlanningService
from ..application.planning_formatter import (
    summarize_scene,
    summarize_safety,
    summarize_decision,
    compact_json,
)
from ..evaluation.planning_quality import compute_planning_quality


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
        "dry_run": bool(getattr(args, "dry_run", False)),
        "inspect_only": bool(getattr(args, "inspect_only", False)),
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
        
        # LLM 调用统计
        "llm_calls": 0,
        "non_llm_frames": 0,
        "llm_call_rate": 0.0,

        "planning_calls": 0,
        "planning_failures": 0,

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

    planning_interval = getattr(args, "planning_interval", 20)
    planning_risk_threshold = getattr(args, "planning_risk_threshold", 0.45)
    planning_time_horizon_s = getattr(args, "planning_time_horizon_s", 3.0)
    use_planning_thread = bool(getattr(args, "use_planning_thread", 1))

    inspect_only = bool(getattr(args, "inspect_only", False))
    dry_run = bool(getattr(args, "dry_run", False))
    event_adapter = build_event_adapter(args.dataset, args.summary_csv)
    thresholds = thresholds_for_dataset(args.dataset)
    planning_service = PlanningService(service.llm)

    frame_metrics_path = os.path.join(logger.run_dir, "frame_metrics.csv")
    episode_summary_path = os.path.join(logger.run_dir, "episode_summary.jsonl")
    profile_trace_path = os.path.join(logger.run_dir, "profile_trace.jsonl")
    trigger_trace_path = os.path.join(logger.run_dir, "trigger_trace.jsonl")
    profile_delta_path = os.path.join(logger.run_dir, "profile_delta.jsonl")
    guardrail_trace_path = os.path.join(logger.run_dir, "guardrail_trace.jsonl")
    planning_trace_path = os.path.join(logger.run_dir, "planning_trace.jsonl")

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
            "thw_s",
            "drac_mps2",
            "dcpa_m",
            "ttca_s",
            "predicted_ttc_s",
            "min_future_distance_m",
            "physical_risk_index",
            "physical_risk_level",

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

            "llm_called",
            "decision_source",
            "planning_active",
            "planning_age_frames",
            "planning_risk_level",
            "planning_strategy",
        ])

    summary = init_summary(
        args,
        template_profile_path=ctx.get("template_profile_path", ""),
    )
    summary.setdefault("planning_calls", 0)
    summary.setdefault("planning_enabled", bool(getattr(args, "use_planning_thread", 0)))

    all_y_true = []
    all_y_pred = []
    global_trigger_stats = Counter()
    global_episode_safety_records = []
    global_alignment_records = []
    global_behavior_records = []
    global_planning_records = []
    global_planning_quality_records = []

    for idx, row in enumerate(event_adapter.iter_rows()):
        if args.limit > 0 and idx >= args.limit:
            break

        summary["rows_seen"] += 1
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

            if hasattr(seq_adapter, "validate_schema"):
                try:
                    seq_adapter.validate_schema()
                except Exception as e:
                    summary["missing_files"] += 1
                    print("[SCHEMA ERROR]", e)
                    continue

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

        frame_safety_metrics_list = []
        decision_list = []
        trigger_list_by_frame = {}
        
        planning_memory = PlanningMemory()
        planning_records = []

        scene_history_for_planning = []
        safety_history_for_planning = []

        last_planning_frame_pos = None

        # ============================================================
        # inspect_only:
        # 只检查 summary → sequence path → scenes 是否能打通
        # 不计算完整 safety metrics
        # 不调用 LLM
        # ============================================================
        if inspect_only:
            append_jsonl(episode_summary_path, {
                "event_index": idx,
                "dataset": args.dataset,
                "mode": args.mode,
                "metadata": metadata,
                "sequence_path": sequence_path,
                "dataset_risk_label": dataset_risk,
                "episode_num_frames": len(scenes),
                "inspect_only": True,
                "dry_run": False,
            })

            summary["total_events"] += 1
            summary["total_frames"] += len(scenes)
            summary["dataset_risk_true"] += int(dataset_risk)
            summary["precision"] = None
            summary["recall"] = None
            summary["f1"] = None
            summary["accuracy"] = None

            print(
                f"[INSPECT] event={idx} "
                f"dataset={args.dataset} "
                f"mode={args.mode} "
                f"frames={len(scenes)} "
                f"risk={dataset_risk} "
                f"sequence_path={sequence_path}"
            )

            continue


        # ============================================================
        # dry_run:
        # 计算完整 safety metrics
        # 不调用 LLM
        # ============================================================
        if dry_run:
            frame_safety_metrics_list = []

            for scene in scenes:
                frame_safety = compute_frame_safety_metrics(scene, thresholds)
                frame_safety_metrics_list.append(frame_safety)

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

                        "" if frame_safety.ttc_s is None else round(frame_safety.ttc_s, 4),
                        "" if frame_safety.thw_s is None else round(frame_safety.thw_s, 4),
                        "" if frame_safety.drac_mps2 is None else round(frame_safety.drac_mps2, 4),
                        "" if frame_safety.dcpa_m is None else round(frame_safety.dcpa_m, 4),
                        "" if frame_safety.ttca_s is None else round(frame_safety.ttca_s, 4),
                        "" if frame_safety.predicted_ttc_s is None else round(frame_safety.predicted_ttc_s, 4),
                        "" if frame_safety.min_future_distance_m is None else round(frame_safety.min_future_distance_m, 4),
                        "" if frame_safety.physical_risk_index is None else round(frame_safety.physical_risk_index, 4),
                        frame_safety.physical_risk_level,

                        "",  # is_violation
                        scene.ego_speed_mps,
                        scene.rel_speed_mps,
                        scene.headway_m,
                        int(scene.vrus_present),
                        0,
                        0,
                        0,
                        0,
                        0,

                        0,              # llm_called
                        "dry_run",      # decision_source
                        0,              # planning_active
                        "",             # planning_age_frames
                        "",             # planning_risk_level
                        "",             # planning_strategy
                    ])

            episode_safety = aggregate_episode_safety_metrics(frame_safety_metrics_list)

            # dry-run 下也进入 global safety 汇总
            global_episode_safety_records.append(asdict(episode_safety))

            append_jsonl(episode_summary_path, {
                "event_index": idx,
                "dataset": args.dataset,
                "mode": args.mode,
                "metadata": metadata,
                "sequence_path": sequence_path,
                "dataset_risk_label": dataset_risk,
                "episode_num_frames": len(scenes),
                "inspect_only": False,
                "dry_run": True,
                "episode_safety": asdict(episode_safety),
            })

            summary["total_events"] += 1
            summary["total_frames"] += len(scenes)
            summary["dataset_risk_true"] += int(dataset_risk)

            print(
                f"[DRY-RUN] event={idx} "
                f"dataset={args.dataset} "
                f"frames={len(scenes)} "
                f"risk={dataset_risk} "
                f"min_ttc={episode_safety.min_ttc_s} "
                f"max_drac={episode_safety.max_drac_mps2} "
                f"min_dcpa={episode_safety.min_dcpa_m} "
                f"risk_exposure={episode_safety.physical_risk_exposure}"
            )

            continue

        for frame_pos, scene in enumerate(scenes):
            # ============================================================
            # 1. 先计算当前帧物理安全指标
            # ============================================================
            frame_safety = compute_frame_safety_metrics(scene, thresholds)
            frame_safety_metrics_list.append(frame_safety)

            scene_history_for_planning.append(summarize_scene(scene))
            safety_history_for_planning.append(summarize_safety(frame_safety))

            # ============================================================
            # 2. Planning Thread 调度
            # ============================================================
            planning_enabled = bool(getattr(args, "use_planning_thread", 0))
            planning_interval = int(getattr(args, "planning_interval", 20))
            planning_min_gap = int(getattr(args, "planning_min_gap", 10))
            planning_risk_threshold = float(getattr(args, "planning_risk_threshold", 0.45))
            planning_time_horizon_s = float(getattr(args, "planning_time_horizon_s", 3.0))
            planning_max_history = int(getattr(args, "planning_max_history", 12))

            current_frame_id = scene.frame_index if scene.frame_index is not None else frame_pos

            should_plan = False

            if planning_enabled and not dry_run and not inspect_only:
                if frame_pos == 0:
                    should_plan = True
                elif planning_interval > 0 and frame_pos % planning_interval == 0:
                    should_plan = True
                elif (
                    frame_safety.physical_risk_index is not None
                    and frame_safety.physical_risk_index >= planning_risk_threshold
                ):
                    should_plan = True
                elif planning_memory.is_stale(current_frame_id):
                    should_plan = True

            can_plan_by_gap = (
                last_planning_frame_pos is None
                or frame_pos - last_planning_frame_pos >= planning_min_gap
            )

            if should_plan and can_plan_by_gap:
                planning_output = planning_service.plan(
                    dataset=args.dataset,
                    driver_type=effective_driver_type,
                    feedback=args.feedback,
                    planning_interval=planning_interval,
                    current_frame=current_frame_id,
                    time_horizon_s=planning_time_horizon_s,
                    recent_scene_summaries=compact_json(
                        scene_history_for_planning,
                        max_items=planning_max_history,
                    ),
                    recent_safety_summaries=compact_json(
                        safety_history_for_planning,
                        max_items=planning_max_history,
                    ),
                    recent_decision_summaries=compact_json(
                        [summarize_decision(d) for d in decision_list],
                        max_items=planning_max_history,
                    ),
                    current_safety_snapshot=compact_json(
                        [summarize_safety(frame_safety)],
                        max_items=1,
                    ),
                )

                planning_memory.update(planning_output, current_frame_id)
                last_planning_frame_pos = frame_pos

                if planning_output.get("diagnostics", {}).get("fallback"):
                    summary["planning_failures"] = summary.get("planning_failures", 0) + 1

                planning_record = {
                    "event_index": idx,
                    "frame_pos": frame_pos,
                    "frame_index": scene.frame_index,
                    "planning": planning_output,
                }

                planning_records.append(planning_record)
                global_planning_records.append(planning_record)

                summary["planning_calls"] = summary.get("planning_calls", 0) + 1
                append_jsonl(planning_trace_path, planning_record)

            # ============================================================
            # 2.1 Planning Hint 给 Reactive Thread
            # ============================================================
            planning_hint = ""
            planning_metadata = {}

            if planning_enabled and planning_memory.last_update_frame is not None:
                planning_hint = planning_memory.to_reactive_hint(
                    current_frame=current_frame_id
                )
                planning_metadata = {
                    "planning_age_frames": current_frame_id - planning_memory.last_update_frame,
                    "last_update_frame": planning_memory.last_update_frame,
                }

            # ============================================================
            # 3. Reactive Thread / LLM 调用策略
            # ============================================================
            call_llm = should_call_llm(
                policy=getattr(args, "llm_policy", "hybrid"),
                frame_pos=frame_pos,
                frame_safety=frame_safety,
                stride=int(getattr(args, "llm_stride", 5)),
                risk_threshold=float(getattr(args, "llm_risk_threshold", 0.35)),
            )

            # dry_run / inspect_only 理论上前面已经 continue，这里再防御一次
            if dry_run or inspect_only:
                call_llm = False

            if call_llm:
                try:
                    result = service.step(
                        scene=scene,
                        driver_type=effective_driver_type,
                        feedback=args.feedback,
                        recent_decisions=recent_decisions,
                        planning_hint=planning_hint,
                        planning_metadata=planning_metadata,
                        frame_safety=frame_safety,
                    )
                    decision = result.decision
                    decision_source = "llm"
                    summary["llm_calls"] += 1

                except Exception as e:
                    print(
                        f"[WARN] service.step failed: "
                        f"event={idx}, frame={scene.frame_index}, error={e}"
                    )
                    result = None
                    decision = fallback_decision_from_physics(frame_safety)
                    decision_source = "physics_fallback_after_llm_error"
                    summary["non_llm_frames"] += 1

            else:
                result = None

                if bool(getattr(args, "reuse_last_decision", 1)) and recent_decisions:
                    decision = dict(recent_decisions[-1])
                    decision["source"] = "reused_last_decision"
                else:
                    decision = fallback_decision_from_physics(frame_safety)

                decision_source = decision.get("source", "physics_fallback")
                summary["non_llm_frames"] += 1

            # 给 decision 统一补充来源字段，后续 planning quality / CSV 会用到
            decision["_source"] = decision_source
            decision["_planning_hint_used"] = bool(planning_hint)
            decision["_planning_age_frames"] = planning_metadata.get("planning_age_frames")

            decision_list.append(decision)

            # ============================================================
            # 4. 基于 decision + scene 计算 step metrics
            # ============================================================
            m = compute_step_metrics(scene, decision, thresholds=thresholds)

            if result is not None:
                trigger_dicts = safe_list_dict(getattr(result, "triggers", []))
                guardrail_dict = safe_dict(getattr(result, "guardrails", {}))
                profile_dict = safe_dict(result.profile)
                evidence_dict = getattr(result, "evidence", {}) or {}
                rules_list = safe_list_dict(getattr(result, "rules", []))
                profile_update = getattr(result, "profile_update", {})
            else:
                trigger_dicts = []
                guardrail_dict = {}
                profile_dict = {}
                evidence_dict = {}
                rules_list = []
                profile_update = {}

            if trigger_dicts:
                trigger_list_by_frame[frame_pos] = trigger_dicts

            if m.ttc_s is not None:
                ttc_values.append(m.ttc_s)
            if m.is_violation is not None:
                violation_flags.append(bool(m.is_violation))

            recent_decisions.append(decision)
            if args.mode == "episode" and args.history_window > 0:
                recent_decisions = recent_decisions[-args.history_window:]
            elif args.mode == "batch":
                recent_decisions = []

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
                "decision": decision,
                "decision_source": decision_source,
                "planning_hint": planning_hint,
                "planning_memory": planning_memory.get() if bool(getattr(args, "use_planning_thread", 0)) else None,
                "triggers": trigger_dicts,
                "trigger_count": len(trigger_dicts),
                "guardrails": guardrail_dict,
                "profile_update": profile_update,
                "evidence": evidence_dict,
                "rules": rules_list,
                "step_metrics": {
                    "ttc_s": m.ttc_s,
                    "thw_s": m.thw_s,
                    "drac_mps2": m.drac_mps2,
                    "dcpa_m": m.dcpa_m,
                    "ttca_s": m.ttca_s,
                    "predicted_ttc_s": m.predicted_ttc_s,
                    "min_future_distance_m": m.min_future_distance_m,
                    "physical_risk_index": m.physical_risk_index,
                    "physical_risk_level": m.physical_risk_level,
                    "is_violation": m.is_violation,
                },
                "frame_safety": asdict(frame_safety),
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
                "profile_update": profile_update,
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
                    "" if m.thw_s is None else round(m.thw_s, 4),
                    "" if m.drac_mps2 is None else round(m.drac_mps2, 4),
                    "" if m.dcpa_m is None else round(m.dcpa_m, 4),
                    "" if m.ttca_s is None else round(m.ttca_s, 4),
                    "" if m.predicted_ttc_s is None else round(m.predicted_ttc_s, 4),
                    "" if m.min_future_distance_m is None else round(m.min_future_distance_m, 4),
                    "" if m.physical_risk_index is None else round(m.physical_risk_index, 4),
                    m.physical_risk_level,

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

                    int(call_llm),
                    decision_source,
                    int(planning_memory.last_update_frame is not None),
                    "" if planning_memory.last_update_frame is None else current_frame_id - planning_memory.last_update_frame,
                    planning_memory.get()
                        .get("risk_forecast", {})
                        .get("risk_level"),
                    planning_memory.get()
                        .get("recommended_strategy", {})
                        .get("strategy"),
                ])

        episode_llm_violation = (sum(violation_flags) > 0) if violation_flags else False
        episode_safety = aggregate_episode_safety_metrics(frame_safety_metrics_list)
        alignment = compute_llm_physics_alignment(
            frame_safety_metrics_list,
            decision_list,
        )
        behavior = compute_behavior_safety_metrics(
            frame_safety_metrics_list,
            decision_list,
            trigger_list_by_frame=trigger_list_by_frame,
        )
        planning_quality = compute_planning_quality(
            planning_records,
            frame_safety_metrics_list,
            decision_list,
            horizon=int(getattr(args, "planning_quality_horizon", 10)),
        )

        global_planning_quality_records.append(planning_quality)
        min_ttc_est = min(ttc_values) if ttc_values else None
        avg_ttc_est = (sum(ttc_values) / len(ttc_values)) if ttc_values else None
        violation_rate = (
            sum(1 for x in violation_flags if x) / len(violation_flags)
            if violation_flags else None
        )

        global_episode_safety_records.append(asdict(episode_safety))
        global_alignment_records.append(asdict(alignment))
        global_behavior_records.append(asdict(behavior))
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

            "episode_safety": asdict(episode_safety),
            "llm_physics_alignment": asdict(alignment),
            "behavior_safety": asdict(behavior),

            "planning_quality": planning_quality,
            "planning_call_count": len(planning_records),

            "trigger_count": episode_trigger_count,
            "trigger_distribution": dict(episode_trigger_stats)
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

    if dry_run or inspect_only:
        summary["classification_skipped"] = True
        summary["classification_skip_reason"] = (
            "dry_run/inspect_only does not call LLM, so no valid model predictions exist."
        )

        summary["episode_llm_violation_true"] = None
        summary["episode_agreement"] = None
        summary["episode_agreement_rate"] = None

        summary["confusion_matrix"] = None
        summary["precision"] = None
        summary["recall"] = None
        summary["f1"] = None
        summary["accuracy"] = None
        summary["total"] = 0

    else:
        summary["classification_skipped"] = False

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

    summary["llm_call_rate"] = (
        summary["llm_calls"] / summary["total_frames"]
        if summary["total_frames"] else 0.0
    )

    def _avg(records, key):
        vals = [r.get(key) for r in records if isinstance(r.get(key), (int, float))]
        return sum(vals) / len(vals) if vals else None

    summary["global_safety"] = {
        "avg_min_ttc_s": _avg(global_episode_safety_records, "min_ttc_s"),
        "avg_avg_ttc_s": _avg(global_episode_safety_records, "avg_ttc_s"),
        "avg_max_drac_mps2": _avg(global_episode_safety_records, "max_drac_mps2"),
        "avg_min_dcpa_m": _avg(global_episode_safety_records, "min_dcpa_m"),
        "avg_min_future_distance_m": _avg(global_episode_safety_records, "min_future_distance_m"),
        "avg_unsafe_ttc_ratio": _avg(global_episode_safety_records, "unsafe_ttc_ratio"),
        "avg_unsafe_thw_ratio": _avg(global_episode_safety_records, "unsafe_thw_ratio"),
        "avg_unsafe_drac_ratio": _avg(global_episode_safety_records, "unsafe_drac_ratio"),
        "avg_unsafe_dcpa_ratio": _avg(global_episode_safety_records, "unsafe_dcpa_ratio"),
        "avg_unsafe_future_distance_ratio": _avg(global_episode_safety_records, "unsafe_future_distance_ratio"),
        "avg_physical_risk_exposure": _avg(global_episode_safety_records, "physical_risk_exposure"),
        "avg_max_physical_risk_index": _avg(global_episode_safety_records, "max_physical_risk_index"),
    }

    summary["global_alignment"] = {
        "avg_alignment_accuracy": _avg(global_alignment_records, "alignment_accuracy"),
        "avg_overreaction_rate": _avg(global_alignment_records, "overreaction_rate"),
        "avg_underreaction_rate": _avg(global_alignment_records, "underreaction_rate"),
        "avg_mean_risk_level_error": _avg(global_alignment_records, "mean_risk_level_error"),
        "avg_llm_violation_rate": _avg(global_alignment_records, "llm_violation_rate"),
    }

    summary["global_behavior"] = {
        "avg_reaction_delay_frames": _avg(global_behavior_records, "reaction_delay_frames"),
        "avg_trigger_delay_frames": _avg(global_behavior_records, "trigger_delay_frames"),
        "avg_decision_flip_rate": _avg(global_behavior_records, "decision_flip_rate"),
        "avg_risk_level_variance": _avg(global_behavior_records, "risk_level_variance"),
    }
    summary["token_time_efficiency"] = {
        "llm_calls": summary.get("llm_calls", 0),
        "planning_calls": summary.get("planning_calls", 0),
        "non_llm_frames": summary.get("non_llm_frames", 0),
        "dry_run_frames": summary.get("dry_run_frames", 0),
        "reactive_frames": summary.get("reactive_frames", 0),
        "total_frames": summary.get("total_frames", 0),

        "reactive_llm_call_rate": (
            summary.get("llm_calls", 0) / summary["reactive_frames"]
            if summary.get("reactive_frames", 0) else 0.0
        ),
        "planning_call_rate": (
            summary.get("planning_calls", 0) / summary["reactive_frames"]
            if summary.get("reactive_frames", 0) else 0.0
        ),
        "non_llm_frame_rate": (
            summary.get("non_llm_frames", 0) / summary["reactive_frames"]
            if summary.get("reactive_frames", 0) else 0.0
        ),
    }
    planning_enabled = bool(getattr(args, "use_planning_thread", 0))
    planning_quality_available = (
        planning_enabled
        and not dry_run
        and not inspect_only
        and len(global_planning_quality_records) > 0
    )

    if dry_run:
        planning_skip_reason = "dry_run skips Planning Thread and Reactive decisions"
    elif inspect_only:
        planning_skip_reason = "inspect_only only validates data loading"
    elif not planning_enabled:
        planning_skip_reason = "planning thread disabled"
    elif not global_planning_quality_records:
        planning_skip_reason = "no planning quality records generated"
    else:
        planning_skip_reason = None

    summary["global_planning"] = {
        "planning_enabled": planning_enabled,
        "planning_quality_available": planning_quality_available,
        "planning_skip_reason": planning_skip_reason,

        "total_planning_calls": summary.get("planning_calls", 0),
        "planning_failures": summary.get("planning_failures", 0),

        "avg_planning_calls_per_event": (
            summary.get("planning_calls", 0) / summary["total_events"]
            if summary["total_events"] else 0.0
        ),

        "num_planning_quality_records": len(global_planning_quality_records),

        "avg_planning_hit_rate": (
            _avg(global_planning_quality_records, "planning_hit_rate")
            if planning_quality_available else None
        ),
        "avg_planning_miss_rate": (
            _avg(global_planning_quality_records, "planning_miss_rate")
            if planning_quality_available else None
        ),
        "avg_planning_false_alarm_rate": (
            _avg(global_planning_quality_records, "planning_false_alarm_rate")
            if planning_quality_available else None
        ),
        "avg_planning_reactive_consistency": (
            _avg(global_planning_quality_records, "planning_reactive_consistency")
            if planning_quality_available else None
        ),
    }

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