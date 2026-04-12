import os
import json
import argparse
from collections import defaultdict


def load_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_jsonl(path: str):
    rows = []
    if not os.path.exists(path):
        return rows
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def group_by_event(rows):
    grouped = defaultdict(list)
    for r in rows:
        grouped[r.get("event_index", -1)].append(r)
    return grouped


def safe_get(d, *keys, default=None):
    cur = d
    for k in keys:
        if not isinstance(cur, dict):
            return default
        cur = cur.get(k, default)
    return cur


def summarize_decisions(decision_rows):
    """
    从单个 episode 的 frame-level decisions 提取摘要
    """
    if not decision_rows:
        return {
            "num_frames": 0,
            "num_violation_frames": 0,
            "violation_rate": 0.0,
            "risk_level_counts": {},
            "trigger_count": 0,
            "first_warning": "",
            "last_recommended_action": "",
        }

    risk_counter = defaultdict(int)
    violation_frames = 0
    trigger_count = 0
    first_warning = ""
    last_action = ""

    for row in decision_rows:
        decision = row.get("decision", {})
        risk_level = str(decision.get("risk_level", "unknown"))
        risk_counter[risk_level] += 1

        if bool(decision.get("is_potential_violation", False)):
            violation_frames += 1

        triggers = row.get("triggers", [])
        if isinstance(triggers, list):
            trigger_count += len(triggers)

        warning = str(decision.get("warning", "") or "").strip()
        if warning and not first_warning:
            first_warning = warning

        action = str(decision.get("recommended_action", "") or "").strip()
        if action:
            last_action = action

    return {
        "num_frames": len(decision_rows),
        "num_violation_frames": violation_frames,
        "violation_rate": violation_frames / len(decision_rows) if decision_rows else 0.0,
        "risk_level_counts": dict(risk_counter),
        "trigger_count": trigger_count,
        "first_warning": first_warning,
        "last_recommended_action": last_action,
    }


def summarize_profile_trace(profile_rows):
    if not profile_rows:
        return {}

    first_profile = profile_rows[0].get("profile", {})
    last_profile = profile_rows[-1].get("profile", {})

    def getv(p, k):
        if isinstance(p, dict):
            return p.get(k, None)
        return None

    return {
        "driver_type": getv(last_profile, "driver_type"),
        "risk_sensitivity_start": getv(first_profile, "risk_sensitivity"),
        "risk_sensitivity_end": getv(last_profile, "risk_sensitivity"),
        "safety_weight_start": getv(first_profile, "safety_weight"),
        "safety_weight_end": getv(last_profile, "safety_weight"),
        "efficiency_weight_start": getv(first_profile, "efficiency_weight"),
        "efficiency_weight_end": getv(last_profile, "efficiency_weight"),
    }


def summarize_trigger_trace(trigger_rows):
    counter = defaultdict(int)
    for row in trigger_rows:
        trig = row.get("trigger", {})
        if isinstance(trig, dict):
            counter[str(trig.get("trigger_type", "unknown"))] += 1
    return dict(counter)


def discover_episode_figures(figures_dir: str, event_index: int):
    names = {
        "main": f"episode_{event_index}_timeline_main.png",
        "profile": f"episode_{event_index}_timeline_profile.png",
        "guardrail": f"episode_{event_index}_timeline_guardrail.png",
        "triggers": f"episode_{event_index}_timeline_triggers.png",
    }

    out = {}
    for k, filename in names.items():
        path = os.path.join(figures_dir, filename)
        if os.path.exists(path):
            out[k] = filename
    return out


def format_json_block(obj):
    return "```json\n" + json.dumps(obj, ensure_ascii=False, indent=2) + "\n```"


def build_case_section(
    item,
    episode_summary_map,
    decisions_map,
    profiles_map,
    triggers_map,
    figures_dir,
):
    event_index = item.get("event_index")
    ep_summary = episode_summary_map.get(event_index, {})
    decision_rows = decisions_map.get(event_index, [])
    profile_rows = profiles_map.get(event_index, [])
    trigger_rows = triggers_map.get(event_index, [])

    decision_summary = summarize_decisions(decision_rows)
    profile_summary = summarize_profile_trace(profile_rows)
    trigger_summary = summarize_trigger_trace(trigger_rows)
    figures = discover_episode_figures(figures_dir, event_index)

    case_type = item.get("_type", "unknown")
    score = item.get("_score", None)

    metadata = ep_summary.get("metadata", {})
    dataset_risk_label = ep_summary.get("dataset_risk_label")
    episode_llm_violation = ep_summary.get("episode_llm_violation")
    episode_num_frames = ep_summary.get("episode_num_frames")
    episode_min_ttc_estimated = ep_summary.get("episode_min_ttc_estimated")
    episode_avg_ttc_estimated = ep_summary.get("episode_avg_ttc_estimated")
    episode_violation_rate = ep_summary.get("episode_violation_rate")

    lines = []
    lines.append(f"## Episode {event_index}")
    lines.append("")
    lines.append(f"- Case type: **{case_type}**")
    if score is not None:
        lines.append(f"- Selection score: **{score:.3f}**")
    lines.append(f"- Dataset risk label: **{dataset_risk_label}**")
    lines.append(f"- Episode LLM violation: **{episode_llm_violation}**")
    lines.append(f"- Number of frames: **{episode_num_frames}**")
    lines.append(f"- Estimated min TTC: **{episode_min_ttc_estimated}**")
    lines.append(f"- Estimated avg TTC: **{episode_avg_ttc_estimated}**")
    lines.append(f"- Episode violation rate: **{episode_violation_rate}**")
    lines.append("")

    if metadata:
        lines.append("### Metadata")
        lines.append("")
        lines.append(format_json_block(metadata))
        lines.append("")

    if decision_summary:
        lines.append("### Decision Summary")
        lines.append("")
        lines.append(format_json_block(decision_summary))
        lines.append("")

    if profile_summary:
        lines.append("### Profile Summary")
        lines.append("")
        lines.append(format_json_block(profile_summary))
        lines.append("")

    if trigger_summary:
        lines.append("### Trigger Summary")
        lines.append("")
        lines.append(format_json_block(trigger_summary))
        lines.append("")

    if figures:
        lines.append("### Figures")
        lines.append("")
        for k, filename in figures.items():
            lines.append(f"#### {k}")
            lines.append(f"![{k}]({filename})")
            lines.append("")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Generate Top-K case report")
    parser.add_argument("--run_dir", type=str, required=True)
    parser.add_argument("--figures_dir", type=str, default="")
    parser.add_argument("--selection_json", type=str, default="")
    parser.add_argument("--output_path", type=str, default="")
    parser.add_argument("--title", type=str, default="Top-K Episode Case Study Report")
    args = parser.parse_args()

    run_dir = args.run_dir
    figures_dir = args.figures_dir or os.path.join(run_dir, "topk_episode_figures")
    selection_json = args.selection_json or os.path.join(figures_dir, "topk_selection.json")
    output_path = args.output_path or os.path.join(figures_dir, "topk_case_report.md")

    if not os.path.exists(selection_json):
        raise RuntimeError(f"Selection file not found: {selection_json}")

    selections = load_json(selection_json)
    episode_summaries = load_jsonl(os.path.join(run_dir, "episode_summary.jsonl"))
    decisions = load_jsonl(os.path.join(run_dir, "decisions.jsonl"))
    profiles = load_jsonl(os.path.join(run_dir, "profile_trace.jsonl"))
    triggers = load_jsonl(os.path.join(run_dir, "trigger_trace.jsonl"))

    episode_summary_map = {r.get("event_index"): r for r in episode_summaries}
    decisions_map = group_by_event(decisions)
    profiles_map = group_by_event(profiles)
    triggers_map = group_by_event(triggers)

    lines = []
    lines.append(f"# {args.title}")
    lines.append("")
    lines.append(f"- Run directory: `{run_dir}`")
    lines.append(f"- Figures directory: `{figures_dir}`")
    lines.append(f"- Number of selected cases: **{len(selections)}**")
    lines.append("")

    # 总览表
    lines.append("## Overview")
    lines.append("")
    lines.append("| event_index | type | score | dataset_risk | llm_violation | min_ttc | violation_rate |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|")

    for item in selections:
        event_index = item.get("event_index")
        ep = episode_summary_map.get(event_index, {})
        lines.append(
            f"| {event_index} | {item.get('_type', 'unknown')} | "
            f"{item.get('_score', 0.0):.3f} | "
            f"{ep.get('dataset_risk_label')} | "
            f"{ep.get('episode_llm_violation')} | "
            f"{ep.get('episode_min_ttc_estimated')} | "
            f"{ep.get('episode_violation_rate')} |"
        )

    lines.append("")

    # 每个 case 的详细章节
    for item in selections:
        lines.append(
            build_case_section(
                item=item,
                episode_summary_map=episode_summary_map,
                decisions_map=decisions_map,
                profiles_map=profiles_map,
                triggers_map=triggers_map,
                figures_dir=figures_dir,
            )
        )
        lines.append("")

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print("Case report saved to:", output_path)


if __name__ == "__main__":
    main()