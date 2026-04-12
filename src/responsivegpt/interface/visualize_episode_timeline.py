import os
import json
import argparse
from collections import defaultdict

import matplotlib.pyplot as plt


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


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def group_by_event(rows, key="event_index"):
    grouped = defaultdict(list)
    for r in rows:
        grouped[r.get(key, -1)].append(r)
    return grouped


def sort_by_frame(rows):
    return sorted(rows, key=lambda x: (x.get("frame_index", -1), x.get("scene", {}).get("frame_index", -1)))


def extract_frame_index(row):
    if "frame_index" in row and row["frame_index"] is not None:
        return row["frame_index"]
    if "scene" in row and isinstance(row["scene"], dict):
        return row["scene"].get("frame_index", None)
    return None


def extract_ttc(row):
    return row.get("step_metrics", {}).get("ttc_s", None)


def extract_violation(row):
    v = row.get("decision", {}).get("is_potential_violation", None)
    if isinstance(v, bool):
        return 1 if v else 0
    return None


def extract_risk_level(row):
    rl = row.get("decision", {}).get("risk_level", "")
    mapping = {"low": 1, "medium": 2, "high": 3}
    return mapping.get(str(rl).lower(), None)


def extract_profile_value(profile_row, key):
    profile = profile_row.get("profile", {})
    if isinstance(profile, dict):
        return profile.get(key, None)
    return None


def extract_guardrail_value(guardrail_row, key):
    guardrails = guardrail_row.get("guardrails", {})
    if isinstance(guardrails, dict):
        return guardrails.get(key, None)
    return None


def plot_episode_timeline(
    event_index: int,
    decision_rows: list,
    profile_rows: list,
    trigger_rows: list,
    guardrail_rows: list,
    output_dir: str,
):
    decision_rows = sort_by_frame(decision_rows)
    profile_rows = sort_by_frame(profile_rows)
    trigger_rows = sort_by_frame(trigger_rows)
    guardrail_rows = sort_by_frame(guardrail_rows)

    frames = []
    ttc_vals = []
    violation_vals = []
    risk_vals = []

    for r in decision_rows:
        f = extract_frame_index(r)
        if f is None:
            continue
        frames.append(f)
        ttc_vals.append(extract_ttc(r))
        violation_vals.append(extract_violation(r))
        risk_vals.append(extract_risk_level(r))

    if not frames:
        return None

    # Trigger points
    trigger_by_frame = defaultdict(list)
    for r in trigger_rows:
        f = extract_frame_index(r)
        trig = r.get("trigger", {})
        trig_type = trig.get("trigger_type", "unknown") if isinstance(trig, dict) else "unknown"
        if f is not None:
            trigger_by_frame[f].append(trig_type)

    trigger_frames = sorted(trigger_by_frame.keys())

    # Profile traces
    p_frames = []
    safety_weights = []
    efficiency_weights = []
    risk_sensitivity = []

    for r in profile_rows:
        f = extract_frame_index(r)
        if f is None:
            continue
        p_frames.append(f)
        safety_weights.append(extract_profile_value(r, "safety_weight"))
        efficiency_weights.append(extract_profile_value(r, "efficiency_weight"))
        risk_sensitivity.append(extract_profile_value(r, "risk_sensitivity"))

    # Guardrail traces
    g_frames = []
    max_speed_vals = []
    min_headway_vals = []

    for r in guardrail_rows:
        f = extract_frame_index(r)
        if f is None:
            continue
        g_frames.append(f)
        max_speed_vals.append(extract_guardrail_value(r, "max_speed"))
        min_headway_vals.append(extract_guardrail_value(r, "min_headway"))

    # ---------- Figure 1: TTC + Violation + Triggers ----------
    plt.figure(figsize=(12, 6))

    plt.plot(frames, ttc_vals, label="TTC (s)")
    plt.plot(frames, violation_vals, label="Violation", linestyle="--")
    plt.plot(frames, risk_vals, label="Risk Level(1/2/3)", linestyle=":")

    for tf in trigger_frames:
        plt.axvline(tf, linestyle="--", alpha=0.5)

    plt.xlabel("Frame")
    plt.ylabel("Value")
    plt.title(f"Episode {event_index}: TTC / Violation / Trigger Timeline")
    plt.legend()
    plt.tight_layout()

    fig1 = os.path.join(output_dir, f"episode_{event_index}_timeline_main.png")
    plt.savefig(fig1, dpi=200)
    plt.close()

    # ---------- Figure 2: Profile evolution ----------
    fig2 = None
    if p_frames:
        plt.figure(figsize=(12, 6))
        plt.plot(p_frames, safety_weights, label="Safety Weight")
        plt.plot(p_frames, efficiency_weights, label="Efficiency Weight")
        plt.plot(p_frames, risk_sensitivity, label="Risk Sensitivity")

        for tf in trigger_frames:
            plt.axvline(tf, linestyle="--", alpha=0.5)

        plt.xlabel("Frame")
        plt.ylabel("Value")
        plt.title(f"Episode {event_index}: Profile Evolution")
        plt.legend()
        plt.tight_layout()

        fig2 = os.path.join(output_dir, f"episode_{event_index}_timeline_profile.png")
        plt.savefig(fig2, dpi=200)
        plt.close()

    # ---------- Figure 3: Guardrail evolution ----------
    fig3 = None
    if g_frames:
        plt.figure(figsize=(12, 6))
        if any(v is not None for v in max_speed_vals):
            plt.plot(g_frames, max_speed_vals, label="Guardrail Max Speed")
        if any(v is not None for v in min_headway_vals):
            plt.plot(g_frames, min_headway_vals, label="Guardrail Min Headway")

        for tf in trigger_frames:
            plt.axvline(tf, linestyle="--", alpha=0.5)

        plt.xlabel("Frame")
        plt.ylabel("Value")
        plt.title(f"Episode {event_index}: Guardrail Evolution")
        plt.legend()
        plt.tight_layout()

        fig3 = os.path.join(output_dir, f"episode_{event_index}_timeline_guardrail.png")
        plt.savefig(fig3, dpi=200)
        plt.close()

    # ---------- Figure 4: Trigger counts by frame ----------
    fig4 = None
    if trigger_frames:
        tf_x = []
        tf_y = []
        for f in trigger_frames:
            tf_x.append(f)
            tf_y.append(len(trigger_by_frame[f]))

        plt.figure(figsize=(12, 4))
        plt.bar(tf_x, tf_y)
        plt.xlabel("Frame")
        plt.ylabel("Trigger Count")
        plt.title(f"Episode {event_index}: Trigger Count by Frame")
        plt.tight_layout()

        fig4 = os.path.join(output_dir, f"episode_{event_index}_timeline_triggers.png")
        plt.savefig(fig4, dpi=200)
        plt.close()

    return {
        "main": fig1,
        "profile": fig2,
        "guardrail": fig3,
        "triggers": fig4,
    }


def main():
    parser = argparse.ArgumentParser(description="Visualize episode timeline from run directory")
    parser.add_argument("--run_dir", type=str, required=True)
    parser.add_argument("--event_index", type=int, default=-1, help="Single event index. -1 means all")
    parser.add_argument("--output_dir", type=str, default="")
    args = parser.parse_args()

    run_dir = args.run_dir
    output_dir = args.output_dir or os.path.join(run_dir, "figures")
    ensure_dir(output_dir)

    decisions = load_jsonl(os.path.join(run_dir, "decisions.jsonl"))
    profiles = load_jsonl(os.path.join(run_dir, "profile_trace.jsonl"))
    triggers = load_jsonl(os.path.join(run_dir, "trigger_trace.jsonl"))
    guardrails = load_jsonl(os.path.join(run_dir, "guardrail_trace.jsonl"))

    decisions_g = group_by_event(decisions)
    profiles_g = group_by_event(profiles)
    triggers_g = group_by_event(triggers)
    guardrails_g = group_by_event(guardrails)

    if args.event_index >= 0:
        event_indices = [args.event_index]
    else:
        event_indices = sorted(decisions_g.keys())

    generated = {}

    for event_index in event_indices:
        figs = plot_episode_timeline(
            event_index=event_index,
            decision_rows=decisions_g.get(event_index, []),
            profile_rows=profiles_g.get(event_index, []),
            trigger_rows=triggers_g.get(event_index, []),
            guardrail_rows=guardrails_g.get(event_index, []),
            output_dir=output_dir,
        )
        if figs:
            generated[event_index] = figs
            print(f"[OK] event {event_index}: {figs}")

    manifest_path = os.path.join(output_dir, "timeline_manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(generated, f, ensure_ascii=False, indent=2)

    print("\nSaved timeline figures to:", output_dir)
    print("Manifest:", manifest_path)


if __name__ == "__main__":
    main()