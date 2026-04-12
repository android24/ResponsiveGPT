import os
import json
import argparse
from collections import defaultdict
from html import escape


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


def format_json_html(obj):
    return f"<pre>{escape(json.dumps(obj, ensure_ascii=False, indent=2))}</pre>"


def summarize_decisions(decision_rows):
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


def relative_path(from_dir: str, to_path: str) -> str:
    return os.path.relpath(to_path, from_dir).replace("\\", "/")


def build_case_card(
    item,
    episode_summary_map,
    decisions_map,
    profiles_map,
    triggers_map,
    figures_dir,
    html_dir,
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

    figs_html = []
    for k, filename in figures.items():
        abs_path = os.path.join(figures_dir, filename)
        src = relative_path(html_dir, abs_path)
        figs_html.append(
            f"""
            <div class="figure-block">
              <h4>{escape(k)}</h4>
              <img src="{escape(src)}" alt="{escape(k)}">
            </div>
            """
        )

    return f"""
    <section class="case-card">
      <h2>Episode {escape(str(event_index))}</h2>
      <div class="meta-grid">
        <div><b>Case type:</b> {escape(str(case_type))}</div>
        <div><b>Selection score:</b> {"" if score is None else f"{score:.3f}"}</div>
        <div><b>Dataset risk label:</b> {escape(str(dataset_risk_label))}</div>
        <div><b>Episode LLM violation:</b> {escape(str(episode_llm_violation))}</div>
        <div><b>Number of frames:</b> {escape(str(episode_num_frames))}</div>
        <div><b>Estimated min TTC:</b> {escape(str(episode_min_ttc_estimated))}</div>
        <div><b>Estimated avg TTC:</b> {escape(str(episode_avg_ttc_estimated))}</div>
        <div><b>Episode violation rate:</b> {escape(str(episode_violation_rate))}</div>
      </div>

      <h3>Metadata</h3>
      {format_json_html(metadata)}

      <h3>Decision Summary</h3>
      {format_json_html(decision_summary)}

      <h3>Profile Summary</h3>
      {format_json_html(profile_summary)}

      <h3>Trigger Summary</h3>
      {format_json_html(trigger_summary)}

      <h3>Figures</h3>
      <div class="figures-grid">
        {"".join(figs_html)}
      </div>
    </section>
    """


def build_overview_table(selections, episode_summary_map):
    rows = []
    for item in selections:
        event_index = item.get("event_index")
        ep = episode_summary_map.get(event_index, {})
        rows.append(
            f"""
            <tr>
              <td>{escape(str(event_index))}</td>
              <td>{escape(str(item.get("_type", "unknown")))}</td>
              <td>{item.get("_score", 0.0):.3f}</td>
              <td>{escape(str(ep.get("dataset_risk_label")))}</td>
              <td>{escape(str(ep.get("episode_llm_violation")))}</td>
              <td>{escape(str(ep.get("episode_min_ttc_estimated")))}</td>
              <td>{escape(str(ep.get("episode_violation_rate")))}</td>
            </tr>
            """
        )

    return f"""
    <table>
      <thead>
        <tr>
          <th>event_index</th>
          <th>type</th>
          <th>score</th>
          <th>dataset_risk</th>
          <th>llm_violation</th>
          <th>min_ttc</th>
          <th>violation_rate</th>
        </tr>
      </thead>
      <tbody>
        {"".join(rows)}
      </tbody>
    </table>
    """


def build_html_page(title, run_dir, figures_dir, selections, episode_summary_map,
                    decisions_map, profiles_map, triggers_map, html_dir):
    overview_table = build_overview_table(selections, episode_summary_map)

    case_cards = []
    for item in selections:
        case_cards.append(
            build_case_card(
                item=item,
                episode_summary_map=episode_summary_map,
                decisions_map=decisions_map,
                profiles_map=profiles_map,
                triggers_map=triggers_map,
                figures_dir=figures_dir,
                html_dir=html_dir,
            )
        )

    return f"""
<!DOCTYPE html>
<html lang="zh-CN">
<head>
  <meta charset="UTF-8">
  <title>{escape(title)}</title>
  <style>
    body {{
      font-family: Arial, sans-serif;
      margin: 24px;
      line-height: 1.6;
      background: #fafafa;
      color: #222;
    }}
    h1, h2, h3, h4 {{
      color: #111;
    }}
    .case-card {{
      background: white;
      border: 1px solid #ddd;
      border-radius: 10px;
      padding: 20px;
      margin-bottom: 28px;
      box-shadow: 0 1px 4px rgba(0,0,0,0.05);
    }}
    .meta-grid {{
      display: grid;
      grid-template-columns: repeat(2, minmax(240px, 1fr));
      gap: 8px 24px;
      margin-bottom: 18px;
    }}
    table {{
      border-collapse: collapse;
      width: 100%;
      background: white;
      margin-bottom: 28px;
    }}
    th, td {{
      border: 1px solid #ccc;
      padding: 8px 10px;
      text-align: left;
    }}
    th {{
      background: #f0f0f0;
    }}
    pre {{
      background: #f6f8fa;
      padding: 12px;
      border-radius: 8px;
      overflow-x: auto;
      border: 1px solid #e5e7eb;
    }}
    .figures-grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(320px, 1fr));
      gap: 18px;
      margin-top: 12px;
    }}
    .figure-block {{
      background: #fff;
      border: 1px solid #eee;
      border-radius: 8px;
      padding: 10px;
    }}
    img {{
      max-width: 100%;
      height: auto;
      display: block;
      border-radius: 6px;
    }}
    .subtle {{
      color: #666;
    }}
  </style>
</head>
<body>
  <h1>{escape(title)}</h1>
  <p class="subtle"><b>Run directory:</b> {escape(run_dir)}</p>
  <p class="subtle"><b>Figures directory:</b> {escape(figures_dir)}</p>
  <p class="subtle"><b>Selected cases:</b> {len(selections)}</p>

  <h2>Overview</h2>
  {overview_table}

  {"".join(case_cards)}
</body>
</html>
"""


def main():
    parser = argparse.ArgumentParser(description="Generate HTML Top-K case report")
    parser.add_argument("--run_dir", type=str, required=True)
    parser.add_argument("--figures_dir", type=str, default="")
    parser.add_argument("--selection_json", type=str, default="")
    parser.add_argument("--output_path", type=str, default="")
    parser.add_argument("--title", type=str, default="Top-K Episode Case Study Report")
    args = parser.parse_args()

    run_dir = args.run_dir
    figures_dir = args.figures_dir or os.path.join(run_dir, "topk_episode_figures")
    selection_json = args.selection_json or os.path.join(figures_dir, "topk_selection.json")
    output_path = args.output_path or os.path.join(figures_dir, "topk_case_report.html")

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

    html_dir = os.path.dirname(output_path)

    html = build_html_page(
        title=args.title,
        run_dir=run_dir,
        figures_dir=figures_dir,
        selections=selections,
        episode_summary_map=episode_summary_map,
        decisions_map=decisions_map,
        profiles_map=profiles_map,
        triggers_map=triggers_map,
        html_dir=html_dir,
    )

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)

    print("HTML case report saved to:", output_path)


if __name__ == "__main__":
    main()