import os
import json
import argparse
import subprocess
from typing import List, Dict, Any


def load_jsonl(path: str) -> List[dict]:
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


def safe_float(v, default=0.0):
    try:
        if v is None:
            return default
        return float(v)
    except Exception:
        return default


def compute_representative_score(row: Dict[str, Any]) -> float:
    """
    分数越高越值得画图
    """
    score = 0.0

    dataset_risk = bool(row.get("dataset_risk_label", False))
    llm_violation = bool(row.get("episode_llm_violation", False))
    min_ttc = row.get("episode_min_ttc_estimated", None)
    violation_rate = row.get("episode_violation_rate", None)
    trigger_count = row.get("trigger_count", 0)

    if dataset_risk:
        score += 10.0
    if llm_violation:
        score += 8.0
    if dataset_risk and llm_violation:
        score += 6.0

    if min_ttc is not None:
        min_ttc = safe_float(min_ttc, 999.0)
        score += max(0.0, 10.0 - min_ttc)

    score += 5.0 * safe_float(violation_rate, 0.0)
    score += 0.2 * safe_float(trigger_count, 0.0)

    return score


def classify_episode_type(row: Dict[str, Any]) -> str:
    """
    用于分层挑选，避免全是同一类
    """
    y_true = bool(row.get("dataset_risk_label", False))
    y_pred = bool(row.get("episode_llm_violation", False))

    if y_true and y_pred:
        return "TP"
    if (not y_true) and y_pred:
        return "FP"
    if y_true and (not y_pred):
        return "FN"
    return "TN"


def select_topk_diverse(rows: List[dict], k: int) -> List[dict]:
    """
    优先保证类别多样性，再按分数排序
    """
    enriched = []
    for row in rows:
        item = dict(row)
        item["_score"] = compute_representative_score(row)
        item["_type"] = classify_episode_type(row)
        enriched.append(item)

    enriched.sort(key=lambda x: x["_score"], reverse=True)

    # 先分桶
    buckets = {"TP": [], "FP": [], "FN": [], "TN": []}
    for item in enriched:
        buckets[item["_type"]].append(item)

    selected = []

    # 第一轮：每类先拿一个
    for t in ["TP", "FN", "FP", "TN"]:
        if buckets[t] and len(selected) < k:
            selected.append(buckets[t].pop(0))

    # 第二轮：剩下按总分补齐
    remaining = []
    for t in buckets:
        remaining.extend(buckets[t])
    remaining.sort(key=lambda x: x["_score"], reverse=True)

    for item in remaining:
        if len(selected) >= k:
            break
        selected.append(item)

    return selected[:k]


def write_selection_manifest(path: str, selected: List[dict]):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(selected, f, ensure_ascii=False, indent=2)


def main():
    parser = argparse.ArgumentParser(description="Select top-K representative episodes and batch plot timelines.")
    parser.add_argument("--run_dir", type=str, required=True)
    parser.add_argument("--top_k", type=int, default=8)
    parser.add_argument("--output_dir", type=str, default="")
    parser.add_argument(
        "--timeline_script",
        type=str,
        default="src/responsivegpt/interface/visualize_episode_timeline.py",
        help="Path to timeline plotting script"
    )
    args = parser.parse_args()

    run_dir = args.run_dir
    output_dir = args.output_dir or os.path.join(run_dir, "topk_episode_figures")
    ensure_dir(output_dir)

    episode_summary_path = os.path.join(run_dir, "episode_summary.jsonl")
    if not os.path.exists(episode_summary_path):
        raise RuntimeError(f"episode_summary.jsonl not found in {run_dir}")

    rows = load_jsonl(episode_summary_path)
    if not rows:
        raise RuntimeError("episode_summary.jsonl is empty")

    selected = select_topk_diverse(rows, args.top_k)

    manifest_path = os.path.join(output_dir, "topk_selection.json")
    write_selection_manifest(manifest_path, selected)

    print(f"Selected {len(selected)} episodes:")
    for item in selected:
        print(
            f"  event_index={item.get('event_index')} "
            f"type={item.get('_type')} "
            f"score={item.get('_score'):.3f} "
            f"min_ttc={item.get('episode_min_ttc_estimated')} "
            f"triggers={item.get('trigger_count', 0)}"
        )

    # 批量调用 timeline 脚本
    for item in selected:
        event_index = item.get("event_index")
        if event_index is None:
            continue

        cmd = [
            "python",
            args.timeline_script,
            "--run_dir", run_dir,
            "--event_index", str(event_index),
            "--output_dir", output_dir,
        ]

        print("Running:", " ".join(cmd))
        subprocess.run(cmd, check=False)

    print("\nTop-K timeline figures saved to:", output_dir)
    print("Selection manifest:", manifest_path)


if __name__ == "__main__":
    main()