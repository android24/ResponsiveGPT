import os
import json
import csv
import argparse
from typing import Dict, Any, List


METRICS = ["precision", "recall", "f1", "accuracy", "agreement_rate"]


def load_json(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def safe_get(d: Dict[str, Any], key: str, default=0.0):
    v = d.get(key, default)
    try:
        return float(v)
    except Exception:
        return default


def summarize_single(summary_path: str, label: str) -> dict:
    s = load_json(summary_path)
    return {
        "label": label,
        "summary_path": summary_path,
        "metrics": {m: safe_get(s, m, 0.0) for m in METRICS},
        "raw": s,
    }


def compare_pair(a: dict, b: dict, label_a: str, label_b: str) -> dict:
    out = {
        "left": label_a,
        "right": label_b,
        "comparison": {}
    }
    for m in METRICS:
        av = safe_get(a, m, 0.0)
        bv = safe_get(b, m, 0.0)
        out["comparison"][m] = {
            label_a: av,
            label_b: bv,
            "absolute_gain": bv - av,
            "relative_gain_percent": ((bv - av) / av * 100.0) if av != 0 else None,
        }
    return out


def write_json(path: str, obj: dict):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def write_table_csv(path: str, rows: List[List[Any]]):
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerows(rows)


def parse_named_paths(items: List[str]) -> Dict[str, str]:
    """
    输入格式：
    label=path
    例如：
    highD_batch=runs/highd_batch/summary.json
    """
    out = {}
    for item in items:
        if "=" not in item:
            raise ValueError(f"Invalid item: {item}. Expected label=path")
        label, path = item.split("=", 1)
        out[label.strip()] = path.strip()
    return out


# =========================
# mode 1: batch 内部对比
# =========================

def run_batch_compare(items: Dict[str, str], output_dir: str):
    singles = {}
    for label, path in items.items():
        singles[label] = summarize_single(path, label)

    json_path = os.path.join(output_dir, "batch_compare.json")
    csv_path = os.path.join(output_dir, "batch_compare.csv")

    payload = {"mode": "batch", "items": singles}
    write_json(json_path, payload)

    rows = [["label"] + METRICS]
    for label, item in singles.items():
        rows.append([label] + [item["metrics"][m] for m in METRICS])
    write_table_csv(csv_path, rows)

    print("Batch comparison written:")
    print(" ", json_path)
    print(" ", csv_path)


# =========================
# mode 2: episode 内部对比
# =========================

def run_episode_compare(items: Dict[str, str], output_dir: str):
    singles = {}
    for label, path in items.items():
        singles[label] = summarize_single(path, label)

    json_path = os.path.join(output_dir, "episode_compare.json")
    csv_path = os.path.join(output_dir, "episode_compare.csv")

    payload = {"mode": "episode", "items": singles}
    write_json(json_path, payload)

    rows = [["label"] + METRICS]
    for label, item in singles.items():
        rows.append([label] + [item["metrics"][m] for m in METRICS])
    write_table_csv(csv_path, rows)

    print("Episode comparison written:")
    print(" ", json_path)
    print(" ", csv_path)


# =========================
# mode 3: batch vs episode 交叉对比
# =========================

def run_cross_compare(batch_items: Dict[str, str], episode_items: Dict[str, str], output_dir: str):
    """
    规则：
    batch_items 和 episode_items 的 label 需要同名
    例如：
      highD=...batch...
      highD=...episode...
    """
    shared = sorted(set(batch_items.keys()) & set(episode_items.keys()))
    if not shared:
        raise RuntimeError("No shared labels between batch and episode inputs.")

    payload = {
        "mode": "cross",
        "comparisons": {}
    }

    rows = [[
        "label", "metric", "batch", "episode", "absolute_gain", "relative_gain_percent"
    ]]

    for label in shared:
        batch_summary = load_json(batch_items[label])
        episode_summary = load_json(episode_items[label])

        comp = compare_pair(batch_summary, episode_summary, "batch", "episode")
        payload["comparisons"][label] = {
            "batch_summary_path": batch_items[label],
            "episode_summary_path": episode_items[label],
            "result": comp
        }

        for m in METRICS:
            item = comp["comparison"][m]
            rows.append([
                label,
                m,
                item["batch"],
                item["episode"],
                item["absolute_gain"],
                "" if item["relative_gain_percent"] is None else item["relative_gain_percent"],
            ])

    json_path = os.path.join(output_dir, "cross_compare.json")
    csv_path = os.path.join(output_dir, "cross_compare.csv")
    write_json(json_path, payload)
    write_table_csv(csv_path, rows)

    print("Cross comparison written:")
    print(" ", json_path)
    print(" ", csv_path)


# =========================
# CLI
# =========================

def main():
    parser = argparse.ArgumentParser(description="Unified experiment comparison runner.")
    parser.add_argument("--mode", type=str, required=True, choices=["batch", "episode", "cross"])
    parser.add_argument("--items", nargs="*", default=[], help="For batch/episode mode: label=path ...")
    parser.add_argument("--batch_items", nargs="*", default=[], help="For cross mode: label=path ...")
    parser.add_argument("--episode_items", nargs="*", default=[], help="For cross mode: label=path ...")
    parser.add_argument("--output_dir", type=str, default="runs/comparison")

    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    if args.mode == "batch":
        items = parse_named_paths(args.items)
        run_batch_compare(items, args.output_dir)

    elif args.mode == "episode":
        items = parse_named_paths(args.items)
        run_episode_compare(items, args.output_dir)

    elif args.mode == "cross":
        batch_items = parse_named_paths(args.batch_items)
        episode_items = parse_named_paths(args.episode_items)
        run_cross_compare(batch_items, episode_items, args.output_dir)


if __name__ == "__main__":
    main()