import os
import json
import argparse

import matplotlib.pyplot as plt


METRICS = ["precision", "recall", "f1", "accuracy", "agreement_rate"]


def load_json(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def plot_single_mode(items: dict, mode_name: str, output_dir: str):
    """
    用于 batch_compare.json / episode_compare.json
    每个 metric 一张图，横轴是 label
    """
    ensure_dir(output_dir)

    labels = list(items.keys())
    for metric in METRICS:
        values = [items[label]["metrics"].get(metric, 0.0) for label in labels]

        plt.figure(figsize=(8, 5))
        plt.bar(labels, values)
        plt.ylim(0, 1.0)
        plt.ylabel(metric)
        plt.title(f"{mode_name}: {metric}")
        plt.xticks(rotation=20)
        plt.tight_layout()

        out_path = os.path.join(output_dir, f"{mode_name}_{metric}.png")
        plt.savefig(out_path, dpi=200)
        plt.close()


def plot_cross_mode(comparisons: dict, output_dir: str):
    """
    用于 cross_compare.json
    每个 metric 一张图：batch vs episode
    """
    ensure_dir(output_dir)

    labels = list(comparisons.keys())

    for metric in METRICS:
        batch_vals = []
        episode_vals = []
        gains = []

        for label in labels:
            item = comparisons[label]["result"]["comparison"][metric]
            batch_vals.append(item["batch"])
            episode_vals.append(item["episode"])
            gains.append(item["absolute_gain"])

        x = range(len(labels))
        width = 0.35

        # batch vs episode
        plt.figure(figsize=(9, 5))
        plt.bar([i - width / 2 for i in x], batch_vals, width=width, label="batch")
        plt.bar([i + width / 2 for i in x], episode_vals, width=width, label="episode")
        plt.xticks(list(x), labels, rotation=20)
        plt.ylim(0, 1.0)
        plt.ylabel(metric)
        plt.title(f"Cross Comparison: {metric}")
        plt.legend()
        plt.tight_layout()

        out_path = os.path.join(output_dir, f"cross_{metric}.png")
        plt.savefig(out_path, dpi=200)
        plt.close()

        # gain plot
        plt.figure(figsize=(9, 5))
        plt.bar(labels, gains)
        plt.axhline(0, linewidth=1)
        plt.ylabel("absolute gain")
        plt.title(f"Episode - Batch Gain: {metric}")
        plt.xticks(rotation=20)
        plt.tight_layout()

        out_path = os.path.join(output_dir, f"cross_gain_{metric}.png")
        plt.savefig(out_path, dpi=200)
        plt.close()


def plot_global_overview(global_overview: dict, output_dir: str):
    """
    从 comparison.json 的 global_overview 画总体平均图
    """
    ensure_dir(output_dir)

    metrics = list(global_overview.keys())
    batch_vals = [global_overview[m]["avg_batch"] for m in metrics]
    episode_vals = [global_overview[m]["avg_episode"] for m in metrics]
    gain_vals = [global_overview[m]["avg_absolute_gain"] for m in metrics]

    x = range(len(metrics))
    width = 0.35

    plt.figure(figsize=(10, 5))
    plt.bar([i - width / 2 for i in x], batch_vals, width=width, label="avg_batch")
    plt.bar([i + width / 2 for i in x], episode_vals, width=width, label="avg_episode")
    plt.xticks(list(x), metrics, rotation=20)
    plt.ylim(0, 1.0)
    plt.ylabel("score")
    plt.title("Global Overview")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "global_overview.png"), dpi=200)
    plt.close()

    plt.figure(figsize=(10, 5))
    plt.bar(metrics, gain_vals)
    plt.axhline(0, linewidth=1)
    plt.ylabel("avg absolute gain")
    plt.title("Global Average Gain")
    plt.xticks(rotation=20)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "global_gain.png"), dpi=200)
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Plot comparison results.")
    parser.add_argument("--json_path", type=str, required=True, help="Path to compare json")
    parser.add_argument("--output_dir", type=str, default="runs/plots")
    args = parser.parse_args()

    ensure_dir(args.output_dir)
    payload = load_json(args.json_path)

    mode = payload.get("mode", "")

    if mode == "batch":
        items = payload["items"]
        plot_single_mode(items, "batch", args.output_dir)

    elif mode == "episode":
        items = payload["items"]
        plot_single_mode(items, "episode", args.output_dir)

    elif mode == "cross":
        comparisons = payload["comparisons"]
        plot_cross_mode(comparisons, args.output_dir)

    else:
        # 兼容 compare_experiments.py 旧版结构
        if "comparisons" in payload and "global_overview" in payload:
            plot_cross_mode(payload["comparisons"], args.output_dir)
            plot_global_overview(payload["global_overview"], args.output_dir)
        else:
            raise RuntimeError(f"Unsupported compare json format: {args.json_path}")

    print("Plots saved to:", args.output_dir)


if __name__ == "__main__":
    main()