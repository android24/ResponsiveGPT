import os
import json
import pandas as pd
import matplotlib.pyplot as plt

from .trigger_analysis import TriggerAnalyzer


class TriggerPlotter:
    def __init__(self, run_dir: str):
        self.run_dir = run_dir
        self.analyzer = TriggerAnalyzer(run_dir)

        self.fig_dir = os.path.join(run_dir, "figures")
        os.makedirs(self.fig_dir, exist_ok=True)

    # --------------------------------------------------
    # 1️⃣ Trigger 分布图
    # --------------------------------------------------
    def plot_trigger_distribution(self):
        stats = self.analyzer.analyze_trigger_distribution()

        df = pd.DataFrame(list(stats.items()), columns=["trigger_type", "count"])
        df = df.sort_values(by="count", ascending=False)

        plt.figure()
        plt.bar(df["trigger_type"], df["count"])
        plt.xticks(rotation=30)
        plt.title("Trigger Distribution")
        plt.xlabel("Trigger Type")
        plt.ylabel("Count")

        path = os.path.join(self.fig_dir, "trigger_distribution.png")
        plt.savefig(path, bbox_inches="tight")
        plt.close()

        return path

    # --------------------------------------------------
    # 2️⃣ Trigger vs TTC
    # --------------------------------------------------
    def plot_trigger_ttc(self):
        stats = self.analyzer.analyze_trigger_ttc()

        df = pd.DataFrame(list(stats.items()), columns=["trigger_type", "avg_ttc"])
        df = df.sort_values(by="avg_ttc")

        plt.figure()
        plt.bar(df["trigger_type"], df["avg_ttc"])
        plt.xticks(rotation=30)
        plt.title("Trigger vs TTC")
        plt.xlabel("Trigger Type")
        plt.ylabel("Average TTC (s)")

        path = os.path.join(self.fig_dir, "trigger_ttc.png")
        plt.savefig(path, bbox_inches="tight")
        plt.close()

        return path

    # --------------------------------------------------
    # 3️⃣ Trigger 提前量（论文核心 ⭐）
    # --------------------------------------------------
    def plot_trigger_lead_time(self):
        result = self.analyzer.analyze_trigger_lead_time()

        if not result or result.get("samples", 0) == 0:
            return None

        # 这里我们重新计算所有 lead times 分布
        frames = self.analyzer.frames

        events = {}
        for r in frames:
            eid = r["metadata"].get("event_id")
            events.setdefault(eid, []).append(r)

        lead_times = []

        for frames in events.values():
            min_ttc = float("inf")
            min_frame = None

            for f in frames:
                ttc = f["step_metrics"].get("ttc_s")
                if ttc is not None and ttc < min_ttc:
                    min_ttc = ttc
                    min_frame = f["scene"].get("frame_index")

            first_trigger = None
            for f in frames:
                if f.get("triggers"):
                    first_trigger = f["scene"].get("frame_index")
                    break

            if min_frame is not None and first_trigger is not None:
                lead_times.append(min_frame - first_trigger)

        if not lead_times:
            return None

        plt.figure()
        plt.hist(lead_times, bins=20)
        plt.title("Trigger Lead Time Distribution")
        plt.xlabel("Lead Time (frames)")
        plt.ylabel("Frequency")

        path = os.path.join(self.fig_dir, "trigger_lead_time.png")
        plt.savefig(path, bbox_inches="tight")
        plt.close()

        return path

    # --------------------------------------------------
    # 4️⃣ Trigger vs Violation
    # --------------------------------------------------
    def plot_trigger_violation(self):
        stats = self.analyzer.analyze_trigger_vs_violation()

        labels = ["With Trigger", "Without Trigger"]
        values = [
            stats["with_trigger_violation_rate"],
            stats["without_trigger_violation_rate"],
        ]

        plt.figure()
        plt.bar(labels, values)
        plt.title("Violation Rate: Trigger vs No Trigger")
        plt.ylabel("Violation Rate")

        path = os.path.join(self.fig_dir, "trigger_violation.png")
        plt.savefig(path, bbox_inches="tight")
        plt.close()

        return path

    # --------------------------------------------------
    # 一键生成全部图
    # --------------------------------------------------
    def plot_all(self):
        paths = {}

        paths["distribution"] = self.plot_trigger_distribution()
        paths["ttc"] = self.plot_trigger_ttc()
        paths["lead_time"] = self.plot_trigger_lead_time()
        paths["violation"] = self.plot_trigger_violation()

        return paths