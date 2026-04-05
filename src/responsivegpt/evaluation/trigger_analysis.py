import json
import os
from collections import defaultdict


def load_jsonl(path):
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            data.append(json.loads(line))
    return data


class TriggerAnalyzer:
    def __init__(self, run_dir: str):
        self.run_dir = run_dir

        self.frame_path = os.path.join(run_dir, "decisions.jsonl")
        self.episode_path = os.path.join(run_dir, "episode_summary.jsonl")

        self.frames = load_jsonl(self.frame_path)

    # --------------------------------------------------
    # 1️⃣ Trigger 分布
    # --------------------------------------------------
    def analyze_trigger_distribution(self):
        stats = defaultdict(int)

        for r in self.frames:
            for t in r.get("triggers", []):
                stats[t.get("trigger_type", "unknown")] += 1

        return stats

    # --------------------------------------------------
    # 2️⃣ Trigger vs TTC
    # --------------------------------------------------
    def analyze_trigger_ttc(self):
        bucket = defaultdict(list)

        for r in self.frames:
            ttc = r["step_metrics"].get("ttc_s")

            if ttc is None:
                continue

            triggers = r.get("triggers", [])

            if not triggers:
                bucket["no_trigger"].append(ttc)
            else:
                for t in triggers:
                    bucket[t["trigger_type"]].append(ttc)

        result = {}
        for k, v in bucket.items():
            if v:
                result[k] = sum(v) / len(v)

        return result

    # --------------------------------------------------
    # 3️⃣ Trigger 提前量（关键论文指标 ⭐）
    # --------------------------------------------------
    def analyze_trigger_lead_time(self):
        """
        lead_time = min_ttc_frame - trigger_frame
        >0 表示提前触发
        """

        events = defaultdict(list)

        for r in self.frames:
            event_id = r["metadata"].get("event_id")
            events[event_id].append(r)

        lead_times = []

        for event_id, frames in events.items():
            # 找最小 TTC 帧
            min_ttc = float("inf")
            min_ttc_frame = None

            for f in frames:
                ttc = f["step_metrics"].get("ttc_s")
                if ttc is not None and ttc < min_ttc:
                    min_ttc = ttc
                    min_ttc_frame = f["scene"].get("frame_index")

            # 找第一次 trigger
            first_trigger_frame = None

            for f in frames:
                if f.get("triggers"):
                    first_trigger_frame = f["scene"].get("frame_index")
                    break

            if min_ttc_frame is not None and first_trigger_frame is not None:
                lead_times.append(min_ttc_frame - first_trigger_frame)

        if not lead_times:
            return {}

        return {
            "avg_lead_time": sum(lead_times) / len(lead_times),
            "positive_ratio": sum(1 for x in lead_times if x > 0) / len(lead_times),
            "samples": len(lead_times),
        }

    # --------------------------------------------------
    # 4️⃣ Trigger vs violation
    # --------------------------------------------------
    def analyze_trigger_vs_violation(self):
        with_trigger = []
        without_trigger = []

        for r in self.frames:
            is_violation = r["step_metrics"].get("is_violation")

            if is_violation is None:
                continue

            if r.get("triggers"):
                with_trigger.append(is_violation)
            else:
                without_trigger.append(is_violation)

        def rate(x):
            return sum(x) / len(x) if x else 0

        return {
            "with_trigger_violation_rate": rate(with_trigger),
            "without_trigger_violation_rate": rate(without_trigger),
            "num_with_trigger": len(with_trigger),
            "num_without_trigger": len(without_trigger),
        }

    # --------------------------------------------------
    # 导出 CSV（画图用）
    # --------------------------------------------------
    def export_csv(self):
        import csv

        # Trigger 分布
        dist = self.analyze_trigger_distribution()
        with open(os.path.join(self.run_dir, "trigger_distribution.csv"), "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["trigger_type", "count"])
            for k, v in dist.items():
                writer.writerow([k, v])

        # TTC
        ttc = self.analyze_trigger_ttc()
        with open(os.path.join(self.run_dir, "trigger_ttc.csv"), "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["trigger_type", "avg_ttc"])
            for k, v in ttc.items():
                writer.writerow([k, v])

        # Lead time
        lead = self.analyze_trigger_lead_time()
        with open(os.path.join(self.run_dir, "trigger_lead_time.json"), "w") as f:
            json.dump(lead, f, indent=2)

        # Violation
        vio = self.analyze_trigger_vs_violation()
        with open(os.path.join(self.run_dir, "trigger_violation.json"), "w") as f:
            json.dump(vio, f, indent=2)

        return {
            "distribution": dist,
            "ttc": ttc,
            "lead_time": lead,
            "violation": vio,
        }