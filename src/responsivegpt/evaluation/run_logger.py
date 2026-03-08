import json
import os
from datetime import datetime

class RunLogger:
    def __init__(self, runs_root: str = "runs", tag: str = "run"):
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_tag = "".join([c if c.isalnum() or c in "-_." else "_" for c in tag])[:64]
        self.run_dir = os.path.join(runs_root, f"{ts}_{safe_tag}")
        os.makedirs(self.run_dir, exist_ok=True)
        self.decisions_path = os.path.join(self.run_dir, "decisions.jsonl")
        self.config_path = os.path.join(self.run_dir, "config.json")

    def write_config(self, config: dict) -> None:
        with open(self.config_path, "w", encoding="utf-8") as f:
            json.dump(config, f, ensure_ascii=False, indent=2)

    def append_decision(self, record: dict) -> None:
        with open(self.decisions_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
