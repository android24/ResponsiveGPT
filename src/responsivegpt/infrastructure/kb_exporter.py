import json
import os

from .kb_seed import default_kb_docs


def export_default_kb_to_json_dir(kb_dir: str) -> None:
    os.makedirs(kb_dir, exist_ok=True)

    law_docs = []
    case_docs = []
    scenario_docs = []

    for d in default_kb_docs():
        item = {
            "id": d.id,
            "kb_type": d.kb_type,
            "title": d.title,
            "text": d.text,
            "scene_type": d.scene_type,
            "event_type": d.event_type,
            "pair_type": d.pair_type,
            "source": d.source,
            "priority": d.priority,
        }

        if d.kb_type == "law":
            law_docs.append(item)
        elif d.kb_type == "case":
            case_docs.append(item)
        elif d.kb_type == "scenario":
            scenario_docs.append(item)

    with open(os.path.join(kb_dir, "law.json"), "w", encoding="utf-8") as f:
        json.dump(law_docs, f, ensure_ascii=False, indent=2)

    with open(os.path.join(kb_dir, "case.json"), "w", encoding="utf-8") as f:
        json.dump(case_docs, f, ensure_ascii=False, indent=2)

    with open(os.path.join(kb_dir, "scenario.json"), "w", encoding="utf-8") as f:
        json.dump(scenario_docs, f, ensure_ascii=False, indent=2)