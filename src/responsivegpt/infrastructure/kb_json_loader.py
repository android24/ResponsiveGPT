import json
import os
from typing import Any

from ..domain.evidence import KnowledgeDoc


REQUIRED_FIELDS = ["id", "kb_type", "title", "text"]
ALLOWED_KB_TYPES = {"law", "case", "scenario"}


def _to_str_or_none(v):
    if v is None:
        return None
    s = str(v).strip()
    return s if s else None


def _to_float(v, default=1.0):
    try:
        return float(v)
    except Exception:
        return default


def validate_doc(raw: dict[str, Any], file_path: str, idx: int) -> None:
    for field in REQUIRED_FIELDS:
        if field not in raw:
            raise ValueError(
                f"Missing required field '{field}' in {file_path} at item index {idx}"
            )

    kb_type = str(raw["kb_type"]).strip()
    if kb_type not in ALLOWED_KB_TYPES:
        raise ValueError(
            f"Invalid kb_type='{kb_type}' in {file_path} at item index {idx}; "
            f"allowed={sorted(ALLOWED_KB_TYPES)}"
        )


def raw_to_doc(raw: dict[str, Any]) -> KnowledgeDoc:
    return KnowledgeDoc(
        id=str(raw["id"]).strip(),
        kb_type=str(raw["kb_type"]).strip(),
        title=str(raw["title"]).strip(),
        text=str(raw["text"]).strip(),
        scene_type=_to_str_or_none(raw.get("scene_type")),
        event_type=_to_str_or_none(raw.get("event_type")),
        pair_type=_to_str_or_none(raw.get("pair_type")),
        source=_to_str_or_none(raw.get("source")),
        priority=_to_float(raw.get("priority", 1.0), default=1.0),
    )


def load_kb_json_file(file_path: str) -> list[KnowledgeDoc]:
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"KB file not found: {file_path}")

    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError(f"KB file must be a JSON list: {file_path}")

    docs = []
    seen_ids = set()

    for idx, raw in enumerate(data):
        if not isinstance(raw, dict):
            raise ValueError(f"Each KB item must be a JSON object in {file_path}, index={idx}")

        validate_doc(raw, file_path, idx)
        doc = raw_to_doc(raw)

        if doc.id in seen_ids:
            raise ValueError(f"Duplicate doc id '{doc.id}' in {file_path}")
        seen_ids.add(doc.id)

        docs.append(doc)

    return docs


def load_kb_json_dir(kb_dir: str) -> list[KnowledgeDoc]:
    """
    约定目录下有:
      - law.json
      - case.json
      - scenario.json
    """
    files = [
        os.path.join(kb_dir, "law.json"),
        os.path.join(kb_dir, "case.json"),
        os.path.join(kb_dir, "scenario.json"),
    ]

    all_docs = []
    global_ids = set()

    for fp in files:
        docs = load_kb_json_file(fp)
        for d in docs:
            if d.id in global_ids:
                raise ValueError(f"Duplicate doc id across KB files: {d.id}")
            global_ids.add(d.id)
            all_docs.append(d)

    return all_docs