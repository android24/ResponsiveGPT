import json
import os
from typing import Any

from ..domain.evidence import KnowledgeDoc


ALLOWED_KB_TYPES = {"law", "case", "scenario", "policy", "safety"}
KB_FILE_ORDER = ("law", "case", "scenario", "policy", "safety")
DEFAULT_KB_DIR = "src/responsivegpt/data/kb"


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


def _to_list(v) -> list[str]:
    if v is None:
        return []
    if isinstance(v, list):
        return [str(x).strip() for x in v if str(x).strip()]
    s = str(v).strip()
    return [s] if s else []


def _legacy_value(raw: dict[str, Any], key: str):
    for container_name in ("legacy_fields", "legacy"):
        container = raw.get(container_name)
        if not isinstance(container, dict):
            continue
        for candidate in (key, f"old_{key}"):
            if candidate in container:
                return container.get(candidate)
    return None


def _doc_id(raw: dict[str, Any]):
    return (
        raw.get("id")
        or raw.get("chunk_id")
        or raw.get("doc_id")
        or _legacy_value(raw, "id")
    )


def _doc_type(raw: dict[str, Any]):
    return (
        raw.get("kb_type")
        or raw.get("doc_type")
        or _legacy_value(raw, "kb_type")
    )


def validate_doc(raw: dict[str, Any], file_path: str, idx: int) -> None:
    if not _doc_id(raw):
        raise ValueError(
            f"Missing required field 'id/chunk_id' in {file_path} at item index {idx}"
        )

    if "title" not in raw:
        raise ValueError(
            f"Missing required field 'title' in {file_path} at item index {idx}"
        )

    if "text" not in raw:
        raise ValueError(
            f"Missing required field 'text' in {file_path} at item index {idx}"
        )

    kb_type = str(_doc_type(raw) or "").strip()
    if kb_type not in ALLOWED_KB_TYPES:
        raise ValueError(
            f"Invalid kb_type/doc_type='{kb_type}' in {file_path} at item index {idx}; "
            f"allowed={sorted(ALLOWED_KB_TYPES)}"
        )


def raw_to_doc(raw: dict[str, Any]) -> KnowledgeDoc:
    priority = (
        raw.get("priority")
        or raw.get("priority_weight")
        or raw.get("original_priority_score")
        or 1.0
    )

    metadata = {}
    for key in ("legacy_fields", "legacy"):
        if isinstance(raw.get(key), dict):
            metadata[key] = raw.get(key)
    if raw.get("evidence_purpose"):
        metadata["evidence_purpose"] = raw.get("evidence_purpose")

    return KnowledgeDoc(
        id=str(_doc_id(raw)).strip(),
        kb_type=str(_doc_type(raw)).strip(),
        title=str(raw["title"]).strip(),
        text=str(raw["text"]).strip(),
        scene_type=_to_str_or_none(raw.get("scene_type") or _legacy_value(raw, "scene_type")),
        event_type=_to_str_or_none(raw.get("event_type") or _legacy_value(raw, "event_type")),
        pair_type=_to_str_or_none(raw.get("pair_type") or _legacy_value(raw, "pair_type")),
        source=_to_str_or_none(raw.get("source")),
        priority=_to_float(priority, default=1.0),
        dataset_tags=_to_list(raw.get("dataset_tags")),
        scenario_tags=_to_list(raw.get("scenario_tags")),
        metric_tags=_to_list(raw.get("metric_tags")),
        risk_tags=_to_list(raw.get("risk_tags")),
        condition=raw.get("condition") if isinstance(raw.get("condition"), dict) else {},
        risk_mechanism=_to_str_or_none(raw.get("risk_mechanism")),
        recommended_action=_to_list(raw.get("recommended_action")),
        forbidden_action=_to_list(raw.get("forbidden_action")),
        severity=_to_str_or_none(raw.get("severity")),
        jurisdiction=_to_str_or_none(raw.get("jurisdiction")),
        metadata=metadata,
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
      - policy.json
      - safety.json
    """
    files = [
        os.path.join(kb_dir, f"{name}.json")
        for name in KB_FILE_ORDER
        if os.path.exists(os.path.join(kb_dir, f"{name}.json"))
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

    if not all_docs:
        raise FileNotFoundError(f"No KB JSON files found in directory: {kb_dir}")

    return all_docs


def resolve_kb_dir(kb_dir: str | None = None) -> str | None:
    candidates = []
    if kb_dir:
        candidates.append(kb_dir)
    candidates.extend([DEFAULT_KB_DIR, "data/kb"])

    seen = set()
    for candidate in candidates:
        if not candidate or candidate in seen:
            continue
        seen.add(candidate)
        if os.path.isdir(candidate):
            return candidate

    return None
