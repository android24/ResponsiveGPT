import json
import math
from dataclasses import dataclass, field


def _safe_float(value, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _compact_text(value, max_chars: int = 240) -> str:
    text = str(value or "")
    if len(text) > max_chars:
        return text[:max_chars] + "..."
    return text


@dataclass
class CaseMemoryMatch:
    source_event_index: int
    similarity: float
    summary: dict

    def to_prompt_hint(self) -> dict:
        return {
            "source_event_index": self.source_event_index,
            "similarity": round(self.similarity, 4),
            "dataset": self.summary.get("dataset"),
            "event_type": self.summary.get("event_type"),
            "dominant_risk_phase": self.summary.get("dominant_risk_phase"),
            "max_physical_risk_index": self.summary.get(
                "max_physical_risk_index"
            ),
            "dominant_action": self.summary.get("dominant_action"),
            "planning_strategy": self.summary.get("planning_strategy"),
            "profile_name": self.summary.get("profile_name"),
        }


@dataclass
class CausalCaseMemory:
    enabled: bool = False
    top_k: int = 3
    min_similarity: float = 0.72
    novelty_threshold: float = 0.45
    records: list[dict] = field(default_factory=list)
    queries: int = 0
    hits: int = 0
    misses: int = 0
    novelty_frames: int = 0
    used_frames: int = 0
    similarity_sum: float = 0.0

    def retrieve(self, query: dict, current_event_index: int) -> dict:
        if not self.enabled:
            return self._empty_result("disabled")

        self.queries += 1
        candidates = [
            record
            for record in self.records
            if int(
                record.get("memory_order", record.get("event_index", -1))
            ) < int(current_event_index)
        ]
        scored = [
            CaseMemoryMatch(
                source_event_index=int(record["event_index"]),
                similarity=self._similarity(query, record),
                summary=record,
            )
            for record in candidates
        ]
        scored.sort(key=lambda item: item.similarity, reverse=True)
        matches = [
            item for item in scored[: max(1, int(self.top_k))]
            if item.similarity >= float(self.min_similarity)
        ]
        best_similarity = scored[0].similarity if scored else 0.0
        novelty_detected = best_similarity < float(self.novelty_threshold)

        if matches:
            self.hits += 1
            self.used_frames += 1
            self.similarity_sum += matches[0].similarity
        else:
            self.misses += 1
        if novelty_detected:
            self.novelty_frames += 1

        return {
            "enabled": True,
            "query_count": self.queries,
            "hit": bool(matches),
            "novelty_detected": bool(novelty_detected),
            "best_similarity": round(best_similarity, 4),
            "matches": [item.to_prompt_hint() for item in matches],
            "leakage_guard": "only_prior_events_no_ground_truth_labels",
        }

    def add_episode(
        self,
        *,
        event_index: int,
        memory_order: int | None = None,
        dataset: str,
        metadata: dict,
        profile_name: str,
        frame_safety_metrics: list,
        decisions: list[dict],
        planning_records: list[dict],
        risk_phase_counts: dict,
    ) -> dict | None:
        if not self.enabled:
            return None
        record = build_case_memory_record(
            event_index=event_index,
            memory_order=memory_order,
            dataset=dataset,
            metadata=metadata,
            profile_name=profile_name,
            frame_safety_metrics=frame_safety_metrics,
            decisions=decisions,
            planning_records=planning_records,
            risk_phase_counts=risk_phase_counts,
        )
        self.records.append(record)
        return record

    def stats(self) -> dict:
        return {
            "enabled": bool(self.enabled),
            "records": len(self.records),
            "queries": self.queries,
            "hits": self.hits,
            "misses": self.misses,
            "used_frames": self.used_frames,
            "novelty_frames": self.novelty_frames,
            "hit_rate": self.hits / self.queries if self.queries else 0.0,
            "novelty_rate": (
                self.novelty_frames / self.queries if self.queries else 0.0
            ),
            "avg_hit_similarity": (
                self.similarity_sum / self.hits if self.hits else 0.0
            ),
            "leakage_guard": "memory_updated_after_episode_only_no_labels",
        }

    def _empty_result(self, reason: str) -> dict:
        return {
            "enabled": bool(self.enabled),
            "hit": False,
            "novelty_detected": False,
            "best_similarity": 0.0,
            "matches": [],
            "reason": reason,
            "leakage_guard": "only_prior_events_no_ground_truth_labels",
        }

    def _similarity(self, query: dict, record: dict) -> float:
        categorical = [
            ("dataset", 0.12),
            ("event_type", 0.12),
            ("pair_type", 0.08),
            ("dominant_risk_phase", 0.16),
            ("profile_name", 0.08),
        ]
        score = 0.0
        weight_sum = 0.0
        for key, weight in categorical:
            weight_sum += weight
            left = str(query.get(key, "") or "").lower()
            right = str(record.get(key, "") or "").lower()
            if left and right and left == right:
                score += weight

        numeric = [
            ("physical_risk_index", "max_physical_risk_index", 0.20, 1.0),
            ("ego_speed_mps", "avg_ego_speed_mps", 0.10, 35.0),
            ("headway_m", "avg_headway_m", 0.08, 80.0),
            ("ttc_s", "min_ttc_s", 0.08, 8.0),
            ("dcpa_m", "min_dcpa_m", 0.08, 8.0),
        ]
        for query_key, record_key, weight, scale in numeric:
            weight_sum += weight
            left = _safe_float(query.get(query_key), None)
            right = _safe_float(record.get(record_key), None)
            if left is None or right is None:
                continue
            score += weight * max(0.0, 1.0 - abs(left - right) / scale)

        return max(0.0, min(1.0, score / weight_sum if weight_sum else 0.0))


def build_case_memory_query(
    *,
    event_index: int,
    memory_order: int | None = None,
    dataset: str,
    metadata: dict,
    scene,
    frame_safety,
    risk_phase: str,
    profile_name: str,
) -> dict:
    return {
        "event_index": int(event_index),
        "memory_order": (
            int(memory_order) if memory_order is not None else int(event_index)
        ),
        "dataset": dataset,
        "event_type": metadata.get("eventType") or metadata.get("event_type") or "",
        "pair_type": metadata.get("pair_type") or "",
        "dominant_risk_phase": risk_phase,
        "profile_name": profile_name,
        "physical_risk_index": getattr(frame_safety, "physical_risk_index", None),
        "ego_speed_mps": getattr(scene, "ego_speed_mps", None),
        "headway_m": getattr(scene, "headway_m", None),
        "ttc_s": getattr(frame_safety, "ttc_s", None),
        "dcpa_m": getattr(frame_safety, "dcpa_m", None),
    }


def build_case_memory_record(
    *,
    event_index: int,
    memory_order: int | None = None,
    dataset: str,
    metadata: dict,
    profile_name: str,
    frame_safety_metrics: list,
    decisions: list[dict],
    planning_records: list[dict],
    risk_phase_counts: dict,
) -> dict:
    risk_indexes = [
        _safe_float(getattr(item, "physical_risk_index", None), None)
        for item in frame_safety_metrics
    ]
    risk_indexes = [value for value in risk_indexes if value is not None]
    ttc_values = [
        _safe_float(getattr(item, "ttc_s", None), None)
        for item in frame_safety_metrics
    ]
    ttc_values = [value for value in ttc_values if value is not None]
    dcpa_values = [
        _safe_float(getattr(item, "dcpa_m", None), None)
        for item in frame_safety_metrics
    ]
    dcpa_values = [value for value in dcpa_values if value is not None]
    actions = [
        str(decision.get("recommended_action", "") or "")
        for decision in decisions
        if isinstance(decision, dict)
    ]
    dominant_action = _dominant_value(actions)
    strategies = []
    for item in planning_records:
        planning = item.get("planning") if isinstance(item, dict) else {}
        strategy = (
            planning.get("recommended_strategy", {}).get("strategy")
            if isinstance(planning, dict)
            else ""
        )
        if strategy:
            strategies.append(str(strategy))

    record = {
        "event_index": int(event_index),
        "memory_order": (
            int(memory_order) if memory_order is not None else int(event_index)
        ),
        "dataset": dataset,
        "event_type": metadata.get("eventType") or metadata.get("event_type") or "",
        "pair_type": metadata.get("pair_type") or "",
        "profile_name": profile_name,
        "dominant_risk_phase": _dominant_value(risk_phase_counts),
        "max_physical_risk_index": max(risk_indexes) if risk_indexes else None,
        "avg_physical_risk_index": (
            sum(risk_indexes) / len(risk_indexes) if risk_indexes else None
        ),
        "min_ttc_s": min(ttc_values) if ttc_values else None,
        "min_dcpa_m": min(dcpa_values) if dcpa_values else None,
        "avg_ego_speed_mps": _avg_scene_value(decisions, "ego_speed_mps"),
        "avg_headway_m": _avg_scene_value(decisions, "headway_m"),
        "dominant_action": _compact_text(dominant_action),
        "planning_strategy": _compact_text(_dominant_value(strategies)),
        "risk_phase_counts": dict(risk_phase_counts or {}),
        "leakage_guard": "no_dataset_risk_label_no_episode_prediction",
    }
    return json.loads(json.dumps(record, ensure_ascii=False))


def case_memory_hint_text(memory_result: dict, max_chars: int = 900) -> str:
    if not memory_result.get("hit"):
        return ""
    payload = {
        "causal_case_memory": {
            "matches": memory_result.get("matches", []),
            "leakage_guard": memory_result.get("leakage_guard"),
        }
    }
    text = json.dumps(payload, ensure_ascii=False, sort_keys=True)
    if len(text) > max_chars:
        text = text[:max_chars] + "..."
    return text


def _dominant_value(values):
    if isinstance(values, dict):
        values = [
            key for key, count in values.items()
            for _ in range(max(0, int(count or 0)))
        ]
    counts = {}
    for value in values or []:
        if value in (None, ""):
            continue
        key = str(value)
        counts[key] = counts.get(key, 0) + 1
    if not counts:
        return ""
    return max(counts.items(), key=lambda item: (item[1], item[0]))[0]


def _avg_scene_value(decisions: list[dict], key: str):
    values = [
        _safe_float(decision.get(key, decision.get(f"_{key}")), None)
        for decision in decisions
        if isinstance(decision, dict)
    ]
    values = [value for value in values if value is not None and math.isfinite(value)]
    return sum(values) / len(values) if values else None
