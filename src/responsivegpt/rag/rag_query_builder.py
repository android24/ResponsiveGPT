from .scenario_inference import infer_scenario_type, dataset_metric_priorities


def build_rag_query(
    *,
    dataset: str,
    scene,
    frame_safety=None,
    metadata=None,
    driver_type: str = "",
    feedback: str = "",
    planning_hint: str = "",
    rag_mode: str = "scenario_metric_aware",
) -> dict:
    metadata = metadata or {}

    event_type = (
        getattr(scene, "event_type", None)
        or metadata.get("eventType")
        or metadata.get("pair_type")
        or ""
    )
    pair_type = metadata.get("pair_type", "")

    scenario_type = infer_scenario_type(
        dataset=dataset,
        event_type=event_type,
        pair_type=pair_type,
        vrus_present=bool(getattr(scene, "vrus_present", False)),
    )

    priorities = dataset_metric_priorities(dataset, scenario_type)

    metric_terms = []
    risk_terms = []

    if frame_safety is not None:
        if getattr(frame_safety, "unsafe_ttc", False):
            metric_terms.append("TTC")
            risk_terms.append("low time to collision")
        if getattr(frame_safety, "unsafe_thw", False):
            metric_terms.append("THW")
            risk_terms.append("short headway")
        if getattr(frame_safety, "unsafe_drac", False):
            metric_terms.append("DRAC")
            risk_terms.append("high required deceleration")
        if getattr(frame_safety, "unsafe_dcpa", False):
            metric_terms.append("DCPA")
            risk_terms.append("small closest approach distance")
        if getattr(frame_safety, "unsafe_future_distance", False):
            metric_terms.append("minimum future distance")
            risk_terms.append("predicted spatial conflict")

        risk_index = getattr(frame_safety, "physical_risk_index", None)
        if risk_index is not None:
            risk_terms.append(f"physical risk index {risk_index:.3f}")

    query_parts = []

    if rag_mode in {"dataset_aware", "scenario_metric_aware", "full"}:
        query_parts.append(f"dataset {dataset}")
        query_parts.append(f"environment scenario {scenario_type}")

    if rag_mode in {"scenario_metric_aware", "full"}:
        query_parts.append(f"event type {event_type}")
        query_parts.append("primary metrics " + " ".join(priorities["primary_metrics"]))
        query_parts.append("risk focus " + " ".join(priorities["risk_focus"]))
        query_parts.append("observed unsafe metrics " + " ".join(metric_terms))
        query_parts.append("risk factors " + " ".join(risk_terms))

    if rag_mode == "full":
        query_parts.append(f"driver type {driver_type}")
        query_parts.append(f"human feedback {feedback}")
        if planning_hint:
            query_parts.append(f"planning hint {planning_hint}")

    if rag_mode == "naive":
        query_parts = [
            f"{dataset} {event_type} risk traffic rule safety decision"
        ]

    query_text = " | ".join([x for x in query_parts if x.strip()])

    return {
        "rag_mode": rag_mode,
        "query_text": query_text,
        "dataset": dataset,
        "scenario_type": scenario_type,
        "event_type": event_type,
        "pair_type": pair_type,
        "metric_terms": metric_terms,
        "risk_terms": risk_terms,
        "primary_metrics": priorities["primary_metrics"],
        "secondary_metrics": priorities["secondary_metrics"],
        "risk_focus": priorities["risk_focus"],
    }