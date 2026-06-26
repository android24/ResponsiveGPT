import hashlib
from itertools import product

from .dataset_registry import resolve_dataset_config
from .job_spec import ExperimentJob


def _as_list(value, default=None):
    if value is None:
        return list(default or [])
    if isinstance(value, list):
        return value
    return [value]


def _stable_id(parts: list[str]) -> str:
    raw = "__".join(str(x) for x in parts)
    suffix = hashlib.sha1(raw.encode("utf-8")).hexdigest()[:8]
    safe = "".join(c if c.isalnum() or c in "-_." else "_" for c in raw)
    safe = safe[:88].strip("_")
    return f"{safe}__{suffix}"


def _variant_name_and_value(variant, *, value_key: str, default_value: str) -> tuple[str, str, dict]:
    if isinstance(variant, dict):
        value = str(variant.get(value_key, default_value))
        name = str(variant.get("name", value))
        return name, value, dict(variant.get("extra_args", {}) or {})
    value = str(variant if variant is not None else default_value)
    return value, value, {}


def _expand_dataset_shards(shard_cfg, dataset: str) -> list[dict]:
    if shard_cfg in (None, "", [], {}):
        return [{"name": "all"}]

    if isinstance(shard_cfg, int):
        if shard_cfg <= 1:
            return [{"name": "all"}]
        return [
            {
                "name": f"shard_{i:03d}_of_{shard_cfg:03d}",
                "shard_id": i,
                "num_shards": shard_cfg,
            }
            for i in range(shard_cfg)
        ]

    if isinstance(shard_cfg, dict):
        if dataset in shard_cfg:
            return _expand_dataset_shards(shard_cfg[dataset], dataset)
        if "num_shards" in shard_cfg:
            num_shards = int(shard_cfg.get("num_shards", 0) or 0)
            if num_shards <= 1:
                return [{"name": "all"}]
            return [
                {
                    "name": f"shard_{i:03d}_of_{num_shards:03d}",
                    "shard_id": i,
                    "num_shards": num_shards,
                }
                for i in range(num_shards)
            ]
        return [{"name": "all"}]

    if isinstance(shard_cfg, list):
        out = []
        for item in shard_cfg:
            if not isinstance(item, dict):
                continue
            item_dataset = item.get("dataset")
            if item_dataset and str(item_dataset) != str(dataset):
                continue
            shard = dict(item)
            shard.pop("dataset", None)
            shard.setdefault("name", f"shard_{shard.get('shard_id', 'x')}_of_{shard.get('num_shards', 'x')}")
            out.append(shard)
        return out or [{"name": "all"}]

    return [{"name": "all"}]


def expand_jobs(config: dict) -> list[ExperimentJob]:
    defaults = dict(config.get("defaults", {}) or {})
    matrix = dict(config.get("matrix", {}) or {})
    experiment_name = str(config["name"])

    configured_datasets = config.get("datasets", {})
    default_dataset_names = configured_datasets if isinstance(configured_datasets, list) else configured_datasets.keys()
    datasets = _as_list(matrix.get("datasets"), default_dataset_names)
    profiles = _as_list(matrix.get("profiles"), ["balanced"])
    rag_variants = _as_list(matrix.get("rag_variants"), [{"name": "full_rag", "rag_mode": "full", "use_retriever": 1}])
    planning_variants = _as_list(matrix.get("planning"), [{"name": "planning_off", "use_planning_thread": 0, "planning_mode": "off"}])
    llm_policy_variants = _as_list(matrix.get("llm_policies"), [{"name": "hybrid", "llm_policy": "hybrid"}])
    default_mode = str(defaults.get("mode", "episode"))
    mode_variants = _as_list(matrix.get("modes"), [default_mode])
    include_mode_in_id = "modes" in matrix

    jobs = []

    for dataset, mode_variant, profile_name, rag, planning, llm_policy in product(
        datasets,
        mode_variants,
        profiles,
        rag_variants,
        planning_variants,
        llm_policy_variants,
    ):
        dataset_cfg = resolve_dataset_config(config, dataset)
        shards = _expand_dataset_shards(matrix.get("shards"), str(dataset))

        for shard in shards:
            rag = dict(rag or {})
            planning = dict(planning or {})
            llm_policy = dict(llm_policy or {})
            shard = dict(shard or {})

            mode_name, mode, mode_extra_args = _variant_name_and_value(
                mode_variant,
                value_key="mode",
                default_value=default_mode,
            )
            rag_mode = str(rag.get("rag_mode", defaults.get("rag_mode", "full")))
            use_retriever = int(rag.get("use_retriever", 0 if rag_mode == "none" else 1))
            require_grounded = int(rag.get("require_grounded_decision", defaults.get("require_grounded_decision", 0)))

            planning_name = str(planning.get("name", "planning"))
            use_planning = int(planning.get("use_planning_thread", defaults.get("use_planning_thread", 0)))
            planning_mode = str(planning.get("planning_mode", "interval_risk" if use_planning else "off"))

            llm_name = str(llm_policy.get("name", llm_policy.get("llm_policy", "hybrid")))
            policy = str(llm_policy.get("llm_policy", defaults.get("llm_policy", "hybrid")))

            rag_name = str(rag.get("name", rag_mode))
            shard_name = str(shard.get("name", "all"))
            id_parts = [experiment_name, dataset]
            if include_mode_in_id:
                id_parts.append(mode_name)
            id_parts.extend([profile_name, rag_name, planning_name, llm_name])
            if shard_name != "all":
                id_parts.append(shard_name)
            job_id = _stable_id(id_parts)
            tag = _stable_id(id_parts)[:64]

            extra_args = {}
            extra_args.update(defaults.get("extra_args", {}) or {})
            extra_args.update(mode_extra_args)
            extra_args.update(rag.get("extra_args", {}) or {})
            extra_args.update(planning.get("extra_args", {}) or {})
            extra_args.update(llm_policy.get("extra_args", {}) or {})
            if "shard_id" in shard and "num_shards" in shard:
                extra_args["shard_id"] = int(shard["shard_id"])
                extra_args["num_shards"] = int(shard["num_shards"])
            if "start_index" in shard:
                extra_args["start_index"] = int(shard["start_index"])
            if "end_index" in shard:
                extra_args["end_index"] = int(shard["end_index"])

            feedback_by_profile = dict(
                defaults.get("feedback_by_profile", {}) or {}
            )
            feedback = str(
                feedback_by_profile.get(
                    str(profile_name),
                    defaults.get(
                        "feedback",
                        "优先安全，避免明显危险操作",
                    ),
                )
            )

            jobs.append(
                ExperimentJob(
                    job_id=job_id,
                    experiment_name=experiment_name,
                    dataset=str(dataset),
                    mode=mode,
                    summary_csv=str(dataset_cfg["summary_csv"]),
                    sequence_root=str(dataset_cfg["sequence_root"]),
                    profile_name=str(profile_name),
                    rag_variant=rag_name,
                    rag_mode=rag_mode,
                    use_retriever=use_retriever,
                    require_grounded_decision=require_grounded,
                    planning_variant=planning_name,
                    use_planning_thread=use_planning,
                    planning_mode=planning_mode,
                    llm_policy_variant=llm_name,
                    llm_policy=policy,
                    llm_stride=int(llm_policy.get("llm_stride", defaults.get("llm_stride", 5))),
                    llm_risk_threshold=float(llm_policy.get("llm_risk_threshold", defaults.get("llm_risk_threshold", 0.35))),
                    limit=int(defaults.get("limit", 0)),
                    model_role=str(defaults.get("model_role", "primary")),
                    feedback=feedback,
                    tag=tag,
                    extra_args=extra_args,
                )
            )

    return jobs
