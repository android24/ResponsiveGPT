from pathlib import Path


def resolve_dataset_config(config: dict, dataset_name: str) -> dict:
    datasets = config.get("datasets", {})
    if isinstance(datasets, list):
        datasets = {name: {} for name in datasets}

    if dataset_name not in datasets:
        raise KeyError(f"Dataset '{dataset_name}' is not defined in experiment config.")

    dataset_cfg = dict(datasets.get(dataset_name) or {})
    registry = config.get("dataset_registry", {})
    if dataset_name in registry:
        base = dict(registry[dataset_name])
        base.update(dataset_cfg)
        dataset_cfg = base

    required = ["summary_csv", "sequence_root"]
    missing = [key for key in required if not dataset_cfg.get(key)]
    if missing:
        raise ValueError(f"Dataset '{dataset_name}' missing required keys: {missing}")

    return dataset_cfg


def path_exists(path: str) -> bool:
    return Path(path).exists()

