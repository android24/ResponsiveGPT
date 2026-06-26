import json
from pathlib import Path


def _load_registry(config: dict, config_path: Path) -> None:
    registry_path = config.get("dataset_registry_path")
    if not registry_path:
        default_path = config_path.parent / "datasets.json"
        registry_path = default_path if default_path.exists() else None

    registry = {}
    if registry_path:
        registry_path = Path(registry_path)
        if not registry_path.is_absolute():
            if not registry_path.exists():
                registry_path = config_path.parent / registry_path
        with registry_path.open("r", encoding="utf-8") as f:
            loaded = json.load(f)
        if not isinstance(loaded, dict):
            raise ValueError(f"Dataset registry must be a JSON object: {registry_path}")
        registry.update(loaded)
        config["dataset_registry_path"] = str(registry_path)

    inline_registry = config.get("dataset_registry", {}) or {}
    if not isinstance(inline_registry, dict):
        raise ValueError("'dataset_registry' must be a JSON object when provided.")
    registry.update(inline_registry)

    if registry:
        config["dataset_registry"] = registry


def load_config(path: str | Path) -> dict:
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as f:
        config = json.load(f)

    if not isinstance(config, dict):
        raise ValueError(f"Experiment config must be a JSON object: {config_path}")

    if not config.get("name"):
        raise ValueError("Experiment config must define a non-empty 'name'.")

    _load_registry(config, config_path)

    return config
