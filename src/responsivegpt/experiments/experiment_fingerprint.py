import csv
import hashlib
import importlib.metadata
import json
import os
import platform
import random
from functools import lru_cache
from pathlib import Path


METHOD_VERSION = "responsivegpt_bsse_v17"
PROJECT_ROOT = Path(__file__).resolve().parents[3]

METHOD_GLOBS = [
    "src/responsivegpt/application/**/*.py",
    "src/responsivegpt/domain/**/*.py",
    "src/responsivegpt/rag/**/*.py",
    "src/responsivegpt/evaluation/**/*.py",
    "src/responsivegpt/infrastructure/*.py",
    "src/responsivegpt/interface/runner_core.py",
    "src/responsivegpt/interface/llm_call_policy.py",
    "src/responsivegpt/interface/experiment_builder.py",
    "src/responsivegpt/interface/run_ablation.py",
    "src/responsivegpt/interface/adapters/**/*.py",
    "src/responsivegpt/experiments/config_loader.py",
    "src/responsivegpt/experiments/dataset_registry.py",
    "src/responsivegpt/experiments/experiment_fingerprint.py",
    "src/responsivegpt/experiments/experiment_matrix.py",
    "src/responsivegpt/experiments/io_utils.py",
    "src/responsivegpt/experiments/job_spec.py",
    "src/responsivegpt/experiments/run_matrix.py",
    "src/responsivegpt/experiments/stratified_sampler.py",
]


@lru_cache(maxsize=None)
def _file_sha256(path: str | Path) -> str:
    path = Path(path)
    if not path.exists() or not path.is_file():
        return ""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_json_sha256(value) -> str:
    canonical = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _json_file_canonical_sha256(path: str | Path) -> str:
    path = Path(path)
    if not path.exists() or not path.is_file():
        return ""
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return ""
    return _canonical_json_sha256(value)


def _model_environment() -> dict[str, str]:
    defaults = {
        "PRIMARY_MODEL": "gpt-5.2",
        "FALLBACK_MODEL": "gpt-4.1",
        "CHEAP_MODEL": "gpt-4o-mini",
        "JIEKOU_BASE_URL": "https://api.jiekou.ai/openai",
        "OLLAMA_BASE_URL": "http://localhost:11434",
        "OLLAMA_EMBED_MODEL": "nomic-embed-text",
        "KB_DIR": "",
        "LLM_MAX_COMPLETION_TOKENS": "2048",
        "LLM_TIMEOUT_S": "120",
        "LLM_MAX_RETRIES": "1",
    }
    values = dict(defaults)
    env_path = PROJECT_ROOT / ".env"
    if env_path.exists():
        for line in env_path.read_text(encoding="utf-8").splitlines():
            if "=" not in line or line.lstrip().startswith("#"):
                continue
            key, value = line.split("=", 1)
            if key.strip() in defaults:
                values[key.strip()] = value.strip()
    for key in defaults:
        if os.environ.get(key):
            values[key] = os.environ[key]
    return values


def _project_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def _display_path(path: Path) -> str:
    return (
        str(path.relative_to(PROJECT_ROOT))
        if path.is_relative_to(PROJECT_ROOT)
        else str(path)
    )


def _resolve_sequence_path(root: Path, ref: str) -> Path:
    ref_path = Path(str(ref).replace("\\", "/"))
    if ref_path.is_absolute():
        return ref_path
    candidates = [root / ref_path]
    if len(ref_path.parts) > 1:
        candidates.append(root.joinpath(*ref_path.parts[1:]))
    candidates.append(root / ref_path.name)
    return next((path for path in candidates if path.exists()), candidates[0])


def _selected_sequence_rows(rows: list[dict], job: dict) -> list[dict]:
    extra_args = job.get("extra_args", {}) or {}
    start_index = int(extra_args.get("start_index", 0) or 0)
    end_index = int(extra_args.get("end_index", -1) or -1)
    shard_id = int(extra_args.get("shard_id", -1) or -1)
    num_shards = int(extra_args.get("num_shards", 0) or 0)
    episode_order_seed = int(extra_args.get("episode_order_seed", 0) or 0)
    limit = int(job.get("limit", 0) or 0)
    shard_enabled = shard_id >= 0 and num_shards > 0

    selected = []
    for index, row in enumerate(rows):
        if index < start_index or (end_index >= 0 and index >= end_index):
            continue
        if shard_enabled and (index % num_shards) != shard_id:
            continue
        selected.append(row)
    if episode_order_seed:
        random.Random(episode_order_seed).shuffle(selected)

    profile_protocol_enabled = bool(
        int(extra_args.get("profile_protocol_enabled", 0) or 0)
    )
    if limit > 0 and not profile_protocol_enabled:
        selected = selected[:limit]
    return selected


def _sequence_hashes(job: dict) -> dict[str, str]:
    summary_path = _project_path(str(job.get("summary_csv", "")))
    root = _project_path(str(job.get("sequence_root", "")))
    if not summary_path.exists():
        return {}
    ref_field = {
        "highd": "clipPath",
        "ind": "scene_file",
        "round": "clip_file",
    }.get(str(job.get("dataset", "")).lower())
    if not ref_field:
        return {}
    hashes = {}
    with summary_path.open("r", encoding="utf-8-sig", newline="") as stream:
        rows = _selected_sequence_rows(list(csv.DictReader(stream)), job)
        for row in rows:
            ref = row.get(ref_field)
            if not ref:
                continue
            path = _resolve_sequence_path(root, ref)
            hashes[_display_path(path)] = _file_sha256(path)
    return hashes


def _resolved_kb_dir(model_environment: dict[str, str]) -> Path | None:
    candidates = []
    configured = str(model_environment.get("KB_DIR", "") or "").strip()
    if configured:
        candidates.append(_project_path(configured))
    candidates.extend([
        PROJECT_ROOT / "src/responsivegpt/data/kb",
        PROJECT_ROOT / "data/kb",
    ])
    return next((path for path in candidates if path.is_dir()), None)


def _kb_hashes(model_environment: dict[str, str]) -> dict[str, str]:
    kb_dir = _resolved_kb_dir(model_environment)
    if kb_dir is None:
        return {}
    return {
        _display_path(path): _file_sha256(path)
        for path in sorted(kb_dir.glob("*.json"))
    }


def _method_source_hashes() -> dict[str, str]:
    paths = set()
    for pattern in METHOD_GLOBS:
        paths.update(
            path
            for path in PROJECT_ROOT.glob(pattern)
            if path.is_file() and "__pycache__" not in path.parts
        )
    return {
        str(path.relative_to(PROJECT_ROOT)): _file_sha256(path)
        for path in sorted(paths, key=lambda item: str(item))
    }


def _runtime_dependencies() -> dict:
    packages = {}
    for name in ("openai", "requests", "pandas", "matplotlib", "httpx"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = "not-installed"
    return {
        "python": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "platform": platform.platform(),
        "packages": packages,
        "requirements_sha256": _file_sha256(
            PROJECT_ROOT / "requirements.txt"
        ),
    }


def build_job_fingerprint(job: dict) -> dict:
    model_environment = _model_environment()
    extra_args = job.get("extra_args", {}) or {}
    profiles_dir = extra_args.get(
        "profiles_dir", "src/responsivegpt/data/profiles"
    )
    payload = {
        "method_version": METHOD_VERSION,
        "job": {
            key: value
            for key, value in job.items()
            if key not in {"job_id", "tag", "experiment_name"}
        },
        "summary_csv_sha256": _file_sha256(
            _project_path(job.get("summary_csv", ""))
        ),
        "profile_sha256": _file_sha256(
            _project_path(profiles_dir)
            / f"{job.get('profile_name', '')}.json"
        ),
        "kb_sha256": _kb_hashes(model_environment),
        "model_environment": model_environment,
        "runtime_dependencies": _runtime_dependencies(),
        "sequence_sha256": _sequence_hashes(job),
        "method_source_sha256": _method_source_hashes(),
    }
    return {
        "method_version": METHOD_VERSION,
        "fingerprint": _canonical_json_sha256(payload),
        "components": payload,
    }


def legacy_result_is_compatible(job: dict) -> bool:
    # BSSE v4 changes shared scheduling, planning, budgeting, and statistical
    # behavior. Results without an exact fingerprint cannot be mixed into the
    # current method, including historical no-RAG and naive-RAG baselines.
    return False


def fingerprint_is_compatible(
    status: dict,
    expected_fingerprint: str | None,
) -> bool:
    if not expected_fingerprint:
        return True
    observed = status.get("experiment_fingerprint")
    if observed:
        return observed == expected_fingerprint
    return legacy_result_is_compatible(status.get("job", {}))


def expected_fingerprints_for_experiment(
    experiment_dir: str | Path,
) -> dict[str, str]:
    from .experiment_matrix import expand_jobs
    from .io_utils import load_json

    experiment_dir = Path(experiment_dir)
    fingerprint_manifest = experiment_dir / "job_fingerprints.json"
    if fingerprint_manifest.exists():
        try:
            cached = load_json(fingerprint_manifest)
        except Exception:
            cached = {}
        manifest_path = experiment_dir / "job_manifest.jsonl"
        snapshot_path = experiment_dir / "config.snapshot.json"
        manifest_matches = (
            not manifest_path.exists()
            or cached.get("job_manifest_sha256")
            == _file_sha256(manifest_path)
        )
        config_matches = (
            not snapshot_path.exists()
            or cached.get("config_sha256")
            == _json_file_canonical_sha256(snapshot_path)
        )
        if (
            cached.get("method_version") == METHOD_VERSION
            and manifest_matches
            and config_matches
        ):
            fingerprints = cached.get("job_fingerprints") or {}
            if isinstance(fingerprints, dict):
                return {
                    str(job_id): str(fingerprint)
                    for job_id, fingerprint in fingerprints.items()
                    if fingerprint
                }

    snapshot = experiment_dir / "config.snapshot.json"
    if not snapshot.exists():
        return {}
    jobs = expand_jobs(load_json(snapshot))
    return {
        job.job_id: build_job_fingerprint(job.to_dict())["fingerprint"]
        for job in jobs
    }
