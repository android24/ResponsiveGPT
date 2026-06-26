import hashlib
import importlib.metadata
import json
import platform
from pathlib import Path

from .io_utils import write_json


ANALYSIS_VERSION = "responsivegpt_analysis_v4"
PROJECT_ROOT = Path(__file__).resolve().parents[3]
ANALYSIS_FILES = (
    "src/responsivegpt/experiments/aggregate_runs.py",
    "src/responsivegpt/experiments/analysis_provenance.py",
    "src/responsivegpt/experiments/make_paper_tables.py",
    "src/responsivegpt/experiments/rag_evidence_audit.py",
    "src/responsivegpt/experiments/report_writer.py",
    "src/responsivegpt/experiments/statistical_tests.py",
    "src/responsivegpt/experiments/validate_runs.py",
    "src/responsivegpt/experiments/weighted_estimator.py",
)


def _file_sha256(path: str | Path) -> str:
    path = Path(path)
    if not path.is_file():
        return ""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _runtime_environment() -> dict:
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
    }


def write_analysis_provenance(
    experiment_dir: str | Path,
    statuses: list[dict],
) -> dict:
    experiment_dir = Path(experiment_dir)
    if not experiment_dir.is_absolute():
        experiment_dir = PROJECT_ROOT / experiment_dir
    analysis_hashes = {
        path: _file_sha256(PROJECT_ROOT / path)
        for path in ANALYSIS_FILES
    }
    missing_analysis_files = [
        path for path, digest in analysis_hashes.items() if not digest
    ]
    if missing_analysis_files:
        raise FileNotFoundError(
            "Analysis provenance source files missing: "
            + ", ".join(missing_analysis_files)
        )
    inputs = []
    for status in sorted(
        statuses, key=lambda item: str(item.get("job_id", ""))
    ):
        run_dir = Path(str(status.get("run_dir", "")))
        if not run_dir.is_absolute():
            run_dir = PROJECT_ROOT / run_dir
        inputs.append({
            "job_id": status.get("job_id"),
            "run_dir": str(run_dir),
            "experiment_fingerprint": status.get(
                "experiment_fingerprint", ""
            ),
            "summary_sha256": _file_sha256(run_dir / "summary.json"),
            "episode_summary_sha256": _file_sha256(
                run_dir / "episode_summary.jsonl"
            ),
            "checkpoint_sha256": _file_sha256(
                run_dir / "episode_checkpoint.json"
            ),
        })
    output_hashes = {}
    for path in sorted(experiment_dir.iterdir()):
        if (
            path.is_file()
            and path.name != "analysis_provenance.json"
            and path.suffix in {".csv", ".json", ".md"}
        ):
            output_hashes[path.name] = _file_sha256(path)
    payload = {
        "analysis_version": ANALYSIS_VERSION,
        "analysis_source_sha256": analysis_hashes,
        "config_snapshot_sha256": _file_sha256(
            experiment_dir / "config.snapshot.json"
        ),
        "job_manifest_sha256": _file_sha256(
            experiment_dir / "job_manifest.jsonl"
        ),
        "job_fingerprints_sha256": _file_sha256(
            experiment_dir / "job_fingerprints.json"
        ),
        "experiment_identity_sha256": _file_sha256(
            experiment_dir / "experiment_identity.json"
        ),
        "requirements_sha256": _file_sha256(
            PROJECT_ROOT / "requirements.txt"
        ),
        "runtime_environment": _runtime_environment(),
        "inputs": inputs,
        "outputs_sha256": output_hashes,
    }
    canonical = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    payload["analysis_fingerprint"] = hashlib.sha256(
        canonical.encode("utf-8")
    ).hexdigest()
    write_json(experiment_dir / "analysis_provenance.json", payload)
    return payload
