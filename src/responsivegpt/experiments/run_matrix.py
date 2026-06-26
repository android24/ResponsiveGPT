import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path

from .aggregate_runs import aggregate_experiment
from .config_loader import load_config
from .experiment_matrix import expand_jobs
from .experiment_fingerprint import (
    METHOD_VERSION,
    build_job_fingerprint,
    fingerprint_is_compatible,
)
from .io_utils import append_jsonl, ensure_dir, load_json, read_jsonl
from .validate_runs import validate_run_dir


DEFAULT_CONFIG = "src/responsivegpt/experiments/configs/paper_cornercase_token_efficiency_smoke.json"


def _canonical_json_bytes(value) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temp_path.write_text(text, encoding="utf-8")
    os.replace(temp_path, path)


def _prepare_experiment_metadata(
    experiment_dir: Path,
    config: dict,
    jobs: list,
    fingerprints: dict[str, dict] | None = None,
) -> dict:
    snapshot_path = experiment_dir / "config.snapshot.json"
    manifest_path = experiment_dir / "job_manifest.jsonl"
    identity_path = experiment_dir / "experiment_identity.json"
    fingerprint_path = experiment_dir / "job_fingerprints.json"
    job_rows = [
        job.to_dict() if hasattr(job, "to_dict") else dict(job)
        for job in jobs
    ]

    if snapshot_path.exists():
        existing = load_json(snapshot_path)
        if _canonical_json_bytes(existing) != _canonical_json_bytes(config):
            raise ValueError(
                "Experiment name already exists with a different config: "
                f"{experiment_dir}. Use a new config name so incompatible "
                "runs cannot be mixed."
            )

    config_bytes = _canonical_json_bytes(config)
    manifest_bytes = b"\n".join(
        _canonical_json_bytes(row) for row in job_rows
    )
    if manifest_bytes:
        manifest_bytes += b"\n"
    identity = {
        "experiment_name": config.get("name"),
        "method_version": METHOD_VERSION,
        "config_sha256": hashlib.sha256(config_bytes).hexdigest(),
        "job_manifest_sha256": hashlib.sha256(
            manifest_bytes
        ).hexdigest(),
        "job_count": len(job_rows),
    }
    if fingerprints:
        fingerprint_manifest = {
            **identity,
            "job_fingerprints": {
                job_id: item["fingerprint"]
                for job_id, item in sorted(fingerprints.items())
            },
            "job_method_versions": {
                job_id: item["method_version"]
                for job_id, item in sorted(fingerprints.items())
            },
        }
    else:
        fingerprint_manifest = {
            **identity,
            "job_fingerprints": {},
            "job_method_versions": {},
        }
    _atomic_write_text(
        snapshot_path,
        json.dumps(config, ensure_ascii=False, indent=2) + "\n",
    )
    _atomic_write_text(
        manifest_path, manifest_bytes.decode("utf-8")
    )
    _atomic_write_text(
        identity_path,
        json.dumps(identity, ensure_ascii=False, indent=2) + "\n",
    )
    _atomic_write_text(
        fingerprint_path,
        json.dumps(
            fingerprint_manifest, ensure_ascii=False, indent=2
        )
        + "\n",
    )
    return identity


def _completed_job_ids(
    status_path: Path,
    expected_fingerprints: dict[str, str] | None = None,
) -> set[str]:
    expected_fingerprints = expected_fingerprints or {}
    completed = set()
    for row in read_jsonl(status_path):
        if row.get("status") != "completed":
            continue
        job_id = row.get("job_id")
        run_dir = row.get("run_dir")
        if not job_id or not run_dir:
            continue
        validation = validate_run_dir(run_dir, row.get("job", {}))
        if not validation.get("execution_valid", validation.get("valid", False)):
            continue
        if expected_fingerprints and str(job_id) not in expected_fingerprints:
            continue
        expected = expected_fingerprints.get(str(job_id))
        if not fingerprint_is_compatible(row, expected):
            continue
        completed.add(str(job_id))
    return completed


def _parse_run_dir(stdout: str) -> str:
    match = re.search(r"Run saved to:\s*(.+)", stdout)
    return match.group(1).strip() if match else ""


def _job_command(
    job,
    fingerprint: dict | None = None,
    *,
    resume_run_dir: str = "",
) -> list[str]:
    cmd = [
        sys.executable,
        "-m",
        "responsivegpt.interface.run_ablation",
        "--dataset",
        job.dataset,
        "--mode",
        job.mode,
        "--summary_csv",
        job.summary_csv,
        "--sequence_root",
        job.sequence_root,
        "--profile_name",
        job.profile_name,
        "--use_retriever",
        str(job.use_retriever),
        "--rag_mode",
        job.rag_mode,
        "--require_grounded_decision",
        str(job.require_grounded_decision),
        "--use_planning_thread",
        str(job.use_planning_thread),
        "--planning_mode",
        job.planning_mode,
        "--llm_policy",
        job.llm_policy,
        "--llm_stride",
        str(job.llm_stride),
        "--llm_risk_threshold",
        str(job.llm_risk_threshold),
        "--limit",
        str(job.limit),
        "--model_role",
        job.model_role,
        "--feedback",
        job.feedback,
        "--tag",
        job.tag,
    ]
    if fingerprint:
        cmd.extend([
            "--experiment_fingerprint",
            str(fingerprint["fingerprint"]),
            "--method_version",
            str(fingerprint["method_version"]),
        ])
    if resume_run_dir:
        cmd.extend(["--resume_run_dir", resume_run_dir])

    for key, value in sorted(job.extra_args.items()):
        flag = "--" + str(key)
        cmd.extend([flag, str(value)])

    return cmd


def _subprocess_env() -> dict:
    env = dict(os.environ)
    src_path = str(Path("src").resolve())
    existing = env.get("PYTHONPATH")
    env["PYTHONPATH"] = src_path if not existing else f"{src_path}{os.pathsep}{existing}"
    env.setdefault("MPLCONFIGDIR", "/private/tmp")
    env.setdefault("PYTHONUNBUFFERED", "1")
    return env


def _pid_is_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


@contextmanager
def _job_run_lock(run_dir: Path):
    lock_path = run_dir / ".job.lock"
    for _ in range(2):
        try:
            fd = os.open(
                lock_path,
                os.O_CREAT | os.O_EXCL | os.O_WRONLY,
                0o644,
            )
        except FileExistsError:
            try:
                lock_data = lock_path.read_text(
                    encoding="utf-8"
                ).strip()
                owner_pid = int(lock_data.splitlines()[0])
            except Exception:
                owner_pid = -1
            if _pid_is_alive(owner_pid):
                raise RuntimeError(
                    f"Job run directory is already locked by PID "
                    f"{owner_pid}: {run_dir}"
                )
            lock_path.unlink(missing_ok=True)
            continue
        else:
            with os.fdopen(fd, "w", encoding="utf-8") as stream:
                stream.write(
                    f"{os.getpid()}\n"
                    f"{datetime.now().isoformat(timespec='seconds')}\n"
                )
            break
    else:
        raise RuntimeError(f"Could not acquire job lock: {run_dir}")
    try:
        yield
    finally:
        lock_path.unlink(missing_ok=True)


def _run_job_streaming(cmd: list[str], stdout_path: Path, stderr_path: Path, timeout_s: int = 0) -> tuple[int, bool]:
    stdout_path.parent.mkdir(parents=True, exist_ok=True)
    stderr_path.parent.mkdir(parents=True, exist_ok=True)
    with stdout_path.open("w", encoding="utf-8") as stdout_f, stderr_path.open("w", encoding="utf-8") as stderr_f:
        proc = subprocess.Popen(
            cmd,
            cwd=Path.cwd(),
            env=_subprocess_env(),
            text=True,
            stdout=stdout_f,
            stderr=stderr_f,
        )
        try:
            return proc.wait(timeout=timeout_s if timeout_s > 0 else None), False
        except subprocess.TimeoutExpired:
            proc.kill()
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                pass
            return proc.returncode if proc.returncode is not None else -9, True


def run_matrix(config_path: str, *, resume: bool = True, dry_run: bool = False) -> Path:
    config = load_config(config_path)
    jobs = expand_jobs(config)
    fingerprints = {
        job.job_id: build_job_fingerprint(job.to_dict())
        for job in jobs
    }
    defaults = dict(config.get("defaults", {}) or {})
    job_timeout_s = int(defaults.get("job_timeout_s", 0) or 0)

    experiment_dir = ensure_dir(Path("runs") / "experiments" / str(config["name"]))
    stdout_dir = ensure_dir(experiment_dir / "stdout")
    stderr_dir = ensure_dir(experiment_dir / "stderr")
    job_runs_dir = ensure_dir(experiment_dir / "job_runs")
    status_path = experiment_dir / "job_status.jsonl"
    _prepare_experiment_metadata(
        experiment_dir, config, jobs, fingerprints
    )

    completed = (
        _completed_job_ids(
            status_path,
            {
                job_id: item["fingerprint"]
                for job_id, item in fingerprints.items()
            },
        )
        if resume
        else set()
    )

    for idx, job in enumerate(jobs, 1):
        fingerprint = fingerprints[job.job_id]
        resume_run_dir = ensure_dir(
            job_runs_dir
            / job.job_id
            / fingerprint["fingerprint"][:12]
        )
        cmd = _job_command(
            job,
            fingerprint,
            resume_run_dir=str(resume_run_dir),
        )
        command_text = " ".join(cmd)

        if job.job_id in completed:
            print(f"[SKIP] {idx}/{len(jobs)} {job.job_id}")
            continue

        print(f"[RUN] {idx}/{len(jobs)} {job.job_id}")
        append_jsonl(status_path, {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "job_id": job.job_id,
            "status": "started",
            "command": command_text,
            "experiment_fingerprint": fingerprint["fingerprint"],
            "method_version": fingerprint["method_version"],
            "run_dir": str(resume_run_dir),
            "job": job.to_dict(),
        })

        if dry_run:
            append_jsonl(status_path, {
                "timestamp": datetime.now().isoformat(timespec="seconds"),
                "job_id": job.job_id,
                "status": "dry_run",
                "command": command_text,
                "experiment_fingerprint": fingerprint["fingerprint"],
                "method_version": fingerprint["method_version"],
                "run_dir": str(resume_run_dir),
                "job": job.to_dict(),
            })
            print(command_text)
            continue

        stdout_path = stdout_dir / f"{job.job_id}.out"
        stderr_path = stderr_dir / f"{job.job_id}.err"
        try:
            with _job_run_lock(resume_run_dir):
                returncode, timed_out = _run_job_streaming(
                    cmd,
                    stdout_path,
                    stderr_path,
                    timeout_s=job_timeout_s,
                )
        except Exception as exc:
            returncode, timed_out = -2, False
            with stderr_path.open("a", encoding="utf-8") as stream:
                stream.write(str(exc) + "\n")

        stdout_text = stdout_path.read_text(encoding="utf-8") if stdout_path.exists() else ""
        run_dir = _parse_run_dir(stdout_text) or str(resume_run_dir)
        valid = False
        validation = {}
        if returncode == 0 and run_dir:
            validation = validate_run_dir(run_dir, job.to_dict())
            valid = bool(validation.get("execution_valid", validation.get("valid")))

        status = "completed" if returncode == 0 else "failed"
        append_jsonl(status_path, {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "job_id": job.job_id,
            "status": status,
            "returncode": returncode,
            "timed_out": timed_out,
            "run_dir": run_dir,
            "valid": valid,
            "execution_valid": valid,
            "quality_gate_pass": validation.get("quality_gate_pass"),
            "experiment_fingerprint": fingerprint["fingerprint"],
            "method_version": fingerprint["method_version"],
            "validation": validation,
            "stdout_path": str(stdout_path),
            "stderr_path": str(stderr_path),
            "job": job.to_dict(),
        })

        if returncode != 0:
            timeout_text = " timed_out=True" if timed_out else ""
            print(f"[FAIL] {job.job_id} returncode={returncode}{timeout_text}")
        elif not valid:
            print(f"[WARN] {job.job_id} finished but validation failed: {validation.get('failure_reasons')}")
        elif validation.get("quality_gate_pass") is False:
            print(
                f"[WARN] {job.job_id} completed with quality-gate findings: "
                f"{validation.get('quality_failures')}"
            )
        else:
            print(f"[OK] {job.job_id} -> {run_dir}")

    if not dry_run:
        aggregate_experiment(experiment_dir)

    return experiment_dir


def main():
    parser = argparse.ArgumentParser(description="Run a ResponsiveGPT experiment matrix.")
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--no_resume", action="store_true")
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()

    experiment_dir = run_matrix(
        args.config,
        resume=not args.no_resume,
        dry_run=args.dry_run,
    )
    print(f"Experiment output: {experiment_dir}")


if __name__ == "__main__":
    main()
