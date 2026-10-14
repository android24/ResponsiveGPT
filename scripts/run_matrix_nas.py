"""Run a matrix while isolating active jobs from transient SMB failures.

Matrix metadata remains on the NAS for compatibility. Each child job runs in a
local staging directory and is verified and atomically copied to the NAS only
after the child exits. This adapter lives outside ``src`` so existing scientific
method fingerprints remain unchanged.
"""

from __future__ import annotations

import errno
import fcntl
import hashlib
import json
import os
import shutil
import sys
import time
from contextlib import contextmanager
from pathlib import Path

from responsivegpt.experiments import aggregate_runs as aggregate_runner
from responsivegpt.experiments import run_matrix as matrix_runner


_TRANSIENT_NETWORK_ERRNOS = {
    errno.EBADF,
    errno.EIO,
    getattr(errno, "ESTALE", 70),
    getattr(errno, "ETIMEDOUT", 60),
    getattr(errno, "ENETDOWN", 50),
    getattr(errno, "ENETUNREACH", 51),
    getattr(errno, "ENOTCONN", 57),
}
_ORIGINAL_READ_TEXT = Path.read_text
_ORIGINAL_RUN_JOB_STREAMING = matrix_runner._run_job_streaming
_ORIGINAL_APPEND_JSONL = matrix_runner.append_jsonl
_ORIGINAL_ATOMIC_WRITE_TEXT = matrix_runner._atomic_write_text
_ORIGINAL_ENSURE_DIR = matrix_runner.ensure_dir
_ORIGINAL_LOAD_JSON = matrix_runner.load_json
_ORIGINAL_READ_JSONL = matrix_runner.read_jsonl
_ORIGINAL_VALIDATE_RUN_DIR = matrix_runner.validate_run_dir
_ORIGINAL_WRITE_ANALYSIS_PROVENANCE = (
    aggregate_runner.write_analysis_provenance
)


def _is_transient(exc: BaseException) -> bool:
    return isinstance(exc, OSError) and exc.errno in _TRANSIENT_NETWORK_ERRNOS


def _assert_nas_mounted() -> None:
    if os.environ.get("RESPONSIVEGPT_NAS_REQUIRE_MOUNT", "1") == "0":
        return
    mount_root = Path(
        os.environ.get("RESPONSIVEGPT_NAS_MOUNT_ROOT", "/Volumes/共享")
    )
    if not os.path.ismount(mount_root):
        raise OSError(errno.ENOTCONN, f"NAS mount is unavailable: {mount_root}")


def _retry_network(operation, description: str):
    """Retry NAS boundary I/O indefinitely unless a finite limit is requested."""
    limit = int(os.environ.get("RESPONSIVEGPT_NAS_RETRY_ATTEMPTS", "0"))
    max_delay = float(os.environ.get("RESPONSIVEGPT_NAS_RETRY_MAX_DELAY", "30"))
    attempt = 0
    while True:
        try:
            _assert_nas_mounted()
            return operation()
        except OSError as exc:
            if not _is_transient(exc) or (limit > 0 and attempt + 1 >= limit):
                raise
            attempt += 1
            delay = min(0.5 * (2 ** min(attempt - 1, 8)), max_delay)
            print(
                f"[WARN] NAS unavailable during {description}; retrying in "
                f"{delay:.1f}s (attempt {attempt})",
                file=sys.stderr,
                flush=True,
            )
            time.sleep(delay)


def _read_text_with_network_retry(self: Path, *args, **kwargs) -> str:
    return _retry_network(
        lambda: _ORIGINAL_READ_TEXT(self, *args, **kwargs), f"read {self}"
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _copy_file_verified(source: Path, destination: Path) -> None:
    """Commit one file atomically and verify the bytes stored on the NAS."""
    source_size = source.stat().st_size
    source_digest = _sha256(source)

    def copy_once() -> None:
        if (
            destination.is_file()
            and destination.stat().st_size == source_size
            and _sha256(destination) == source_digest
        ):
            return
        destination.parent.mkdir(parents=True, exist_ok=True)
        temp_path = destination.with_name(f".{destination.name}.{os.getpid()}.tmp")
        try:
            shutil.copy2(source, temp_path)
            if temp_path.stat().st_size != source_size:
                raise OSError(errno.EIO, f"size mismatch copying {destination}")
            if _sha256(temp_path) != source_digest:
                raise OSError(errno.EIO, f"checksum mismatch copying {destination}")
            os.replace(temp_path, destination)
        finally:
            try:
                temp_path.unlink(missing_ok=True)
            except OSError:
                pass

    _retry_network(copy_once, f"verified copy to {destination}")


def _sync_tree_verified(source_root: Path, destination_root: Path) -> None:
    for source in sorted(source_root.rglob("*")):
        relative = source.relative_to(source_root)
        destination = destination_root / relative
        if source.is_dir():
            _retry_network(
                lambda path=destination: path.mkdir(parents=True, exist_ok=True),
                f"create directory {destination}",
            )
        elif source.is_file():
            _copy_file_verified(source, destination)


def _hydrate_local_run(nas_run_dir: Path, local_run_dir: Path) -> None:
    # Retained local state may be newer after an interrupted transfer. Never
    # overwrite it with the older NAS checkpoint.
    if local_run_dir.exists() and any(local_run_dir.iterdir()):
        print(f"[RESUME-LOCAL] {local_run_dir}", flush=True)
        return
    local_run_dir.mkdir(parents=True, exist_ok=True)
    # run_matrix creates this directory before entering the adapter, including
    # for a fresh job. Treat disappearance as a network boundary failure.
    print(f"[HYDRATE] {nas_run_dir} -> {local_run_dir}", flush=True)
    _retry_network(
        lambda: shutil.copytree(nas_run_dir, local_run_dir, dirs_exist_ok=True),
        f"hydrate {nas_run_dir}",
    )
    (local_run_dir / ".job.lock").unlink(missing_ok=True)


def _replace_arg(cmd: list[str], flag: str, value: str) -> list[str]:
    rewritten = list(cmd)
    index = rewritten.index(flag)
    rewritten[index + 1] = value
    return rewritten


def _run_job_locally_then_sync(
    cmd: list[str],
    stdout_path: Path,
    stderr_path: Path,
    timeout_s: int = 0,
) -> tuple[int, bool]:
    if "--resume_run_dir" not in cmd:
        raise RuntimeError("NAS adapter requires --resume_run_dir")

    nas_run_dir = Path(cmd[cmd.index("--resume_run_dir") + 1])
    identity = hashlib.sha256(str(nas_run_dir).encode("utf-8")).hexdigest()[:20]
    staging_root = Path(
        os.environ.get(
            "RESPONSIVEGPT_NAS_JOB_STAGING_ROOT",
            "/private/tmp/responsivegpt-nas-jobs",
        )
    )
    job_root = staging_root / identity
    local_run_dir = job_root / "run"
    local_stdout = job_root / "stdout.out"
    local_stderr = job_root / "stderr.err"
    job_root.mkdir(parents=True, exist_ok=True)
    minimum_free_gb = float(os.environ.get("RESPONSIVEGPT_NAS_MIN_FREE_GB", "5"))
    free_bytes = shutil.disk_usage(staging_root).free
    if free_bytes < minimum_free_gb * 1024**3:
        raise RuntimeError(
            f"Local NAS staging has only {free_bytes / 1024**3:.1f} GiB free; "
            f"at least {minimum_free_gb:.1f} GiB is required"
        )

    _hydrate_local_run(nas_run_dir, local_run_dir)
    local_cmd = _replace_arg(cmd, "--resume_run_dir", str(local_run_dir))
    returncode, timed_out = _ORIGINAL_RUN_JOB_STREAMING(
        local_cmd, local_stdout, local_stderr, timeout_s=timeout_s
    )

    # Preserve failed-job checkpoints too. NAS loss may delay this boundary
    # sync, but it cannot interrupt the child computation.
    _sync_tree_verified(local_run_dir, nas_run_dir)

    stdout_text = local_stdout.read_text(encoding="utf-8")
    stdout_text = stdout_text.replace(str(local_run_dir), str(nas_run_dir))
    rewritten_stdout = job_root / "stdout.nas.out"
    rewritten_stdout.write_text(stdout_text, encoding="utf-8")
    _copy_file_verified(rewritten_stdout, stdout_path)
    _copy_file_verified(local_stderr, stderr_path)

    # The complete job is durable on NAS; local usage stays bounded to one job.
    shutil.rmtree(job_root)
    return returncode, timed_out


def _patch_matrix_boundary_io() -> None:
    Path.read_text = _read_text_with_network_retry
    matrix_runner._run_job_streaming = _run_job_locally_then_sync
    matrix_runner.append_jsonl = _append_jsonl_atomically
    matrix_runner._atomic_write_text = lambda path, value: _retry_network(
        lambda: _ORIGINAL_ATOMIC_WRITE_TEXT(path, value), f"write {path}"
    )
    matrix_runner.ensure_dir = lambda path: _retry_network(
        lambda: _ORIGINAL_ENSURE_DIR(path), f"create directory {path}"
    )
    matrix_runner.load_json = lambda path: _retry_network(
        lambda: _ORIGINAL_LOAD_JSON(path), f"load {path}"
    )
    matrix_runner.read_jsonl = lambda path: _retry_network(
        lambda: _ORIGINAL_READ_JSONL(path), f"read {path}"
    )
    matrix_runner.validate_run_dir = lambda run_dir, job: _retry_network(
        lambda: _ORIGINAL_VALIDATE_RUN_DIR(run_dir, job), f"validate {run_dir}"
    )
    matrix_runner._job_run_lock = _local_job_lock
    aggregate_runner.write_analysis_provenance = (
        _write_analysis_provenance_with_runtime_paths
    )


def _runtime_absolute(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else (Path.cwd() / path).resolve()


def _write_analysis_provenance_with_runtime_paths(
    experiment_dir: str | Path,
    statuses: list[dict],
) -> dict:
    """Keep NAS-relative status paths from being anchored to the source tree."""
    resolved_statuses = []
    for status in statuses:
        resolved = dict(status)
        run_dir = str(resolved.get("run_dir", "") or "")
        if run_dir:
            resolved["run_dir"] = str(_runtime_absolute(run_dir))
        resolved_statuses.append(resolved)
    resolved_experiment_dir = _runtime_absolute(experiment_dir)
    return _retry_network(
        lambda: _ORIGINAL_WRITE_ANALYSIS_PROVENANCE(
            resolved_experiment_dir,
            resolved_statuses,
        ),
        f"write analysis provenance for {resolved_experiment_dir}",
    )


def _assert_runtime_targets_nas() -> None:
    configured = os.environ.get("RESPONSIVEGPT_NAS_RUNS_ROOT", "").strip()
    if not configured:
        raise RuntimeError(
            "NAS adapter must be launched through scripts/run_matrix_nas.sh; "
            "RESPONSIVEGPT_NAS_RUNS_ROOT is not configured"
        )
    expected = Path(configured).resolve()
    actual = (Path.cwd() / "runs").resolve()
    if actual != expected:
        raise RuntimeError(
            "NAS adapter must be launched through scripts/run_matrix_nas.sh; "
            f"current ./runs resolves to {actual}, expected {expected}"
        )


def _append_jsonl_atomically(path, obj) -> None:
    path = Path(path)

    def commit() -> None:
        existing = _ORIGINAL_READ_TEXT(path, encoding="utf-8") if path.exists() else ""
        if existing and not existing.endswith("\n"):
            existing += "\n"
        value = existing + json.dumps(obj, ensure_ascii=False) + "\n"
        _ORIGINAL_ATOMIC_WRITE_TEXT(path, value)

    _retry_network(commit, f"append {path}")


@contextmanager
def _local_job_lock(run_dir: Path):
    lock_root = Path(os.environ.get("RESPONSIVEGPT_NAS_LOCK_ROOT", "/private/tmp"))
    lock_root.mkdir(parents=True, exist_ok=True)
    identity = hashlib.sha256(str(run_dir).encode("utf-8")).hexdigest()[:20]
    lock_path = lock_root / f"responsivegpt-nas-job-{identity}.lock"
    with lock_path.open("a+", encoding="utf-8") as stream:
        try:
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(
                f"This NAS job is already running (lock: {lock_path})"
            ) from exc
        try:
            yield
        finally:
            fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


@contextmanager
def _matrix_instance_lock():
    lock_root = Path(os.environ.get("RESPONSIVEGPT_NAS_LOCK_ROOT", "/private/tmp"))
    lock_root.mkdir(parents=True, exist_ok=True)
    try:
        config_index = sys.argv.index("--config") + 1
        matrix_identity = str(Path(sys.argv[config_index]).resolve())
    except (ValueError, IndexError):
        matrix_identity = "default"
    identity = hashlib.sha256(matrix_identity.encode("utf-8")).hexdigest()[:16]
    lock_path = lock_root / f"responsivegpt-nas-matrix-{identity}.lock"
    with lock_path.open("a+", encoding="utf-8") as stream:
        try:
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(
                f"This NAS matrix command is already running (lock: {lock_path})"
            ) from exc
        stream.seek(0)
        stream.truncate()
        stream.write(f"{os.getpid()}\n")
        stream.flush()
        try:
            yield
        finally:
            fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def main() -> None:
    _assert_nas_mounted()
    _assert_runtime_targets_nas()
    _patch_matrix_boundary_io()
    with _matrix_instance_lock():
        matrix_runner.main()


if __name__ == "__main__":
    main()
