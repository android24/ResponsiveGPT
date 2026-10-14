#!/bin/sh
set -eu

script_dir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
repo_root=$(dirname "$script_dir")

# This wrapper opts into NAS output without changing run_matrix.py. It runs from
# a small scratch directory where ./runs points to the mounted Synology share.
# The regular run_matrix command still uses the repository-local ./runs path.
nas_runs_root="${RESPONSIVEGPT_NAS_RUNS_ROOT:-/Volumes/共享/论文/Papers/202606/ResponsiveGPT/runs}"
nas_mount_root="${RESPONSIVEGPT_NAS_MOUNT_ROOT:-/Volumes/共享}"
runtime_root="${RESPONSIVEGPT_NAS_RUNTIME_ROOT:-/private/tmp/responsivegpt-nas-runtime}"
keepalive_interval="${RESPONSIVEGPT_NAS_KEEPALIVE_INTERVAL:-60}"
python_bin="${RESPONSIVEGPT_PYTHON:-}"

if [ -z "$python_bin" ]; then
    if [ -x /opt/anaconda3/bin/python3 ]; then
        python_bin=/opt/anaconda3/bin/python3
    else
        python_bin=python3
    fi
fi

if [ "$#" -lt 1 ]; then
    echo "Usage: $0 <matrix-config.json> [run_matrix options]" >&2
    exit 2
fi

if [ ! -d "$nas_runs_root" ]; then
    echo "NAS runs directory is unavailable: $nas_runs_root" >&2
    echo "Mount the Synology shared folder before starting the experiment." >&2
    exit 1
fi

if ! mount | grep -F " on $nas_mount_root " >/dev/null 2>&1; then
    echo "NAS share is not mounted at: $nas_mount_root" >&2
    exit 1
fi

if [ ! -w "$nas_runs_root" ]; then
    echo "NAS runs directory is not writable: $nas_runs_root" >&2
    exit 1
fi

config_path="$1"
shift

case "$config_path" in
    "$repo_root"/*) config_path=${config_path#"$repo_root"/} ;;
esac

mkdir -p "$runtime_root"
chmod 700 "$runtime_root"
ln -sfn "$repo_root/src" "$runtime_root/src"
ln -sfn "$repo_root/data" "$runtime_root/data"
ln -sfn "$nas_runs_root" "$runtime_root/runs"

# Experiment credential loading is cwd-relative. Expose the local private
# configuration inside the user-only runtime directory without copying secrets
# to the NAS.
if [ -f "$repo_root/.env" ]; then
    ln -sfn "$repo_root/.env" "$runtime_root/.env"
fi
if [ -d "$repo_root/config" ]; then
    ln -sfn "$repo_root/config" "$runtime_root/config"
fi

if [ ! -f "$runtime_root/.env" ] && \
   [ ! -f "$runtime_root/config/accounts.local.json" ] && \
   [ -z "${JIEKOU_API_KEY:-}" ]; then
    echo "Missing local LLM credentials (.env, accounts.local.json, or JIEKOU_API_KEY)." >&2
    exit 1
fi

cd "$runtime_root"

export RESPONSIVEGPT_NAS_RUNS_ROOT="$nas_runs_root"
export RESPONSIVEGPT_NAS_MOUNT_ROOT="$nas_mount_root"

keepalive_file="$nas_runs_root/.responsivegpt_smb_keepalive"
keepalive_loop() {
    while :; do
        if ! date -u '+%Y-%m-%dT%H:%M:%SZ' > "$keepalive_file"; then
            echo "[WARN] NAS keepalive write failed: $keepalive_file" >&2
        fi
        sleep "$keepalive_interval"
    done
}

keepalive_loop &
keepalive_pid=$!
cleanup_keepalive() {
    kill "$keepalive_pid" 2>/dev/null || true
    wait "$keepalive_pid" 2>/dev/null || true
}
trap cleanup_keepalive EXIT HUP INT TERM

PYTHONPATH="$repo_root/src${PYTHONPATH:+:$PYTHONPATH}" \
"$python_bin" "$repo_root/scripts/run_matrix_nas.py" \
    --config "$config_path" "$@"
