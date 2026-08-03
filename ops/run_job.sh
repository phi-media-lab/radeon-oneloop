#!/usr/bin/env bash
set -uo pipefail

usage() {
  printf 'usage: %s <role> <config> <formal:true|false> <dataset-hash-or-null> -- <command...>\n' "$0" >&2
  exit 64
}
[[ $# -ge 6 ]] || usage
role=$1
config=$2
formal=$3
dataset_hash=$4
shift 4
[[ ${1:-} == -- ]] || usage
shift
[[ $# -gt 0 ]] || usage
[[ $formal == true || $formal == false ]] || usage
[[ -f $config ]] || { printf 'missing config: %s\n' "$config" >&2; exit 66; }

repo_root=$(git -C "$(dirname "$0")" rev-parse --show-toplevel)
commit=$(git -C "$repo_root" rev-parse HEAD)
cd "$repo_root"
export PYTHONPATH="$repo_root/src:$repo_root${PYTHONPATH:+:$PYTHONPATH}"
config_hash=$(sha256sum "$config" | awk '{print $1}')
seed=${ONELOOP_SEED:-20260803}
run_root=${ONELOOP_RUN_ROOT:-/root/radeon-oneloop-runs/jobs}
run_id="$(date -u +%Y%m%dT%H%M%SZ)_${role}_${commit:0:7}_${seed}"
run_dir="$run_root/$run_id"
mkdir -p "$run_root"
if ! mkdir "$run_dir" 2>/dev/null; then
  printf 'run directory already exists: %s\n' "$run_dir" >&2
  exit 73
fi
export ONELOOP_RUN_DIR="$run_dir"

exec 9>/tmp/radeon-oneloop-gpu0.lock
if ! flock -n 9; then
  printf '%s\n' 'GPU0 is locked by another Radeon OneLoop job' >&2
  exit 75
fi

python_bin=${ONELOOP_PYTHON:-/root/radeon-oneloop-env/rocm721-py312/bin/python}
hardware_json=$("$repo_root/ops/assert_single_radeon.sh" gfx1100 "$python_bin") || exit $?
gpu_uid=$(rocm-smi --showuniqueid --csv 2>/dev/null | awk -F, 'NR>1 {gsub(/[[:space:]]/, "", $2); if ($2 != "") {print $2; exit}}')
if [[ -z $gpu_uid ]]; then
  gpu_uid=$(rocm-smi --showuniqueid 2>/dev/null | awk '/Unique ID/ {print $NF; exit}')
fi
if [[ $formal == true ]]; then
  [[ ${ONELOOP_FORMAL_HOST:-} == radeon-c ]] || {
    printf '%s\n' 'formal jobs require ONELOOP_FORMAL_HOST=radeon-c' >&2
    exit 78
  }
  [[ $gpu_uid == 0x153f7d55778ab659 ]] || {
    printf 'formal GPU UID mismatch: %s\n' "$gpu_uid" >&2
    exit 78
  }
fi

printf '%s\n' "$hardware_json" > "$run_dir/hardware.json"
printf '%q ' "$@" > "$run_dir/command.sh"
printf '\n' >> "$run_dir/command.sh"
chmod +x "$run_dir/command.sh"
"$python_bin" -m pip freeze | sort > "$run_dir/environment.txt"
cp "$config" "$run_dir/config.yaml"

started=$(date -u +%Y-%m-%dT%H:%M:%SZ)
"$python_bin" - "$run_dir/manifest.json" "$run_id" "${ONELOOP_FORMAL_HOST:-shadow}" "$formal" "$role" "$commit" "$config_hash" "$dataset_hash" "$seed" "$gpu_uid" "$run_dir" "$started" <<'PY'
import json, sys
(
    path, job_id, host, formal, role, commit, config_hash, dataset_hash,
    seed, gpu_uid, artifact_dir, started,
) = sys.argv[1:]
value = {
    "job_id": job_id,
    "host": host,
    "formal": formal == "true",
    "role": role,
    "git_commit": commit,
    "config_hash": config_hash,
    "dataset_hash": None if dataset_hash == "null" else dataset_hash,
    "parent_checkpoint": None,
    "seed": int(seed),
    "gpu_uid": gpu_uid or None,
    "gfx_target": "gfx1100",
    "status": "running",
    "started_at": started,
    "finished_at": None,
    "artifact_dir": artifact_dir,
    "notes": None,
}
open(path, "w").write(json.dumps(value, indent=2) + "\n")
PY

set +e
sampler_stop="$run_dir/.sampler_stop"
(
  printf 'timestamp_utc\tdevice_sample\n'
  while [[ ! -e $sampler_stop ]]; do
    printf '%s\t' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    rocm-smi --showmemuse --showuse --csv 2>/dev/null | tail -n +2 | tr '\n' ';'
    printf '\n'
    sleep 1
  done
) > "$run_dir/gpu_samples.tsv" &
sampler_pid=$!
stop_sampler() {
  touch "$sampler_stop"
  wait "$sampler_pid" 2>/dev/null || true
}
trap stop_sampler EXIT INT TERM
"$@" > >(tee "$run_dir/stdout.log") 2> >(tee "$run_dir/stderr.log" >&2)
exit_code=$?
stop_sampler
trap - EXIT INT TERM
set -e
finished=$(date -u +%Y-%m-%dT%H:%M:%SZ)
status=failed
marker=FAILED
if [[ $exit_code -eq 0 ]]; then
  status=done
  marker=DONE
fi
"$python_bin" - "$run_dir/manifest.json" "$status" "$finished" <<'PY'
import json, sys
path, status, finished = sys.argv[1:]
value = json.load(open(path))
value["status"] = status
value["finished_at"] = finished
open(path, "w").write(json.dumps(value, indent=2) + "\n")
PY
if [[ -n ${ONELOOP_METRICS_PATH:-} && -f ${ONELOOP_METRICS_PATH} ]]; then
  cp "$ONELOOP_METRICS_PATH" "$run_dir/metrics.json"
elif [[ ! -f $run_dir/metrics.json ]]; then
  printf '{"schema_version":"radeon_oneloop.empty_metrics.v1","exit_code":%d}\n' "$exit_code" > "$run_dir/metrics.json"
fi
find "$run_dir" -maxdepth 1 -type f ! -name hashes.sha256 -print0 \
  | sort -z | xargs -0 sha256sum > "$run_dir/hashes.sha256"
touch "$run_dir/$marker"
printf 'run_dir=%s\nstatus=%s\nexit_code=%d\n' "$run_dir" "$status" "$exit_code"
exit "$exit_code"
