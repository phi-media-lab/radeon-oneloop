#!/usr/bin/env bash
set -euo pipefail

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
python_bin=${ONELOOP_ROCM_PYTHON:-/home/amd/.venvs/rocm-probe-312/bin/python}
run_root=${ONELOOP_LIVE_RUN_ROOT:-$repo_root/runs}
source_run=${ONELOOP_MGPBD_BUNNY_SOURCE_RUN:?set ONELOOP_MGPBD_BUNNY_SOURCE_RUN}
seed=${ONELOOP_MGPBD_BUNNY_SEED:-20260806}
run_id="$(date -u +%Y%m%dT%H%M%SZ)_${BASHPID}_amd_genesis_mgpbd_bunny_bridge"
run_dir=$run_root/$run_id
mkdir -p "$run_dir/artifacts" "$run_dir/xdg-runtime"
chmod 700 "$run_dir/xdg-runtime"

finalize() {
  status=$?
  trap - EXIT INT TERM
  if [[ -d $run_dir ]]; then
    (
      cd "$run_dir"
      find . -type f ! -name hashes.sha256 ! -name DONE ! -name FAILED \
        -print0 | sort -z | xargs -0 sha256sum >hashes.sha256
    ) || status=1
    if [[ $status -eq 0 ]]; then
      touch "$run_dir/DONE"
    else
      touch "$run_dir/FAILED"
    fi
  fi
  exit "$status"
}
trap finalize EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

exec 9>/tmp/radeon-oneloop-gpu0.lock
if ! flock -n 9; then
  printf '%s\n' 'GPU0 is locked by another Radeon OneLoop job' >&2
  exit 75
fi

runner_sha=$(sha256sum "$repo_root/sim/genesis_so101/mgpbd_bunny_genesis_smoke.py" | cut -d' ' -f1)
source_checkpoint_sha=$(sha256sum "$source_run/artifacts/last_safe_state.npz" | cut -d' ' -f1)
printf '%s\n' \
  'schema_version: radeon_oneloop.amd_genesis_mgpbd_bunny_bridge_run.v1' \
  'formal: false' \
  'host_role: amd_apu_nonformal_genesis_mgpbd_bridge' \
  'physical_robot_output: false' \
  'physical_leader_read: false' \
  'contact_enabled: false' \
  'gravity_enabled: false' \
  'integration_enabled: false' \
  'physics_owner: custom_MGPBD_checkpoint' \
  'Genesis_role: custom_vverts_boundary_renderer' \
  "source_run: $(basename "$source_run")" \
  "source_checkpoint_sha256: $source_checkpoint_sha" \
  "runner_sha256: $runner_sha" \
  "seed: $seed" \
  >"$run_dir/manifest.yaml"

export PYTHONPATH="$repo_root/src:$repo_root"
export LD_LIBRARY_PATH="/opt/rocm-7.2.1/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export XDG_RUNTIME_DIR="$run_dir/xdg-runtime"
command=(
  "$python_bin" -m sim.genesis_so101.mgpbd_bunny_genesis_smoke
  --source-run "$source_run"
  --output "$run_dir/artifacts"
  --seed "$seed"
)
printf '%q ' "${command[@]}" >"$run_dir/command.txt"
printf '\n' >>"$run_dir/command.txt"
"${command[@]}" >"$run_dir/stdout.log" 2>"$run_dir/stderr.log"

if "$python_bin" - "$run_dir/artifacts/metrics.json" <<'PY'
import json
import sys
from pathlib import Path

payload = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
raise SystemExit(0 if payload.get("passed") is True else 1)
PY
then
  touch "$run_dir/GATE_PASSED"
else
  touch "$run_dir/GATE_FAILED"
  exit 1
fi

printf 'AMD Genesis MGPBD bunny bridge passed: %s\n' "$run_dir"
