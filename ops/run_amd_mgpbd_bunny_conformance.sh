#!/usr/bin/env bash
set -euo pipefail

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
python_bin=${ONELOOP_ROCM_PYTHON:-/home/amd/.venvs/rocm-probe-312/bin/python}
run_root=${ONELOOP_LIVE_RUN_ROOT:-$repo_root/runs}
model=${ONELOOP_MGPBD_BUNNY_MODEL:-bunny_small}
mode=${ONELOOP_MGPBD_BUNNY_MODE:-projection}
contract=${ONELOOP_MGPBD_BUNNY_CONTRACT:-official_fidelity}
profile=${ONELOOP_MGPBD_BUNNY_PROFILE:-public_matrix_ua}
timeout_s=${ONELOOP_MGPBD_BUNNY_TIMEOUT_S:-7200}
seed=${ONELOOP_MGPBD_BUNNY_SEED:-20260806}
initial_height_ratio=${ONELOOP_MGPBD_BUNNY_INITIAL_HEIGHT_RATIO:-0.25}
numerical_dtype=${ONELOOP_MGPBD_NUMERICAL_DTYPE:-float32}
soc_admm_beta=${ONELOOP_MGPBD_SOC_ADMM_BETA:-1e-4}
soc_admm_maximum_iterations=${ONELOOP_MGPBD_SOC_ADMM_MAXIMUM_ITERATIONS:-2000}
soc_admm_pcg_maximum_iterations=${ONELOOP_MGPBD_SOC_ADMM_PCG_MAXIMUM_ITERATIONS:-2000}
soc_admm_pcg_relative_tolerance=${ONELOOP_MGPBD_SOC_ADMM_PCG_RELATIVE_TOLERANCE:-1.5e-5}
soc_admm_required_consecutive_gate_passes=${ONELOOP_MGPBD_SOC_ADMM_REQUIRED_CONSECUTIVE_GATE_PASSES:-2}
if [[ -n ${ONELOOP_MGPBD_DIRECT_LINEAR_ORACLE:-} ]]; then
  direct_linear_oracle=$ONELOOP_MGPBD_DIRECT_LINEAR_ORACLE
elif [[ $model == bunny_small && $mode == projection ]]; then
  direct_linear_oracle=true
else
  direct_linear_oracle=false
fi
if [[ -n ${ONELOOP_MGPBD_BUNNY_FRAMES:-} ]]; then
  frames=$ONELOOP_MGPBD_BUNNY_FRAMES
elif [[ $mode == projection ]]; then
  frames=1
else
  frames=100
fi
reference_commit=06761eb38dee8fb4165c6b9df8212c4f1744d131
scene_sha=59caf188ea939cae78be1ffb885c2a31197c06267bdda95c977ff6910e5882d7
case "$model" in
  bunny_small)
    node_sha=d5c4f1d0af593074920180e7a6088fcf3e6edd083726aa8c644b3157f994a89d
    element_sha=cffdc6168764863049f1bf729e5f02ba1de7af348a2528ed16a90e9baed6ae7e
    face_sha=8a8e5b6529f3eee316a4efc8f3fe8dbbe2073aac1f5c54db4fadc77f2cf6ca73
    ;;
  bunnyBig)
    node_sha=98298f7f1310d5bc75d01a0941cdeec0e760ca42b5841e8bac392ff2ad854543
    element_sha=8a6d6cd0f52b1e7f1c2db61c380f34b78fad4fbff6ac3d3b009e5e031b49bccf
    face_sha=
    ;;
  *)
    printf 'Unsupported bunny model: %s\n' "$model" >&2
    exit 64
    ;;
esac

run_id="$(date -u +%Y%m%dT%H%M%SZ)_${BASHPID}_amd_mgpbd_${model}_${contract}_${mode}"
run_dir=$run_root/$run_id
reference_root=$run_dir/inputs/mgpbd-$reference_commit
model_dir=$reference_root/data/model/$model
scene_dir=$reference_root/data/scene/bunny_squash
mkdir -p "$run_dir/artifacts" "$model_dir" "$scene_dir" "$run_dir/xdg-runtime"
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

raw_root=https://raw.githubusercontent.com/chunleili/mgpbd/$reference_commit
curl --fail --location --retry 3 --silent --show-error \
  "$raw_root/data/model/$model/$model.node" \
  --output "$model_dir/$model.node"
curl --fail --location --retry 3 --silent --show-error \
  "$raw_root/data/model/$model/$model.ele" \
  --output "$model_dir/$model.ele"
if [[ -n $face_sha ]]; then
  curl --fail --location --retry 3 --silent --show-error \
    "$raw_root/data/model/$model/$model.face" \
    --output "$model_dir/$model.face"
fi
curl --fail --location --retry 3 --silent --show-error \
  "$raw_root/data/scene/bunny_squash/bunny_squash.json" \
  --output "$scene_dir/bunny_squash.json"

printf '%s  %s\n' "$node_sha" "$model_dir/$model.node" | sha256sum --check --status
printf '%s  %s\n' "$element_sha" "$model_dir/$model.ele" | sha256sum --check --status
if [[ -n $face_sha ]]; then
  printf '%s  %s\n' "$face_sha" "$model_dir/$model.face" | sha256sum --check --status
fi
printf '%s  %s\n' "$scene_sha" "$scene_dir/bunny_squash.json" \
  | sha256sum --check --status

printf '%s\n' \
  'schema_version: radeon_oneloop.amd_mgpbd_bunny_conformance_run.v3' \
  'formal: false' \
  'host_role: amd_apu_nonformal_mgpbd_reference_conformance' \
  'physical_robot_output: false' \
  'physical_leader_read: false' \
  'genesis_enabled: false' \
  'contact_enabled: false' \
  'hardware_output_enabled: false' \
  'isolated_direction_smoke: false' \
  "reference_commit: $reference_commit" \
  "model: $model" \
  "mode: $mode" \
  "contract: $contract" \
  "profile: $profile" \
  "frames: $frames" \
  "seed: $seed" \
  "initial_height_ratio: $initial_height_ratio" \
  "numerical_dtype: $numerical_dtype" \
  "direct_linear_oracle: $direct_linear_oracle" \
  "soc_admm_beta: $soc_admm_beta" \
  "soc_admm_maximum_iterations: $soc_admm_maximum_iterations" \
  "soc_admm_pcg_maximum_iterations: $soc_admm_pcg_maximum_iterations" \
  "soc_admm_pcg_relative_tolerance: $soc_admm_pcg_relative_tolerance" \
  "soc_admm_required_consecutive_gate_passes: $soc_admm_required_consecutive_gate_passes" \
  "node_sha256: $node_sha" \
  "element_sha256: $element_sha" \
  "scene_sha256: $scene_sha" \
  >"$run_dir/manifest.yaml"

if repo_commit_detected=$(git -C "$repo_root" rev-parse HEAD 2>/dev/null); then
  repo_commit=${ONELOOP_SOURCE_REPO_COMMIT:-$repo_commit_detected}
  repo_vcs_available=true
  if [[ -n $(git -C "$repo_root" status --porcelain 2>/dev/null) ]]; then
    repo_dirty=true
  else
    repo_dirty=false
  fi
else
  repo_commit=${ONELOOP_SOURCE_REPO_COMMIT:-unavailable}
  repo_vcs_available=false
  repo_dirty=unknown
fi
runner_sha=$(sha256sum "$repo_root/sim/genesis_so101/mgpbd_bunny_conformance.py" | cut -d' ' -f1)
projector_sha=$(sha256sum "$repo_root/sim/genesis_so101/mgpbd_tet.py" | cut -d' ' -f1)
reference_io_sha=$(sha256sum "$repo_root/sim/genesis_so101/mgpbd_reference_io.py" | cut -d' ' -f1)
soc_admm_sha=$(sha256sum "$repo_root/sim/genesis_so101/mgpbd_soc_admm.py" | cut -d' ' -f1)
wrapper_sha=$(sha256sum "$repo_root/ops/run_amd_mgpbd_bunny_conformance.sh" | cut -d' ' -f1)
printf '%s\n' \
  "repo_commit: $repo_commit" \
  "repo_vcs_available: $repo_vcs_available" \
  "repo_dirty: $repo_dirty" \
  "runner_sha256: $runner_sha" \
  "projector_sha256: $projector_sha" \
  "reference_io_sha256: $reference_io_sha" \
  "soc_admm_sha256: $soc_admm_sha" \
  "wrapper_sha256: $wrapper_sha" \
  >>"$run_dir/manifest.yaml"

export PYTHONPATH="$repo_root/src:$repo_root"
export LD_LIBRARY_PATH="/opt/rocm-7.2.1/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export XDG_RUNTIME_DIR="$run_dir/xdg-runtime"
"$python_bin" - <<'PY' >"$run_dir/environment.json"
import json
import platform
import numpy
import pyamg
import scipy
import torch

if torch.version.hip is None:
    raise RuntimeError("AMD conformance runner requires a ROCm/HIP Torch build")

print(json.dumps({
    "platform": platform.platform(),
    "python": platform.python_version(),
    "torch": torch.__version__,
    "hip": torch.version.hip,
    "numpy": numpy.__version__,
    "scipy": scipy.__version__,
    "pyamg": pyamg.__version__,
    "device": torch.cuda.get_device_name(0),
    "device_properties": str(torch.cuda.get_device_properties(0)),
}, indent=2, sort_keys=True))
PY

command=(
  "$python_bin" -m sim.genesis_so101.mgpbd_bunny_conformance
  --reference-root "$reference_root" \
  --model "$model" \
  --mode "$mode" \
  --contract "$contract" \
  --frames "$frames" \
  --profile "$profile" \
  --seed "$seed" \
  --initial-height-ratio "$initial_height_ratio" \
  --numerical-dtype "$numerical_dtype" \
  --device cuda \
  --output "$run_dir/artifacts"
)
if [[ $profile == orientation_safe_soc_matrix_free ]]; then
  command+=(
    --soc-admm-beta "$soc_admm_beta"
    --soc-admm-maximum-iterations "$soc_admm_maximum_iterations"
    --soc-admm-pcg-maximum-iterations "$soc_admm_pcg_maximum_iterations"
    --soc-admm-pcg-relative-tolerance "$soc_admm_pcg_relative_tolerance"
    --soc-admm-required-consecutive-gate-passes "$soc_admm_required_consecutive_gate_passes"
  )
fi
if [[ $direct_linear_oracle == true ]]; then
  command+=(--direct-linear-oracle)
fi
printf '%q ' "${command[@]}" >"$run_dir/command.txt"
printf '\n' >>"$run_dir/command.txt"

timeout --signal=TERM --kill-after=30 "$timeout_s" \
  "${command[@]}" \
  >"$run_dir/stdout.log" 2>"$run_dir/stderr.log"

if "$python_bin" - "$run_dir/artifacts/metrics.json" <<'PY'
import json
import sys
from pathlib import Path

payload = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
raise SystemExit(0 if payload.get("passed") is True else 1)
PY
then
  touch "$run_dir/GATE_PASSED"
  gate_status=passed
else
  touch "$run_dir/GATE_FAILED"
  gate_status=failed
fi

printf 'AMD MGPBD bunny experiment completed (gate %s): %s\n' \
  "$gate_status" "$run_dir"
