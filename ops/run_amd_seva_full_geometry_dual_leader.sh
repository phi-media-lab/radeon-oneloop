#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 4 ]]; then
  printf 'usage: %s LEFT_PORT RIGHT_PORT LEFT_CALIBRATION_ID RIGHT_CALIBRATION_ID\n' "$0" >&2
  exit 64
fi

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
if [[ ${ONELOOP_PROJECT_OWNER_VISUAL_CONFIRMATION:-} != accepted ]]; then
  printf '%s\n' 'set ONELOOP_PROJECT_OWNER_VISUAL_CONFIRMATION=accepted only after reviewing the new live asset' >&2
  exit 65
fi
: "${ONELOOP_OBSERVED_CORE_ROOT:?set ONELOOP_OBSERVED_CORE_ROOT to the reviewed full-geometry asset}"

expected_ply_sha256=ad538d0f1d4da96293aed7de5f9f33030435870c1c4339187f48c9dfa25bb4f2
actual_ply_sha256=$(sha256sum "$ONELOOP_OBSERVED_CORE_ROOT/appearance_full_geometry_canonical.ply" | awk '{print $1}')
if [[ "$actual_ply_sha256" != "$expected_ply_sha256" ]]; then
  printf 'unexpected full-geometry PLY hash: %s\n' "$actual_ply_sha256" >&2
  exit 66
fi

export ONELOOP_FULL_GEOMETRY_CANDIDATE=1
export ONELOOP_COMPLETED_APPEARANCE=0
export ONELOOP_GENERATED_FILL_ENABLED=0
export ONELOOP_LIVE_CANDIDATE_NONFORMAL=0
export ONELOOP_FINAL_TASK_RECORDING=0
export ONELOOP_SHOW_PRESENTER=${ONELOOP_SHOW_PRESENTER:-1}
export ONELOOP_SHOW_GENESIS_VIEWER=0
export ONELOOP_RECORD_VIDEO=1
export ONELOOP_RENDER_HZ=${ONELOOP_RENDER_HZ:-8}
export ONELOOP_LIVE_DURATION_S=${ONELOOP_LIVE_DURATION_S:-300}
export ONELOOP_LIVE_TIMEOUT_S=${ONELOOP_LIVE_TIMEOUT_S:-360}
export ONELOOP_LIVE_RUN_ROOT=${ONELOOP_LIVE_RUN_ROOT:-/home/amd/radeon-oneloop-runs/seva_full_geometry_dual_leader}
export ONELOOP_FAULT_EXIT_AFTER_FRAMES=0

exec "$repo_root/ops/run_amd_decoupled_gaussian_live_gate.sh" "$@"
