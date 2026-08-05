#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 4 ]]; then
  printf 'usage: %s LEFT_PORT RIGHT_PORT LEFT_CALIBRATION_ID RIGHT_CALIBRATION_ID\n' "$0" >&2
  exit 64
fi

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
: "${ONELOOP_OBSERVED_CORE_ROOT:?set ONELOOP_OBSERVED_CORE_ROOT to the content-verified asset directory}"

export ONELOOP_SHOW_PRESENTER=${ONELOOP_SHOW_PRESENTER:-1}
# The authoritative Genesis viewer contains only the physics/debug scene and
# must not be confused with the separate Gaussian presenter.
export ONELOOP_SHOW_GENESIS_VIEWER=${ONELOOP_SHOW_GENESIS_VIEWER:-0}
export ONELOOP_RENDER_HZ=${ONELOOP_RENDER_HZ:-8}
export ONELOOP_LIVE_DURATION_S=${ONELOOP_LIVE_DURATION_S:-1800}
export ONELOOP_LIVE_TIMEOUT_S=${ONELOOP_LIVE_TIMEOUT_S:-1860}
export ONELOOP_RECORD_VIDEO=${ONELOOP_RECORD_VIDEO:-0}
export ONELOOP_LIVE_CANDIDATE_NONFORMAL=${ONELOOP_LIVE_CANDIDATE_NONFORMAL:-0}
export ONELOOP_FAULT_EXIT_AFTER_FRAMES=0

exec "$repo_root/ops/run_amd_decoupled_gaussian_live_gate.sh" "$@"
