#!/usr/bin/env bash
set -euo pipefail

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
reviewed_root=${ONELOOP_M1_REVIEWED_ROOT:?set ONELOOP_M1_REVIEWED_ROOT to the reviewed four-view artifact}
run_root=${ONELOOP_FOUR_VIEW_INPUT_RUN_ROOT:-$repo_root/runs/four_view_generation_input}
python_bin=${ONELOOP_PYTHON:-python3}
run_id="$(date -u +%Y%m%dT%H%M%SZ)_${BASHPID}_phi_four_view_input"
run_dir="$run_root/$run_id"

mkdir -p "$run_root"

"$python_bin" -m gaussian.prepare_four_view_generation \
  --reviewed-root "$reviewed_root" \
  --output "$run_dir" \
  --image-size 576 \
  --target-frames 49 \
  --elevation-deg 0 \
  --camera-radius 2 \
  --horizontal-fov-deg 50

"$python_bin" -m gaussian.prepare_four_view_generation \
  --reviewed-root "$reviewed_root" \
  --output "$run_dir" \
  --validate-only >/dev/null

printf 'Four-view generation input complete: %s\n' "$run_dir"
