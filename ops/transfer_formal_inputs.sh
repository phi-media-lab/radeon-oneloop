#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  printf 'usage: %s <radeon-f|radeon-c>\n' "$0" >&2
  exit 64
fi
destination=$1
case "$destination" in
  radeon-f|radeon-c) ;;
  *) printf 'unsupported destination: %s\n' "$destination" >&2; exit 64 ;;
esac

dest_root=/root/radeon-oneloop-data/sources
bc_src=/home/amd/.cache/huggingface/lerobot/fbsh96/so101-sock-ball-handover-v1
hil_src=/home/amd/.cache/huggingface/lerobot/fbsh96/so101_handover_hil_rlready_batch1_batch2_20260506

ssh "$destination" "mkdir -p '$dest_root'"
if ! ssh "$destination" "test -f '$dest_root/bc_seed/meta/info.json' && [[ \$(find '$dest_root/bc_seed/videos' -type f -name '*.mp4' 2>/dev/null | wc -l) -eq 3 ]]"; then
  ssh amd "tar -C '$bc_src' -cf - data meta videos" \
    | ssh "$destination" "mkdir -p '$dest_root/bc_seed' && tar -C '$dest_root/bc_seed' -xf -"
fi
if ! ssh "$destination" "test -f '$dest_root/hil_batch1_batch2/meta/info.json' && [[ \$(find '$dest_root/hil_batch1_batch2/videos' -type f -name '*.mp4' 2>/dev/null | wc -l) -eq 48 ]]"; then
  ssh amd "tar -C '$hil_src' -cf - data meta videos makermods_hil" \
    | ssh "$destination" "mkdir -p '$dest_root/hil_batch1_batch2' && tar -C '$dest_root/hil_batch1_batch2' -xf -"
fi

ssh "$destination" "set -e; \
  printf '%s  %s\n' 2e7db73c99f95bb7ff403f1f2ba630750dbc1bb07d4b52e2b300704bd220999b '$dest_root/bc_seed/meta/info.json' | sha256sum -c -; \
  printf '%s  %s\n' a15975d734013f3c45e9ec8869573ac09d52a8f6ae9e47b68ebd0bf28380a64f '$dest_root/bc_seed/data/chunk-000/file-000.parquet' | sha256sum -c -; \
  printf '%s  %s\n' e198927b1fc7f0d7566e5a4b622872ba3f2a0bafa58010f5d900441e6debcb3d '$dest_root/hil_batch1_batch2/meta/info.json' | sha256sum -c -; \
  printf '%s  %s\n' dafbbf6db47685ed433b7e2f4383191f7498b3ae037289f6c0fc7e77e1f0f88b '$dest_root/hil_batch1_batch2/data/chunk-000/file-000.parquet' | sha256sum -c -; \
  [[ \$(find '$dest_root/bc_seed/videos' -type f -name '*.mp4' | wc -l) -eq 3 ]]; \
  [[ \$(find '$dest_root/hil_batch1_batch2/videos' -type f -name '*.mp4' | wc -l) -eq 48 ]]; \
  test -f '$dest_root/hil_batch1_batch2/makermods_hil/combined_hil_batch1_batch2_phase_aware_awr_v2_20260507/handover_rl_seed_manifest_v0.jsonl'; \
  du -sh '$dest_root/bc_seed' '$dest_root/hil_batch1_batch2'"
