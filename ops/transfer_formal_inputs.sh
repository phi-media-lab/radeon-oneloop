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
hil_src=/mnt/models_alehe/phi-fbsh/so101_handover_hil_rl/datasets/fbsh96/so101_handover_hil_rlready_batch1_batch2_20260506

ssh "$destination" "mkdir -p '$dest_root'"
if ! ssh "$destination" "test -f '$dest_root/bc_seed/meta/info.json'"; then
  ssh amd "tar -C '$bc_src' -cf - data meta videos" \
    | ssh "$destination" "mkdir -p '$dest_root/bc_seed' && tar -C '$dest_root/bc_seed' -xf -"
fi
if ! ssh "$destination" "test -f '$dest_root/hil_batch1_batch2/meta/info.json'"; then
  ssh -o RemoteCommand=none -o RequestTTY=no phi-amd-work \
    "tar -C '$hil_src' -cf - data meta videos makermods_hil" \
    | ssh "$destination" "mkdir -p '$dest_root/hil_batch1_batch2' && tar -C '$dest_root/hil_batch1_batch2' -xf -"
fi

ssh "$destination" "set -e; test -f '$dest_root/bc_seed/data/chunk-000/file-000.parquet'; test -f '$dest_root/hil_batch1_batch2/data/chunk-000/file-000.parquet'; du -sh '$dest_root/bc_seed' '$dest_root/hil_batch1_batch2'"

