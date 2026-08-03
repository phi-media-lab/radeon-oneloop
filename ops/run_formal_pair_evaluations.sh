#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 4 ]]; then
  printf 'usage: %s <baseline-checkpoint-dir> <baseline-sha256> <phase-checkpoint-dir> <phase-sha256>\n' "$0" >&2
  exit 64
fi

baseline_checkpoint=$1
baseline_sha=$2
phase_checkpoint=$3
phase_sha=$4
for checkpoint in "$baseline_checkpoint" "$phase_checkpoint"; do
  [[ $checkpoint == /* ]] || {
    printf 'checkpoint path must be absolute on the formal host: %s\n' "$checkpoint" >&2
    exit 64
  }
done
for digest in "$baseline_sha" "$phase_sha"; do
  [[ $digest =~ ^[0-9a-f]{64}$ ]] || {
    printf 'checkpoint digest must be lowercase SHA-256: %s\n' "$digest" >&2
    exit 64
  }
done

repo_root=$(git rev-parse --show-toplevel)
host=${ONELOOP_EVAL_HOST:-radeon-c}
python_bin=/root/radeon-oneloop-env/rocm721-py312/bin/python
dataset_root=/root/radeon-oneloop-data/formal_handover_v1
dataset_sha=ba18dd207ffd00c562a7ad18c831508d0529cd4d8d7b478a9b2f6d46618489cf

run_latency() {
  local checkpoint=$1
  local digest=$2
  local config=$3
  ONELOOP_PARENT_CHECKPOINT="$digest" \
    "$repo_root/ops/dispatch.sh" "$host" act_eval "$config" true "$dataset_sha" -- \
    "$python_bin" -m evaluation.policy_latency \
    --checkpoint "$checkpoint" \
    --dataset-root "$dataset_root" \
    --frame-index 0 --warmup 20 --iterations 200
}

run_reconstruction() {
  local checkpoint=$1
  local digest=$2
  local config=$3
  ONELOOP_PARENT_CHECKPOINT="$digest" \
    "$repo_root/ops/dispatch.sh" "$host" act_eval "$config" true "$dataset_sha" -- \
    "$python_bin" -m evaluation.action_reconstruction \
    --checkpoint "$checkpoint" \
    --dataset-root "$dataset_root" \
    --samples-per-role 256 --batch-size 16 --num-workers 4
}

run_latency "$baseline_checkpoint" "$baseline_sha" configs/act_baseline.yaml
run_latency "$phase_checkpoint" "$phase_sha" configs/act_phase_aware.yaml
run_reconstruction "$baseline_checkpoint" "$baseline_sha" configs/act_baseline.yaml
run_reconstruction "$phase_checkpoint" "$phase_sha" configs/act_phase_aware.yaml
