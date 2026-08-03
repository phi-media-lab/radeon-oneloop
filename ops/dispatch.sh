#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 7 ]]; then
  printf 'usage: %s <host> <role> <config-relative> <formal:true|false> <dataset-hash-or-null> -- <command...>\n' "$0" >&2
  exit 64
fi
host=$1
role=$2
config=$3
formal=$4
dataset_hash=$5
shift 5
[[ $1 == -- ]]
shift
repo_root=$(git rev-parse --show-toplevel)
commit=$(git -C "$repo_root" rev-parse HEAD)
remote_repo="/root/radeon-oneloop/src/$commit"

quote() { printf '%q' "$1"; }
remote_command="$(quote "$remote_repo/ops/run_job.sh") $(quote "$role") $(quote "$remote_repo/$config") $(quote "$formal") $(quote "$dataset_hash") --"
for argument in "$@"; do
  remote_command+=" $(quote "$argument")"
done
if [[ $formal == true ]]; then
  remote_command="ONELOOP_FORMAL_HOST=radeon-c $remote_command"
else
  remote_command="ONELOOP_FORMAL_HOST=$host $remote_command"
fi
ssh "$host" "$remote_command"
