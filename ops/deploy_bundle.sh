#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  printf 'usage: %s <ssh-host>\n' "$0" >&2
  exit 64
fi
host=$1
repo_root=$(git rev-parse --show-toplevel)
cd "$repo_root"
if [[ -n "$(git status --porcelain)" ]]; then
  printf '%s\n' 'refusing to deploy a dirty worktree; commit the exact source first' >&2
  exit 65
fi
commit=$(git rev-parse HEAD)
bundle=$(mktemp "${TMPDIR:-/tmp}/radeon-oneloop.${commit:0:12}.XXXXXX.bundle")
trap 'rm -f "$bundle"' EXIT
git bundle create "$bundle" HEAD
remote_base=/root/radeon-oneloop
remote_src="$remote_base/src/$commit"
remote_bundle="$remote_base/bundles/$commit.bundle"
ssh "$host" "mkdir -p '$remote_base/src' '$remote_base/bundles'"
scp -q "$bundle" "$host:$remote_bundle"
ssh "$host" "set -e; if [[ ! -d '$remote_src/.git' ]]; then git clone -q '$remote_bundle' '$remote_src'; fi; [[ \$(git -C '$remote_src' rev-parse HEAD) == '$commit' ]]; ln -sfn '$remote_src' '$remote_base/current'; printf '%s\\n' '$commit'"

