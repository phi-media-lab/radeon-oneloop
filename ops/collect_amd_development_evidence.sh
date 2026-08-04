#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 5 || $# -gt 6 ]]; then
  printf 'usage: %s <ssh-host> <absolute-remote-run-dir> <orbit|live> <public-label> <expected-ply-sha256> [visual-review]\n' "$0" >&2
  exit 64
fi

ssh_host=$1
remote_dir=${2%/}
mode=$3
label=$4
expected_ply_sha=$5
visual_review=${6:-}
repo_root=$(git rev-parse --show-toplevel)
destination="$repo_root/artifacts/development/$label"

[[ $remote_dir == /home/amd/radeon-oneloop-runs/* ]] || {
  printf 'remote run must be below /home/amd/radeon-oneloop-runs: %s\n' "$remote_dir" >&2
  exit 64
}
[[ $mode == orbit || $mode == live ]] || {
  printf 'unsupported development evidence mode: %s\n' "$mode" >&2
  exit 64
}
[[ $label =~ ^[a-z0-9][a-z0-9_-]*$ ]] || {
  printf 'invalid public label: %s\n' "$label" >&2
  exit 64
}
[[ $expected_ply_sha =~ ^[0-9a-f]{64}$ ]] || {
  printf '%s\n' 'expected PLY SHA-256 must be lowercase hexadecimal' >&2
  exit 64
}
[[ ! -e $destination ]] || {
  printf 'destination exists; evidence collection never overwrites: %s\n' "$destination" >&2
  exit 73
}
if [[ $mode == orbit && -z $visual_review ]]; then
  printf '%s\n' 'orbit evidence requires a human visual-review label' >&2
  exit 64
fi

ssh "$ssh_host" "set -e
test -f '$remote_dir/DONE'
test ! -e '$remote_dir/FAILED'
cd '$remote_dir'
sha256sum -c hashes.sha256 >/dev/null"

temporary=$(mktemp -d /tmp/radeon-oneloop-development-evidence.XXXXXX)
source_dir="$temporary/$(basename "$remote_dir")"
if [[ $mode == orbit ]]; then
  files=(
    DONE
    manifest.yaml
    hashes.sha256
    artifacts/metrics.json
    artifacts/orbit_contact_sheet.png
    artifacts/orbit_360.mp4
  )
else
  files=(
    DONE
    manifest.yaml
    hashes.sha256
    gate.json
    consumer/metrics.json
    renderer/READY
    renderer/metrics.json
    renderer/live_gaussian_first.png
    renderer/live_gaussian_final.png
    renderer/live_gaussian.mp4
    publisher.log
  )
fi

for relative in "${files[@]}"; do
  mkdir -p "$source_dir/$(dirname "$relative")"
  scp -q "$ssh_host:$remote_dir/$relative" "$source_dir/$relative"
done

PYTHONPATH="$repo_root/src:$repo_root" "$repo_root/.venv/bin/python" \
  -m gaussian.development_evidence \
  --source "$source_dir" \
  --mode "$mode" \
  --expected-ply-sha256 "$expected_ply_sha" \
  --visual-review "$visual_review" \
  --output "$source_dir/summary.json" \
  >"$temporary/validation.log"

mkdir -p "$destination"
public_files=(DONE manifest.yaml summary.json)
if [[ $mode == orbit ]]; then
  public_files+=(
    artifacts/metrics.json
    artifacts/orbit_contact_sheet.png
    artifacts/orbit_360.mp4
  )
else
  public_files+=(
    gate.json
    consumer/metrics.json
    renderer/READY
    renderer/metrics.json
    renderer/live_gaussian_first.png
    renderer/live_gaussian_final.png
    renderer/live_gaussian.mp4
  )
fi
for relative in "${public_files[@]}"; do
  mkdir -p "$destination/$(dirname "$relative")"
  cp "$source_dir/$relative" "$destination/$relative"
done
cp "$source_dir/hashes.sha256" "$destination/source_hashes.sha256"
(
  cd "$destination"
  find . -type f ! -name collected.sha256 -print0 \
    | sort -z \
    | xargs -0 sha256sum >collected.sha256
)
printf 'collected redacted AMD development evidence: %s\n' "$destination"
