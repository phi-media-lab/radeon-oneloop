#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 3 ]]; then
  printf 'usage: %s <ssh-host> <absolute-remote-job-dir> <public-label>\n' "$0" >&2
  exit 64
fi

ssh_host=$1
remote_dir=${2%/}
label=$3
repo_root=$(git rev-parse --show-toplevel)
destination="$repo_root/artifacts/formal/$label"

[[ $remote_dir == /root/radeon-oneloop-runs/jobs/* ]] || {
  printf 'remote job must be below /root/radeon-oneloop-runs/jobs: %s\n' "$remote_dir" >&2
  exit 64
}
[[ $label =~ ^[a-z0-9][a-z0-9_-]*$ ]] || {
  printf 'invalid public label: %s\n' "$label" >&2
  exit 64
}
[[ ! -e $destination ]] || {
  printf 'destination exists; evidence collection never overwrites: %s\n' "$destination" >&2
  exit 73
}

ssh "$ssh_host" python3 - "$remote_dir" <<'PY'
import json
import pathlib
import sys

root = pathlib.Path(sys.argv[1])
manifest = json.loads((root / "manifest.json").read_text())
if manifest.get("formal") is not True:
    raise SystemExit("refusing to publish a non-formal job")
if manifest.get("host") != "radeon-c":
    raise SystemExit(f"unexpected formal role: {manifest.get('host')!r}")
if manifest.get("gpu_uid") != "0x153f7d55778ab659":
    raise SystemExit(f"unexpected GPU UID: {manifest.get('gpu_uid')!r}")
if manifest.get("status") != "done" or not (root / "DONE").is_file():
    raise SystemExit("formal job is not complete")
PY

files=(
  DONE
  command.sh
  config.yaml
  environment.txt
  gpu_samples.tsv
  hardware.json
  hashes.sha256
  manifest.json
  metrics.json
  stderr.log
  stdout.log
)
mkdir -p "$destination"
for name in "${files[@]}"; do
  scp -q "$ssh_host:$remote_dir/$name" "$destination/$name"
done

(
  cd "$destination"
  sha256sum "${files[@]}" > collected.sha256
)
printf 'collected formal evidence: %s\n' "$destination"
