#!/usr/bin/env bash
set -euo pipefail

env_root=${1:-/root/radeon-oneloop-env/rocm721-py312}
source_root=${2:-/root/radeon-oneloop-deps}
log_root=${3:-/root/radeon-oneloop-runs/environment}
commit=e26c254938c81ff85998cd357a9e005e255d9b03
archive_sha=1b0248769a1b37dff0d1f435762eb6078461d90d5d3f1a7f816604fa643e6400
archive="$source_root/vksplat-$commit.tar.gz"
checkout="$source_root/vksplat-$commit"
python_bin="$env_root/bin/python"
pip_bin="$env_root/bin/pip"

[[ -x $python_bin ]]
mkdir -p "$source_root" "$log_root"
run_dir="$log_root/$(date -u +%Y%m%dT%H%M%SZ)_bootstrap_vksplat"
mkdir "$run_dir"
exec > >(tee "$run_dir/stdout.log") 2> >(tee "$run_dir/stderr.log" >&2)
trap 'code=$?; if [[ $code -ne 0 ]]; then touch "$run_dir/FAILED"; fi' EXIT

export DEBIAN_FRONTEND=noninteractive
apt-get update
apt-get install -y --no-install-recommends libglm-dev libvulkan-dev vulkan-tools

if [[ ! -s $archive ]]; then
  curl -fL --retry 5 --retry-delay 2 -o "$archive.part" \
    "https://codeload.github.com/harry7557558/vksplat/tar.gz/$commit"
  mv "$archive.part" "$archive"
fi
printf '%s  %s\n' "$archive_sha" "$archive" | sha256sum -c -
if [[ ! -d $checkout ]]; then
  extract_tmp=$(mktemp -d "$source_root/.vksplat-extract.XXXXXX")
  tar -xzf "$archive" -C "$extract_tmp"
  extracted=$(find "$extract_tmp" -mindepth 1 -maxdepth 1 -type d | head -n 1)
  mv "$extracted" "$checkout"
  rmdir "$extract_tmp"
fi

"$pip_bin" install 'pybind11>=2.11.1,<3' 'tqdm>=4.66,<5'
"$pip_bin" install --no-deps --no-build-isolation -e "$checkout/vksplat"
xdg_runtime="$run_dir/xdg-runtime"
mkdir "$xdg_runtime"
chmod 700 "$xdg_runtime"
XDG_RUNTIME_DIR="$xdg_runtime" vulkaninfo --summary | tee "$run_dir/vulkaninfo_summary.txt"
"$python_bin" - "$checkout/vksplat/shader" <<'PY' | tee "$run_dir/vksplat_smoke.json"
import json
import sys
import vksplat

module = vksplat.VkSplat()
module.initialize(sys.argv[1], -1)
module.cleanup()
print(json.dumps({"vksplat_import": True, "vulkan_initialize": True}, sort_keys=True))
PY
"$pip_bin" freeze | sort > "$run_dir/pip_freeze.txt"
sha256sum "$run_dir"/*.txt "$run_dir"/*.json > "$run_dir/hashes.sha256"
touch "$run_dir/DONE"
trap - EXIT
printf 'VkSplat bootstrap passed: %s\n' "$run_dir"
