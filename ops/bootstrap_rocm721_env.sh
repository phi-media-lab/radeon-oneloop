#!/usr/bin/env bash
set -euo pipefail

install_root=${1:-/root/radeon-oneloop-env/rocm721-py312}
wheel_root=${2:-/root/radeon-oneloop-wheelhouse/rocm721-py312}
log_root=${3:-/root/radeon-oneloop-runs/environment}

mkdir -p "$install_root" "$wheel_root" "$log_root"
exec 9>/tmp/radeon-oneloop-gpu0.lock
if ! flock -n 9; then
  printf '%s\n' 'GPU0 is locked by another Radeon OneLoop job' >&2
  exit 75
fi

run_id="$(date -u +%Y%m%dT%H%M%SZ)_bootstrap_rocm721"
run_dir="$log_root/$run_id"
mkdir -p "$run_dir"
exec > >(tee "$run_dir/stdout.log") 2> >(tee "$run_dir/stderr.log" >&2)

mark_failure() {
  exit_code=$?
  if [[ $exit_code -ne 0 ]]; then
    touch "$run_dir/FAILED"
  fi
}
trap mark_failure EXIT

printf 'run_id=%s\n' "$run_id"
printf 'host=%s\n' "$(hostname -f 2>/dev/null || hostname)"
printf 'install_root=%s\n' "$install_root"
printf 'wheel_root=%s\n' "$wheel_root"

[[ "$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')" == "3.12" ]]
[[ "$(readlink -f /opt/rocm)" == "/opt/rocm-7.2.1" ]]
[[ "$(rocminfo 2>/dev/null | awk '/^[[:space:]]*Name:[[:space:]]*gfx/{print $2}' | sort -u)" == "gfx1100" ]]
[[ "$(rocminfo 2>/dev/null | awk '/^[[:space:]]*Name:[[:space:]]*gfx/{n++} END{print n+0}')" == "1" ]]

export DEBIAN_FRONTEND=noninteractive
apt-get update
apt-get install -y --no-install-recommends \
  build-essential \
  cmake \
  git \
  libegl1 \
  libgl1 \
  libglfw3 \
  libsm6 \
  libxext6 \
  libxrender1 \
  ninja-build \
  python3-dev \
  python3-venv \
  vulkan-tools

python3 -m venv "$install_root"
python_bin="$install_root/bin/python"
pip_bin="$install_root/bin/pip"

"$python_bin" -m pip install --upgrade pip wheel setuptools

cat > "$run_dir/constraints.txt" <<'EOF'
numpy==1.26.4
EOF

download_wheel() {
  local output_name=$1
  local source_url=$2
  if [[ ! -s "$wheel_root/$output_name" ]]; then
    curl -fL --retry 5 --retry-delay 2 -o "$wheel_root/$output_name.part" "$source_url"
    mv "$wheel_root/$output_name.part" "$wheel_root/$output_name"
  fi
}

download_wheel \
  torch-2.9.1+rocm7.2.1.lw.gitff65f5bc-cp312-cp312-linux_x86_64.whl \
  'https://repo.radeon.com/rocm/manylinux/rocm-rel-7.2.1/torch-2.9.1%2Brocm7.2.1.lw.gitff65f5bc-cp312-cp312-linux_x86_64.whl'
download_wheel \
  torchvision-0.24.0+rocm7.2.1.gitb919bd0c-cp312-cp312-linux_x86_64.whl \
  'https://repo.radeon.com/rocm/manylinux/rocm-rel-7.2.1/torchvision-0.24.0%2Brocm7.2.1.gitb919bd0c-cp312-cp312-linux_x86_64.whl'
download_wheel \
  torchaudio-2.9.0+rocm7.2.1.gite3c6ee2b-cp312-cp312-linux_x86_64.whl \
  'https://repo.radeon.com/rocm/manylinux/rocm-rel-7.2.1/torchaudio-2.9.0%2Brocm7.2.1.gite3c6ee2b-cp312-cp312-linux_x86_64.whl'
download_wheel \
  triton-3.5.1+rocm7.2.1.gita272dfa8-cp312-cp312-linux_x86_64.whl \
  'https://repo.radeon.com/rocm/manylinux/rocm-rel-7.2.1/triton-3.5.1%2Brocm7.2.1.gita272dfa8-cp312-cp312-linux_x86_64.whl'

"$pip_bin" install -c "$run_dir/constraints.txt" \
  "$wheel_root/torch-2.9.1+rocm7.2.1.lw.gitff65f5bc-cp312-cp312-linux_x86_64.whl" \
  "$wheel_root/torchvision-0.24.0+rocm7.2.1.gitb919bd0c-cp312-cp312-linux_x86_64.whl" \
  "$wheel_root/torchaudio-2.9.0+rocm7.2.1.gite3c6ee2b-cp312-cp312-linux_x86_64.whl" \
  "$wheel_root/triton-3.5.1+rocm7.2.1.gita272dfa8-cp312-cp312-linux_x86_64.whl"

"$pip_bin" install -c "$run_dir/constraints.txt" \
  'https://github.com/Genesis-Embodied-AI/genesis-world/archive/refs/tags/v1.3.1.tar.gz'

export TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1
"$python_bin" - <<'PY' | tee "$run_dir/torch_smoke.json"
import json
import torch

assert torch.version.hip is not None, torch.__version__
assert torch.cuda.is_available()
assert torch.cuda.device_count() == 1
a = torch.randn((1024, 1024), device="cuda")
b = torch.randn((1024, 1024), device="cuda")
c = a @ b
torch.cuda.synchronize()
assert torch.isfinite(c).all().item()
print(json.dumps({
    "torch": torch.__version__,
    "hip": torch.version.hip,
    "device_count": torch.cuda.device_count(),
    "device_0": torch.cuda.get_device_name(0),
    "matmul_shape": list(c.shape),
    "matmul_finite": True,
}, sort_keys=True))
PY

"$python_bin" - <<'PY' | tee "$run_dir/genesis_smoke.json"
import json
import genesis as gs

gs.init(backend=gs.amdgpu, seed=20260803)
assert gs.backend == gs.amdgpu
print(json.dumps({
    "genesis": getattr(gs, "__version__", "unknown"),
    "backend": str(gs.backend),
    "device": str(gs.device),
}, sort_keys=True))
PY

xdg_runtime="$run_dir/xdg-runtime"
mkdir -p "$xdg_runtime"
chmod 700 "$xdg_runtime"
XDG_RUNTIME_DIR="$xdg_runtime" vulkaninfo --summary | tee "$run_dir/vulkaninfo_summary.txt"

"$pip_bin" freeze | sort > "$run_dir/pip_freeze.txt"
/opt/rocm/bin/rocminfo > "$run_dir/rocminfo.txt"
rocm-smi --showproductname --showuniqueid --showmeminfo vram > "$run_dir/rocm_smi.txt"
sha256sum "$run_dir"/*.txt "$run_dir"/*.json > "$run_dir/hashes.sha256"
touch "$run_dir/DONE"
trap - EXIT
printf 'environment bootstrap passed: %s\n' "$run_dir"
