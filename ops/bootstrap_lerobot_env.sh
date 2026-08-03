#!/usr/bin/env bash
set -euo pipefail

env_root=${1:-/root/radeon-oneloop-env/rocm721-py312}
source_root=${2:-/root/radeon-oneloop-deps}
project_root=${3:-/root/radeon-oneloop/current}
log_root=${4:-/root/radeon-oneloop-runs/environment}

lerobot_commit=d3bee432ab26bab857b232cebefdc57327060ea8
lerobot_url="https://codeload.github.com/phi-media-lab/Evo-RL-Phi/tar.gz/$lerobot_commit"
lerobot_archive_sha=13b11882d6f3ed5a55ee9461c8e21457afa24e32aa7318b17c2409af7a682c87
archive="$source_root/Evo-RL-Phi-$lerobot_commit.tar.gz"
checkout="$source_root/Evo-RL-Phi-$lerobot_commit"
python_bin="$env_root/bin/python"
pip_bin="$env_root/bin/pip"

[[ -x "$python_bin" ]]
mkdir -p "$source_root" "$log_root"
run_id="$(date -u +%Y%m%dT%H%M%SZ)_bootstrap_lerobot"
run_dir="$log_root/$run_id"
mkdir -p "$run_dir"
exec > >(tee "$run_dir/stdout.log") 2> >(tee "$run_dir/stderr.log" >&2)
trap 'code=$?; if [[ $code -ne 0 ]]; then touch "$run_dir/FAILED"; fi' EXIT

cat > "$run_dir/constraints.txt" <<'EOF'
numpy==1.26.4
opencv-python-headless==4.11.0.86
packaging==25.0
EOF
export PIP_CONSTRAINT="$run_dir/constraints.txt"
export PIP_INDEX_URL='https://pypi.tuna.tsinghua.edu.cn/simple'
export PIP_TRUSTED_HOST='pypi.tuna.tsinghua.edu.cn'

export DEBIAN_FRONTEND=noninteractive
apt-get update
apt-get install -y --no-install-recommends ffmpeg libavcodec-dev libavformat-dev libavutil-dev

if [[ ! -s "$archive" ]]; then
  curl -fL --retry 5 --retry-delay 2 -o "$archive.part" "$lerobot_url"
  mv "$archive.part" "$archive"
fi
printf '%s  %s\n' "$lerobot_archive_sha" "$archive" \
  | tee "$run_dir/lerobot_archive.sha256" | sha256sum -c -
if [[ ! -d "$checkout" ]]; then
  extract_tmp=$(mktemp -d "$source_root/.lerobot-extract.XXXXXX")
  trap 'code=$?; if [[ -n "${extract_tmp:-}" && -d "$extract_tmp" ]]; then rm -rf "$extract_tmp"; fi; if [[ $code -ne 0 ]]; then touch "$run_dir/FAILED"; fi' EXIT
  tar -xzf "$archive" -C "$extract_tmp"
  extracted=$(find "$extract_tmp" -mindepth 1 -maxdepth 1 -type d | head -n 1)
  mv "$extracted" "$checkout"
  rmdir "$extract_tmp"
  extract_tmp=
fi

"$pip_bin" install \
  'accelerate>=1.10,<2' \
  'av>=15,<16' \
  'datasets>=4.0,<4.2' \
  'deepdiff>=7,<9' \
  'diffusers>=0.27.2,<0.36' \
  'draccus==0.10.0' \
  'einops>=0.8,<0.9' \
  'gymnasium>=1.1.1,<2' \
  'huggingface-hub[hf-transfer]>=0.34.2,<0.36' \
  'imageio[ffmpeg]>=2.34,<3' \
  'jsonlines>=4,<5' \
  'opencv-python-headless>=4.9,<4.13' \
  'packaging>=24.2,<26' \
  'pandas>=2.2,<3' \
  'pyarrow>=17,<24' \
  'PyYAML>=6.0.2,<7' \
  'rerun-sdk>=0.24,<0.27' \
  'termcolor>=2.4,<4' \
  'wandb>=0.24,<0.25'

"$pip_bin" install --no-deps -e "$checkout"
"$pip_bin" install -e "$project_root[data]"

export PYTHONPATH="$checkout/src:$project_root/src${PYTHONPATH:+:$PYTHONPATH}"
"$python_bin" - <<'PY' | tee "$run_dir/lerobot_import.json"
import json
import lerobot
import torch
from lerobot.configs.train import TrainPipelineConfig
from lerobot.policies.act.modeling_act import ACTPolicy

assert torch.cuda.is_available()
assert torch.cuda.device_count() == 1
print(json.dumps({
    "lerobot": getattr(lerobot, "__version__", "0.4.4-source"),
    "torch": torch.__version__,
    "hip": torch.version.hip,
    "device": torch.cuda.get_device_name(0),
    "train_config_import": TrainPipelineConfig.__name__,
    "act_policy_import": ACTPolicy.__name__,
}, sort_keys=True))
PY

PYTHONPATH="$project_root/src" "$python_bin" -m unittest discover -s "$project_root/tests" -v \
  | tee "$run_dir/project_tests.txt"
"$pip_bin" freeze | sort > "$run_dir/pip_freeze.txt"
sha256sum "$run_dir"/*.txt "$run_dir"/*.json "$run_dir"/*.sha256 > "$run_dir/hashes.sha256"
touch "$run_dir/DONE"
trap - EXIT
printf 'LeRobot environment bootstrap passed: %s\n' "$run_dir"
