#!/usr/bin/env bash
set +e

printf 'timestamp\t%s\n' "$(date -Is)"
printf 'hostname\t%s\n' "$(hostname -f 2>/dev/null || hostname)"
printf 'os\t%s\n' "$(. /etc/os-release 2>/dev/null; printf '%s %s' "$NAME" "$VERSION_ID")"
printf 'kernel\t%s\n' "$(uname -r)"
printf 'arch\t%s\n' "$(uname -m)"
printf 'python\t%s\n' "$(python3 --version 2>&1)"
printf 'rocm_path\t%s\n' "$(readlink -f /opt/rocm 2>/dev/null || printf absent)"
printf 'hipcc\t%s\n' "$(/opt/rocm/bin/hipcc --version 2>/dev/null | head -n 1 || printf absent)"
printf 'gfx_targets\t%s\n' "$(rocminfo 2>/dev/null | awk '/^[[:space:]]*Name:[[:space:]]*gfx/{print $2}' | sort -u | paste -sd, -)"
printf 'rocm_gpu_agents\t%s\n' "$(rocminfo 2>/dev/null | awk '/^[[:space:]]*Name:[[:space:]]*gfx/{n++} END{print n+0}')"
printf 'rocm_smi_devices\t%s\n' "$(rocm-smi -i 2>/dev/null | grep -o 'GPU\[[0-9][0-9]*\]' | sort -u | wc -l | tr -d ' ')"
printf 'gpu_pci\t%s\n' "$(lspci -nn 2>/dev/null | grep -Ei 'VGA|Display' | paste -sd'|' -)"
printf 'radeon_icd\t%s\n' "$(find /usr/share/vulkan/icd.d -maxdepth 1 -type f -iname '*radeon*' -print 2>/dev/null | paste -sd, -)"
printf 'vulkaninfo\t%s\n' "$(command -v vulkaninfo 2>/dev/null || printf absent)"
printf 'render_nodes\t%s\n' "$(find /dev/dri -maxdepth 1 -name 'renderD*' -print 2>/dev/null | sort | paste -sd, -)"
for render_node in /dev/dri/renderD*; do
  [[ -e "$render_node" ]] || continue
  render_name=$(basename "$render_node")
  printf 'render_node_sysfs\t%s=%s\n' "$render_node" "$(readlink -f "/sys/class/drm/$render_name/device" 2>/dev/null)"
done
printf 'torch_genesis\t'
python3 - <<'PY'
import importlib.util
import json

out = {}
for name in ("torch", "genesis"):
    out[name] = importlib.util.find_spec(name) is not None

if out["torch"]:
    import torch

    out["torch_version"] = torch.__version__
    out["torch_hip"] = torch.version.hip
    out["torch_device_available"] = torch.cuda.is_available()
    if torch.cuda.is_available():
        out["torch_device_count"] = torch.cuda.device_count()
        out["torch_device_0"] = torch.cuda.get_device_name(0)

print(json.dumps(out, sort_keys=True))
PY

printf '%s\n' 'rocm_smi_begin'
rocm-smi --showproductname --showuniqueid --showmeminfo vram 2>&1 | sed -n '1,80p'
printf '%s\n' 'rocm_smi_end'
