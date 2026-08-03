#!/usr/bin/env bash
set -euo pipefail

expected_gfx=${1:-gfx1100}
python_bin=${2:-/root/radeon-oneloop-env/rocm721-py312/bin/python}
rocminfo_agents=$(rocminfo 2>/dev/null | awk '/^[[:space:]]*Name:[[:space:]]*gfx/{print $2}')
[[ "$(printf '%s\n' "$rocminfo_agents" | sed '/^$/d' | wc -l | tr -d ' ')" == 1 ]]
[[ "$rocminfo_agents" == "$expected_gfx" ]]
[[ "$(find /dev/dri -maxdepth 1 -name 'renderD*' 2>/dev/null | wc -l | tr -d ' ')" == 1 ]]

"$python_bin" - "$expected_gfx" <<'PY'
import json
import sys
import torch

expected = sys.argv[1]
assert torch.version.hip is not None
assert torch.cuda.is_available()
assert torch.cuda.device_count() == 1
props = torch.cuda.get_device_properties(0)
arch = str(getattr(props, "gcnArchName", ""))
assert arch.startswith(expected), (expected, arch)
print(json.dumps({
    "torch": torch.__version__,
    "hip": torch.version.hip,
    "device_count": torch.cuda.device_count(),
    "device_name": torch.cuda.get_device_name(0),
    "gcn_arch": arch,
    "total_memory": int(props.total_memory),
}, sort_keys=True))
PY

