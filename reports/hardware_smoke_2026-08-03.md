# gfx1100 hardware smoke — 2026-08-03

> Status: pre-install evidence. No training or environment workload was started.
> The reusable command is `ops/remote_hardware_smoke.sh`.

## Summary

Both cloud aliases expose one ROCm `gfx1100` agent with approximately 48 GiB
VRAM and have a Mesa RADV ICD. They are distinct GPUs, as shown by their ROCm
unique IDs. Neither machine currently has PyTorch, Genesis, or `vulkaninfo`, so
the formal environment gate is **not yet passed**.

| Field | `radeon-c` — formal candidate | `radeon-f` — shadow candidate |
|---|---|---|
| Timestamp | 2026-08-03 14:05:38 UTC | 2026-08-03 14:05:37 UTC |
| Hostname | `u-9752-9c0a120c` | `u-15493-28a03719` |
| OS / kernel | Ubuntu 24.04 / 6.8.0-79 | Ubuntu 24.04 / 6.8.0-79 |
| Python | 3.12.3 | 3.12.3 |
| ROCm | 7.2.1 | 7.2.1 |
| HIP | 7.2.53211 | 7.2.53211 |
| ROCm target | `gfx1100` | `gfx1100` |
| ROCm GPU agents | 1 | 1 |
| ROCm-SMI device | `GPU[0]` | `GPU[0]` |
| ROCm unique ID | `0x153f7d55778ab659` | `0xd9aa7556136c1abf` |
| VRAM | 51,522,830,336 bytes (~47.98 GiB) | 51,522,830,336 bytes (~47.98 GiB) |
| Exposed render node | `/dev/dri/renderD131` | `/dev/dri/renderD130` |
| Render-node PCI path | ends at `0000:63:00.0` | ends at `0000:43:00.0` |
| RADV ICD | `/usr/share/vulkan/icd.d/radeon_icd.json` | same |
| `vulkaninfo` | absent | absent |
| PyTorch | absent | absent |
| Genesis | absent | absent |

## Device-isolation interpretation

The physical host PCI inventory lists eight AMD display functions plus an
ASPEED display controller. This is not evidence that the project process can
use eight GPUs. The current session has one `/dev/dri/renderD*` device node,
one `rocminfo` gfx agent, and ROCm-SMI reports only `GPU[0]`.

The formal single-GPU assertion must still be repeated after PyTorch and
`vulkaninfo` are installed. That later gate must record:

1. PyTorch device count/name and ROCm unique identity;
2. Genesis backend and selected device;
3. Vulkan device name, UUID, and PCI information; and
4. the mapping from Vulkan to the exposed render node/ROCm GPU.

Until that cross-API assertion passes, the report must say "one ROCm-visible
gfx1100 candidate" rather than claiming complete ROCm/Vulkan identity proof.

## Gate result

| Check | Result |
|---|---|
| SSH and read-only probe | PASS |
| ROCm 7.2.1 / HIP available | PASS |
| One ROCm-visible `gfx1100` agent | PASS |
| Distinct formal and shadow GPU UIDs | PASS |
| Radeon Vulkan ICD present | PASS |
| Vulkan device enumeration | BLOCKED — `vulkaninfo` absent |
| PyTorch ROCm import/device | BLOCKED — PyTorch absent |
| Genesis AMD import/init | BLOCKED — Genesis absent |
| Gate A formal environment | NOT PASSED |

## Next gate

Install and pin the supported ROCm/PyTorch environment and Vulkan tools on
`radeon-f` first. After clean preflight passes, reproduce the same installation
on `radeon-c`, install pinned Genesis, and rerun the same script plus actual
PyTorch, Genesis, and Vulkan device assertions.
