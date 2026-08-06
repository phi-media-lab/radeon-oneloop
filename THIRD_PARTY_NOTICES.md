# Third-party notices

This inventory is intentionally conservative. Dependency names do not imply
that their source code, models, datasets, or assets are redistributed here.
Before bundling an artifact, add its exact version, source URL, license, and
redistribution status to this file.

| Project or asset | Expected use | License/status | Bundled now |
|---|---|---|---:|
| Genesis `v1.3.1` | Physics environment and rendering | Apache-2.0; installed from the official release tag | No |
| PyTorch `2.9.1+rocm7.2.1` and ROCm `7.2.1` | ACT training and inference | Upstream licenses; official AMD wheels installed separately | No |
| `phi-media-lab/Evo-RL-Phi@d3bee432` | LeRobot 0.4.4 plus ACT-AWR loss support | Apache-2.0; installed from pinned source | No |
| SO-ARM100 / SO-101 simulation assets `1b74d9fc` | Robot MJCF/STL | Apache-2.0; downloaded separately with size and SHA-256 verification | No |
| VkSplat | Gaussian optimization and rendering | Apache-2.0 | No |
| COLMAP | Camera reconstruction | BSD-3-Clause | No |
| Existing HIL datasets | Policy training/evaluation | Team-collected; access-controlled; redistribution disabled | No |
| Workspace images | Gaussian optimization | Team-created; release decision pending | No |
| Stable Virtual Camera v1.1 (`e538e251`) | Internal novel-view and Real2Sim R&D only | Stability AI Non-Commercial License; upstream states outputs follow the same restriction; excluded from competition submission without separate written clearance | No |
| Microsoft TRELLIS.2 | Generate a textured object candidate from team-owned photographs | MIT source repository; hosted inference and model terms remain upstream; weights and service code are not bundled | No |
| MGPBD reference repository (`06761eb3`) | Hash-pinned bunny topology and paper-level conformance reference | Public repository declares no license; upstream source/data are fetched only into an untracked run directory and are not redistributed | No |
| FreeTimeGS/Open-d4rt artifacts | Prior-work appendix only | Do not redistribute until audited | No |

The Genesis Nyx renderer is not part of the Radeon formal path and is not
bundled.
