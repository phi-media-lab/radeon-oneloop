# Radeon OneLoop

**Single-Radeon Phase-Aware HIL Bimanual Handover**

Radeon OneLoop is a Track 3 embodied-intelligence project for the AMD Radeon
Hackathon 2026. The target task is a real SO-101 bimanual handover: one arm
grasps and transfers an object, the second arm receives it, and the object is
placed at the target.

The formal profile uses exactly one AMD Radeon GPU (`radeon-c`, `gfx1100`) for:

- Genesis environment execution;
- baseline and phase-aware ACT training;
- real-time ACT inference.

A CPU-only edge process handles cameras, robot I/O, timeouts, limits, watchdog,
and emergency stop. Development may use other AMD machines, but their results
are never mixed into the formal result lineage.

## Scope freeze

The competition build contains three integrated deliverables:

1. a reproducible minimal Genesis environment for the SO-101 handover;
2. a fair baseline-versus-phase-aware ACT comparison; and
3. a fail-closed CPU edge and measured Radeon ACT inference path for the real
   bimanual robot.

The repository retains a gated VkSplat experiment, but no calibrated SO-101
workspace capture was available at scope freeze. Gaussian results are therefore
not a competition deliverable or an ACT observation dependency in this release.
Dynamic 4D Gaussian models, Genesis/GS real-time compositing, NPU inference,
and multi-GPU training are explicitly out of scope.

## Repository status

The reproducibility core is implemented: pinned ROCm/Genesis/LeRobot
bootstraps, immutable SSH deployment, single-Radeon assertions, a verified
124-episode dataset builder, phase-aware target generation, fair ACT command
generation, a dual SO-101 Genesis scene, and CPU-edge safety contracts. Remote
smokes and formal training are tracked as evidence gates; this README does
**not** claim a task-success result until its formal registry entries exist.

## Formal evidence rule

Only runs listed in [`ops/formal_run_registry.yaml`](ops/formal_run_registry.yaml)
may populate the technical report. A formal entry must reference the `radeon-c`
hardware identity, Git commit, configuration hash, dataset hash, checkpoint
lineage, raw logs, and metrics.

## Layout

```text
configs/       Frozen experiment profiles
data/          Dataset contract and immutable registry
sim/           Minimal Genesis SO-101 environment
policy/        ACT training and inference
gaussian/      Gated future VkSplat experiment (not a formal deliverable)
runtime/       CPU-edge protocol and safety
evaluation/    Simulation, real-robot, latency, and fidelity metrics
ops/           Job manifests, formal registry, and validation
reports/       Technical report sources
submission/    Official PR entry and demo-video plan
```

## Local validation

```bash
./ops/validate_scaffold.sh
```

See the workstream READMEs for the exact remote environment, dataset, Genesis,
and ACT commands.

## License

Project-authored code is licensed under Apache-2.0. Third-party code, models,
datasets, and assets retain their own licenses; see
[`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md).
