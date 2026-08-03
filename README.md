# Radeon OneLoop

**Phase-Aware HIL Bimanual Handover with a Calibrated Gaussian Workspace Twin**

Radeon OneLoop is a Track 3 embodied-intelligence project for the AMD Radeon
Hackathon 2026. The target task is a real SO-101 bimanual handover: one arm
grasps and transfers an object, the second arm receives it, and the object is
placed at the target.

The formal profile uses exactly one AMD Radeon GPU (`radeon-c`, `gfx1100`) for:

- Genesis environment execution;
- baseline and phase-aware ACT training;
- real-time ACT inference; and
- VkSplat Gaussian workspace optimization and rendering.

A CPU-only edge process handles cameras, robot I/O, timeouts, limits, watchdog,
and emergency stop. Development may use other AMD machines, but their results
are never mixed into the formal result lineage.

## Scope freeze

The competition build contains three deliverables:

1. a reproducible minimal Genesis environment for the SO-101 handover;
2. a fair baseline-versus-phase-aware ACT comparison; and
3. a calibrated static Gaussian workspace twin for visualization and
   synchronized trajectory replay.

The Gaussian renderer is not an ACT observation dependency in this release.
Dynamic 4D Gaussian models, Genesis/GS real-time compositing, NPU inference, and
multi-GPU training are explicitly out of scope.

## Repository status

This is the initial project scaffold. It defines the execution contracts,
formal-run registry, data schema, and workstream boundaries. It does **not** yet
claim a completed training run, task-success result, or Gaussian benchmark.

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
gaussian/      VkSplat preparation, training, rendering, and calibration
runtime/       CPU-edge protocol and safety
evaluation/    Simulation, real-robot, latency, and fidelity metrics
ops/           Job manifests, formal registry, and validation
reports/       Technical report sources
submission/    Official PR entry and demo-video plan
```

## Bootstrap validation

```bash
./ops/validate_scaffold.sh
```

Hardware, environment, training, and evaluation commands will be added only
after they pass the corresponding project gate.

## License

Project-authored code is licensed under Apache-2.0. Third-party code, models,
datasets, and assets retain their own licenses; see
[`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md).
