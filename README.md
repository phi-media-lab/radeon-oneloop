# Radeon OneLoop

**A staged Real2Sim2Real loop for bimanual handover on AMD Radeon**

Radeon OneLoop is a Track 3 embodied-intelligence entry for the AMD Radeon
Hackathon 2026. It connects a proven real-robot learning loop to a new
Real2Sim path for the same dual-SO-101 handover cell and plush object.

The submission has two auditable evidence chains:

1. **Real -> learn -> real.** We combine 84 team demonstrations with 40
   reviewed policy/HIL episodes, train a baseline and a correction-aware ACT
   policy from scratch on one `gfx1100` Radeon, run inference through ROCm,
   and preserve a reviewed 37/45 physical handover result from the inherited
   closed loop.
2. **Real -> sim.** Reviewed real photographs of the team-owned object produce a TRELLIS.2 textured
   asset, which is canonicalized to the measured 95 mm object frame and
   ingested by Genesis. In parallel, our Torch/ROCm MGPBD implementation and
   exact Genesis state bridge are validated on a clean tetrahedral bunny.

This is a staged Real2Sim2Real system, not a claim that every experimental
component has already passed the same formal gate. The competition result is
the single-Radeon learning/deployment loop. The Real2Sim extension establishes
the appearance, metric, solver, and renderer interfaces needed to generate the
next simulated intervention curriculum; deformable doll contact and subsequent
sim-to-real retraining remain the next controlled experiment.

## OneLoop architecture

```text
team-owned physical cell
        |
        +-- demonstrations + reviewed interventions
        |          -> immutable 124-episode dataset
        |          -> baseline / phase-aware ACT on one Radeon
        |          -> ROCm inference + CPU safety edge
        |          -> physical dual-arm handover-to-place
        |
        `-- reviewed real object photographs
                   -> TRELLIS.2 appearance asset
                   -> 95 mm canonicalization
                   -> ROCm MGPBD state + Genesis AMD bridge
                   -> simulated interventions (next evaluation)
                   -> same policy and safety interfaces
```

The TRELLIS.2 generation receipt used its official hosted inference endpoint;
we do **not** claim that TRELLIS.2 inference itself ran on Radeon. The Radeon
evidence covers asset preparation and Genesis execution, plus the MGPBD solver
and MGPBD-to-Genesis bridge.

## Single-Radeon boundary

The formal profile assigns exactly one exposed `gfx1100` device on `radeon-c`
to all accelerator work. Genesis and PyTorch use ROCm/HIP; the CPU only handles
dataset decoding, cameras, robot I/O, timeouts, limits, watchdog, and emergency
stop.

```text
real demonstrations + reviewed HIL corrections
                       |
                       v
             immutable 124-episode dataset
                       |
          +------------+-------------+
          |                          |
          v                          v
  ACT baseline                  phase-aware ACT
  weight = 1                    correction = 4
                                failed prefix = 0.05
          |                          |
          +------------+-------------+
                       |
                       v
        one gfx1100 Radeon: Genesis + training + inference
                       |
                       v
        CPU edge: validation, watchdog, robot I/O, E-stop
```

Other AMD systems were used only for read-only prior-art inventory or shadow
preflight. Their checkpoints and performance numbers are excluded from the
formal lineage. The machine-readable enforcement lives in
[`configs/formal_radeon_only.yaml`](configs/formal_radeon_only.yaml) and
[`ops/run_job.sh`](ops/run_job.sh).

## Frozen experiment

The input is 124 team-collected real episodes: 84 human demonstrations and 40
reviewed HIL rollouts, totaling 178,465 frames at 30 Hz. It is deliberately
train-only; the repository therefore does not present reconstruction error or
training loss as task success.

Both ACT policies use the same default LeRobot architecture, optimizer,
observations, actions, batch size 16, seed `20260803`, and 10,000 update steps.
The only intended difference is the phase-aware loss weight. Positive weights
are normalized to mean one:

| Segment role | Frames | Raw weight | Gradient-mass share |
|---|---:|---:|---:|
| Human demonstration | 116,550 | 1.00 | 54.24% |
| Successful policy | 34,101 | 1.00 | 15.87% |
| Failed policy prefix | 11,908 | 0.05 | 0.28% |
| Human correction | 15,906 | 4.00 | 29.61% |

The checkpoint rule was predeclared before the formal pair: use step 10,000,
with no post-hoc search over intermediate training loss.

## What is measured

The formal result table is populated only after a run is registered in
[`ops/formal_run_registry.yaml`](ops/formal_run_registry.yaml). Each entry must
bind the host, GPU UID, Git commit, config hash, dataset hash, seed, checkpoint
hash, raw logs, and metrics.

Completed platform evidence already establishes:

- ROCm 7.2.1, HIP 7.2, AMD PyTorch 2.9.1 and Genesis 1.3.1;
- one ROCm-visible `gfx1100` Radeon with 51,522,830,336 bytes of VRAM;
- a 1,000-step dual-arm Genesis run with two 480×640 RGB observations; and
- median / p95 / p99 all-step simulation latency of 4.27 / 5.02 / 7.02 ms
  in the camera-corrected formal run.

PyTorch intentionally exposes ROCm tensors through its CUDA-compatible Python
API, so logs may print `cuda:0` or configs may say `device: cuda`. In this
profile that string identifies the single HIP-backed Radeon device: the same
records show `torch.version.hip`, `gfx1100`, the AMD device name and Radeon GPU
UID. No NVIDIA driver, CUDA runtime or NVIDIA package is installed.

The scripted Genesis sweep is an environment, control, and observation test;
it is not counted as a handover success. Likewise, the action-reconstruction
diagnostic uses deterministic training-set frames and is not a validation
metric. The inherited 37/45 reviewed real-robot result is reported separately
as pre-competition evidence and never attributed to the new formal
checkpoints.

The inspected formal camera pair is published at
[`artifacts/formal/genesis_camera_corrected/camera_pair.png`](artifacts/formal/genesis_camera_corrected/camera_pair.png),
alongside its run manifest, raw timing metrics, GPU samples and hashes.

## Formal result snapshot

| Metric | Baseline ACT | Phase-aware ACT |
|---|---:|---:|
| Training time, 10,000 updates | 87.75 min | 87.68 min |
| Mean sampled GPU use | 98.59% | 98.45% |
| Full 100-action chunk, p50 / p95 | 18.11 / 46.84 ms | 18.55 / 45.73 ms |
| Queued action dispatch, p50 | 1.233 ms | 1.233 ms |
| Equal-role train-frame chunk L1 | 0.09177 | 0.10479 |
| Correction-frame chunk L1 | 0.11957 | **0.11712** |

Phase-aware ACT reduced the correction-frame chunk diagnostic by 2.05%, its
intended target, while worsening the equal-role aggregate by 14.19% and the
three non-correction roles. Even on corrections, first-action L1 worsened from
0.09009 to 0.09626. This is evidence of a targeted tradeoff, not a blanket
improvement. All reconstruction values use training frames; neither model has
a new task-success or generalization claim.

The final checkpoints are content-addressed as
`7c8f2089…29dc79` (baseline) and `3ae18054…2721d4` (phase-aware).
The full digests, raw logs and metrics live under
[`artifacts/formal`](artifacts/formal). The English technical report is
[`output/pdf/radeon-oneloop-technical-report.pdf`](output/pdf/radeon-oneloop-technical-report.pdf),
and the public demo is attached to release `v1.0.0`.

## Real2Sim2Real evidence ledger

The two branches are joined by the same measured object, workspace, 12-DoF
robot state, dual-camera observations, and CPU safety protocol.

| Stage | Frozen evidence | What it establishes |
|---|---|---|
| Real data | 84 demonstrations + 40 reviewed HIL episodes; 124 episodes, 178,465 frames | Team-owned physical data and explicit correction lineage |
| Learn on Radeon | Two 10,000-step, from-scratch ACT runs on one `gfx1100` | Reproducible baseline/phase-aware comparison on ROCm |
| HIL learning signal | Correction-frame chunk L1 `0.11957 -> 0.11712` | 2.05% targeted diagnostic improvement, not task success |
| Real execution | Reviewed historical ledger: 37/45 (82.22%); [51.14 s physical handover-to-place video](https://github.com/phi-media-lab/radeon-oneloop/releases/download/v1.0.0/amd-hackathon.mp4) | The inherited policy/HIL/deployment loop operated on the real cell; the video is one execution, not a success-rate estimate |
| Real appearance -> sim asset | TRELLIS.2 output: 221,670 vertices / 293,972 faces; canonical height 95 mm | A textured, metrically aligned visual asset; rear geometry remains generative and the mesh is visual-only |
| MGPBD -> Genesis on AMD | `T152831`: 2,992 volume vertices / 12,298 tets, 2,000 boundary vertices / 3,996 faces, zero open/non-manifold edges, exact rest mapping | A coherent closed-boundary state bridge from the ROCm MGPBD checkpoint into Genesis |

The 37/45 result predates the new formal checkpoints, so it is not presented
as their success rate. The full MGPBD P0a2 solve stopped safely after 31
accepted outer updates when outer 32 missed its stationarity gate; `T152831`
replays that safe checkpoint and proves the bridge, not realtime doll contact.
These boundaries are documented in
[`reports/historical_real_robot_evidence.md`](reports/historical_real_robot_evidence.md)
and
[`reports/mgpbd_forensic_audit_and_recovery_plan_2026-08-06.md`](reports/mgpbd_forensic_audit_and_recovery_plan_2026-08-06.md).
The compact `T152831` manifest, metrics, hashes, boundary mesh, and three
stage-distinct captures are preserved under
[`artifacts/development/amd_genesis_mgpbd_bunny_bridge_20260806T152831Z`](artifacts/development/amd_genesis_mgpbd_bunny_bridge_20260806T152831Z).

## Reproduce

### 1. Validate the source tree

Requires Python 3.12. No GPU or private dataset is needed for this step.

```bash
git clone https://github.com/phi-media-lab/radeon-oneloop.git
cd radeon-oneloop
python3 -m venv .venv
. .venv/bin/activate
pip install -e '.[test]'
./ops/validate_scaffold.sh
```

### 2. Build the Radeon environment

The tested host is Ubuntu 24.04 with `/opt/rocm-7.2.1`, Python 3.12 and exactly
one visible `gfx1100` agent. The bootstrap downloads hash-pinned official AMD
PyTorch wheels and the Genesis 1.3.1 wheel, rejects NVIDIA dependencies, runs a
GPU matrix multiplication, initializes `gs.amdgpu`, and captures ROCm and
Vulkan identity evidence.

```bash
sudo bash ops/bootstrap_rocm721_env.sh
sudo bash ops/bootstrap_lerobot_env.sh
```

The LeRobot source dependency is pinned to
`phi-media-lab/Evo-RL-Phi@d3bee432ab26bab857b232cebefdc57327060ea8` and
verified by a deterministic source-tree hash before installation.

### 3. Supply and verify data

The raw videos are access-controlled and are not redistributed. Authorized
reviewers can place the two registered source datasets under the paths shown
in [`data/README.md`](data/README.md), then run:

```bash
ONELOOP_PYTHON=/root/radeon-oneloop-env/rocm721-py312/bin/python \
  bash ops/build_formal_dataset.sh
```

The transaction refuses to overwrite an existing output, validates camera and
12-DoF contracts, remaps video indices, generates the phase sidecar, and
checks the resulting dataset hash
`ba18dd207ffd00c562a7ad18c831508d0529cd4d8d7b478a9b2f6d46618489cf`.

### 4. Run Genesis and the paired ACT experiment

Every GPU job acquires a single-device lock and emits a self-contained evidence
directory (`manifest.json`, exact command, environment, hardware, metrics,
one-second GPU samples, hashes, and a terminal marker). See
[`ops/README.md`](ops/README.md) for remote dispatch examples.

The policy commands can be inspected without launching training:

```bash
oneloop-train-command \
  --config configs/act_baseline.yaml \
  --paired-config configs/act_phase_aware.yaml \
  --dataset-root /root/radeon-oneloop-data/formal_handover_v1 \
  --output-dir /root/radeon-oneloop-runs/artifacts/act-baseline

oneloop-train-command \
  --config configs/act_phase_aware.yaml \
  --paired-config configs/act_baseline.yaml \
  --dataset-root /root/radeon-oneloop-data/formal_handover_v1 \
  --output-dir /root/radeon-oneloop-runs/artifacts/act-phase-aware
```

Add `--execute` only after the hardware assertion passes. Formal runs also
require `ONELOOP_FORMAL_HOST=radeon-c`; an unexpected hostname role, GPU count,
GPU UID, or `gfx` target fails closed.

### 5. Evaluate inference

```bash
python -m evaluation.policy_latency \
  --checkpoint CHECKPOINT/pretrained_model \
  --dataset-root /root/radeon-oneloop-data/formal_handover_v1 \
  --warmup 20 --iterations 200

python -m evaluation.action_reconstruction \
  --checkpoint CHECKPOINT/pretrained_model \
  --dataset-root /root/radeon-oneloop-data/formal_handover_v1 \
  --samples-per-role 256
```

The latency benchmark separates full 100-action chunk generation from cheap
queued-action dispatch, synchronizes the GPU around every sample, and records
peak allocated VRAM. It uses a real dataset observation and never relabels
latency as robot success.

## Safety boundary

[`src/radeon_oneloop/runtime_protocol.py`](src/radeon_oneloop/runtime_protocol.py)
implements a small dependency-free safety kernel. It rejects stale
observations, stale or reordered commands, mismatched sequence IDs, empty
chunks, non-finite or out-of-range joints, and excessive per-step deltas. Any
validation failure latches an E-stop; recovery requires a new controller after
physical reset. This code does not replace manufacturer limits or an operator
with access to the physical E-stop.

## Deliberate scope cuts

The repository retains a gated VkSplat experiment, but no calibrated static
multi-view capture of this SO-101 workspace existed at scope freeze. Corgi,
synthetic and dynamic robot-video assets were rejected as substitutes, so
Gaussian rendering is not an ACT observation or a competition result.
Multi-GPU training, MI300X checkpoints, Radeon 890M/NPU performance, Genesis
Nyx, and remote inference APIs are also outside the formal profile.

## Upstream AMD support contribution

The formal bring-up exposed that Genesis' checked-in AMD Docker path still
targets an older ROCm/PyTorch/Genesis tuple. We filed
[Genesis issue #3163](https://github.com/Genesis-Embodied-AI/genesis-world/issues/3163)
with the verified ROCm 7.2.1 / Genesis 1.3.1 `gfx1100` evidence, the exact
version gap, and a proposed image smoke test. The issue explicitly distinguishes
our native-host verification from an unperformed Docker build.

## Repository map

```text
configs/       Frozen single-Radeon and paired experiment profiles
data/          Dataset contract and immutable registry (no raw data)
sim/           Dual SO-101 Genesis scene and hash-pinned assets
policy/        ACT training contract
runtime/       CPU-edge protocol and safety boundary
evaluation/    Latency and stratified reconstruction diagnostics
ops/           Environment, immutable jobs, evidence and validation
reports/       Technical report source and evidence interpretation
submission/    Official PR text and 3–5 minute video plan
gaussian/      Deferred VkSplat gate, excluded from formal results
```

## License and data

Project-authored code is Apache-2.0. Third-party code, models, datasets, and
assets retain their own licenses; see
[`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md). No raw dataset, checkpoint,
personal information, credential, or private SSH configuration is included.
