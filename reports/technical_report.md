# Radeon OneLoop

## Single-Radeon Phase-Aware HIL Bimanual Handover

**AMD Radeon Hackathon 2026 — Track 3: Physical AI**

**Team:** Phi Media Lab

**Formal profile:** one AMD Radeon `gfx1100`, ROCm 7.2.1

**Report status:** formal ACT results pending; completed claims below are backed
by immutable run records.

## Abstract

Radeon OneLoop is an auditable learning-and-deployment pipeline for a difficult
SO-101 bimanual handover. The left arm must grasp and present a soft object,
the right arm must receive it, and the system must place it in a target zone.
Our main technical idea is phase-aware learning from human intervention: keep
the architecture, optimizer and data fixed, but allocate more loss mass to the
frames where a human corrected the policy and almost none to a failed
autonomous prefix. The experiment compares this method against a uniform ACT
baseline, initialized from scratch under an identical 10,000-step budget.

The competition boundary is intentionally strict. A single Radeon `gfx1100`
executes the Genesis environment, both model-training jobs and policy
inference. The CPU performs decoding, camera and robot I/O, validation,
watchdog and emergency-stop duties only. Every accelerator job is tied to a
GPU UID, immutable source commit, config hash, dataset hash, seed and raw
evidence directory. The real dataset contains 124 episodes and 178,465 frames;
15,906 are reviewed intervention/correction frames. The formal dataset is
train-only, so this report separates training-set diagnostics from task
success. A reviewed 37/45 real-robot result is included only as historical
evidence for the inherited closed-loop system, never as performance of the new
formal checkpoints.

## 1. Problem and motivation

Bimanual handover combines perception, long-horizon coordination and contact.
The receiving arm must arrive while the giving arm still stabilizes the
object; release must occur only after a reliable grasp; and a small error early
in the sequence can become an irrecoverable drop. These characteristics make
the task a useful test of whether human interventions can be turned into
targeted learning signal rather than simply appended to a dataset.

The physical platform consists of two six-command SO-101 arms and two RGB
cameras. Each policy observation contains the 12 joint/gripper values plus a
front and hand camera image. Each action contains five joint commands and one
gripper command for the left arm followed by the same six commands for the
right arm. The control contract is 30 Hz. ACT generates chunks of 100 actions,
allowing the edge process to dispatch queued actions while the next chunk is
computed.

The object and target vary less than they would in an unconstrained household
setting. This is deliberate: the entry studies reliable phase transitions and
intervention learning in a complete real pipeline, rather than claiming
general-purpose manipulation.

## 2. Compliance and system architecture

### 2.1 One-card boundary

The formal host exposes one ROCm agent and one PyTorch device:

| Item | Formal value |
|---|---|
| ROCm / HIP | 7.2.1 / 7.2.53211 |
| PyTorch | 2.9.1+rocm7.2.1 |
| Genesis | 1.3.1, `gs.amdgpu` |
| Target | `gfx1100` |
| Reported name | AMD Radeon Graphics |
| GPU UID | `0x153f7d55778ab659` |
| VRAM | 51,522,830,336 bytes (47.98 GiB) |
| Visible accelerator count | 1 |

PyTorch uses its CUDA-compatible Python namespace for ROCm devices. Therefore
Genesis/PyTorch logs can display `cuda:0` even though the underlying runtime is
HIP. The simultaneous `torch.version.hip`, `gfx1100`, AMD device name, ROCm
UID and dependency audit establish that this is the Radeon device; the
bootstrap rejects NVIDIA packages and no CUDA runtime is present.

The same device executes:

1. Genesis scene build, rigid-body stepping and camera rendering;
2. ACT baseline training;
3. phase-aware ACT training; and
4. full-chunk ACT inference and queued-action dispatch.

`radeon-f` is an equivalent but distinct shadow GPU used to fail fast on clean
environment and two-step smoke tests. A Radeon APU machine and an MI300X
machine were inspected for prior work only. Their checkpoints, frame rates and
training results are forbidden formal sources.

### 2.2 Runtime split

```text
                         one Radeon gfx1100
              +----------------------------------+
              | Genesis AMD environment          |
              | ACT baseline + phase-aware train |
              | ACT chunk inference              |
              +----------------+-----------------+
                               |
             observation ID / action chunk / timestamps
                               |
              +----------------v-----------------+
              | CPU edge                          |
              | cameras, joint I/O, limits        |
              | watchdog, timeout, physical E-stop|
              +----------------+-----------------+
                               |
                        two SO-101 arms
```

The CPU is not an alternative inference path. The formal profile explicitly
prohibits CUDA, an NPU, a second GPU, a software renderer and remote inference
APIs.

### 2.3 Evidence discipline

Every GPU command runs under an exclusive file lock. The launcher asserts one
PyTorch device, a `gfx1100` architecture and the expected formal UID. It then
writes the exact command, environment freeze, hardware record, configuration,
one-second ROCm samples, stdout/stderr, metrics, hashes and `DONE` or `FAILED`.
Failed and aborted runs are retained. Only registered runs may populate result
tables.

## 3. Data and contracts

### 3.1 Sources

The immutable dataset builder merges two team-collected sources without
mutating either:

| Source | Episodes | Frames | Role |
|---|---:|---:|---|
| Human behavior demonstrations | 84 | 116,550 | Train |
| Reviewed HIL policy rollouts | 40 | 61,915 | Train |
| **Combined** | **124** | **178,465** | **Train only** |

The builder verifies two `480 × 640 × 3` RGB streams, the 30 Hz rate, the
12-value action and state order, task tables, video references, contiguous
episode indices and exact HIL manifest coverage. It emits a file-level SHA-256
ledger and refuses to overwrite a prior output. The resulting dataset hash is:

```text
ba18dd207ffd00c562a7ad18c831508d0529cd4d8d7b478a9b2f6d46618489cf
```

The raw dataset is access-controlled because it contains team-recorded robot
video and has not been cleared for public redistribution. The schema, builder,
source hashes and derived hashes are public.

### 3.2 Action and observation contract

The frozen action order is:

```text
left:  shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll, gripper
right: shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll, gripper
```

Real revolute joints are expressed in degrees, while the gripper uses closed =
0 and open = 100. The Genesis adapter converts revolute joints to radians and
maps the gripper monotonically into the model joint range. Round-trip and
ordering tests run without robotics or ML dependencies.

### 3.3 Intervention labels

The 40 HIL episodes were human-reviewed. Successful policy frames receive the
ordinary behavior weight. In failed episodes, explicit correction intervals
identify where the operator took over; the preceding autonomous segment is
labeled as a failed policy prefix.

| Role | Frames | Raw weight | Normalized gradient-mass share |
|---|---:|---:|---:|
| Behavior demonstration | 116,550 | 1.00 | 54.24% |
| Successful policy | 34,101 | 1.00 | 15.87% |
| Failed policy prefix | 11,908 | 0.05 | 0.28% |
| Human correction | 15,906 | 4.00 | 29.61% |

Weights are normalized over positive frames so the positive mean is one. This
keeps the overall loss scale comparable while changing which frames dominate
the gradient. The sidecar hash is:

```text
b34f5559db6d697890a08b6a7e49b52f3a916537c3328aea72c8470bb90bec57
```

## 4. Genesis environment

The simulation component is deliberately minimal. It downloads and verifies
the official SO-101 MJCF and mesh assets, instantiates two arms, a table, a
rigid object and two cameras, and exposes the same state and image keys as the
real dataset. It includes reset, position control, deterministic stepping,
attached-camera motion and an explicit placement predicate.

The formal smoke ran 1,000 steps on `gs.amdgpu`, exercised both arms, rendered
the front and attached hand camera, and validated finite 12-value state. Scene
build took 144.49 s, dominated by one-time Genesis compilation. Non-render
step latency was:

| Statistic | Time |
|---|---:|
| Median | 4.02 ms |
| p95 | 4.58 ms |
| p99 | 5.22 ms |

The all-step mean is not used as the steady-state result because it includes
two explicit high-resolution renders. The scripted motion is a build/control
test and does not execute a learned handover; `task_success=false` is recorded
in the formal metrics.

## 5. Policy method

### 5.1 Baseline

The baseline is the default LeRobot ACT architecture (51.6 million parameters
in the smoke evidence), initialized randomly and trained on every frame with
uniform sample contribution. Its role is a reproducible control, not a
historical checkpoint.

### 5.2 Phase-aware ACT

The phase-aware model uses the same architecture, random initialization,
observations, action targets, optimizer defaults, batch size, worker count,
seed and 10,000-step budget. Only the per-frame ACT loss weighting changes.
The patched training path looks up each sampled dataset index in the immutable
sidecar, applies the weight to the per-item loss and normalizes nonzero batch
weights. A startup audit reports coverage and weight statistics before the
first optimizer update.

The design encodes a simple hypothesis: the short human correction after a
failure contains more information about recovery and transition timing than a
long autonomous prefix already known to be unsuccessful. Weighting avoids
duplicating frames and preserves an exact baseline comparison.

### 5.3 Selection rule

The dataset has no held-out validation split, and the Genesis scene is not a
calibrated replica suitable for choosing a real-task checkpoint. We therefore
predeclared the final checkpoint at step 10,000 as the only candidate for each
method. Checkpoints saved every 1,000 steps remain debugging artifacts and are
not searched after observing loss.

## 6. CPU-edge safety

The dependency-free safety controller uses monotonically increasing sequence
IDs and monotonic-clock timestamps. Before any chunk can reach robot I/O it
requires:

- an armed, non-latched controller;
- a current observation and command;
- exact observation/action sequence correspondence;
- a non-empty, finite 12-DoF action chunk;
- joint and gripper values inside configured limits; and
- per-joint delta below the configured bound.

A stale observation, stale command, reordered packet, mismatch or limit
violation latches E-stop. Software recovery is not permitted: a new controller
must be created after physical reset. These checks supplement rather than
replace manufacturer safety limits, workspace clearance and an operator with
access to the physical stop.

## 7. Evaluation protocol

### 7.1 Training and runtime measurements

Both formal runs use the same machine, dataset hash, seed, batch size and
training budget. We report elapsed wall time, terminal training loss,
checkpoint hash and externally sampled peak GPU memory. These are systems and
optimization measurements, not robot-success estimates.

For inference, a real dataset observation is decoded into the same camera and
state dictionary used at runtime. After 20 warm-up calls, 200 synchronized
full-chunk calls measure model execution plus preprocessing/postprocessing.
The policy is then invoked without reset to measure queued action dispatch.
The report includes mean, median, p95 and p99 latency, action shape, finiteness,
chunk horizon and peak allocated VRAM.

### 7.2 Action reconstruction diagnostic

The evaluator selects up to 256 evenly spaced frame indices from each segment
role. It compares the predicted normalized ACT action chunk and the first
action against dataset targets. Selection is deterministic and identical for
both checkpoints. Because all samples are training frames, the values are
reported as reconstruction diagnostics only. They cannot estimate
generalization or closed-loop task success.

### 7.3 Real-robot boundary

The source project contains a complete physical perception-policy-control
loop. A prior reviewed 45-episode batch recorded 37 successes and eight
`handover_failed` outcomes (82.22%). Hashes of the run, annotations, action log,
launch ledger and checkpoint were captured from a read-only evidence host.
That checkpoint predates this formal experiment and is neither a parent nor a
result. No success rate for the new checkpoints is claimed without new,
supervised physical rollouts.

## 8. Formal results

### 8.1 Environment

| Run | Result | Formal evidence |
|---|---|---|
| ROCm/PyTorch/Genesis clean bootstrap | PASS | one `gfx1100`, AMD matmul, `gs.amdgpu`, Vulkan enumeration |
| Dataset build | PASS | 124 episodes, 178,465 frames, exact derived hash |
| Genesis dual-arm smoke | PASS | 1,000 steps, two cameras, 12-value state |

### 8.2 Paired ACT experiment

| Metric | Baseline ACT | Phase-aware ACT |
|---|---:|---:|
| Updates | 10,000 (running) | Pending |
| Training wall time | Pending | Pending |
| Terminal training loss | Pending | Pending |
| Peak sampled VRAM | Pending | Pending |
| Step-10,000 checkpoint SHA-256 | Pending | Pending |

### 8.3 Inference and reconstruction

| Metric | Baseline ACT | Phase-aware ACT |
|---|---:|---:|
| Full 100-action chunk, median | Pending | Pending |
| Full chunk, p95 | Pending | Pending |
| Queued action dispatch, median | Pending | Pending |
| Peak allocated inference VRAM | Pending | Pending |
| Stratified normalized action L1 | Pending | Pending |

No value in these tables will be filled from a shadow host, smoke checkpoint,
MI300X, APU/NPU, post-hoc checkpoint choice or unregistered directory.

## 9. Negative results and scope decisions

### 9.1 Gaussian workspace twin

We evaluated adding a Vulkan/RADV Gaussian workspace representation through
VkSplat. Existing machines contained a corgi example, synthetic cameras and
unrelated 3D/4D Gaussian research, but not a calibrated static multi-view
capture of the competition SO-101 workspace. Substituting those assets would
make an attractive demo without demonstrating the same task. The branch was
therefore gated out before formal optimization. Its hash-pinned runner and
capture schema remain as reproducible future work, but Gaussian output is not
an observation, result or claimed contribution in this release.

### 9.2 Aborted checkpoint-selection run

An initial formal baseline was stopped at step 250 after discovering that its
configuration text described fixed validation seeds even though the dataset
has no validation split. The run remains marked failed. Before restarting, the
selection rule was changed to the predeclared final training step and covered
by a regression test. Preserving this negative run is part of the evidence
policy.

### 9.3 General limitations

- The formal dataset is train-only and cannot quantify generalization.
- The Genesis scene verifies AMD execution and interfaces but is not a
  calibrated sim-to-real benchmark.
- Historical physical success does not measure either new checkpoint.
- Raw team video cannot be redistributed, limiting one-command external data
  reproduction; authorized reviewers can reproduce from registered hashes.
- The task is constrained to one object family and workspace.
- The CPU-edge safety kernel has unit tests but no safety certification.

## 10. Reproducibility

The public repository contains:

- exact environment and dependency bootstraps with wheel/source hashes;
- immutable data merger, phase-target builder and public source/derived hashes;
- frozen baseline, phase-aware and single-Radeon configurations;
- a hash-verified SO-101 Genesis asset fetcher and minimal scene;
- fair-pair checks and exact LeRobot command generation;
- latency, reconstruction and safety evaluators;
- job manifests, GPU sampling, terminal markers and SHA-256 ledgers; and
- unit and shell validation invoked by `./ops/validate_scaffold.sh`.

The root README gives clean-clone commands and makes the access-controlled data
boundary explicit. Project code is Apache-2.0; dependency and asset status is
listed in `THIRD_PARTY_NOTICES.md`.

## 11. Upstream community contribution

During the competition we filed
[Genesis issue #3163](https://github.com/Genesis-Embodied-AI/genesis-world/issues/3163)
to improve the project's AMD support path. The checked-in
`docker/Dockerfile.amdgpu` still describes a ROCm 6.4.1 / PyTorch 2.6 /
Genesis 0.2.1-era target, while Radeon OneLoop provides public native-host
evidence for ROCm 7.2.1, AMD PyTorch 2.9.1 and Genesis 1.3.1 on `gfx1100`.
The report proposes an explicit supported tuple, pinned compatibility-sensitive
packages and a `gs.amdgpu` scene smoke. It also states that we did not build the
current Dockerfile, avoiding an unverified container claim. A code PR is
offered once maintainers select the intended base-image tag.

## 12. Team contribution

Phi Media Lab developed the real SO-101 collection and HIL workflow, reviewed
the intervention episodes, implemented the phase-aware ACT extension, built
the single-Radeon evidence path, integrated the Genesis environment, and
authored the safety and evaluation tooling. Upstream Genesis, PyTorch, ROCm,
LeRobot and SO-101 assets remain credited to their respective authors.

## References

1. AMD-DEV-CONTEST, *Radeon Hackathon 2026-07 official repository*.
2. Genesis Authors, *Genesis: A Universal and Generative Physics Engine for
   Robotics and Beyond*, Genesis 1.3.1 documentation and source.
3. Zhao et al., *Learning Fine-Grained Bimanual Manipulation with Low-Cost
   Hardware*, ACT / ALOHA.
4. Hugging Face, *LeRobot*.
5. AMD, *ROCm 7.2.1 Radeon documentation and PyTorch wheels*.
