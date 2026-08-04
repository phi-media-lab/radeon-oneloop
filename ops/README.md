# Operations

Each remote job must acquire `/tmp/radeon-oneloop-gpu0.lock`, run from an
immutable Git commit, and write to its own `runs/<job_id>/` directory.

Required output:

```text
manifest.json
command.sh
stdout.log
stderr.log
environment.txt
hardware.json
metrics.json
hashes.sha256
DONE or FAILED
```

Only the main integration thread may add a run to
`formal_run_registry.yaml`. Failed runs remain preserved.

## Environment preflight

Run the pinned ROCm 7.2.1 / Python 3.12 environment bootstrap on the shadow
host first:

```bash
ssh radeon-f 'bash -s' < ops/bootstrap_rocm721_env.sh
```

The script installs the official AMD PyTorch 2.9.1 wheel set in an isolated
venv, pins Genesis v1.3.1, verifies one visible gfx1100 device with a GPU
matmul, initializes Genesis with `gs.amdgpu`, and records Vulkan enumeration.
Only reproduce it on `radeon-c` after the shadow run passes.

After committing an exact source revision, deploy it without a GitHub token and
install the pinned public LeRobot dependency:

```bash
./ops/deploy_bundle.sh radeon-f
ssh radeon-f 'bash /root/radeon-oneloop/current/ops/bootstrap_lerobot_env.sh'
```

Transfer the two access-controlled source datasets through the local SSH
control plane and verify their expected metadata:

```bash
./ops/transfer_formal_inputs.sh radeon-f
```

All GPU commands run through `run_job.sh`; `dispatch.sh` supplies the immutable
remote checkout and rejects formal identity mismatches. For example:

```bash
./ops/dispatch.sh radeon-f genesis_smoke configs/genesis_minimal.yaml false null -- \
  /root/radeon-oneloop-env/rocm721-py312/bin/python -m \
  sim.genesis_so101.scripted_smoke --asset-root \
  /root/radeon-oneloop-data/assets/so101 --output \
  /root/radeon-oneloop-runs/genesis-shadow --steps 1000
```

Evaluation jobs bind their input model in the manifest as well as in the exact
command. Supply the deterministic checkpoint-tree digest when dispatching:

```bash
ONELOOP_PARENT_CHECKPOINT=64_HEX_SHA256 \
  ./ops/dispatch.sh radeon-c act_eval configs/act_baseline.yaml true DATASET_SHA256 -- \
  /root/radeon-oneloop-env/rocm721-py312/bin/python -m evaluation.policy_latency \
  --checkpoint /absolute/checkpoint/pretrained_model \
  --dataset-root /root/radeon-oneloop-data/formal_handover_v1
```

After hashing both predeclared final checkpoints, the complete matched latency
and stratified-reconstruction suite can be dispatched sequentially with:

```bash
./ops/run_formal_pair_evaluations.sh \
  /absolute/baseline/checkpoints/010000/pretrained_model BASELINE_SHA256 \
  /absolute/phase/checkpoints/010000/pretrained_model PHASE_SHA256
```

## Nonformal Vista4D surface-carrier branch

The private reviewed-photo root is supplied only at execution time. The AMD
carrier and portable visual conversion remain `formal: false`:

```bash
ONELOOP_M1_MANIFEST=/path/to/reviewed_m1/manifest.json \
  ./ops/run_amd_surface_carrier.sh

ONELOOP_OBSERVED_CORE_ROOT=/path/to/observed_core \
ONELOOP_VISTA4D_SURFACE_CARRIER_ROOT=/path/to/carrier/artifacts \
  ./ops/run_amd_surface_carrier_glb.sh

ONELOOP_SURFACE_CARRIER_ROOT=/path/to/carrier/artifacts \
  ./ops/run_amd_vista4d_mask_alignment.sh
```

The current carrier-conditioned Vista4D runs failed their identity gate. Do
not feed them into `completion_candidate.py` or replace the Genesis collision
proxy with the exported GLB.
