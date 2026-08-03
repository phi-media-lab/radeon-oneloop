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
