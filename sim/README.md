# Simulation workstream

Implement only the minimal Genesis SO-101 bimanual scene required for build,
reset, deterministic stepping, native camera observations, joint/control tests,
and closed-loop evaluation. Large-scale domain randomization and Gaussian
camera integration are out of scope for the competition build.

`sim/genesis_so101` implements a deterministic dual-arm scene from the pinned
official SO-101 MJCF. Assets are not committed: `fetch_assets.py` downloads all
14 files and rejects any size or SHA-256 mismatch. The scene converts LeRobot
joint degrees and 0..100 gripper values to Genesis radians, exposes the frozen
12-value state and two 480x640 camera keys, and provides an explicit placement
success predicate.

```bash
PYTHONPATH=src:. python -m sim.genesis_so101.scripted_smoke \
  --asset-root sim/genesis_so101/assets/so101 \
  --output runs/genesis-smoke --steps 1000
```

The scripted sweep is a build/control/camera test, not a handover result.

Historical Real2Sim trajectories exported by `gaussian.hil_capture` can be
replayed without opening any physical device:

```bash
PYTHONPATH=src:. python -m sim.genesis_so101.replay_hil \
  --trajectory /path/to/workspace/trajectories/episode_000.npz \
  --asset-root sim/genesis_so101/assets/so101 \
  --output /path/to/replay \
  --workspace-texture /path/to/genesis_table_texture.png \
  --front-camera-calibration /path/to/front_camera_calibration.json \
  --render-fps 10 --record-video
```

The replay clamps historical calibration overshoot to the pinned MJCF limits,
records both gripper transforms and the attached hand-camera transform, and
records the content hash of the HIL-derived object proxy. It remains ineligible
for formal handover-success metrics until its size, mass and material response
are measured rather than estimated.

The optional front-camera file is accepted only if its own geometry and
reprojection quality gate passed. Genesis consumes its position, look-at, up
vector, and vertical field of view; the replay records the calibration hash.

For an interactive, offline-only inspection, loop the same trajectory in the
Genesis viewer. This opens no serial device and sends no physical command:

```bash
PYTHONPATH=src:. python -m sim.genesis_so101.view_hil \
  --trajectory /path/to/workspace/trajectories/episode_000.npz \
  --asset-root sim/genesis_so101/assets/so101 \
  --workspace-texture /path/to/genesis_table_texture.png \
  --front-camera-calibration /path/to/front_camera_calibration.json \
  --output /path/to/viewer-run --loop
```

## HIL-derived handover object

The historical object is the MINISO/Disney Mickey Fun Crash Series
`Graffiti Mickey` vinyl-plush pendant. The procedural asset separates its
colored **physics/debug** visual, closed plush-body collision/PBD mesh, and
flexible display-only strap/keyring. It does not use 3DGS and is not the final
photorealistic Real2Sim appearance. Regenerate all outputs from the single
parameter source with:

```bash
PYTHONPATH=src:. python -m sim.genesis_so101.handover_asset \
  --config configs/handover_object.json
```

The dual-arm scene convexifies the accessory-free hybrid visual for stable
real-time grasp and contact-feedback work. A separate plush-body-only PBD
profile checks that the same body dimensions, mass prior and soft parameters
build and deform on the AMD backend. Genesis
1.3.1 uses PyMeshLab while tetrahedralizing this mesh; install
`pymeshlab==2025.7.post1` in the host's Genesis environment, then run:

```bash
ONELOOP_SOFT_OBJECT_STEPS=240 ops/run_amd_soft_object_smoke.sh
```

This wrapper acquires the shared GPU lock and writes a nonformal manifest,
metrics, captures, hashes and a terminal marker. The soft-object result is a
feasibility smoke, not a calibrated hybrid-material benchmark.

## Live dual-leader bridge

The live bridge deliberately uses two processes so serial/LeRobot dependencies
remain isolated from the ROCm/Genesis environment. It supports both localhost
and split-host operation with the same UDP wire contract. In its default
`monitor` mode, the publisher only opens the two calibrated leader buses and
reads positions; it never sends a command to a physical follower.

Leader calibration can report a few degrees beyond the pinned MJCF limits.
The consumer retains the raw 12-DoF packet, explicitly clamps only the virtual
command to those per-joint model limits, and reports every clamped value plus
the resulting simulated-state tracking error.

Start the Genesis consumer first:

```bash
PYTHONPATH=src:. python -m sim.genesis_so101.live_teleop \
  --asset-root sim/genesis_so101/assets/so101 \
  --output runs/genesis-live \
  --bind-host 127.0.0.1 --port 58081 \
  --duration-s 30 --watchdog-ms 250
```

Then start the serial publisher in the LeRobot environment. Use stable
`/dev/serial/by-id/...` paths supplied at runtime; hardware serial numbers are
not stored in this repository:

```bash
PYTHONPATH=src:. python -m sim.genesis_so101.leader_publisher \
  --left-port /dev/serial/by-id/LEFT \
  --right-port /dev/serial/by-id/RIGHT \
  --left-id CALIBRATION_ID_LEFT \
  --right-id CALIBRATION_ID_RIGHT \
  --destination-host 127.0.0.1 --destination-port 58081 \
  --hz 30 --duration-s 30
```

For split-host operation, bind the consumer to the intended network interface,
set `--source-host` to the leader host address, and point the publisher's
`--destination-host` at the Genesis host. Firewall exposure should be limited
to that single UDP source and port. Receiver-local arrival time drives the
watchdog because monotonic clocks cannot be compared across hosts.

On the `amd` APU host, the wrapper below acquires the shared GPU lock, launches
both processes, marks the run `formal: false`, and writes logs, metrics, hashes,
and an atomic completion marker. Its four arguments remain host-local so USB
serial identifiers and calibration names never enter version control.

```bash
ONELOOP_LIVE_DURATION_S=30 ops/run_amd_live_bridge.sh \
  /dev/serial/by-id/LEFT /dev/serial/by-id/RIGHT \
  CALIBRATION_ID_LEFT CALIBRATION_ID_RIGHT
```

Set `ONELOOP_SHOW_VIEWER=1` from the active desktop session to open the live
Genesis viewer. The wrapper then preserves that session's runtime directory;
headless runs continue to use an isolated one.

For the photorealistic object demo, do not use the authoritative Genesis debug
viewer as the appearance window. The object is deliberately invisible there;
that process owns physics only. The loopback Gaussian presenter is the single
object-appearance view. After reviewing the hardware-free full-chain output,
launch the exact-hash SEVA candidate with:

```bash
ONELOOP_PROJECT_OWNER_VISUAL_CONFIRMATION=accepted \
ONELOOP_OBSERVED_CORE_ROOT=/path/to/full_geometry_asset \
ops/run_amd_seva_full_geometry_dual_leader.sh \
  /dev/serial/by-id/LEFT /dev/serial/by-id/RIGHT \
  CALIBRATION_ID_LEFT CALIBRATION_ID_RIGHT
```

The receipt variable is intentionally required. The wrapper disables the
legacy debug viewer and completed-appearance path, opens only the Gaussian
presenter, reads the two leaders in monitor mode, and never commands a motor.

## Haptic return path

The consumer can return contact-gated simulated joint reaction efforts to the
leader host at 30 Hz. The runner defaults to protocol monitoring and records
`physical_output_commands: false`; it never enables leader torque unless the
explicit single-joint bench gate is selected. That gate is limited to one
non-gripper motor, 30/1000 torque, one degree of offset and ten seconds of
output. The staged position-impedance design, limits, watchdog, estop
requirements and hardware bring-up gates are documented in
[`HAPTICS.md`](HAPTICS.md).
