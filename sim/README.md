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
