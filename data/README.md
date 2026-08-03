# Data contract

Raw and derived data are intentionally excluded from Git. Register every
dataset version in `registry.yaml` using immutable hashes and documented
provenance before a formal job consumes it.

The policy contract must freeze:

- left/right camera keys, resolution, color order, and normalization;
- joint-state and 12-DoF action ordering;
- gripper conversion between the real robot and simulation;
- control rate and action-chunk semantics;
- phase labels and train/evaluation split; and
- consent, license, and redistribution status.

Never include private paths, credentials, participant identifiers, or raw
images in the public registry.

The frozen formal input is 124 real episodes (84 human demonstrations plus 40
reviewed HIL episodes). Build it without mutating either source:

```bash
oneloop-merge-data \
  --bc-root /root/radeon-oneloop-data/sources/bc_seed \
  --hil-root /root/radeon-oneloop-data/sources/hil_batch1_batch2 \
  --hil-manifest /root/radeon-oneloop-data/sources/hil_batch1_batch2/makermods_hil/combined_hil_batch1_batch2_phase_aware_awr_v2_20260507/handover_rl_seed_manifest_v0.jsonl \
  --output /root/radeon-oneloop-data/formal_handover_v1
```

The builder verifies the two camera keys, 480x640 RGB shape, 30 Hz control
rate, action/state order, source counts, tasks, videos, and manifest coverage.
It emits a full file hash ledger and a unified episode manifest.
