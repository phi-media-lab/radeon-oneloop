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
