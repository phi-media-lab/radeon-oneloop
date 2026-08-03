# Policy workstream

The formal policy family is ACT. Maintain exactly two comparable formal
experiments: baseline and phase-aware. Both must use the same architecture,
data split, observation/action contract, and evaluation protocol; the
phase-aware run may change only the documented sampling or weighting method.

The frozen method is per-frame loss weighting: behavior demonstrations and
successful policy frames have weight 1; failed autonomous prefixes have 0.05;
human recovery/correction frames have 4; unusable failures have 0. Positive
weights are normalized to mean one. Both policies start from random
initialization on `radeon-c`; historical MI300X checkpoints are prohibited.

Generate the phase sidecar, then print or execute the exact LeRobot command:

```bash
oneloop-build-targets --dataset-parquet DATA/data/chunk-000/file-000.parquet \
  --episode-manifest DATA/oneloop/episode_manifest.jsonl \
  --output-parquet DATA/oneloop/phase_targets.parquet \
  --report DATA/oneloop/phase_targets_report.json

oneloop-train-command --config configs/act_baseline.yaml \
  --paired-config configs/act_phase_aware.yaml --dataset-root DATA \
  --output-dir RUN/output
```
