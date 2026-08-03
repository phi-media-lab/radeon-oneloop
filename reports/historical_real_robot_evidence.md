# Historical real-robot evidence

This ledger records inherited evidence from the complete SO-101 HIL pipeline
that motivated Radeon OneLoop. It is **prior, non-formal evidence**: the policy
was trained before the competition formal profile was frozen and is not used as
a checkpoint parent or as a result in the controlled baseline-versus-phase-aware
comparison.

## Reviewed 45-episode batch

- Run: `hil_20260508_103950_so101-handover-act-awr-v2-phase-aware-batch1-batch2-10000steps-20260507`
- Hardware mode recorded by the run: real SO-101 bimanual robot, optimized
  inference, 100-action chunks, human outcome review.
- Status: completed, 45/45 episodes annotated.
- Outcome: 37 success, 8 `handover_failed` (37/45 = 82.22%).
- Annotation semantics: successful episodes explicitly state that no operator
  intervention was recorded; failures were marked by the operator after
  observation.

Source hashes on the read-only `amd` evidence host:

```text
b337f0195109b88b00904666d91f1ef23c96c08ae695b190cd0c9ff998282c30  hil_run.json
dbc3ba5b240f2d7c4f42416336c4bc45f5ec2d36be94b1489931fbff9612bfd9  annotations.json
1f6d2626b849bbf5969305b8e736e7d8f9627688d3c57ed975f5f5d0c8601a74  record_action_log.jsonl
420b42b62c9925bb3ca26f6c13534158a7789a62d60c81e159865a6b74085718  rollout_launches.jsonl
1ffd4c733bf33b5203ef0be8c4051351c2cb83bed4cb0e995f1ceffe156175d3  model.safetensors
```

## Submission boundary

The 37/45 result establishes that the source project has operated as a complete
real perception-policy-control loop. It does not establish the performance of
the new formal checkpoints. Formal training, single-Radeon latency, and any new
closed-loop evaluation are reported separately through
`ops/formal_run_registry.yaml`; no historical MI300X or APU number may be copied
into the formal comparison table.
