# Repository agent guidance

## Scope

Radeon OneLoop has one formal task, one formal GPU, and two formal ACT models.
Do not expand the competition scope without updating the project plan and
removing an equivalent-cost item.

## Formal evidence

- Only `radeon-c / GPU0 / gfx1100` can produce formal metrics.
- Runs from `radeon-f`, `amd`, and `phi-amd-work` must set `formal: false`.
- Never copy a number into a report unless its job is listed in
  `ops/formal_run_registry.yaml` and its raw artifact exists.
- Never inherit an MI300X checkpoint into the formal ACT lineage.
- Preserve failed runs and negative results.

## Workstream ownership

- `sim/`: minimal Genesis environment only.
- `policy/`: baseline and phase-aware ACT only.
- `gaussian/`: static calibrated workspace twin; not a policy input.
- `runtime/`: CPU-edge protocol and safety; no secondary accelerator.
- `reports/` and `submission/`: English public deliverables.

Keep concurrent agents in separate branches/worktrees and non-overlapping
directories. The main agent owns integration and the formal registry.

## Safety and secrets

- Do not commit SSH configuration, tokens, private keys, raw robot data, or
  private dataset paths.
- Real-robot commands require timeout, joint/action limits, watchdog, and
  emergency-stop behavior.
- Do not start two GPU jobs on one host; use the host GPU lock.
- Long remote jobs must write a manifest, logs, metrics, hashes, and an atomic
  `DONE` or `FAILED` marker.

## Validation

Run `./ops/validate_scaffold.sh` before committing structural changes. Add the
smallest relevant test with each implementation change and record the exact
formal command before launching a long run.
