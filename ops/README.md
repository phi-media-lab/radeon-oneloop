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
