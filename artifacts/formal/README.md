# Formal evidence

This directory is the public, machine-readable subset of the immutable run
directories on the formal `radeon-c` host. A result belongs here only when the
job manifest says `formal: true`, `host: radeon-c`, status `done`, and GPU UID
`0x153f7d55778ab659`.

`ops/collect_formal_job.sh` enforces those conditions and refuses to overwrite
an existing evidence directory. Each published job contains the exact command,
frozen configuration and environment, hardware identity, one-second GPU
samples, stdout/stderr, metrics, original hash ledger, a collection hash ledger
and the `DONE` marker.

Checkpoints and access-controlled robot data are intentionally not committed.
Their deterministic tree or dataset hashes are recorded in the formal registry
and technical report. Failed runs remain on the formal host and are summarized
in the report, but a `FAILED` job cannot pass the public collector.

The root of truth for accepted claims is
[`ops/formal_run_registry.yaml`](../../ops/formal_run_registry.yaml).
