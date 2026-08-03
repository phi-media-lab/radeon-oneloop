# Radeon OneLoop demo video

Target duration: **4:00**. Spoken language and all on-screen text: English.
Only team-cleared robot footage may be uploaded; dataset videos remain private.

## Shot list and narration

### 0:00–0:25 — The task

**Picture:** Title, then a cleared real SO-101 handover clip. Pause briefly at
left grasp, transfer, right receive, and placement.

**Narration:** “Radeon OneLoop learns a contact-rich bimanual handover from
real demonstrations and reviewed human interventions. The challenge is not
only reaching the object. The two arms must coordinate exactly when control
and support transfer.”

**Overlay:** `Radeon OneLoop · Track 3 · Phi Media Lab`

### 0:25–0:55 — One-Radeon proof

**Picture:** Formal hardware record and a simple architecture animation:
dataset → Genesis / ACT training / ACT inference on Radeon → CPU safety edge →
two arms.

**Narration:** “Our formal accelerator path uses exactly one Radeon gfx1100.
Genesis runs through its AMD backend, both ACT variants train from scratch
through ROCm, and policy chunks execute on the same device. The CPU handles
cameras, robot I/O, validation, watchdog and emergency stop—never a second
inference path.”

**Overlay:** ROCm 7.2.1 · PyTorch 2.9.1 · Genesis 1.3.1 · one visible GPU

### 0:55–1:25 — Genesis environment

**Picture:** Front and hand-camera frames from the formal Genesis run; animate
the deterministic joint sweep beside the run manifest.

**Narration:** “A hash-verified SO-101 model gives us a reproducible dual-arm
environment with the same 12-value state and two 480 by 640 image keys as the
real pipeline. The 1,000-step formal run records a 4.02 millisecond median and
5.22 millisecond p99 non-render physics step. This is an environment and
interface test, not a simulated handover claim.”

### 1:25–2:05 — Phase-aware HIL

**Picture:** Four-row data chart, then emphasize corrections and dim failed
prefixes. Show the fixed baseline/phase-aware configs side by side.

**Narration:** “The immutable training set contains 124 real episodes and
178,465 frames. Human corrections are short but informative. Uniform training
lets long failed prefixes dominate. Our phase-aware objective gives correction
frames weight four, failed prefixes point zero five, and all successful or
demonstration frames weight one. Positive weights are normalized, and every
other training setting remains identical.”

**Overlay:** corrections: 15,906 frames → 29.61% of gradient mass

### 2:05–2:40 — Formal paired training

**Picture:** Synchronized loss curves, 10,000-step markers, GPU utilization and
the two checkpoint hashes.

**Narration template:** “Both policies start from random initialization with
the same seed, batch size, architecture, optimizer and 10,000-step budget on
the formal Radeon. [Insert final wall time, loss and memory comparison.] We
predeclared the final-step checkpoint; there is no post-hoc search on training
loss.”

### 2:40–3:10 — Real-time path and safety

**Picture:** Full-chunk and queued-dispatch latency plots; inject a stale
sequence in a terminal demo and show the controller latch E-stop.

**Narration template:** “ACT generates 100-action chunks. A real observation
produces a full chunk in [baseline / phase-aware latency], while queued actions
dispatch in [latency]. The edge rejects stale or reordered packets, mismatched
observations, unsafe joints and excessive deltas. Any violation latches the
stop.”

### 3:10–3:35 — Physical evidence, honestly bounded

**Picture:** Cleared examples from the historical reviewed batch: one success
and one `handover_failed`, labeled `prior non-formal pipeline evidence`.

**Narration:** “The inherited physical loop previously completed 37 of 45
reviewed handovers. We show that only to prove the full robot loop exists. It
predates the formal checkpoints and is not copied into our controlled result
table. Reconstruction on training frames is likewise reported only as a
diagnostic.”

### 3:35–4:00 — Reproduce and close

**Picture:** Public repository, `./ops/validate_scaffold.sh`, formal evidence
index, report, license.

**Narration:** “The public repository includes hash-pinned environments and
assets, the immutable data builder, frozen experiment pair, raw run lineage,
inference evaluators and safety tests. Radeon OneLoop shows how one Radeon can
carry a physical-AI project from environment, through learning, to real-time
deployment—without hiding negative results or borrowing another accelerator.”

## Required pre-export replacements

- Replace both narration templates with registered formal numbers.
- Use only cleared real-robot clips; otherwise substitute a task diagram and
  retain the historical text label.
- Show the full 64-character checkpoint hashes at least once.
- End card: source URL, report URL, Apache-2.0, and the public video URL.
- Export 1080p H.264/AAC, 24 or 30 fps, 3–5 minutes; verify audio and anonymous
  link access before filing the official PR.
