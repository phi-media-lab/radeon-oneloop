# Learned-mesh candidate review — 2026-08-05

## Decision

Hunyuan3D-2mv seed `10030` is selected only as nonformal generated
conditioning. Seeds `10028` and `10029` pass the same topology checklist but
are retained as dominated alternatives. None of the three is observed or
final metric geometry, a collision mesh, held-out evidence, or formal
single-Radeon lineage.

The sweep starts directly from the four reviewed real photographs. It does
not use the rejected seed `10027` mesh or any derivative. Exact hashes and
decisions are bound in
[`learned_mesh_candidate_review_2026-08-05.json`](learned_mesh_candidate_review_2026-08-05.json).

## Candidate comparison

| Seed | Mean/min four-view silhouette IoU | Metric extents (mm) | Topology review | Decision |
| --- | --- | --- | --- | --- |
| `10028` | `0.77174 / 0.74719` | `84.72 × 79.52 × 95.00` | all seven checks pass | superseded valid candidate |
| `10029` | `0.77457 / 0.75148` | `85.40 × 80.57 × 95.00` | all seven checks pass | superseded valid candidate |
| `10030` | `0.77664 / 0.75134` | `84.97 × 79.31 × 95.00` | all seven checks pass | accepted conditioning only |

The numeric rank alone is not an acceptance gate. The 12-angle continuous
orbit must also show one front face, no face on the rear hemisphere, two
correctly asymmetric ears, one rear strap, and no duplicate or discontinuous
body surface. Two private HIL rear/top exemplars confirm the same oval white
body and cyan-ear identity qualitatively. Their camera poses are unknown, so
they do not contribute a calibrated metric.

## Remaining defects and boundary

All candidates use four-view vertex-color projection, which visibly blurs
inter-cardinal appearance. The generated hidden-side shape is not metric
truth, and the unseen underside contains no identity texture. The selected
orbit may enter a Vista4D/direct-appearance A/B only after its independent
review hash is verified. It may never initialize physics collision geometry.

SEVA remains the preferred four-image novel-view generator. The current phi
installation has an invalid stored Hugging Face credential and no local v1.1
weights; no credential is recorded in this repository or in experiment logs.
