# MGPBD plush-physics forensic audit and recovery plan — 2026-08-06

## Decision

The current doll result does **not** establish that MGPBD is unsuitable for a
tightly filled plush object.  It combines five independently invalidated
layers: an inflated convex physical proxy, a topologically fragmented visual
LOD, an under-converged global solve, a post-solve contact projection, and
whole-object grasp transport heuristics.  The prior "water-ball" judgement is
therefore withdrawn.

MGPBD remains a primary material-kernel candidate, but it is not yet the
integrated doll mainline.  It is evaluated as a homogenized high-stiffness ARAP
tetrahedral continuum.  The current conformance result proves that a scalar
line search plus determinant guard is insufficient for our stricter local
quality contract.  A scalable constrained direction has now been implemented
as a matrix-free all-tetrahedron SOC-ADMM solve and installed in the complete
P0a2 nonlinear projector.  Its full `bunny_small` gate is still in progress;
there is therefore no `bunnyBig`, doll, contact, or realtime conclusion yet.
The granular solver is retained only as a later material ablation, and the
rigid model remains a demo fallback.  No claim is made that MGPBD resolves
individual stuffing grains.

This work is development-only (`formal: false`).  Only `radeon-c / GPU0 /
gfx1100` may later produce formal competition metrics.  The conformance phase
does not access either USB leader and produces no physical robot output.

## Upstream reference boundary

The reference is Li et al., *MGPBD: Multigrid Preconditioned Position Based
Dynamics*, the authors' project page, and public repository:

- paper: <https://arxiv.org/abs/2505.13390>;
- project: <https://chunleili.github.io/project-page-mgpbd/>;
- public code: <https://github.com/chunleili/mgpbd>;
- audited public commit: `06761eb38dee8fb4165c6b9df8212c4f1744d131`;
- public bunny scene:
  `data/scene/bunny_squash/bunny_squash.json`.

The upstream repository currently contains no visible license file.  Its code
and bunny data must not be bundled into the submission or this repository.
Development runs may fetch the two pinned TetGen input files into an untracked
run directory and record their hashes.  Our implementation remains clean-room.

There is one upstream configuration ambiguity that the execution matrix must
preserve.  The paper describes a six-component bootstrapped near-kernel AMG
candidate, while the current public `bunny_squash.json` does not set
`build_P_method` and therefore inherits the public runner's plain `UA` default.
Neither result may be silently labelled as the other.

## What the official bunny actually demonstrates

The public `bunny_squash` scene uses:

| Property | Public value |
| --- | ---: |
| Model | `bunnyBig` |
| Vertices | 60,678 |
| Tetrahedra | 270,199 |
| Timestep | 0.01 s |
| Numeric `mu` | `1e9` |
| Particle mass | 1.0 per vertex (public default) |
| Gravity | disabled |
| Reinitialization | every vertex Y set to rest-state `y_min` |
| Nonlinear iterations | at most 20 per frame |
| Inner linear iterations | at most 100, tolerance `1e-5` |
| Public line-search first trial | 1.0 |
| Contact | none in the bunny scene |

The public code uses backtracking line search and first tries a unit step.  Its
`omega=0.1` default is used when line search is disabled; the paper separately
reports fixed soft-body relaxation near 0.1.  Public-as-is line search and the
paper fixed-relaxation variant must therefore remain separate configurations.

The public animation runs 100 dynamic frames.  Every frame performs zero-
gravity, unit-retention semi-Euler prediction, global projection, and velocity
update.  It visually demonstrates recovery of a coherent, high-stiffness ARAP
tetrahedral body from an extreme zero-thickness initialization.  The public
solver has no per-tetrahedron determinant constraint, however, so the video is
not evidence that every intermediate tetrahedron remains positively oriented.
It does not demonstrate frictional grasping by two moving robot grippers.  The
paper's separate static SDF collision response is not a ready-made two-gripper
contact solver.

The public runner does **not** export a boundary-only render mesh.
`build_face_indices()` emits all four statically ordered faces for every
tetrahedron without cancelling shared faces, and the PLY writer reuses that
connectivity while updating all physical vertex positions.  For `bunnyBig`
this is 1,080,796 triangle records (`4 * 270,199`), including duplicated
internal faces, rather than the 69,630 triangles of the clean physical
boundary.  All 60,678 physical vertices are written.  There is no separate
embedded display cage, but the public visualization topology is not a clean
boundary either.

The public entry point only writes headless PLY sequences.  The project-page
video metadata identifies Houdini 20.5.470, while its scene, culling,
sidedness, normal, and material settings are not published.  Consequently,
the visually intact video is neither a signed-volume certificate nor a
reproducible rendering baseline for our Genesis surface.

## Forensic comparison

| Layer | Official bunny | Current doll path | Consequence |
| --- | --- | --- | --- |
| Physical shape | Boundary-conforming bunny volume | Convex appearance hull, scaled by 1.01 and then 1.045 | Early invisible contact and loss of concave silhouette |
| Resolution | 270,199 tets | 9,390 tets | Coarse, non-local indentation |
| Display surface | All four faces per physical tet, including duplicated internal faces; static PLY connectivity | 37,393-face realtime LOD embedded in tets | Render topology and culling are not comparable; doll LOD cracks already exist at rest |
| Material solve | High-stiffness, up to 20 nonlinear iterations | Numeric `mu=150000`, one nonlinear iteration | Numerical softness and incomplete equilibrium |
| Linear solve | Up to 100 iterations, `1e-5` target | Eight PCG iterations, observed residual about 0.09–0.10 | The accepted state is materially under-converged |
| AMG hierarchy | Public matrix-derived UA; paper also reports near-kernel candidate | Static unweighted tet-topology aggregation for dense meshes | Poor preconditioning is exposed by the short PCG budget |
| Contact | No contact | Custom boundary sample projection after ARAP | Collision and elasticity pull against each other across frames |
| Control timing | Offline frame budget | One 1/120 s physical step per 4.5–8 Hz wall-clock loop | Moving grippers appear to jump between poses |
| Grasp transport | None | Coarse rigid transport and center lock | A passing lift does not prove frictional contact |

The numeric stiffness values are not directly comparable as measured material
moduli because the upstream object coordinates, particle masses, and our SI
geometry/mass convention differ.  Exact upstream conformance and SI doll
calibration are therefore separate experiments.

## Root-cause evidence

### A. The physical volume is an inflated convex blob

`ops/build_plush_collision_proxy.py` starts from `visual.convex_hull` and
applies a 1.01 scale.  `ops/build_mgpbd_dense_volume.py` remeshes that convex
surface and applies another 1.045 outward scale.  The accepted volume has
2.269 mm p01 visual-to-physics clearance and fills every concavity.

In the latest grasp gate, the inflated physical body already had 537 contacts
and 3.35 mm penetration against the left fixed collider while the rendered
surface penetration was only 0.13 mm.  The solver was responding to geometry
the operator could not see.

### B. The realtime visual asset is statically fragmented

The complete TRELLIS.2 asset has 293,972 faces.  After welding coincident
positions and removing geometric duplicates, the development builder audit
shows that it is still an open, non-manifold, multi-component triangle soup,
not the boundary of one solid.  This corrects the earlier claim that the weld
alone removed every open boundary edge.  The development-only source receipt
is `/tmp/oneloop_p1_wrap_a/receipt.json` on `amd`; it is not formal evidence.
The 37,393-face realtime LOD fails under two deliberately different topology
audits.  Positional welding while retaining the authored face list gives
1,367 vertex-connected components, 48,215 open edges, and 463 non-manifold
edges.  The P1 cleaner additionally removes degenerate/duplicate faces and
then measures edge-connected components; that stricter pipeline gives 7,456
components, 48,947 open edges, and 314 non-manifold edges.  Component counts
from these two definitions must not be mixed, but both independently reject
the LOD as an intact deforming skin.

These counts apply to
`graffiti_mickey_trellis2_real_front_seed12345_visual_realtime.obj`, SHA-256
`4b0a5a5e8b277e48841ff4de56aba068825478ee6b306bd95ab39608172d3ebc`.
The 293,972-face full visual (SHA-256
`5d5984ddb4146aa2da8ba8e6c7dfc0c67c0844eaa97522d6795afea97bb65063`)
is the appearance source, not a directly tetrahedralizable solid: after the
same P1 cleaner it still has 459 open edges, 4,675 non-manifold edges, and 102
edge-connected face components.  It therefore needs a separate closed outer
wrap followed by barycentric appearance binding.

`gaussian/decimate_textured_mesh.py` uses texture QEM with boundary
preservation disabled.  Both current MGPBD launchers explicitly select this
broken LOD.  No physics parameter can repair a surface that is disconnected
before simulation.  The full visual has already passed a static FEM tet
embedding test with zero visible face flips.

The full triangle soup also contains buried or overlapping sheets.  It is
therefore incorrect to require a small bidirectional Hausdorff distance
between every authored visual triangle and the physical outer boundary.  The
outer-volume contract is instead
`S_visual` strictly contained in `Omega_physical`, together with a one-way
complete-surface bound from `boundary(Omega_physical)` to `S_visual`.  A large
clearance from an internal visual sheet to the outer boundary is diagnostic,
not a geometry rejection.  This gate selects a geometric volume candidate;
it does not validate gripper contact, friction, or penetration.

### C. The global material solve is under-converged

`sim/genesis_so101/scene.py` currently configures one nonlinear iteration,
eight PCG iterations, one smoother sweep, and relaxation 0.60.  The latest
grasp run terminated its inner solve near relative residual 0.10.  This is a
numerically soft approximation, not the converged ARAP response shown by the
reference bunny.

The dense branch in `sim/genesis_so101/mgpbd_tet.py` builds a reusable
unweighted hierarchy from shared-tet topology with strength threshold forced
to zero.  The public reference UA instead derives aggregation from the current
dual matrix.  This is primarily a convergence-rate difference only if the
linear solve reaches tolerance; with eight iterations it changes the accepted
deformation.

The conformance audit also found two previously unrecorded differences:

1. the public bunny default uses uniform per-particle mass 1.0, while our
   projector always creates volume-weighted lumped masses from a total mass;
2. our projector globally clips an ARAP correction to 12 mm, while the public
   squash recovery has no equivalent live-contact displacement cap.

The conformance runner must explicitly select uniform masses and disable the
live correction cap without changing the existing robot-scene defaults.

### C.1 Exact squash is a singular fidelity probe, not a zero-inversion gate

The exact public reinitialization sets every vertex Y to `y_min`; all 12,298
small-bunny tetrahedra therefore start at determinant zero with rank-two
deformation gradients.  This is not merely a difficult linear solve.  On the
small bunny, a CPU sparse direct solve of the first unscaled global system
reached relative residual `1.73e-12`, yet five tetrahedra still had a
wrong-orientation directional derivative.  For every tested positive global
step (`1e-9` through `1.0`), those same five tetrahedra opened with negative
signed volume.  A scalar backtracking line search can only reduce this step to
zero; it cannot change the direction.

The three first Radeon projection probes used all 20 outer iterations and all
100 PCG iterations per outer.  Their recursive relative residuals remained
between approximately `0.006` and `0.040`, their final dual norms remained
`9.2` to `14.0` times above the configured stopping target, and they ended
with 7, 9, and 47 inverted tetrahedra respectively.  A separate 1,000-PCG
probe reached `1e-5` on the first linear solve but retained the five core
wrong-way elements.  Thus inner under-convergence adds inversions, but it is
not the source of the first five.

The P0 contract is consequently split:

- `official_fidelity` retains the exact flat initialization and public
  multiplier/position line-search semantics.  It records inversions and
  recovery as observations, not upstream acceptance requirements.
- `orientation_safe_recovery` is our downstream physical-safety extension. It
  starts from a strictly positive affine squash, uses determinant-feasible
  backtracking, and accepts or rejects the position and multiplier step
  atomically.  Only this contract may gate zero inversion for the doll path.

This distinction explains how the official bunny can look visually intact
while a strict element-level audit fails.  A small number of inverted
tetrahedra, especially interior ones, need not change the silhouette.  The
public PLY also contains duplicated internal faces, and the unpublished
Houdini culling/sidedness settings prevent a stronger inference.  Later frames
may recover some elements, but the public runner does not record that.  The
visual fact does not make the topology-safe requirement wrong for gripper
contact; it places the requirement in our downstream extension rather than in
the upstream fidelity claim.

The public implementation records neither signed rest orientation nor current
signed-volume ratios.  Its optional `calc_strain()` reports the maximum
singular-value ARAP residual, is disabled in the bunny configuration, and
cannot detect orientation reversal: reflection and its positive-orientation
counterpart have the same singular values.  The bunny loop converges on dual
residual or time, not on maximum strain.

### C.2 The live doll does not run the audited SOC path

`ops/run_amd_mgpbd_live.sh` selects `ONELOOP_PLUSH_PHYSICS_MODE=mgpbd` but does
not enable the constrained direction.  `scene.py` therefore constructs the
legacy live profile: relaxation 0.60, one nonlinear iteration, eight PCG
iterations, `mu=150000`, moving contact, gravity, correction clipping, and the
grasp helpers.  The P0a2 SOC profile instead uses up to 60 nonlinear outers,
up to thousands of inner ADMM/PCG iterations, `mu=1e9`, no gravity, no
contact, and no correction clip.  The currently visible doll has never run
the audited algorithm/configuration responsible for the new bunny evidence.

There is also an intentional integration block.  The contact solver requires
a finite collision-correction bound, while the SOC direction forbids any
post-direction clip.  A callback applied after the audited line search could
change the accepted position without rechecking Armijo, orientation, strain,
or the paired multiplier transaction.  The projector now rejects any
post-iteration callback when SOC/SQP is active.  Contact remains blocked until
it is included in the trial state and re-audited before commit, or split into
an explicitly labelled contact step followed by a new material projection and
full audit.

### D. Contact is a post-solve heuristic, not a coupled constraint

The current code samples boundary vertices, triangle centroids, and edge
midpoints, projects penetrated samples out of convex gripper hulls, and
scatters the full sample displacement to tet vertices.  This occurs after an
accepted global ARAP update.  With one outer iteration there is no same-frame
elastic re-equilibration after the contact correction.

The positive-volume barrier then decomposes the candidate into whole-body
translation plus local deformation.  If the local update would invert a tet,
it can reject all local deformation while retaining the translation.  The
latest nominally passing gate recorded:

- 1,738 barrier activations;
- minimum accepted local line fraction 0;
- worst unlimited candidate signed-volume ratio -1.052;
- repeated accepted ratios at the configured 0.20 floor.

These are collision failures hidden by rollback, not evidence of stable
contact.

### E. The latest lift gate is dominated by hidden transport

The latest passing development run was
`runs/20260806T043903Z_869924_amd_mgpbd_grasp_gate`:

- reported net lift: 54.21 mm;
- coarse transport active: 590 frames;
- center lock active: 511 frames;
- cumulative center-lock Z correction: +0.313048 m;
- cumulative coarse-transport Z correction: +0.00058 m;
- p95 step time: 122.63 ms.

Cumulative corrections are not net displacement, but their frequency and
magnitude prove that the object was continually repositioned to compensate for
missing support/contact.  `synthetic_attachment: false` is therefore too weak
to justify the old `object_lifts_without_tether` claim.

### F. The integration clock is not 120 Hz

The scene calls `object.step_simulation()` once per controller iteration; it
does not execute four 120 Hz substeps for each 30 Hz leader update.  The latest
live path achieved approximately 4.47 Hz with 153 ms p50 and 187 ms p95 step
times, and almost every input frame hit a tracking clamp.  A 1/120 s
integrator receiving gripper poses at this wall-clock cadence observes large
discrete collider jumps.

### G. Existing gates validate survival, not fidelity

The dense benchmark checks counts, finite timings, rest embedding, and final
tet orientation.  It does not fail on an unmet PCG residual, barrier
activation, hidden transport, full-trajectory penetration, visible face
flips, visual area collapse, missing substeps, or missed real-time budget.
Endpoint-only contact checks can pass after severe mid-trajectory penetration.

## Recovery execution plan

Every phase is fail-closed.  Infrastructure/input/evidence failure produces a
`FAILED` run.  A complete numerical experiment produces `DONE` plus either
`GATE_PASSED` or `GATE_FAILED`; an expected negative result is never confused
with a crashed job.  Every non-passing run is preserved as evidence and the
next phase remains blocked until the relevant safety contract passes.

### P0 — Reference conformance, no Genesis and no contact

#### P0a1: small-bunny official-fidelity projection

Use the upstream `bunny_small` mesh (2,992 vertices / 12,298 tets) with the
public exact squash initialization.  Run one projection with no integration to
isolate TetGen ingestion, mass convention, ARAP, matrix-derived UA, PCG,
line-search, and outer dual stopping.  This is a kernel contract, not the
complete public animation and not a replacement for `bunnyBig`.

Required configuration:

- `dt=0.01`, numeric `mu=1e9`, uniform particle mass 1.0;
- gravity/contact/integration disabled for this one projection only;
- up to 20 nonlinear iterations;
- public-as-is dual-objective backtracking beginning at 1.0;
- strict objective decrease, minimum trial `1e-9`, and full upstream
  multiplier update even when the accepted position step is zero;
- PCG maximum 100 and relative tolerance `1e-5`;
- no live 12 mm correction cap;
- exact upstream asset hashes recorded in the run manifest.

Required checks:

- finite positions and metrics;
- closed two-manifold physical boundary;
- every nonlinear outer refreshes numeric `A` and all `P^T A P` levels while
  reusing only the permitted prolongator structure;
- recurrence and true physical residuals are both reported;
- level-zero sparse action agrees with `J M^-1 J^T + alpha`;
- configured inner/outer targets are reported as a stricter clean-room quality
  result, not mislabelled as an upstream binary pass condition;
- inverted/collapsed counts, ARAP norm, maximum strain, and recovery are
  reported as observations rather than official acceptance conditions;
- normalized rest-shape recovery error and extent recovery are reported;
- no contact, transport, center lock, table, Genesis, viewer, serial, or USB
  code is imported.

#### P0a2: small-bunny orientation-safe projection

Repeat the kernel probe from a rest-oriented affine squash with initial height
ratio 0.25.  A direct linear oracle shows that this seed admits the complete
first global direction; ratios 0.10 and below force severe backtracking and
are continuation stress tests, not the first safety baseline.  Increase the inner/outer budgets for correctness rather than
matching the public cap.  Every candidate records its minimum signed-volume
ratio; backtracking must retain a positive accepted ratio, and rejection must
roll back both position and multiplier.  This is our safety extension and is
the first contract that requires zero inverted/collapsed tetrahedra, true
linear convergence, outer convergence, decreasing ARAP norm and maximum
strain, nondegenerate boundary faces, and mass-center conservation.

The executed safety contract additionally uses the downstream-only trust
filter `max_t ||sigma(F_t) - 1||_2 <= 1.0`.  Together with positive
determinant, this excludes the observed positive-volume but approximately
tenfold principal stretch.  It is an absolute feasible region rather than a
monotonic maximum-strain requirement: five small-bunny elements have a
wrong-way first derivative, so requiring the maximum to decrease on every
infinitesimal step would reject all progress.  This filter is not part of
public MGPBD and must be labelled as such in every receipt.

The executable P0a2 replacement no longer scales an already-invalid legacy
direction.  At every nonlinear outer it freezes the current closest proper
rotation `R_t` and solves the convex direction problem

```text
min_d  1/2 d^T M d + 1/2 (q + J d)^T alpha^-1 (q + J d)

subject to ||F_t - R_t + K_t d||_F <= 0.989  for every tetrahedron t.
```

Here `q = C + alpha * lambda` is the current material residual.  The resulting
MGPBD multiplier update is reconstructed as
`delta_lambda = -alpha^-1 (q + J d)`.  Distance to `SO(3)` is no greater than
distance to the frozen proper rotation, so the work-ball constraint also
majorizes the true ARAP constraint.  The proof radius is
`min(1, 1 - (1e-6)^(1/3)) = 0.99`; a per-block primal tolerance of `2e-4`
leaves the accepted `.989 -> .99` determinant-proof margin intact.  The outer
line search may shorten this direction, but it must apply exactly the same
fraction to position and multiplier and must roll both back together.

#### P0a3: two 100-frame dynamic recoveries

Run both contracts with the public frame loop:

1. `old = x`;
2. `predicted = x + dt * velocity` with zero gravity and retention 1.0;
3. project with fresh per-frame Lagrange multipliers;
4. `velocity = (projected - old) / dt`.

Save per-frame outer dual, PCG true/recurrence residual, RAP currency, ARAP,
volume, extent, and rigid-aligned recovery histories.  Exact-flat fidelity may
contain transient inversions; `orientation_safe_soc_recovery` may not.  The
retired scalar strain-trust path is not the orientation-safe dynamic contract.
A single projection cannot establish dynamic shape recovery.

Run the paper fixed-relaxation `omega=0.1` configuration only as a separately
labelled ablation.  Likewise, the paper's per-resolution time-budget results
are a separate ablation; the current public JSON uses iteration/dual stopping.

#### P0b: full public bunny

Repeat both the one-frame resource probe and the complete 100-frame dynamic
recovery with `bunnyBig` (60,678 vertices / 270,199 tets).  Record peak GPU
memory, hierarchy construction time, per-outer convergence, and total time.
This is expected to be offline; performance is not a correctness failure.

#### P0c: hierarchy ablation

On identical input and budget compare:

1. public-repository-as-is, matrix-derived plain UA;
2. the paper's six-component near-kernel construction;
3. our current static tet-topology UA.

Do not claim paper conformance from topology-UA alone.  The acceptance basis is
residual and recovered state, not the name of the hierarchy.

### P1 — Doll elasticity with no robot contact

Build a watertight, boundary-conforming volume from the intact full TRELLIS.2
asset using an SDF/wrap plus quality tetrahedralization.  A convex hull is
forbidden.  The complete visual triangle soup must be strictly contained, and
the one-way physical-boundary-to-visual distance must be at most 0.5 mm over
the complete physical boundary before and after tetrahedralization.  This is
a geometry-candidate gate only; contact validation remains in P3.

First render the physical tet boundary itself.  Run no-contact squash recovery
and slow parallel-platen squeeze/hold/release at several resolutions (initially
about 9k, 50k, and 200k tets) and calibrated dimensionless stiffnesses.

P1 passes only with zero inversions, zero barrier use, a converged solve,
localized indentation, full resting support, and shape recovery after release.

### P2 — Intact appearance binding

Bind the complete 293,972-face TRELLIS.2 visual through exact tetrahedral
barycentric embedding and local deformation-gradient normal transport.  The
current realtime LOD is quarantined.  Any future LOD must pass topology gates
before it enters a physics run.

P2 passes only with zero visible face flips, zero degenerate visible faces,
continuous appearance throughout squeeze/release, measured rest reconstruction
error below 0.1 micrometre, and maximum dynamic separation below 0.1 mm for
every rest-coincident or rest-near seam pair.  Face-flip and area checks alone
do not certify that independently bound triangle-soup components stay joined.

### P3 — Contact ladder, still no robot hardware

Disable coarse transport, center lock, jaw limiter, and translation-preserving
barrier fallback.  Use only straight distal finger pads:

1. static platen;
2. one slowly moving pad;
3. two slowly moving parallel pads;
4. scripted simulated SO-101;
5. simulated dual-leader trajectory replay.

Contact must be iterated with the ARAP material solve, not appended after its
last iteration.  Collider poses are interpolated over physical substeps.  At a
30 Hz control rate, four actual 120 Hz physics steps are required.

P3 gates the full trajectory: maximum physical and visible penetration at
most 0.5 mm for platen tests and at most 1.0 mm for SO-101, zero inversion,
positive barrier step fraction if a barrier remains, and lift caused only by
resolved contact/friction.  All transport/lock counters must remain zero.

### P4 — Interactive optimization and guarded operator validation

Only after P3 correctness may resolution, hierarchy reuse, render update rate,
and GPU kernels be optimized.  Maintain two explicit modes if necessary:

- high-quality reference mode, allowed to be offline;
- reduced interactive mode, accepted only if it independently passes the same
  topology/contact gates at its chosen resolution.

The public paper reports offline runtimes for its larger examples.  The
current APU's 4.5–8 Hz result does not justify sacrificing topology or adding
hidden grasp transport.  If real-time requires new Taichi/C++/ROCm kernels,
that is a performance task after correctness, not a material-model change.

Physical leader read and motor output remain disabled throughout P0–P3.  P4
operator validation may read the two leaders only through the existing
watchdog and action-limit path.  Motor output remains disabled unless a
separate explicit haptics trial is authorized.

## P0 execution record

### Superseded exact-flat development probes

The following AMD APU (`gfx1150`, ROCm 7.2.1) small-bunny projections are
preserved as negative evidence:

| Run suffix | Profile | PCG/outer result | Final inverted tets |
| --- | --- | --- | ---: |
| `075252Z_1027923` | public plain UA | 20 x 100 exhausted; final recursive residual `0.0197` | 9 |
| `075417Z_1028053` | diagonally equilibrated plain UA | 20 x 100 exhausted; no material improvement | 7 |
| `075800Z_1028453` | fixed `omega=0.1` | 20 x 100 exhausted; worse recovery | 47 |

All 12,298 constraints were active at every outer in these runs, so the later
stale-RAP fix does not alter their negative conclusion.  Their old `FAILED`
marker conflated a complete negative numerical experiment with infrastructure
failure; schema v2 now separates those states.

The old-source 100-frame job
`20260806T080048Z_1028717_amd_mgpbd_bunny_small_trajectory` was terminated at
source-drift detection and is marked `FAILED`.  It is not cited as solver
evidence.

### Schema-v2 orientation-safe height-0.10 probe

Run
`20260806T083327Z_1029720_amd_mgpbd_bunny_small_orientation_safe_recovery_projection`
is the first complete schema-v2 receipt.  It has `DONE + GATE_FAILED`, verified
artifact hashes, recorded source SHA-256 values, and recorded ROCm/Torch,
NumPy, SciPy, and PyAMG versions.  No Genesis, contact, leader, or motor path
was enabled.

Positive findings:

- all 60 outer solves used a freshly refreshed numeric RAP hierarchy;
- level-zero/physical-operator relative mismatch was at most `9.70e-6`;
- every accepted state remained positive; the minimum accepted signed-volume
  ratio was `3.16e-6`;
- mass-center drift was `2.93e-8` rest-box diagonals;
- source/configuration/evidence contract checks all passed.

Blocking findings:

- recursive residuals reached approximately `1e-5`, but independently
  recomputed true residuals remained `3.10e-5` to `6.18e-5`;
- 1,031 determinant backtracks reduced some accepted steps to
  `1.91e-6` after full candidates reached ratio `-674`;
- aggregate ARAP decreased from `99.81` to `71.23`, but the worst element
  strain increased from `0.900` to `2.492`;
- the final height recovered only from 0.10 to 0.296 of rest, and the outer
  dual target was not reached.

This is a successful falsification of the 0.10 first-baseline choice, not a
material-model failure.  The implementation now performs periodic true-
residual replacement/restart.  Follow-up probes use a 0.25 positive-height
seed for which the direct direction oracle admits a full first step.

### Clean official-fidelity receipts

Run
`20260806T084753Z_1030503_amd_mgpbd_bunny_small_official_fidelity_projection`
is a complete schema-v2 `DONE + GATE_FAILED` receipt with verified hashes.  All
contract checks passed: pinned inputs, clean-room plain-UA recipe, strict
public line-search semantics, 20 current RAP refreshes, level-zero/physical
operator agreement, source hashes, and the no-Genesis/no-contact/no-hardware
boundary.

It deliberately does not pass the stricter clean-room numerical quality
gate.  All 20 inner solves exhausted 100 PCG iterations with true residuals
`0.00846` to `0.03375`; outer dual decreased from `110.90` to `10.32` but did
not meet its `1%` target.  The final state had 11 inverted tetrahedra and
worst ARAP strain `5.53`; these are observations, not falsely attributed
upstream binary acceptance criteria.  Its direct CPU oracle reached
`1.51e-12` relative residual and proved the same five tetrahedra invert for
every positive tested step from exact flatness.

After the multiplier-transaction and finite-value audit, run
`20260806T090449Z_1031495_amd_mgpbd_bunny_small_official_fidelity_projection`
repeated the experiment from source hashes recorded in its manifest.  It is a
verified `DONE + GATE_FAILED` receipt with every contract check passing.  All
20 outers refreshed current RAP, all 20 recorded an
`accepted_full_trial_multiplier` transaction with selected fraction 1.0 and
zero transaction error, and the maximum level-zero/physical-operator mismatch
was `9.52e-6`.  The strict clean-room quality result remained negative: true
PCG residual reached as high as `0.03250`, the outer target was not met, and
the final state had 9 inverted tetrahedra with maximum ARAP strain `4.19`.
This updated result supersedes the precise final counts of the earlier source
while preserving its conclusion.

### Orientation-safe height-0.25 probes

Two source-hashed schema-v2 experiments isolate multiplier semantics:

1. `20260806T083949Z_1030053...` scaled multiplier and position by the same
   accepted line step.  All true linear residuals passed (`<=9.88e-6`) and all
   accepted states stayed positive, but the worst ARAP strain rose from 0.75
   to 8.72 and the outer dual stalled at 43.14.
2. `20260806T084506Z_1030259...` restored the public accepted-step behavior:
   full multiplier update with a fractional position step, while retaining
   atomic rollback for an infeasible step.  All true linear residuals passed
   (`<=9.99e-6`), the final minimum signed-volume ratio was `0.00172`, and no
   tet inverted.  Nevertheless, the first full global direction raised the
   worst ARAP strain from 0.75 to 10.42; after 60 outers it remained 7.61 and
   the dual was still 29.73.

The 0.25 direct oracle reached `9.79e-13` and kept every tet positive even at
full step (minimum ratio 0.0489), so determinant feasibility alone is not the
remaining issue.  The scalar global L2 merit accepts a direction that reduces
aggregate ARAP while severely distorting a few small, ill-conditioned
boundary tetrahedra.  A new direction/trust-region formulation must address
per-element quality; more PCG, a smaller omega, or another determinant-only
backtrack is now ruled out.

### Orientation-safe strain-trust receipt

Run
`20260806T090042Z_1031314_amd_mgpbd_bunny_small_orientation_safe_recovery_projection`
is the source-hashed execution of the absolute `Cmax <= 1.0` trust filter.  It
is a verified `DONE + GATE_FAILED` result with all contract checks passing.
No Genesis, contact, integration, leader read, or motor output was enabled.

The implementation behaved exactly as predicted by the direct oracle:

- outer 1 rejected the unit through 1/8 trials and accepted step `1/16`;
- ARAP L2 decreased `83.17 -> 78.16`, maximum strain changed only
  `0.750 -> 0.763`, minimum signed-volume ratio remained `0.237`, and dual L2
  decreased to `75.93`;
- all 60 true PCG residuals were at most `9.96e-6`, and all 60 current RAP
  checks passed;
- all 60 states remained positive and within the trust region; every
  multiplier transaction used the complete accepted `delta lambda`;
- there were zero rejected outer transactions, but 922 strain backtracks and
  559 coincident orientation backtracks.

The filter then reached its intended boundary and exposed the remaining
direction defect.  By outer 5, maximum strain was `0.993`; later values stayed
near `0.99999`, accepted steps fell mostly to `1.9e-6`--`1.5e-5`, and the
minimum volume ratio approached `0.00156`.  After 60 outers, ARAP L2 was
`70.13`, height had recovered only `0.250 -> 0.301`, and dual L2 remained
`35.22`.  The only failed quality checks were outer convergence and coherent
ARAP-maximum/height recovery.

This is not a trust-filter failure.  It is a controlled proof that the
unconstrained MGPBD search direction points outside the local feasible cone
once a small boundary tet reaches `C=1`.  A scalar step can only approach zero
there.  It established the need for a constrained direction; continuing to
raise PCG iterations, tune `omega`, or relax only the determinant threshold
is unsupported.

### Matrix-free all-tet constrained-direction recovery

The first explicit active-set/SQP prototype was rejected.  Its dense Schur
rows, retained coupling columns, and repeated host synchronizations do not
scale to `bunnyBig`, and a locally converged tangent-cut problem did not imply
that the nonlinear per-tet constraints were satisfied.

The replacement is a matrix-free SOC-ADMM direction.  For each tetrahedron it
freezes the current closest proper rotation and constrains the complete affine
candidate deformation gradient to a Frobenius ball strictly inside the unit
radius.  This is conservative with respect to the true closest-rotation ARAP
constraint.  The strict radius also supplies a determinant lower bound, so
orientation safety is a consequence of the solved convex subproblem rather
than a post-step rollback.

The Torch matrix-free solve agrees with an independently assembled SciPy
direct/SOC oracle on a shared-face two-tetrahedron fixture.  A historical
single-direction Radeon `bunny_small` smoke also passed its fail-closed
primal, dual, stationarity, normal-cone, material-coupling, ARAP, and
signed-volume checks.  The raw development receipt is preserved under
`runs/20260806_mgpbd_soc_admm_bunny_small_rocm_v5/receipt.json` and explicitly
claims neither trajectory, contact, nor realtime behavior.  This validates
the historical replacement direction kernel only; it does not contain the
current solver source hash and is not a whole-projector parity receipt.  The
SOC direction is now integrated at every nonlinear outer, completely bypasses
the legacy dual PCG/AMG and retired active-set paths, reconstructs a consistent
`delta_lambda`, and enters the coupled outer Armijo transaction.  Its current
whole-`bunny_small` P0a2 execution is still pending.  It has not run on
`bunnyBig` or with gripper contact.

The isolated Radeon trace suggested that many warm ADMM linear solves are
short, but the complete projector cold-starts a new SOC problem at every
nonlinear outer and later outers already require hundreds of Jacobi-PCG
iterations during KKT polish.  Selective vertex UA-AMG is therefore a design
hypothesis, not a settled priority.  It must be evaluated on the complete
P0a2 trace and must not replace the cheap warm path without a measured
wall-time and residual win.

The integrated P0a2 experiments below are diagnostic receipts rather than
passes.  Manual invalidation or termination is recorded as `FAILED`; a
numerical non-pass that reaches a complete receipt is recorded as
`DONE + GATE_FAILED`.

| Run suffix | Auditable conclusion |
| --- | --- |
| `T122503` | Outer 1 completed; outer 2 reached the 2,000-ADMM limit.  This proves that the integrated path was exercised, not that P0a2 passed. |
| `T124120` | An unconditional full-FP64 attempt was too slow and was manually terminated. |
| `T124909` | Fixed-order gather removed one source of ROCm nondeterminism; outer 2 still ended at stationarity `0.00470494` after 4,000 ADMM iterations. |
| `T125727` | Exposed a tolerance mismatch: an RHS-relative PCG target of `9.28785e-4` admitted a true FP32 residual of `2.32506e-3`. |
| `T130219` | Stationarity passed but primal feasibility stopped at `3.54434e-4 > 2e-4`. |
| `T130452`, `T130632` | Reproduced finite FP32 true-residual floors near `1.23128e-4` and `1.24380e-4`, both above the required `1.16098e-4` target. |
| `T131006`, `T131250` | Promoting only each PCG direction and casting it back every iteration did not remove the FP32 ADMM-state floor; both were manually terminated. |
| `T131745` | Full-state FP64 continuation existed but still carried an obsolete low-beta detour and unbounded tightening; it was manually terminated. |
| `T132006` | Ran the pre-audit source through five completed SOC directions and into direction 6, then was deliberately stopped after the normal-cone, receipt, and callback defects below were found.  The wrapper preserved hashes and a truthful `FAILED` marker; there is no numerical gate receipt. |
| `T135501` | Reached direction 6 with the corrected physical-dual gate, then was manually invalidated after the recursive-work-radius and dimensionally inconsistent PCG-tightening defects were identified; preserved as `FAILED`. |
| `T141421` | Completed six directions and correctly returned `DONE + GATE_FAILED`: FP64 stationarity was `4.99772e-4`, but the mandatory FP32 accepted-dtype re-audit measured `5.00021e-4 > 5e-4`.  Every other re-audit check passed. |
| `T142153` | Correctly returned `DONE + GATE_FAILED` after 14 accepted nonlinear outers.  Outer 15 reached the 4,000-ADMM limit with stationarity `8.820e-4 > 5e-4`; primal `5.78e-7`, dual `6.39e-6`, physical normal-cone `1.68e-11`, and proof radius `0.989000618 <= 0.99` all passed.  Its target trace exposed a deterministic inexact-solve scheduling cycle rather than a physical-constraint failure: from ADMM 2009 through 4000 the force-space PCG target alternated strictly between about `6.18e-5` and disabled. |
| `T143742` | Correctly returned `DONE + GATE_FAILED` after 31 atomically accepted outers; direction 32 exhausted 4,000 ADMM iterations with stationarity `1.16335e-3 > 5e-4`.  Primal `3.81e-7`, dual `1.17e-5`, physical normal-cone `8.04e-12`, proof radius `0.98900038`, and every PCG target passed.  The latch fix is independently verified: all 32 directions entered KKT polish and none ever returned to `None`.  The checkpoint now truthfully contains accepted outer 31: minimum determinant ratio `0.245395`, zero inverted/collapsed tets, no degenerate boundary faces, and Y extent recovery from `0.25` to `0.739668` of rest. |

The historical isolated Radeon smoke used 2,992 vertices and 12,298
tetrahedra, took 20.2947 s, allocated a peak 43,420,672 bytes, and executed
1,455 ADMM plus 13,121 PCG iterations.  Its final true ARAP value was
`0.8348000`, its minimum signed-volume ratio was `0.1937393`, and it reported
zero inverted or collapsed tetrahedra.  It is only a single direction, does
not record the solver source hash, and therefore cannot replace a corrected
whole-projector run.

### Correctness fixes applied during P0

The conformance implementation now includes these auditable corrections that
do not alter the live scene defaults:

1. every nonlinear outer refreshes the current fine matrix and all RAP values
   once a hierarchy exists;
2. convergence uses the independently recomputed unscaled physical residual,
   with periodic FP32 residual replacement/restart;
3. the PCG recurrence checks both current `rz` and newly computed `next_rz`,
   and all merit/orientation comparators reject non-finite values;
4. rejected-multiplier policy is explicit.  A single selected scalar drives
   the actual tensor transaction and the receipt records its vector-relative
   error.  Public fidelity retains full `delta lambda`; orientation-safe
   rejection rolls both position and multiplier back atomically;
5. the matrix-free transpose uses a fixed-order vertex-incidence gather rather
   than nondeterministic ROCm atomic reduction;
6. PCG accepts a dimensionally consistent force-space target recomputed from
   the current true KKT stationarity budget.  It may relax as well as tighten;
   the obsolete dimensionless-primal comparison and unconditional 25-step
   halving are removed.  Adaptive beta is guarded by both the ADMM dual and
   KKT stationarity scores;
7. once KKT polish begins, the penalty is frozen.  A reproducible finite FP32
   residual floor promotes the complete `d/z/u`, operator, and input state to
   FP64 rather than promoting only one PCG vector;
8. an FP64 continuation is cast back only once and must re-pass primal, dual,
   stationarity, physical-`y` normal-cone, SOC proof, material coupling, true
   ARAP, determinant, objective, and finite checks in the caller's dtype;
9. the runner now requires the accepted-dtype re-audit whenever precision
   continuation is reported, and rejects a constrained post-iteration callback
   that could mutate the position after the atomic Armijo commit;
10. an exact stationary `(0 -> 0)` constrained direction is separately
    accepted, while every other non-descent direction remains fail-closed;
11. the outer accepted-state ARAP gate is no larger than the SOC work radius,
    so every committed state is a feasible zero-direction start for the next
    nonlinear outer;
12. the first numerical pass retains an equal-or-tighter PCG target for its
    required confirmation pass.  FP64 continuation uses a recorded 1% tighter
    internal stationarity target before its one-time FP32 cast and unchanged
    accepted-dtype gate;
13. KKT polish is now a latched phase.  Once an absolute force-space PCG target
    first becomes necessary, a transiently exhausted *current* dual-motion
    budget cannot release the next solve back to the much looser RHS-relative
    default.  A later positive force-space budget may still safely recompute
    and relax the target, and the exact stationarity gate remains unchanged;
14. a failed nonlinear direction now checkpoints the most recent atomically
    accepted outer state and its outer index.  The previous failure artifact
    misleadingly named `last_safe_state` contained the frame input rather than
    the 14 accepted internal updates, so it cannot support a final-shape claim;
15. the final SOC contract predicate now distinguishes the recursively
    feasible accepted-outer limit (`strain_trust_filter_maximum <= 0.989` work
    radius) from the independent accepted-candidate true-ARAP audit limit
    (`1.0`).  The stale predicate required both limits to equal `1.0`, which
    was incompatible with the corrected profile even if every numerical solve
    passed.

### Genesis bunny state bridge

The clean `bunny_small` physical boundary can be rendered in Genesis without
converting the solver to Genesis FEM.  MGPBD remains the owner of the
tetrahedral state; Genesis owns only one fixed, collision-disabled
`custom_vverts` boundary entity.  This is the same ownership split required by
the doll mainline, but the bunny needs no barycentric appearance skin because
its physical boundary is already the displayed surface.

Development run `T152147` initially appeared to pass, but all three captures
had the same hash because `set_vverts()` did not force a rasterizer refresh.
That result is rejected despite its stale wrapper marker.  The gate now
requires stage-distinct image hashes and measured pixel change as well as
topology and binding checks.

Corrected run `T152831` returned `DONE + GATE_PASSED` on `amd` with the final
lazy-import-compatible source (the preceding `T152408` run produced the same
validated geometry and pixels):

- 2,992 volume vertices, 2,000 boundary vertices, and 3,996 boundary faces;
- one edge-connected component, zero open edges, and zero non-manifold edges;
- exact Genesis rest-state vertex mapping, maximum error `0 m`;
- zero degenerate faces in rest, squashed, and accepted-outer states;
- rest-to-squash changed-pixel fraction `0.25864`;
- squash-to-accepted-outer changed-pixel fraction `0.19206`.

The replay source is the safe outer-31 checkpoint from `T143742`, not a full
P0a2 pass.  The bridge therefore proves coherent MGPBD-state rendering in
Genesis, not full nonlinear convergence, dynamics, contact, or realtime
performance.  Live `solve` mode remains blocked on a passing P0a2 kernel and
must use `VolumetricMGPBDProjector`; the existing contact-enabled
`MGPBDTetSolver.step()` is contract-incompatible with the constrained SOC
path.

## Current status

| Item | Status |
| --- | --- |
| Forensic audit | Complete |
| Existing grasp pass | Rejected as physical evidence |
| Broken realtime LOD | Rejected for MGPBD mainline |
| Inflated convex doll volume | Rejected for material/contact validation |
| P0 schema-v3 runner | Implemented; historical schema-v2 receipts remain immutable; corrected ROCm unit/integration regressions pass |
| Exact-flat official fidelity | Updated clean schema-v2 `DONE + GATE_FAILED`; contract valid, strict numerical gate negative; public multiplier transaction verified |
| Orientation-safe 0.10 | Complete, `DONE + GATE_FAILED`; diagnosed above |
| Orientation-safe 0.25 | Scalar strain-trust execution stalls at the `Cmax=1` boundary; the replacement SOC direction passes the synthetic CPU oracle and historical isolated Radeon kernel gate |
| SOC constrained direction | Implemented and integrated into every P0a2 nonlinear outer; corrected whole-`bunny_small` gate has not yet passed |
| Pre-audit full P0a2 | `T132006` deliberately invalidated and stopped; truthful `FAILED`, no numerical verdict |
| Corrected full P0a2 | `T143742` proves the latch and safe checkpoint through 31 accepted outers, but direction 32 still fails stationarity; full P0a2 remains negative |
| Genesis `bunny_small` bridge | `T152831` `DONE + GATE_PASSED`; exact 2,000-vertex custom-vvert mapping and coherent rest/squash/accepted replay; renderer bridge only |
| P0b `bunnyBig` | Not started; blocked on corrected P0a2 `DONE + GATE_PASSED` |
| P1 boundary-conforming doll/no-contact | Blocked on corrected P0a2 and `bunnyBig` |
| P2–P4 appearance/contact/live | Blocked; contact coupling and intact visual binding remain independent gates |
