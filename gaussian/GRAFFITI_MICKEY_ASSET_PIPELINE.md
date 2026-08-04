# Graffiti Mickey Real2Sim asset pipeline

> **Implementation status:** active. M1 evidence/masks, the MI300X
> VGGT-Omega metric pose initializer, the four-view real-only VkSplat
> appearance QA, the first MI300X SHARP alignment/fusion ablations, and a
> UniSHARP appearance-proposal audit have executable runners and immutable
> remote evidence.
>
> Existing product photographs, historical HIL frames, and the 95 mm product
> specification are sufficient to start. A new capture is useful only as a
> later quality upgrade; it is not an input gate. Generated content may fill
> unobserved regions, but it is never treated as measured evidence.

## 1. Deliverable and boundaries

The deliverable is a metric, poseable visual-and-physical digital twin of the
MINISO Disney Mickey Fun Crash Series **Graffiti Mickey** used in the dual-arm
handover task. It has three intentionally separate representations:

1. **Observed appearance:** Gaussian primitives optimized only from real
   photographs. This is the source of visual truth and the only appearance
   layer used for real-view quality metrics.
2. **Generated completion:** low-confidence geometry and appearance for the
   bottom and remaining occluded or poorly covered regions. Every generated
   region retains its provenance and can be disabled.
3. **Physics representation:** the existing rigid collision proxy plus the
   qualitative PBD plush-body proxy. Physics does not depend on Gaussian
   geometry and remains stable when appearance assets change.

The fused visual is for rendering and demonstration. It is not a policy
observation unless a later experiment explicitly adds and validates that path.
The current procedural OBJ remains the collision/debug baseline; it must not be
presented as a 3DGS reconstruction.

## 2. Evidence and confidence tiers

All source pixels remain in the private sibling data root. Only relative source
IDs, hashes, licenses/usage notes, and derived run metadata enter this public
repository.

| Tier | Inputs | Permitted use | Weight |
| --- | --- | --- | --- |
| A | Continuous front/right/rear/left views of the standard variant | Pose recovery, observed-core training, held-out real evaluation | Highest |
| B | Confirmed-variant front-top, top/keyring, rear/tag, catalog and boxed views | Sparse-region supervision and identity checks | Medium |
| C | 1,020 HIL images from 17 successful episodes and two cameras | Task-domain color/shape checks, deformation envelope, carefully selected extra observations | Low |
| G | Generated mesh, splats, or pseudo-views | Fill only where real-view visibility is insufficient | Confidence dependent |

Multi-view photometric losses must not silently mix different physical units.
The continuous tier-A set is the coherent reconstruction anchor. Tier-B and
tier-C images carry an `instance_id`; views from another unit supervise only
shared identity, silhouette, or coarse material priors unless a compatibility
check shows that their visible geometry matches the anchor unit.

The reported two-graffiti-ear manufacturing error is excluded from canonical
geometry. The standard front-view identity constraint is viewer-left
black/pink graffiti ear and viewer-right cyan-blue ear.

The 95 mm product height is the initial metric anchor. It is an actual product
specification, not an inferred Gaussian scale. Its provenance and uncertainty
remain recorded so a later direct measurement can supersede it without
changing the coordinate contract.

## 3. Canonical coordinates and asset contract

The object coordinate system is fixed before reconstruction:

- `+Y`: canonical front;
- `+Z`: up;
- `+X`: viewer-left for a front camera on `+Y` looking at the origin;
- origin: center of the plush body, not the keyring or overall bounding box;
- metric unit: metre.

The private run root is structured as follows:

```text
graffiti_mickey_asset_v1/
  00_sources/
    observed_anchor/       # tier A, immutable
    observed_support/      # tier B, immutable
    hil/                   # tier C, immutable selection or links
    generated/             # tier G, never mixed with observed sources
    source_manifest.jsonl
  01_normalized/
    rgb/
    masks/
    alpha/
    metadata.jsonl
    normalization.json
  02_pose_init/
    vggt_omega/
    manual_ring/
    cameras_observed.json
    sparse_points.ply
    quality.json
  03_observed_core/
    input/
    runs/
    appearance_observed.ply
    visibility_field.npz
    manifest.json
  04_generated_fill/
    candidates/
    accepted/
      completion_mesh.glb
      pseudo_views/
      confidence/
    manifest.json
  05_fusion/
    appearance_observed.ply
    appearance_fill.ply
    appearance_fused.ply
    visual_mesh.glb
    fusion_manifest.json
  06_metric_alignment/
    similarity_transform.json
    dimensions.json
    alignment_report.json
  07_genesis/
    physics_rigid.obj
    physics_plush_pbd.obj
    appearance_binding.json
  08_eval/
    heldout_real/
    identity/
    genesis/
    metrics.json
  DONE
```

Observed and generated Gaussians are preserved as separate PLY files even
after fusion. A custom per-Gaussian provenance attribute is optional because
some trainers/renderers may discard it; the two-file partition and fusion
manifest are mandatory.

## 4. End-to-end stages

### P0 — Freeze evidence and provenance

1. Copy or link the current 14 selected product/reference images and selected
   HIL frames into the immutable source layout.
2. Record SHA-256, source URL or dataset episode/frame, instance identity,
   view label, tier, usage note, and `observed`/`generated` provenance.
3. Preserve original pixels. Every crop, mask, color adjustment, or generated
   derivative receives a new hash and parent IDs.
4. Validate the identity checklist before any optimization: standard ears,
   pink face plate, two moving-eye housings, black nose, white plush body,
   white hands, yellow shoes, rear seam/tag, strap, and keyring.

**Gate P0:** all selected sources hash successfully; no manufacturing-error
image is marked as canonical; no generated image appears in an observed split;
every photometric training view has an explicit compatible `instance_id`.

### P1 — Normalize real observations

1. Correct EXIF orientation and lens rotation without resizing the immutable
   originals.
2. Segment the object with an automatic mask proposal followed by visual QA.
   Keep soft alpha at fur/strap boundaries and hard masks for pose estimation.
3. Produce lossless normalized crops with a consistent margin. Store crop and
   resize transforms so camera parameters can be mapped back to source pixels.
4. Apply only reversible exposure/white-balance normalization. Do not erase
   graffiti, seams, labels, shadows that describe shape, or material response.
5. Label view direction, image tier, likely camera model, image instance, and
   which physical regions are visible.
6. From HIL data, select sharp, minimally occluded frames with maximum novel
   view coverage. Do not treat robot-occluded silhouettes as complete object
   masks. Retain deformation-state labels.

**Gate P1:** masks pass visual review; the anchor set covers four sides; all
normalization transforms are invertible from metadata.

### P2 — Pose and coarse-geometry initialization

The primary implemented initializer is the existing **VGGT-Omega-1B-512**
stack on the MI300X. It exports learned camera extrinsics/intrinsics, masked
depth, a spatially sampled object point cloud, proper-Sim(3) canonicalization,
and reprojection/handedness gates. MASt3R-SfM and InstantSplat-style DUSt3R
remain comparison branches. Conventional COLMAP is retained only as a
diagnostic because sparse product views and historical dynamic HIL footage
have already shown weak registration.

1. Estimate relative poses and dense point maps from tier-A masked views, then
   add tier-B observations only when they improve consistency.
2. Solve a similarity alignment into canonical axes. Use the 95 mm product
   height to resolve scale.
3. Reject mirrored solutions by enforcing the standard left/right ear colors.
4. Maintain a deterministic manual-ring fallback with nominal canonical camera
   azimuths `0/-90/180/+90` degrees for the listing's
   `front/right/rear/left` files. The source right/left names describe the
   listing sequence, while the blue-ear and black-graffiti-ear identity cues
   establish the canonical side. Optimize small pose and focal-length
   corrections against silhouettes rather than pretending the nominal views
   are exact.
5. The procedural physics mesh may provide a scale and silhouette prior, but
   never appearance supervision. A generated coarse mesh may also regularize
   unobserved geometry, with an explicit generated-prior flag.

**Gate P2:** all four tier-A views have valid poses; no front/back or mirror
ambiguity remains; projected silhouettes are visually consistent; metric
height resolves to 95 mm within the declared tolerance.

### P3 — Train the observed-core Gaussian asset

1. Initialize points/Gaussians from P2 dense points or the aligned coarse
   surface.
2. Optimize only against real masked views. Tier A receives the highest image
   weight, tier B medium weight, and selected tier C low weight.
3. Begin with low-order spherical harmonics to prevent sparse-view color
   overfitting. Use opacity, scale, density, and foreground/background
   regularization.
4. Record per-region visibility from the real camera frusta. This field, not a
   hand-authored region list, determines where generation may contribute.
5. Run leave-one-anchor-out validation: train four variants, each withholding
   one real anchor view, then render the withheld view.
6. The competition-facing optimization/rendering path uses the pinned
   VkSplat/Vulkan stack on a Radeon GPU. Pose initialization may be an offline
   nonformal preprocessing branch but must be declared in the manifest.

**Gate P3:** recognizable and correctly oriented in every real anchor view;
no duplicated ears/limbs; withheld real-view metrics and silhouette quality
beat the procedural proxy baseline; output is useful with generated fill
disabled.

The current nonformal observed-core run contains 30,000 Gaussians optimized
only from the four masked real anchor views. VkSplat writes this PLY after a
dataparser similarity normalization; `canonicalize_vksplat_ply.py` now applies
the exact inverse similarity to positions, Gaussian scales, and quaternions.
Direct RADV rendering with the accepted 95 mm cameras is visually accepted in
all four directions. This canonical observed core is the current release
baseline. The audit is explicitly a training-view coordinate/render check,
not a held-out metric claim.

### P4 — Generate missing geometry and views

The first candidate reuses the existing MI300X ROCm **SHARP → 3DGS** practice.
SHARP has already produced per-image Gaussian PLY files on this machine, and
the existing research workspace includes audited conversion from SHARP's
OpenCV source-camera coordinates into a calibrated Gaussian world. For this
object, run SHARP independently on the four neutral-background tier-A views,
transform each PLY through the P2 cameras into the canonical object frame, and
retain only cross-view-consistent generated surfaces.

The implemented `sharp_object_fusion.py` does this with pixel-corresponded
SHARP/VGGT depth alignment, a trimmed proper Sim(3), full covariance-frame
rotation, observed-silhouette carving, and multi-source voxel-neighborhood
support. The default keeps only the maximum-opacity visible SHARP layer per
pixel. The all-eight-layer ablation is preserved as a rejected result because
interior generated layers leak through the sparse exterior during rendering.
Likewise, global one-Gaussian-per-voxel reduction is an ablation, not the
default: it was geometrically coherent but too sparse at 512 x 512. The
current default keeps all cross-source-supported visible-surface Gaussians,
bounded by a configurable maximum.

The second implemented candidate is **UniSHARP**, reusing the existing ROCm
checkpoint and AMD-native `gsplat` path on the MI300X. Its four 768 x 768
per-anchor runs each emit 1,179,648 Gaussians and preserve the toy's fur,
asymmetric ears, face, rear strap, and local-view identity well. Its first
metric-fit audit, however, measured 5.3--6.8 mm median residual and
10.7--16.2 degree rotation correction against the accepted VGGT-Omega frame;
it therefore failed the metric-geometry gate. UniSHARP is accepted only as a
generated **appearance/pseudo-view proposal**. It does not replace SHARP/VGGT
geometry, the observed core, or any held-out-real sample.

The enhanced UniSHARP run completed in 43 seconds on the MI300X and retained
80 lossless local pseudo-views (20 per anchor) with crop-corrected intrinsics,
exact generator-local camera transforms, source hashes, and explicit
non-observed/nonformal eligibility. A direct hybrid ablation then used Apple
SHARP as the geometry carrier and donated UniSHARP SH0 color plus opacity. It
passed the SHARP metric gate with 124,420 Gaussians, but the RADV/VkSplat audit
showed essentially unchanged exterior holes. That direct PLY is rejected.
The result isolates the bottleneck: UniSHARP must supervise a separately
optimized confidence-masked fill layer; appearance-field substitution cannot
repair absent geometry.

A follow-up low-confidence branch relaxed cross-source support from two views
to one while retaining two-view silhouette support. Its 260,942-Gaussian RADV
render restored a continuous recognizable shell, but exposed peripheral
floaters and duplicated ear layers. Metric-depth pruning now keeps every
cross-source Gaussian and keeps a single-source Gaussian only when it is
within 8 mm of its source-view VGGT surface and does not sit more than 4 mm in
front of any non-source real surface. The current v2 result contains 229,576
Gaussians and is accepted for **real-photo refinement and observed-visibility
masking**, not as a final appearance asset.

The third implemented generation interface is **Vista4D**. It consumes a
49-frame source video, a point-grounded render, alpha/motion masks, and target
cameras and emits a generated appearance video.  It does not emit a completed
PLY or mesh and the released interface does not directly accept four still
images. The first same-source Gaussian baseline completed on MI300X and is
accepted only as an executable reshooting-interface proof. A fixed-
seed A/B that inserted the four reviewed real views at frames 0/12/24/37
preserved those frames but failed to propagate their detail between anchors
and increased temporal p95 residual from 0.03697 to 0.09882.  Sparse real
keyframes are therefore a preserved negative control.

The subsequent procedural 95 mm surface-carrier branch is also a negative
control. Its silhouette fit and real-color coverage are measurements of the
carrier, not evidence that the photographed object's geometry was recovered.
Because the distorted procedural geometry directly controls depth, silhouette,
and point conditioning, neither it nor its Vista4D derivatives may initialize
the final mesh or Gaussian. Portable Genesis PLY/GLB probes also fail the
visual gate.

The corrected mainline takes the four reviewed real photographs as the only
identity input. Stable Virtual Camera v1.1 is the preferred camera-controlled
orbit-video generator. Hunyuan3D-2mv is a separately gated multi-candidate
fallback, never an automatically trusted complete-mesh prior. Four-view
alignment, continuous-orbit topology review, and private HIL identity review
must all pass before optional Vista4D reshooting or Gaussian distillation.
NoPoSplat or FreeSplatter remains an optional comparison.

The original Hunyuan seed `10027` and all descendants are quarantined by exact
run IDs, manifest/asset hashes, and hybrid dataset hash. This includes its
aligned and real-projected mesh, the derived Vista4D video, and the 5k/15k
VkSplat runs. Numeric convergence cannot rehabilitate a rejected geometry
prior. The replacement seed sweep `10028`/`10029`/`10030` starts directly from
the four real images. Seed `10030` is the current conditioning-only selection;
the other two are preserved as valid but dominated candidates.
TRELLIS-family models likewise remain optional rather than the default.

The executed seed-`10030` fallback does not use the mesh as Gaussian geometry.
It initializes 30,000 frozen Gaussians from the four observed masks, fits only
appearance, and partitions the result by visibility in the four real cameras.
Only 1,224 Gaussians visible in at most one real anchor survive as a generated
fill layer; 28,776 overlapping observed support are rejected. The fused
31,224-Gaussian preview preserves the observed core bit-for-bit and is accepted
only as a default-off nonformal toggle. Its separate Vista4D reshoot is
quarantined for front-identity persistence and rear/ear drift.

1. Generate multiple deterministic candidates. For SHARP, treat the four
   input views as four independently generated Gaussian hypotheses; for a
   stochastic mesh model, use multiple explicit seeds. Produce only the
   missing bottom/oblique pseudo-views needed by the visibility field.
2. Score candidates against all real silhouettes, landmarks, colors, and the
   95 mm dimension contract before accepting any generated region.
3. Enforce hard identity constraints:
   - viewer-left graffiti ear and viewer-right cyan-blue ear;
   - one face plate, two eyes, one nose, two hands, and two shoes;
   - no extra or fused limbs, ears, straps, rings, logos, or text;
   - observed seams, tag placement, shoe placement, and rear shape cannot be
     overwritten by a generated alternative;
   - unobserved text or product labels remain blank/neutral rather than being
     hallucinated.
4. Derive a spatial confidence map from real-view coverage, agreement between
   seeds, reprojection consistency, and distance to observed surfaces.
5. Keep rejected seeds and rejection reasons for auditability.

**Gate P4:** a completion candidate agrees with every tier-A silhouette and
identity constraint. Otherwise the system ships the observed core plus a
neutral low-detail underside; generation is never a blocking dependency.

### P5 — Confidence-aware fusion

1. Align the generated mesh/field to the observed core using canonical axes,
   the metric height, masked silhouette optimization, and surface registration.
2. Real observations dominate every surface with adequate visibility. The
   generated branch contributes only below a configured visibility threshold.
3. Taper generated opacity near observed/generated seams. Penalize depth and
   normal discontinuity, but do not blur observed texture to hide a poor fit.
4. If pseudo-views are used during a final optimization, give their losses a
   much lower weight and mask them to missing regions. They never modify an
   observed pixel footprint with adequate coverage.
5. Export:
   - `appearance_observed.ply`: real-only truth layer;
   - `appearance_fill.ply`: generated-only completion layer;
   - `appearance_fused.ply`: convenience rendering asset;
   - `visual_mesh.glb`: optional baked/extracted compatibility asset;
   - `fusion_manifest.json`: hashes, transforms, thresholds, source weights,
     and per-region provenance.

**Gate P5:** disabling `appearance_fill.ply` changes only low-visibility
regions; observed-region quality does not regress relative to P3.

### P6 — Metric alignment and physics binding

1. Store a single similarity transform from reconstruction coordinates to the
   canonical metric object frame. Do not scale the physics and appearance
   assets independently.
2. Bind the appearance root to the Genesis rigid object's pose. The PBD branch
   remains a separate qualitative material experiment until rigid-to-soft
   attachment and measured compression calibration exist.
3. Keep the current closed, simplified collision surface. Neither splat
   density nor an extracted photogrammetry mesh becomes the real-time collision
   mesh by default.
4. Validate appearance/physics alignment at canonical front, side, rear, and
   grasp poses. Record dimension and transform residuals.

**Proposed gate P6:** overall height error at most 2 mm or 3%, whichever is
larger; appearance-to-collision registration within 2 mm translation and 2
degrees rotation at the root. These are engineering acceptance thresholds,
not claims about source-spec precision.

### P7 — Genesis rendering integration

Use a tiered integration so photoreal appearance cannot destabilize control:

1. **Control/debug cameras:** Genesis rasterizer plus the lightweight debug
   mesh. This is the reliable teleoperation and haptics path.
2. **Preferred demo path:** test Genesis Nyx `LightFieldAsset` loading of the
   Gaussian PLY in an isolated environment. Gate it on Radeon execution,
   simulated-geometry occlusion, and the ability to apply the object's
   per-frame rigid transform. Current public documentation demonstrates light
   fields collected at scene build, so dynamic attachment must be proven rather
   than assumed.
3. **Dynamic fallback:** transform the Genesis camera into object coordinates
   with `T_object_camera = inverse(T_world_object) * T_world_camera`, render the
   object offscreen with VkSplat, then depth/alpha composite it with the Genesis
   RGB/depth output.
4. **Portable fallback:** use `visual_mesh.glb` baked from the final appearance
   asset in the standard Genesis rasterizer.

The controller, leader bridge, force-feedback loop, and policy inputs remain
independent of the demo renderer. A renderer crash may drop to the debug mesh
without stopping the simulation.

**Gate P7:** a 10-second dual-arm teleoperation smoke test completes without a
renderer-induced simulation failure; appearance follows the physics pose;
occlusion ordering is correct in the two grippers and tabletop contacts.

### P8 — Evaluation and release

Only held-out **real** images count toward PSNR, SSIM, LPIPS, silhouette IoU,
or identity evaluation. Generated pseudo-views can test continuity but cannot
inflate reported reconstruction metrics.

Release checks:

- source split integrity and full hash verification;
- four-of-four tier-A pose registration;
- leave-one-anchor-out real-view metrics for P3 and P5;
- silhouette IoU target of 0.95 on manually reviewed tier-A masks, reported as
  a target until achieved;
- metric dimension and root-pose thresholds from P6;
- identity checklist with front/side/rear renders;
- Radeon render FPS, latency, and VRAM at the declared demo resolution;
- rigid collision, PBD smoke, dual-arm teleoperation, and haptic safety
  regressions remain passing;
- `DONE` is written only after all selected gates pass; otherwise a `FAILED`
  artifact records the stage and reason.

## 5. Experiment matrix

| ID | Pose/init | Appearance | Completion | Purpose |
| --- | --- | --- | --- | --- |
| A0 | Hand-authored | Procedural OBJ | None | Current physics/debug baseline |
| A1 | Manual ring | VkSplat real views | None | Deterministic minimum observed core |
| A2 | MI300X VGGT-Omega + silhouette refinement | VkSplat real views | None | Primary observed-core candidate |
| A3a | A2 | VkSplat real views | MI300X SHARP metric geometry prior | Implemented; direct-appearance ablations rejected |
| A3b | A2 | VkSplat real views | MI300X UniSHARP pseudo-view proposal | Implemented; 80 lossless views retained, metric geometry rejected |
| A3b-delta | A2 | VkSplat real views | SHARP geometry + UniSHARP direct color/opacity donation | Implemented negative control; direct PLY rejected after RADV audit |
| A3b-prune | A2 | VkSplat real views | Single-source fill + VGGT depth/front-conflict pruning | 229,576-Gaussian candidate accepted for refinement, not final release |
| A3c | A2 | VkSplat real views | MI300X Hunyuan3D-2mv fill | Optional future coherent-mesh comparison |
| A4 | Pose-free splat initializer | Real-view refinement | NoPoSplat/FreeSplatter fill | Research comparison |

A3a/A3b/A3c are selected only if they improve missing-region continuity without
reducing held-out real-view metrics or failing the identity checklist.
Otherwise A2 is the release asset. This decision rule prevents a visually
plausible generated model from replacing the actual product.

## 6. Four-machine AMD allocation

The machines accelerate independent experiments, but the competition's
single-Radeon execution claim remains explicit.

| Resource | Role | Competition status |
| --- | --- | --- |
| `amd` | Mask/data preprocessing, CPU pose experiments, Genesis integration, teleoperation and haptic smoke tests | Nonformal development |
| `phi-amd-work`, MI300X `gfx942` | Primary generator-to-GS lab: VGGT-Omega pose/depth, SHARP metric geometry hypotheses, UniSHARP appearance proposals, confidence fusion, optional future mesh generation/extraction | Nonformal development |
| `radeon-f` | VkSplat configuration sweeps and renderer integration experiments | Nonformal development |
| `radeon-c`, GPU0 `gfx1100` | One pinned final VkSplat optimization/render and declared measurements | Formal single-Radeon run |

The MI300X is a major acceleration resource, but it is not the declared formal
competition GPU. Its existing ROCm-native SHARP, UniSHARP, `gsplat`, and
FreeTimeGS practices are therefore used aggressively for development,
candidate generation, coordinate conversion, confidence filtering, and
ablations. The formal submission must
still remain reproducible with A1 or A2 and the generated fill disabled. If
A3a/A3b is shown, it is labeled as an optional visual enhancement unless its
entire declared path is accepted by the competition compute rules.

## 7. Implementation work packages

The following additions turn this plan into an executable pipeline:

1. `gaussian/object_asset_manifest.schema.json` — provenance, split, pose,
   metric anchor, generation lineage, and acceptance schema.
2. `gaussian/prepare_object_views.py` — immutable hashing, crop/mask transforms,
   metadata and real/generated split enforcement.
3. `gaussian/object_pose_init.py` and
   `gaussian/vggt_omega_object_pose.py` — deterministic manual ring,
   MI300X VGGT-Omega inference, similarity alignment, spatial point sampling,
   reprojection checks and mirror rejection.
4. `gaussian/object_colmap_export.py`, `gaussian/vksplat_train.py`, and
   `ops/run_object_vksplat.sh` — masked real-only datasets, all-train visual
   probes, leave-one-out negative controls, low-order SH, hashes, and pinned
   VkSplat execution.
5. `gaussian/sharp_object_fusion.py` and
   `ops/run_phi_sharp_fusion.sh` — MI300X SHARP-family/VGGT metric alignment,
   silhouette carving, cross-source support, generated-only PLY and provenance.
6. `gaussian/vksplat_render_ply.py` and
   `ops/run_object_vksplat_render.sh` — direct nonformal RADV rendering of a
   generated PLY for visual acceptance without retraining.
7. `ops/run_phi_unisharp_object_baseline.sh` and
   `gaussian/unisharp_infer_with_frames.py` — MI300X UniSHARP hypotheses plus
   lossless pseudo-views with exact local camera metadata.
8. `gaussian/record_generated_fill_audit.py`,
   `gaussian/gaussian_appearance_delta.py`, and
   `gaussian/record_generated_render_audit.py` — immutable split decisions,
   bandwidth-efficient exact appearance deltas, and RADV visual ablation
   evidence.
9. `gaussian/chunk_artifact.py` and
   `gaussian/prune_generated_confidence.py` — hashed parallel transport plus
   VGGT metric-depth/front-conflict pruning for low-confidence single-source
   Gaussians.
10. `gaussian/canonicalize_vksplat_ply.py` and
    `gaussian/record_observed_canonical_audit.py` — exact inverse VkSplat
    dataparser similarity and immutable four-direction canonical render audit.
11. `gaussian/prepare_four_view_generation.py`,
    `ops/run_phi_prepare_four_view_generation.sh`, and
    `gaussian/FOUR_VIEW_GENERATIVE_REAL2SIM.md` — geometry-free, hash-bound
    four-observed-view inputs for SEVA, Hunyuan3D-2mv, and the downstream
    Vista4D handoff.
12. `gaussian/prepare_vista4d_object_input.py`,
    `ops/run_amd_vista4d_object_input.sh`,
    `ops/run_phi_vista4d_object_completion.sh`, and
    `gaussian/audit_vista4d_completion.py` — exact Vista4D object contract,
    fixed-seed MI300X generation, and temporal/identity audit with generated
    content excluded from formal and held-out-real evidence.
13. `gaussian/completion_candidate.py` — immutable observed/generated PLY,
    confidence, source-label, transform, and hash contract for a later
    Vista4D-depth lifting stage. The current Vista4D audits do not authorize
    this stage.
14. `gaussian/surface_carrier.py`,
    `gaussian/audit_vista4d_mask_alignment.py`, and
    `gaussian/export_surface_carrier_glb.py` — bounded AMD-ROCm carrier fit,
    exact support-threshold measurement, and portable visual conversion. The
    GLB path is retained as rejected visual evidence, not a live default.
15. `gaussian/align_object_metric.py` — canonical-axis and product-dimension
    transform with validation report.
16. `sim/genesis_so101/gaussian_appearance.py` — Nyx capability probe,
    VkSplat-composite path and GLB fallback behind one binding interface.
17. `configs/graffiti_mickey_asset.yaml` — pinned inputs, thresholds, machine
    role, formal flag and output locations.
18. Unit tests for provenance leakage, mirror rejection, generated-region
    masks, metric alignment, quaternion/covariance synchronization, PLY
    decoding and renderer fallback.
19. `gaussian/export_observed_initialization.py`,
    `gaussian/prune_generated_fill_visibility.py`, and
    `gaussian/fuse_gaussian_layers.py` — observed-only geometry initialization,
    real-camera visibility partitioning, and bit-preserving layered preview.
20. `gaussian/audit_layered_gaussian_fusion.py` and the explicit Genesis
    `layered-preview` loader — no-harm A/B, default-off runtime binding, and
    independent-collision enforcement.

## 8. Execution order and exit criteria

The shortest path to a credible asset is P0 → P1 → P2 → P3 → P6 → P7. P4 and
P5 run after P3 and can improve the underside without blocking integration.

Milestones:

1. **M1 — dataset ready:** source manifest, normalized images, masks and
   canonical identity contract pass P0/P1.
2. **M2 — observed twin:** A1 and A2 evaluated; best real-only Gaussian passes
   P2/P3 and is metric aligned.
3. **M3 — learned completion:** SEVA multi-view video is preferred; externally
   reviewed Hunyuan3D-2mv candidates are a fallback. Every proposal is aligned
   to the four reviewed observations and screened over a continuous orbit
   before optional Vista4D reshooting. A3 branches enter fusion only if their
   own P4/P5 gates pass and no quarantine token appears in their lineage.
4. **M4 — dynamic demo:** physics-driven appearance runs in Genesis with
   correct occlusion and safe fallback.
5. **M5 — formal package:** the selected asset path is rerun on `radeon-c`
   GPU0, hashes and real-view metrics are frozen, and nonformal steps are
   clearly separated.

The pipeline is complete when the object is recognizable and metrically
aligned from all observed directions, the uncertain underside is either
confidence-labeled completion or neutral geometry, Genesis manipulation remains
stable, and every published result can distinguish observed evidence,
generated content, physics priors, and formal Radeon measurements.

## 9. Upstream implementations to evaluate

- MASt3R / MASt3R-SfM: <https://github.com/naver/mast3r>
- InstantSplat: <https://instantsplat.github.io/>
- NoPoSplat: <https://github.com/cvg/NoPoSplat>
- FreeSplatter: <https://openaccess.thecvf.com/content/ICCV2025/papers/Xu_FreeSplatter_Pose-free_Gaussian_Splatting_for_Sparse-view_3D_Reconstruction_ICCV_2025_paper.pdf>
- Hunyuan3D-2 and Hunyuan3D-2mv: <https://github.com/Tencent-Hunyuan/Hunyuan3D-2>
- Stable Virtual Camera: <https://github.com/Stability-AI/stable-virtual-camera>
- AIHoloImager staged mesh workflow: <https://github.com/gongminmin/AIHoloImager>
- Vista4D video reshooting/completion: <https://github.com/Eyeline-Labs/Vista4D>
- Genesis Nyx renderer: <https://genesis-world.readthedocs.io/en/latest/user_guide/rendering/nyx_renderer.html>
