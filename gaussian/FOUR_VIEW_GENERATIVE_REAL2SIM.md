# Four-view generative Real2Sim mainline

## Corrected boundary

The four reviewed photographs are the only identity authority for generative
completion. No procedural OBJ, surface carrier, observed Gaussian, or depth-
lifted approximation may enter the upstream generation stage as complete
geometry. The August 4 surface-carrier branch is retained only as a negative
control: its geometry is distorted, so projecting real color onto it cannot
turn it into a reconstruction of the photographed object.

Vista4D is a downstream video-reshooting model. The released interface takes a
source video, point-grounded renders, masks, and target cameras; it does not
directly turn four still photographs into a complete object. The released
repository also marks its I2V checkpoint as unreleased. It is optional after a
source orbit exists and is never an upstream geometry source.

## Executable architecture

```text
four reviewed same-instance photographs
  -> hash/provenance gate and white-background normalization
  -> deterministic observed-mask visual hull (initial Gaussian geometry only)
  -> SEVA four-input camera-controlled orbit (primary appearance branch)
  -> generated-orbit audit and explicit human identity review
  -> frozen-geometry Gaussian appearance fit with high-weight real anchors
  -> observed-core / generated-fill visibility partition
  -> appearance Gaussian + independent conservative collision asset
  -> pose-synchronous Genesis teleoperation demo

parallel A/B only:
  four photographs -> Hunyuan3D-2mv mesh -> real-view alignment -> orbit render
  -> optional Vista4D reshoot -> the same provenance and visibility gates
```

SEVA is the primary video-first path because its `img2trajvid` task supports
unordered sparse input views and an ordered target trajectory. Hunyuan3D-2mv
is an independent mesh-first baseline, not an input to SEVA and not a source
of collision geometry. This retains the useful AIHoloImager comparison—learned
mesh followed by real-view alignment—without contaminating the primary orbit
with a generated mesh or inheriting its Windows/CUDA implementation.

## Stage contracts

### G0: four-view input

`prepare_four_view_generation.py` requires front, right, rear, and left Tier-A
views from one physical instance. Every RGB, neutral RGB, hard mask, and soft
alpha file must match the reviewed M1 manifest and hash index. The output
contains:

- SEVA `ReconfusionParser` scene with four observed inputs and 49 ordered dummy
  targets;
- exact OpenGL camera transforms and intrinsics for a closed 0–360 degree
  orbit;
- Hunyuan `front/left/back/right` RGBA inputs;
- immutable provenance, hashes, and a terminal marker.

The nominal product orbit is not misrepresented as photogrammetric camera
calibration. Camera, scale, and mesh alignment remain explicit downstream
optimization variables.

### G1: SEVA appearance prior

Run Stable Virtual Camera v1.1 with `task=img2trajvid`, four inputs, the
two-pass sparse-view sampler, fixed seed, and the bound target cameras. Record
the model revision, checkpoint hash, command, environment, per-frame output,
video, runtime, and terminal status. All output frames are generated Tier-G
evidence even where they coincide with an observed azimuth.

The executable `phi-amd-work` path deliberately uses the interactive
Hugging Face CLI credential only; tokens are never accepted as arguments or
environment variables:

```bash
hf auth login

# The upstream environment currently omits this imported runtime dependency.
uv pip install --python SEVA_PYTHON scipy==1.14.1

./ops/run_phi_seva_until_review.sh \
  FOUR_VIEW_INPUT PIPELINE_RUN_ROOT SEVA_CHECKOUT LOCAL_MODEL_ROOT \
  MODEL_INSTALL_RUN_ROOT SEVA_PYTHON
```

The installer pins model revision
`e538e251c1009e9a41cf8b7fee5f21332a1960de`, records file hashes, and fails
before inference if authorization or either required model file is missing.
The pipeline writes `REVIEW_REQUIRED.json` and stops after the 49-frame numeric
audit; successful generation is not automatic approval.

The current `phi-amd-work` execution is at that review stop. Install run
`seva_model_install_20260804T215409Z_1434707` is account-bound to `fbsh96`.
Recovery pipeline `seva_primary_recovered_20260804T233837Z_1438301` reuses the
completed 49-frame inference from parent run
`seva_primary_20260804T222216Z_1436441` without rerunning the model; the parent
failed only because the original recorder required `4x4` matrices while SEVA
emits valid `3x4` extrinsics. The fixed recorder appends the homogeneous row
and measured a maximum camera error of `5.24e-08`.

Two upstream compatibility details are explicit rather than hidden in the
environment: the official fixed model revision contains a legitimate empty
`config.yaml`, so installation validates its exact upstream size while still
requiring a nonempty checkpoint; and the deleted SD2.1-base VAE locator is
replaced by the reviewed `ops/patches/seva_official_vae_31f26fde.patch`. That
patch pins the official `stabilityai/sd-vae-ft-mse` revision and the runner
verifies the expected VAE file hashes. The runtime also records SciPy and uses
a finite `10800 s` timeout because the fixed workload is 300 diffusion steps.

The audit reports real-anchor silhouette IoU `0.981943` mean / `0.976890`
minimum, adjacent-frame foreground IoU `0.943569` mean / `0.904279` minimum,
and first/last foreground IoU `0.961873`. These are consistency diagnostics,
not a held-out-real-view quality claim. Explicit review of identity, adjacent
motion, cyclic seam, background stability, and private HIL rear/top exemplars
is still required.

### G2: SEVA-to-Gaussian distillation

Audit all 49 generated frames before training. The dataset repeats the four
real anchors so real observations outnumber generated pseudo-views about two
to one, excludes generated copies of the four anchor azimuths, initializes all
30,000 centers from the observed-mask visual hull, and freezes means, scales,
quaternions, refinement, and higher SH. Generated imagery can fit appearance;
it cannot create or move observed geometry.

After the reviewer creates a hash-bound accepted review with
`gaussian.record_seva_orbit_review`, the downstream dataset has one guarded
entry point:

```bash
./ops/run_phi_seva_after_review.sh \
  UNTIL_REVIEW_PIPELINE ACCEPTED_REVIEW_JSON OBSERVED_INITIALIZATION \
  PSEUDOVIEW_DATASET_ROOT SEVA_PYTHON
```

This wrapper re-resolves generation and audit paths from the review request,
requires `accepted_low_confidence_pseudoviews`, checks that the review binds the
same audit metrics hash, and only then invokes the pseudo-view COLMAP builder.
Rejected reviews and mismatched pipeline paths fail closed.

### G3: independent Hunyuan mesh-first A/B

Run Hunyuan3D-2mv with all four named RGBA views and fixed seed. Export the raw
mesh before texture generation. A generated mesh is an independent proposal,
not metric truth, an observed-core initializer, or a physics asset. It must
pass topology sanity checks and four-view alignment before it may be rendered
into its own comparison branch.

### G4: real-view alignment and visibility partition

Optimize similarity, camera residuals, and bounded deformation against the
four reviewed masks and images. Real-view silhouette and identity losses have
authority; generated inter-view imagery may regularize unseen surfaces but
cannot overwrite observed regions. Preserve raw, aligned, observed-core, and
generated-fill artifacts independently. The current Hunyuan fallback keeps
only Gaussians visible in at most one real anchor as a toggleable fill layer.

### G5: optional Vista4D ablation

Vista4D may reshoot an accepted SEVA orbit or an accepted aligned-mesh orbit
into the exact 49-frame target cameras. Its output must pass a new review and
is never accepted merely because its source passed. If used for Gaussian
training, preserve separate provenance weights:

```text
L = lambda_real * L_real + lambda_generated * L_generated
    + L_mask + L_temporal + L_loop,
where lambda_real > lambda_generated.
```

Generated frames never populate observed or held-out-real metrics.

### G6: Genesis assets

The appearance Gaussian and physical collision geometry remain separate. The
Gaussian follows the simulated rigid-body pose and provides visual identity.
A conservative simplified mesh or ellipsoid proxy provides collision, mass,
friction, and contact stability. Generated visual detail cannot silently alter
physics.

## Promotion gates

A candidate is promoted only when all applicable gates pass:

1. four real-anchor silhouette and identity reprojection;
2. no front logo, ear, limb, or keyring identity drift;
3. adjacent-view temporal consistency and 0/360 loop closure;
4. explicit generated/observed provenance for every frame and surface region;
5. metric height of 95 mm within the declared uncertainty;
6. stable Genesis contact and pose-synchronous rendering;
7. a replayable single-`radeon-c` path for every formal competition claim.

MI300X runs accelerate nonformal model and parameter selection. They cannot
provide formal metrics or checkpoints for the formal ACT lineage.

## Preserved negative controls

- observed-Gaussian same-source Vista4D: executable interface baseline, no
  completion claim;
- four sparse real keyframes inserted into an observed orbit: identity does
  not propagate and temporal residual worsens;
- procedural 95 mm surface carrier and both conditioned Vista4D videos:
  invalid complete-geometry prior plus measured overexposure/identity loss;
- surface-carrier PLY/GLB Genesis previews: rejected visual fallback.
- Hunyuan seed-10030 Vista4D reshoot: rejected because the front identity
  persists around the orbit and the ears/rear drift despite stable execution.

These artifacts remain reproducible evidence of why the corrected mainline is
necessary; none is deleted or relabeled as successful reconstruction.
