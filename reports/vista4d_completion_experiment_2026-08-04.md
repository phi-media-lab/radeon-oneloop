# Vista4D object-completion experiment — 2026-08-04

## 2026-08-05 correction

The procedural surface carrier used in this experiment is not a neutral
conditioning canvas. Its distorted geometry directly determines depth,
silhouette, and point conditioning, so real-photo color projection cannot make
it a reconstruction of the photographed object. The carrier and both
carrier-conditioned videos are therefore rejected as complete-prior inputs and
preserved only as negative controls. The corrected mainline starts from the
four reviewed photographs with no inherited geometry, using a multi-view
video generator and a learned multi-view mesh generator before any Vista4D
reshooting.

## Decision

The MI300X video-reshooting practice relevant to this project is **Vista4D**.
The first end-to-end interface experiments run successfully on the
Graffiti Mickey asset, but they also establish a strict interface boundary:
Vista4D generates a novel-view **video proposal** from a source video, a
point-cloud-render video, masks, and target cameras.  It does not directly
emit a completed point cloud, mesh, or Gaussian PLY.

The same-source Gaussian run is accepted as the executable baseline.  The
four-sparse-real-keyframe run is retained as a negative control because it
copies the real identity at four frames without propagating the improvement
through the orbit and increases temporal discontinuity.  A subsequent
metric surface-carrier branch attempted the proposed AIHoloImager-style loop,
but it used procedural rather than learned complete geometry. Its two Vista4D
outputs are also rejected for depth lifting: they improve
silhouette continuity while overexposing and blurring the identity.  None of
the Vista4D results is a metric-geometry asset or held-out-real evidence.

## Bound input lineage

- observed geometry-frozen Gaussian:
  `dc4de9a0a3f4dadf62a4c03c2be939a9b178d284254c9e833a2e19941e41793b`;
- 49-frame closed canonical orbit, 672 x 384, 24 fps;
- exact per-frame point-render alpha masks and zero motion masks;
- target camera file with 49 `cam_c2w` matrices and 49
  `[fx, fy, cx, cy]` intrinsics;
- four reviewed SAM3 real-photo identity anchors from
  `m1_reviewed_orbitfix_20260804T043427Z` for the A/B branch;
- Vista4D 384p49 checkpoint and Wan2.1-T2V-14B base on one MI300X VF;
- seed `10027` held constant between the two branches.

During preparation, the older `amd` M1 directory was found to contain an
early mask failure that included the product-photo support surface.  It was
not reused.  The reviewed SAM3 files were transferred from the hash-bound M1
artifact, verified before and after transfer, and then used for the A/B input.

## Executed A/B

| Condition | Run | Runtime | Generated video SHA-256 | Decision |
| --- | --- | ---: | --- | --- |
| observed Gaussian orbit used as both source and point condition | `vista4d_object_20260804T142413Z_1397758_seed10027` | 746 s | `17c346276ceb01f54c32e0fc780bb297950afa27dad88064ce727291827c07f0` | accepted executable baseline; visually stable but only mild smoothing |
| same orbit plus reviewed real frames at indices 0/12/24/37 | `vista4d_object_20260804T145734Z_1400842_seed10027` | 726 s | `c982a4f15f251e763b19047af84bf63a8562b8cbdfaf306a2352df31e8122108` | rejected as the source construction for the next run; real frames are preserved locally but create temporal impulses |

The baseline audit metrics SHA-256 is
`78cd91ac4dc23792347e0613e3871bae2351630f46c863dbbd2657c265f6d10b`.
The final reviewed-keyframe audit v2 metrics SHA-256 is
`59c54f64e047f3031d875a33882d55eac8fcefc69caadfad97f906e86dbdb0cb`.

The numeric audit supports the visual decision:

| Audit metric | Same-source baseline | Sparse-real-keyframe A/B |
| --- | ---: | ---: |
| observed-support RGB MAE, mean | 0.11591 | 0.11798 |
| background white MAE outside dilated support, mean | 0.02128 | 0.02148 |
| temporal delta residual vs conditioning, mean | 0.03270 | 0.04174 |
| temporal delta residual vs conditioning, p95 | 0.03697 | 0.09882 |
| first/last RGB MAE | 0.07035 | 0.08509 |

These are diagnostic, nonformal metrics.  The observed-support number measures
preservation, not correctness of a hidden surface.  Human review found that
the second branch retained the face, asymmetric ears, rear `MICKEY` strap, and
keyring at the four source frames, but the intervening views remained close to
the original Gaussian and the transitions became less smooth.

## AIHoloImager mapping

[AIHoloImager](https://github.com/gongminmin/AIHoloImager) is useful as an
architecture reference rather than a dependency.  Its documented data flow
is SfM plus masking/delighting, a complete AI mesh, differentiable pose/shape
optimization, and texture projection.  The repository currently presents a
Windows application and uses openMVG, rembg, Intrinsic, and TRELLIS; importing
the whole application is therefore not the shortest AMD/ROCm path.

The OneLoop equivalents should be:

| AIHoloImager stage | OneLoop implementation |
| --- | --- |
| SfM and camera poses | reviewed M1 cameras plus VGGT-Omega / canonical four-view alignment |
| foreground mask and delighting | reviewed SAM3 mask/soft alpha plus neutral real-photo inputs |
| complete AI mesh | learned multi-view mesh from Hunyuan3D-2mv; never the procedural plush proxy or surface carrier |
| differentiable alignment | real-mask silhouette, 95 mm scale, camera, and photometric optimization |
| texture projection | real photos dominate observed regions; generated content is confined to low-visibility regions |
| novel-view polishing | Vista4D over a *continuous* 49-frame render, not four isolated photo impulses |

## Complete surface-carrier execution

Run `20260804T153528Z_201147_amd_surface_carrier` attempted to construct the missing
continuous source rather than asking Vista4D to infer it from four temporal
impulses.  The carrier is a complete 12,078-vertex / 24,044-triangle surface
in the accepted canonical frame. It is now classified as invalid for this
role because its procedural shape is materially distorted. Its height is
exactly 95 mm; the final
extents are 87.938 x 74.030 x 95.000 mm.  A bounded AMD-ROCm silhouette fit
completed in 10.58 s and produced four-view IoUs of 0.7964, 0.7113, 0.7980,
and 0.7141.  Real photographs supply color to 82.12% of vertices; the
remaining 17.88% is explicitly procedural fallback.

This is an offline source carrier, not reconstructed truth.  Its PLY SHA-256
is `4bb613c66c39bdd7fd4be10e4035e0f224bdf015fcaf67fa359bdf827d96705b`;
manifest SHA-256 is
`c30186ef633f786995651dc3f5b4021f4cc04d2e4fb7aca8d04b991dbfc9d5de`.
The first attempted run is preserved as `FAILED` because the deployed AMD
checkout lacked an already-existing pose dependency; the corrected run binds
all sources and succeeds.

The carrier then supplies every frame of a 49-frame source video while the
geometry-frozen Gaussian remains the independent point condition.  A mask
alignment sweep found that point alpha threshold 0.50 improves mean
point/carrier silhouette IoU from 0.68555 to 0.70619, but also confirms that a
material mismatch remains.  Audit metrics SHA-256:
`4e0c8d91279ff316cbba497373037ad3d280368fc16d54790c3086db9566e45e`.

## Surface-carrier Vista4D A/B

Both runs hold seed `10027`, prompt identity, camera trajectory, and model
checkpoint fixed.  The second run changes only the measured point-mask
threshold and explicit classifier-free guidance:

| Condition | Run | Generated video SHA-256 | Key audit result | Decision |
| --- | --- | --- | --- | --- |
| carrier source, point alpha 0.001, default CFG 5 | `vista4d_object_20260804T154158Z_1404244_seed10027` | `a85c93aecd70fcf61cb985c23d88a170ac858586667efad47f43851bfbd275da` | source IoU 0.82811; source RGB MAE 0.27525; closure 0.06006 | rejected for depth lift; continuous but fuzzy, overexposed, identity and strap drift |
| carrier source, point alpha 0.50, explicit CFG 3 | `vista4d_object_20260804T161211Z_1407492_seed10027` | `3e2063f09915c03e6ad200ade2d228c3ca2689cb451cf357cb51e3c3610c1a3d` | source IoU 0.83728; source RGB MAE 0.28117; closure 0.05709 | rejected for depth lift; temporal/background improve, face identity worsens |

The lower-CFG branch reduces background MAE from 0.03911 to 0.02610 and
temporal residual against the source from 0.12432 to 0.10198, but observed RGB
MAE worsens from 0.25952 to 0.29321.  That trade is unacceptable for this
identity-sensitive object.  The corresponding audit metric hashes are
`037950f6e8d299da083b6ae710cbc429648d179c3ee0afb147633fb535bd04fd`
and
`bdbf21f95f5c3bc15ed0ae4398acf957aae46f14f8d3714455ba69c0b648b27a`.

Genesis 1.3.1 does not ingest PLY directly.  Portable GLB conversions were
therefore tested separately.  Vertex-color GLB loads but renders white;
texture-atlas and quantized-material variants load on the AMD backend but
produce unacceptable faceting/color breakup.  These technical successes are
preserved as negative visual evidence, and the carrier is not installed as the
live visual fallback.  The stable procedural proxy remains the collision and
debug layer; the accepted observed Gaussian remains the pose-synchronous
appearance layer.

## Correct next execution order

The carrier and fixed-seed A/B are complete negative controls. Because the
carrier is an invalid geometry prior and neither Vista4D output passes identity
and temporal/background gates together, their depth-lifting and generated-
Gaussian stages must not run. The corrected order is now:

1. Freeze the surface carrier, mask-alignment sweep, Vista4D A/B, and GLB
   preview failures as a reproducible negative-control branch.
2. Build a geometry-free contract from the four reviewed same-instance photos.
3. Generate a controlled orbit-video prior with Stable Virtual Camera and a
   complete learned mesh with Hunyuan3D-2mv.
4. Align the learned mesh to all four real images before rendering a continuous
   source for optional Vista4D reshooting.
5. Keep the existing rigid proxy authoritative for collision and fallback
   rendering; do not substitute any generated or poorly rendered mesh.
6. Keep the geometry-frozen observed Gaussian as the explicit nonformal
   continuous-view demo appearance, with the registered real-only asset as the
   formal lineage anchor.
7. Capture reserved real inter-anchor views and the final read-only-leader
   approach–grasp–handover–release recording.
8. Package Vista4D as an honest conditional completion ablation: executable,
   measured, useful for continuity, and rejected when it harms identity.

This preserves the useful AIHoloImager principle—complete geometry first,
real-view alignment second, texture projection last—while respecting the
actual Vista4D interface and refusing to promote a visually attractive but
identity-inaccurate generation into the simulator's physical truth.
