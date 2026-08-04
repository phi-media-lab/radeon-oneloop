# Formal geometry-frozen object execution plan — 2026-08-05

This run is restricted to `radeon-c` GPU0/gfx1100 and the existing formal-
input-eligible four-real-view dataset. Generated views, generated geometry,
learned depth, MI300X checkpoints, Hunyuan/Vista4D artifacts, and generated
fill are excluded. The run fits observed-photo DC color and opacity while
freezing means, scales, quaternions, refinement, and higher SH.

The exact training command, recorded before execution, is:

```bash
ONELOOP_FORMAL_HOST=radeon-c \
ONELOOP_SEED=20260804 \
ONELOOP_OBJECT_DATASET=/root/radeon-oneloop-data/object_assets/graffiti_mickey_asset_v1/formal_inputs/manual_ring_visual_hull_r160_v1 \
./ops/run_job.sh \
  gaussian_train \
  configs/gaussian_object_formal_geometry_frozen_train.yaml \
  true \
  682b65e97653ffe08e469496bb0554f349aeff103ddf8e57f1e4857f8c04534e \
  -- ./ops/run_formal_object_geometry_frozen_train.sh
```

The resulting checkpoint is not accepted until its manifest, metrics, source
hashes, and terminal marker pass review. A later render job must bind the exact
checkpoint SHA-256 and remains limited to anchor-view registration/rendering;
neither job makes a held-out or novel-view quality claim.

The accepted training job is
`20260804T204531Z_gaussian_train_14053af_20260804`; its exact checkpoint
SHA-256 is
`e9be3a2df4c1ca7fcfddc86deee4c366a2f941f66a881e41d13367c329aff378`.
The exact render command, recorded before execution, is:

```bash
ONELOOP_FORMAL_HOST=radeon-c \
ONELOOP_SEED=20260804 \
ONELOOP_PARENT_CHECKPOINT=e9be3a2df4c1ca7fcfddc86deee4c366a2f941f66a881e41d13367c329aff378 \
ONELOOP_PARENT_TRAIN_RUN=/root/radeon-oneloop-runs/jobs/20260804T204531Z_gaussian_train_14053af_20260804 \
ONELOOP_OBJECT_DATASET_MANIFEST=/root/radeon-oneloop-data/object_assets/graffiti_mickey_asset_v1/formal_inputs/manual_ring_visual_hull_r160_v1/dataset_manifest.json \
ONELOOP_OBJECT_CAMERAS=/root/radeon-oneloop-data/object_assets/graffiti_mickey_asset_v1/formal_inputs/manual_ring_pose_v1/cameras_observed.json \
./ops/run_job.sh \
  gaussian_render \
  configs/gaussian_object_formal_geometry_frozen_render.yaml \
  true \
  682b65e97653ffe08e469496bb0554f349aeff103ddf8e57f1e4857f8c04534e \
  -- ./ops/run_formal_object_geometry_frozen_render.sh
```

The command completed successfully as job
`20260804T205059Z_gaussian_render_e149f01_20260804`. Its public manifest and
metrics SHA-256 values are respectively
`4f9493f6e8fbbb0393d75522fed4a7c6b73d603fbb0b371d33c2e5c3ade720fa`
and `d1ee2de51cca6c04fc83a092976b1e1a08f67802b44d9cf92ec282fd32f16ff4`.
All formal gates passed; the canonical PLY SHA-256 is
`7f01c1e6d8253d7f15162e2cb51e18845676fa1015983266b7d356d9b21aa706`.
This closes the registered training-and-anchor-render execution step without
expanding the claim to held-out or novel-view quality.
