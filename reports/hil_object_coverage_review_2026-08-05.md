# HIL object-view coverage review — 2026-08-05

## Decision

The private HIL recordings are accepted as **real task-domain and rear/top
identity evidence**. They are rejected as calibrated object-centric training
views, metric geometry, and full-orbit held-out evidence.

This decision follows a complete phase audit of all 24 successful episodes in
the immutable 40-episode HIL dataset. The audit sampled both cameras at 12
deterministic phase positions per episode: 576 images total. No raw image or
private dataset path is committed to this repository.

## Bound evidence

- run ID: `20260804T175724Z_208999_amd_hil_object_coverage`
- host role: `amd`, `formal=false`
- dataset `info.json` SHA-256:
  `e198927b1fc7f0d7566e5a4b622872ba3f2a0bafa58010f5d900441e6debcb3d`
- audit manifest SHA-256:
  `b1e41cdb5752d99d622577d47f9d731c353a4ce05a3d4501457be84c5f1880e4`
- frame index SHA-256:
  `e516888501d5b47c34ee13420df23065ee27e734e081448989a06bc635bfbf13`
- front-camera sheet SHA-256:
  `191af8c936a6c63c6aedf12524c7778b81b3b5eb079d8f6e1a8c2c5e25a156cd`
- hand-camera sheet SHA-256:
  `5e87055483101a339dcabb83b2203b97dfff8dce1960d97ff7b34a84b138fa4d`
- artifact hash index SHA-256:
  `3263e11d12627cb8e471435adc48e278c20d47370d83e78c286a3f09302dc216`

## Visual findings

The 24 successful episodes repeat a consistent handover choreography rather
than an object turntable capture.

| Camera | Real coverage | Limitation | Allowed role |
| --- | --- | --- | --- |
| `hand_cam` | rear/top plush surface, MINISO/Disney tag, cyan ear, gripper-contact deformation | unknown object pose; moving wrist hand-eye calibration failed its earlier metric gate; object leaves view after transfer | private rear/top identity check, deformation envelope, qualitative generated-view validation |
| `front_cam` | task-space silhouette and the object between both SO-101 arms | object is small and substantially robot-occluded; no clean texture-bearing object view | task-domain integration and occlusion evidence only |

Three immutable hand-camera exemplars bind the strongest rear/top evidence:

| Episode / phase / frame | Image SHA-256 | Review |
| --- | --- | --- |
| `e012 / p01 / f000128` | `e3f9f078e5ff3b0f7db623f1125692085ed7594db24b7dd1e9cf2a1e42780d4f` | nearly complete undeformed rear/top body, tag and cyan ear; accepted identity exemplar |
| `e010 / p02 / f000256` | `a6d4fe8268f3e300c8c3d4d115f1e2bf764a39da2af0d8a892956180d3dbe85a` | close rear/top view with tag and cyan ear; accepted identity exemplar |
| `e000 / p05 / f000644` | `b09446ea047888532a39651dca7fb16fc2cc565e5ab34351b8f7ba96f0e2b149` | gripper-occluded rear/top during contact; accepted deformation exemplar only |

The first two exemplars are reserved for real-image qualitative comparison and
must not be used to train the completion that they review. Their camera poses
are unknown, so comparisons may optimize only a diagnostic 2D alignment and
must not report calibrated novel-view metrics.

## Reconstruction consequence

The real-only baseline remains the four reviewed front/right/rear/left product
views and the real-only Gaussian trained from those views. HIL adds a top/rear
identity gate but does not close the unobserved inter-cardinal and underside
sectors. The generated branch may therefore fill:

- inter-cardinal side transitions between the four real anchors;
- the underside and other genuinely unobserved regions;
- an elevated orbit needed by the wrist-camera demo, subject to the two
  reserved rear/top exemplars.

Generated pixels remain lower confidence than all observed pixels. The
procedural OBJ is excluded from this lineage; the HIL audit does not change the
stable, separately provenance-bound Genesis collision proxy.
