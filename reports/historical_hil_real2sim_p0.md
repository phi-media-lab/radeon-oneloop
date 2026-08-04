# Historical HIL Real2Sim P0

## Outcome

The reviewed historical handover data now supports a reproducible, non-formal
Genesis P0 reconstruction of the parts directly constrained by the capture:

- a cleaned fixed-camera tabletop appearance built from all 24 reviewed
  successful episodes;
- the three observed target regions, provisionally placed in the dual-SO-101
  scene from the existing 0.40 m base separation;
- a P0 fixed-camera view alignment from the three target quadrilaterals;
- the synchronized 12-DoF historical state/action trajectory;
- two Genesis SO-101 arms and the separately documented HIL-derived rigid
  object proxy;
- offline front and wrist camera rendering on the AMD GPU backend.

Raw LeRobot data and machine-specific paths remain outside the repository.
Every exported/reconstructed workspace and replay records input hashes and a
terminal marker.

## Evidence snapshot

The source batch contains 40 reviewed episodes and 61,915 frames at 30 Hz,
including 24 successful and 16 unsuccessful episodes. Both RGB streams are
640×480. The fixed front stream contains a 640×360 active image inside 60-pixel
top and bottom letterbox bands.

The multi-episode fixed-camera reconstruction used 1,156 sampled success
images and composited 600 of them. Its temporal absolute-deviation median is
5 intensity levels and p95 is 26. The three yellow regions provide 12 planar
corner correspondences for the P0 camera fit:

- focal length: 666.30 px under the fixed square-pixel, centered-principal-
  point, zero-distortion model;
- vertical field of view: 39.62 degrees;
- reprojection error: 8.01 px median and 14.15 px p95;
- status: `accepted_p0_view_alignment`, not surveyed calibration.

In a Genesis render, the two fully visible square centers are within about
4.1 px and 4.3 px of the real fixed-camera median. The third square is partly
occluded by the simulated arm at the sampled pose and is excluded by the
full-quadrilateral detector.

The complete episode-000 replay ran for 47.23 seconds on `gs.amdgpu`: 1,418
source frames, 5,668 120 Hz simulation steps, and 237 recorded dual-camera
frames. It completed with hashes and `DONE`. It sent no command to physical
hardware and remains `formal: false` because `amd` is not the designated
competition metric host.

## Negative evidence and boundary

Incremental COLMAP is not a valid reconstruction route for this capture. The
scene is nearly planar: one retained attempt registered only 2 images; the
fixed-intrinsic retry registered 4 images and 1 sparse point. These failed runs
are preserved rather than hidden or made to pass by weakening their gates.

The wrist camera has strong motion and supports target tracking and a planar
relative trajectory, but the hand-eye solve against the current MJCF forward
kinematics is rejected. Rotation residuals are a few degrees while translation
scale is physically implausible. Therefore the existing capture does **not**
directly support any of the following claims:

- a metric 3DGS/NeRF reconstruction;
- a calibrated wrist-camera extrinsic;
- real object mass, friction, compliance, or contact-force identification;
- photorealistic novel views of regions never visible in the source video;
- a simulated handover-success metric equivalent to the real task.

A short surveyed marker capture plus verified real-to-MJCF joint calibration
is the minimum next acquisition needed to remove the wrist-camera metric
boundary. Object material parameters require a separate physical measurement.
