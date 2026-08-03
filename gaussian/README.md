# Gaussian workspace workstream

Build a static, calibrated Gaussian representation of the real handover
workspace using VkSplat/Vulkan RADV. The competition deliverable is a visual
twin and synchronized trajectory replay, not a policy observation, collision
model, dynamic 4DGS system, or online correction pipeline.

The implementation is pinned to VkSplat commit
`e26c254938c81ff85998cd357a9e005e255d9b03`. Bootstrap the Vulkan extension,
then train from an immutable COLMAP dataset:

```bash
bash ops/bootstrap_vksplat.sh
python -m gaussian.vksplat_train \
  --source /root/radeon-oneloop-deps/vksplat-e26c254938c81ff85998cd357a9e005e255d9b03 \
  --dataset /root/radeon-oneloop-data/gaussian/workspace_v1 \
  --output /root/radeon-oneloop-runs/gaussian/workspace_v1 \
  --steps 30000 --evaluate
```

`workspace_capture.schema.json` freezes the capture identity, camera model,
image count, and metric scale anchor. The runner validates the COLMAP model,
hashes every source image and calibration file, disables the viewer, and emits
the splat hash, quality metrics, training time, and Vulkan VRAM evidence.
