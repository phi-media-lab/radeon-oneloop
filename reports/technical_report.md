# Radeon OneLoop Technical Report

> Status: scaffold. Every quantitative claim must link to a registered formal
> run and raw evidence before submission.

## Abstract

TBD after formal evaluation.

## 1. Problem and real-robot setup

## 2. Single-Radeon system architecture

## 3. Dataset and phase definition

## 4. Minimal Genesis environment

## 5. Baseline and phase-aware ACT

## 6. CPU-edge safety runtime

## 7. Deferred Gaussian workspace experiment

The repository includes a hash-pinned VkSplat/Vulkan RADV runner and a capture
contract. It is excluded from the competition architecture and result tables
because no calibrated, static multi-view capture of the SO-101 workspace was
available at scope freeze. Corgi, synthetic, and dynamic robot-video assets
were explicitly rejected as substitutes.

## 8. Experiments and results

The formal dataset is train-only; no independent validation or simulation
success split exists in this release. Both paired runs therefore use the
predeclared final checkpoint at step 10,000. Intermediate checkpoints are
retained for debugging but are not searched or selected after observing their
training losses.

## 9. Limitations and negative results

## 10. Reproducibility

## 11. Team contributions

## 12. Licenses and upstream contributions
