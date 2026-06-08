# Git History Landmines

This file summarizes prior optimization attempts found in git history. Treat these as constraints before making new performance changes.

## Major Pattern

The bad outcomes cluster around GPU lifetime, acceleration-structure mutation, cross-queue ownership, driver-specific AS features, and overly broad optimization bundles. New work should be smaller, measured, and easier to revert.

## Reverted Or Risky Bundles

### Performance Audit Bundle

- MCVR `3cc6437` - `Performance audit + displacement edge fade unification`
- MCVR `86f88b5` - reverted it
- MCVR `e1d74e5` - reapplied it
- Radiance `77cd97b` - Java/UI side of the same broad performance bundle
- Radiance `f2e7e1d` - reverted it
- Radiance `fdaa984` - reapplied it

This bundle mixed payload packing, descriptor batching, TLAS update mode, memory gating, metadata pooling, entity pooling, shader changes, displacement changes, and UI/chunk loading changes. Lesson: do not bundle many unrelated optimizations together. Land one hypothesis at a time with a capture bundle.

### BLAS Compaction

- MCVR `cbda5e9` measured compaction savings.
- MCVR `c63e908` enabled chunk BLAS compaction and reported large VRAM savings.
- MCVR `377bb98` disabled compaction due Blackwell/driver `595.97` TDR/crash and added cross-queue GC sync.
- MCVR `a4e11e2` again records compaction disabled because of corrupt BVH from `vkCmdCopyAccelerationStructureKHR`.

Lesson: do not re-enable BLAS compaction as a quick win. If revisited, isolate it behind a feature flag, require current-driver validation, and test lifetime/resource handoff heavily.

### Mega-BLAS

- MCVR `c31bb01` added mega-BLAS infrastructure.
- MCVR `c4322bf` integrated distant chunk merging into mega-BLASes, disabled by default.
- Radiance `c93534f` removed the mega-BLAS option because synchronous fence stalls caused TDR risk on driver `595.97`.
- MCVR `a4e11e2` also records mega-BLAS removal as a driver workaround.

Lesson: do not enable mega-BLAS broadly until async/incremental build, cross-queue visibility, and fence-stall behavior are solved.

### Partitioned TLAS

- MCVR `104c207` added `VK_NV_partitioned_acceleration_structure` and claimed lower TLAS cost.
- MCVR `fa1d48a` records that crash still reproduced.
- MCVR `1f152c6` disabled PTLAS pending driver fix after Aftermath found AS build using a freed BLAS address.

Lesson: PTLAS is not a safe first step. The standard TLAS path is stable; revisit PTLAS only after lifetime tracking and driver behavior are proven.

## Lifetime And Queue Hazards

- MCVR `02aa491` fixed `DEVICE_LOST` from staging buffers destroyed before GPU copy commands executed.
- MCVR `47e896d` fixed BLAS references being freed while the GPU still traced them and fixed a command buffer reset while pending.
- MCVR `4560f93` mitigated render-distance change crashes with BLAS thread stop and `vkDeviceWaitIdle`, but notes a full fix needs frame-indexed deletion or timeline lifetimes.
- MCVR `3258b7f` saved vertex/index buffers in TLAS snapshots to prevent BDA read-after-destroy.
- MCVR `3be7de8` fixed TLAS UPDATE corruption caused by entity BLAS handles changing without being tracked.
- MCVR `420f54a` fixed entity buffer lifetime by saving batch refs per swapchain context.

Lesson: every AS, BDA buffer, staging buffer, and descriptor-visible resource must remain alive until the consuming GPU context has completed. Optimizations that reduce allocations must not shorten lifetimes.

## SHARC Hazards

- MCVR `d4e433d` fixed SHARC runtime enable crash from null BDA push constants.
- MCVR `feaf7df` fixed SHARC runtime enable crash from null SBT pointers.
- MCVR `729ed97` added `MCVR_ENABLE_SHARC` and `SHARC_COMPILE_ENABLE` gates, defaulting off because driver `595.97` crashed compiling SHARC Int64-atomics+BDA code.

Lesson: report SHARC truthfully as option requested, compiled support, initialized, and active. Do not turn it on globally without a current-driver compile and runtime validation pass.

## Known Good Or Useful Wins

- Radiance `7649d2f` fixed a 30-40 ms Java chunk rebuild sort bottleneck with bounded selection.
- Radiance `54b39b8` purged stale rebuild queue entries and added queue diagnostics.
- MCVR `7045d46` added full-frame CPU timing and exposed that Java work was once the dominant bottleneck.
- MCVR `d99b760` moved chunk build work to a dedicated BLAS builder thread and async pipeline.
- MCVR `a19e7dd` improved TLAS instance population, but this area is explicitly sensitive to thread-safety and lifetime.

Lesson: instrumentation and bounded queue work have been safer than driver-facing AS feature experiments.

## Rules For The Next Patch

- One optimization hypothesis per commit.
- No AS feature re-enablement without a feature flag and explicit before/after capture.
- No lifetime-shortening changes without context/fence/timeline proof.
- No cross-queue change without resource visibility and ownership notes.
- No broad shader payload or vertex format rewrite before counters identify a dominant cause.
- Prefer instrumentation first, because the old failures were too broad to attribute cleanly.

