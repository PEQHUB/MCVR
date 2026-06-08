# No-Go Guardrails

These are not permanent bans. They are changes that should only happen with a focused plan, visual validation, and before/after data.

## Do Not Use For Perf Claims

- Graphics Capture timing. It has perturbed Radiance before.
- Validation-layer runs.
- Nsight shader-debug-info builds unless compared against normal Release.
- Minimized/off-screen/waiting-for-framebuffer runs.

## Do Not Remove Correctness Paths Casually

- Texture upload/layout synchronization.
- Stale-texture fallback that keeps geometry visible.
- BLAS/buffer lifetime rules for frame-stable GPU references.
- Any-hit alpha behavior without OMM or another complete replacement.
- Physical-quality paths; make them settings or quality tiers instead.

## Do Not Change Casually

- `PBRTriangle` 96-byte stride.
- `emissiveBlockType` shader packing.
- `TextureMapEntry.properties` bit meanings.
- Chunk render-data producer/consumer stability.
- WorldPrepare parallelization, especially around chunk scan/fill.
- Mega-BLAS enablement while it can synchronously wait on a secondary-queue fence.
- Global SHARC compile enablement while the CMake driver-compiler warning remains unresolved.

## Required Before Risky Changes

- A normal Release baseline.
- A rollback path.
- A visual checklist.
- A deployed build.
- A capture bundle containing options, feature truth states, GPU profile, frame histogram, and scene complexity.

