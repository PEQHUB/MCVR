# Phase 2: Main Trace Cost Reduction

Goal: reduce the current about 8.7 ms `RT.MainTrace` cost without removing high-quality paths.

## Evidence

- `RT.MainTrace` is the largest measured cost.
- Main trace dispatch is in `C:\RadSER\MCVR\src\core\render\modules\world\ray_tracing\ray_tracing_module.cpp`.
- Shader entry is `C:\RadSER\MCVR\src\shader\world\ray_tracing\world.rgen`.
- Main bounce loop runs up to `pc.numRayBounces` in `world.rgen`.
- The 11:28 options used `rayBounces=12`.
- Dense guide-buffer writes and reflection-probe rays are active code paths.

## Leads

### Hand Pass

- New floor sweeps show disabling the hand pass saved `2.526 ms` in the open/plains scene and `7.291 ms` in the heavier sky-lit occluded scene.
- Feature-truth dumps from those scenes did not show `entityFlag_hand_count`, so the first suspected bug is a full-screen `HAND_MASK` query even when the TLAS has zero HAND instances.
- First safe fix: pass the current HAND instance count from `WorldPrepare` to `world.rgen` and skip the hand loop when it is zero.
- Later, if HAND instances exist and remain expensive, add conservative screen-region scissoring or hand-specific bounce/lighting controls.

### Secondary Direct Lighting

- Disabling the secondary sun/direct block saved `3.008 ms` in the open/plains scene and `9.380 ms` in the sky-lit occluded scene.
- This is not dead work when the sky is offscreen but contributing. Treat it as a quality/importance-sampling problem, not a blind removal.
- Candidate setting: `Secondary Direct Lighting Quality`, with full quality still available.

### Bounce Count And Termination

- Captured sweeps show 12 to 4 bounces saved `1.692 ms` in the open/plains scene and `7.859 ms` in the heavier sky-lit occluded scene.
- Bounce count is a real quality/performance lever in occluded views, but should not be the first global default reduction.
- Tune Russian roulette or adaptive termination by lobe/roughness/path throughput.
- Preserve an offline/reference mode for ground truth.

### Reflection Probe Rays

- `world.rgen` traces an extra reflection-probe ray for glossy surfaces.
- Floor sweeps showed only `0.070-0.151 ms` saved by disabling the dense reflection probe; this is not the current priority.
- Gate by denoiser mode, roughness, material class, and screen-space importance.
- Test checkerboard or half-rate probe generation.
- Add a quality setting for dense reflection motion vectors.

### Guide Buffer Writes

- Many full-resolution storage images are written for denoiser/upscaler guidance.
- Skip outputs that are not consumed by the active mode.
- Pack related low-precision masks where possible.
- Add A/B toggles for guide-output groups.

### POM And Displacement

- Material hit shaders include shader displacement/POM logic.
- Add counters for primary POM steps, secondary self-shadow steps, and POM-eligible pixels.
- Gate expensive displacement by distance, material class, and quality tier.

### Material Divergence

- `world_solid_transparent.rchit` is large and branch-heavy.
- Use shader profiler data to identify hot functions once GPU Trace source export is available.
- Consider specialized hit shader permutations only after counters prove a dominant path.

## Tests

- Each change gets normal Release before/after numbers.
- Compare internal 1920x1080 run and at least one higher internal resolution setting.
- Capture frameHistogram and gpuProfile after warmup.
- Verify no guide-output toggle breaks DLSS-RR stability, reflections, glass, water, particles, or emissive surfaces.

## Exit Criteria

- `RT.MainTrace` drops materially while the high-quality/reference path remains available.
- Each new quality lever has a documented visual/perf tradeoff.
