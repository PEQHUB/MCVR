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

### Bounce Count And Termination

- Run a bounce sweep: 2, 4, 6, 8, 12, 16.
- Record `RT.MainTrace`, visible noise, DLSS-RR behavior, and material correctness.
- Tune Russian roulette or adaptive termination by lobe/roughness/path throughput.
- Preserve an offline/reference mode for ground truth.

### Reflection Probe Rays

- `world.rgen` traces an extra reflection-probe ray for glossy surfaces.
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

