# Radiance RT Performance Review

Date: 2026-05-28

This folder is the working plan for restoring Radiance performance. It breaks the current leads into phases so each change can be measured, deployed, and kept or reverted independently.

## Current Read

The fresh 11:28 run is GPU-bound inside ray tracing:

- `TOTAL`: about 13.24 ms
- `RayTracing`: about 11.08 ms
- `RT.MainTrace`: about 8.70 ms
- `RT.BLAS_TLAS`: about 2.21 ms
- `DLSS-RR`: about 1.91 ms
- `RT.TexturePublish`: 0.00 ms
- `RT.ImageBarriers`: 0.001 ms
- Render diag reports `RT mainTrace begin w=1920 h=1080`
- Scene: about 4,180 TLAS instances, about 4,162 live BLAS, 113 lights

This is not primarily Java, texture upload, image barrier, present, or CPU world-prep cost. The main work is `RT.MainTrace`, followed by acceleration-structure work.

## Phase Order

1. Instrumentation and truth reporting
2. Main trace cost reduction
3. Acceleration-structure cost reduction
4. SHARC and radiance-cache decision
5. Alpha, OMM, SER, geometry, and shader divergence
6. Settings, validation, deployment, and no-go guardrails

## Files

- [evidence/2026-05-28-1128-run.md](evidence/2026-05-28-1128-run.md) - measured facts from the latest run.
- [evidence/profiler-gaps.md](evidence/profiler-gaps.md) - what the current logs do and do not prove.
- [evidence/git-history-landmines.md](evidence/git-history-landmines.md) - prior optimization attempts that caused crashes, TDRs, or reversions.
- [agent-findings-compiled.md](agent-findings-compiled.md) - all completed agent findings folded into one reference.
- [phases/01-instrumentation.md](phases/01-instrumentation.md) - make future captures precise.
- [phases/02-main-trace.md](phases/02-main-trace.md) - reduce the 8.7 ms main RT pass.
- [phases/03-acceleration-structures.md](phases/03-acceleration-structures.md) - reduce the 2.2 ms AS block.
- [phases/04-sharc-radiance-cache.md](phases/04-sharc-radiance-cache.md) - decide how SHARC should be compiled, exposed, and tuned.
- [phases/05-alpha-ser-geometry.md](phases/05-alpha-ser-geometry.md) - handle any-hit, OMM, SER, vertex formats, and divergence.
- [phases/06-settings-validation.md](phases/06-settings-validation.md) - quality ladders, test matrix, deployment discipline.
- [references/external-research.md](references/external-research.md) - Vulkan, NVIDIA, AMD, and research references.
- [references/code-map.md](references/code-map.md) - relevant code entry points.
- [no-go-guardrails.md](no-go-guardrails.md) - things that should not be changed casually.

## Success Criteria

- Native profiler reports exclusive child timings for the RT pass.
- The debug UI and capture context expose whether a feature is compiled, initialized, and active, not only whether the Java option is true.
- `RT.MainTrace` has per-feature A/B numbers for bounces, reflection probe, guide writes, POM/displacement, ReSTIR, SER, OMM, and radiance-cache use.
- `RT.BLAS_TLAS` is split enough to know whether TLAS update, TLAS build, chunk BLAS, entity BLAS, SBT setup, or metadata upload is responsible.
- Every optimization has a before/after capture with normal Release timing, not Graphics Capture timing.
- Physical accuracy paths remain available; performance wins become settings, quality tiers, or adaptive policies.
