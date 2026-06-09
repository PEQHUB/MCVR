# Agent Findings Compiled

This file folds the completed agent findings into the review. The artifact-inspection agent was shut down without a final report; artifact findings in this folder come from local analysis of the 11:28 logs and bundle. The code-path agent completed and its findings are compiled below.

## Code Path Map

- Nested GPU profiling starts in `C:\RadSER\MCVR\src\core\render\render_framework.cpp`.
- Per-world-module scopes are added in `C:\RadSER\MCVR\src\core\render\pipeline.cpp`.
- RT adds nested `ScopedGpuProfile` scopes in `C:\RadSER\MCVR\src\core\render\modules\world\ray_tracing\ray_tracing_module.cpp`.
- RT frame order is:
  - atmosphere
  - `WorldPrepare` BLAS/TLAS
  - descriptor publish
  - image barriers
  - optional SHARC update/resolve
  - main trace
  - optional ReSTIR spatial
  - optional offline accumulation
- Main trace dispatch is in `ray_tracing_module.cpp` and enters `C:\RadSER\MCVR\src\shader\world\ray_tracing\world.rgen`.
- Material work is in `C:\RadSER\MCVR\src\shader\world\ray_tracing\world_solid_transparent.rchit`.
- Alpha any-hit is in `C:\RadSER\MCVR\src\shader\world\ray_tracing\world_transparent.rahit`.
- Texture publish is in `ray_tracing_module.cpp`; descriptor helpers call `vkUpdateDescriptorSets` in `C:\RadSER\MCVR\src\core\vulkan\descriptor.cpp`.

## High-Confidence Leads

- SHARC resolve is O(capacity) every SHARC frame. C++ dispatches `(capacity + 63) / 64`, and the shader resolves one index per invocation. A touched/active-entry list, lower default capacity, or resolve budget would directly reduce this if SHARC is enabled.
- SHARC update reuses the heavy main closest-hit stack while only clearing ReSTIR bits. A SHARC-specific lightweight hit shader could skip POM, guide writes, and some direct-light extras.
- Glossy reflection motion/probe path adds an extra ray for low roughness. Gate it by denoiser mode, quality, roughness, or checkerboard it.
- Full-resolution guide/output image bandwidth is large. Skip or pack outputs not consumed by the active denoiser/upscaler mode.
- `WorldPrepare` appends `per_frame_world_prepare.log` every frame. Guard this behind a diagnostic flag for a clean CPU win.
- TLAS instance data is rebuilt into a fresh host-visible buffer each frame. A persistent grow-only instance buffer should reduce churn.
- Metadata buffers reallocate on exact size changes. Grow-only capacity would avoid churn as visible chunk/entity counts fluctuate.
- SBT hit setup runs for both main and SHARC every frame. Cache by geometry/signature generation when the hit layout is unchanged.
- Area-light CPU sorting and shader fallback loops are worth tightening when area lights are enabled. Use partial top-N selection and a stable-id lookup table.
- Vertex compaction support exists, but chunk build forces full 96-byte vertices due layout issues. Fixing the layout could reduce bandwidth, but requires careful validation.

## Risky Or No-Go Findings

- Do not naively re-parallelize `WorldPrepare` chunk scan/fill; code comments reference a previous `WP:chunkPass1` hang.
- Do not drop stale-texture fallback. The code counts stale texture chunks but keeps them visible.
- Do not remove any-hit alpha tests without a complete OMM/alpha replacement.
- Do not casually enable SHARC compile globally. The CMake option is off and notes a driver compiler issue.
- Do not change `PBRTriangle` stride or vertex formats casually.
- Do not broadly enable mega-BLAS until async/incremental build is handled; current code can synchronously wait on a secondary-queue fence.

## Missing Instrumentation

- Make profiler output hierarchical or exclusive.
- Split `RT.BLAS_TLAS` into chunk BLAS, entity BLAS, TLAS update/build, SBT setup, metadata upload, and mega-BLAS wait.
- Add CPU timers for descriptor updates and texture publish.
- Add SHARC counters: active entries, touched entries, query attempts/hits, resolve count, update rays selected, and early-return rate.
- Add shader feature counters: POM steps, any-hit alpha ignores, ReSTIR candidates/shadow rays, reflection-probe rays, and area-light fallback usage.

