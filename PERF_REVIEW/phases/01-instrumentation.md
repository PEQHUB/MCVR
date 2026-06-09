# Phase 1: Instrumentation And Truth Reporting

Goal: make the next run answer exactly which code path is slow.

## Why First

The current data identifies `RT.MainTrace` and `RT.BLAS_TLAS`, but not the internal causes. Optimizing without this split risks changing the wrong thing or hiding a regression.

## Changes

- Add hierarchical/exclusive GPU profiler output.
- Split `RT.BLAS_TLAS` into:
  - `RT.WP.TLASBuild`
  - `RT.WP.TLASUpdate`
  - `RT.WP.ChunkBLAS`
  - `RT.WP.EntityBLAS`
  - `RT.WP.MetadataUpload`
  - `RT.WP.SBTSetup`
  - `RT.WP.MegaBLAS`
- Add CPU timers for:
  - descriptor updates
  - texture descriptor publication
  - SBT setup
  - per-frame log write cost
  - TLAS instance-buffer upload
- Add feature truth state to debug UI and capture context:
  - Java option requested
  - native CMake compiled support
  - pipeline/resource initialized
  - active this frame
- Add shader counters behind a debug toggle:
  - bounce depth histogram
  - reflection probe rays
  - any-hit alpha attempts and ignores
  - POM/displacement steps
  - guide buffer stores by output
  - ReSTIR candidate/shadow rays
  - SHARC query attempts, hits, skips, and misses

## Likely Files

- `C:\RadSER\MCVR\src\core\render\render_framework.cpp`
- `C:\RadSER\MCVR\src\core\render\pipeline.cpp`
- `C:\RadSER\MCVR\src\core\render\modules\world\ray_tracing\ray_tracing_module.cpp`
- `C:\RadSER\MCVR\src\core\render\modules\world\ray_tracing\submodules\world_prepare.cpp`
- `C:\RadSER\MCVR\src\core\middleware\com_radiance_client_proxy_vulkan_RendererProxy.cpp`
- `C:\RadSER\Radiance\src\main\java\com\radiance\client\debug\DebugRuntimeDiagnostics.java`
- `C:\RadSER\Radiance\src\main\java\com\radiance\client\gui\unified\populators\DebugPopulator.java`

## Tests

- Build and deploy.
- In-game debug menu can show active/compiled states.
- `gpuProfile frames=120` reports child scopes without parent double-count confusion.
- `captureContext` includes feature truth states.
- Normal Release timing is unchanged or overhead is disabled by default.

## Exit Criteria

- A new run can attribute at least 90 percent of `RT.MainTrace` and `RT.BLAS_TLAS` to named subfeatures or named unresolved buckets.

