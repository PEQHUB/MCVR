# Profiler Gaps

## What Worked

- Native nested GPU scopes are visible in logs and the `.ngfx-gputrace` binary contains the top-level marker names.
- `gpu_profile_latest.txt` gives enough data to identify `RT.MainTrace` and `RT.BLAS_TLAS` as the current priority.
- The capture bundle gives options, window state, GPU, driver, scene complexity, and git commits.
- Frame logs rule out Java, present, texture upload, and image barriers as the primary steady-state issue.

## What Did Not Work

- The `.ngfx-gputrace` file did not leave a companion CSV, JSON, or shader-profiler export that can be parsed locally.
- Shader-level source hotspots still need either Nsight GUI inspection or GPU Trace auto-export configured at capture time.
- `gpuProfile` is flat. Parent and child scopes can be double counted if read naively.
- `RT.BLAS_TLAS` is too broad. It currently wraps `worldPrepareContext->render()` and does not distinguish TLAS update from full build, entity BLAS, chunk BLAS, SBT setup, metadata upload, or mega-BLAS work.
- `RT.TexturePublish` measures GPU upload/barrier work, not CPU descriptor update time.
- The UI option state can be misleading. SHARC was shown enabled by options but compiled out by CMake.

## Required Instrumentation

- Hierarchical/exclusive GPU profiler output.
- Split acceleration-structure GPU scopes.
- CPU timers for descriptor updates, texture descriptor publication, SBT setup, and per-frame logging.
- Shader counters behind a debug toggle:
  - actual bounce-depth histogram
  - reflection-probe rays
  - any-hit alpha invocations and ignored intersections
  - POM/displacement steps and self-shadow steps
  - ReSTIR candidates and shadow rays
  - SHARC query attempts, hits, misses, skipped-by-roughness, skipped-by-voxel-size
  - guide image writes by output class

## Capture Discipline

- Use GPU Trace and native timing for performance claims.
- Use Graphics Capture for resource/state debugging only, because it has perturbed this app before.
- Store every capture bundle under `C:\RadSER\results\nsight\<timestamp>\`.
- Keep validation layers and shader debug-info builds out of final performance claims.
- Compare normal Release against the Nsight source-attribution build when shader debug info is used.

