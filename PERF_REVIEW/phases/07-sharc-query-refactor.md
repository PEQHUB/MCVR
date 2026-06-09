# Phase 7: SHARC Query Refactor

Goal: recover SHARC for performance and cave/global-illumination quality without compiling the query path into the current `world.rgen` megakernel.

## Current Truth

Local test system:

```text
GPU: NVIDIA GeForce RTX 5090
Driver: 596.49
nvgpucomp64.dll: 32.0.15.9649
```

Stable:

- SHARC buffers can allocate and initialize.
- SHARC update RT pipeline can create and dispatch.
- SHARC resolve compute pipeline can create and dispatch.
- Main trace query off is boot-safe.

Crash:

- `MCVR_SHARC_MAIN_TRACE_QUERY_MODE=radiance` crashed in `nvgpucomp64.dll` during world RT pipeline creation.
- `MCVR_SHARC_MAIN_TRACE_QUERY_MODE=hash` also crashed in `nvgpucomp64.dll` during world RT pipeline creation.
- Both crashes happened immediately after `World RT pipeline create start`, before query warmup, cache contents, update/resolve dispatch, or runtime activation could matter.

Conclusion: the blocker is not SHARC as a cache. The blocker is compiling SHARC hash/query logic into the large production raygen pipeline.

## Documentation Signals

- NVIDIA SHARC integration describes three conceptual render-time pieces: update, resolve, and render/query. We do not have to place the query inside the existing all-feature main raygen.
- NVIDIA RTX best practices favor simpler shaders, limiting divergence, avoiding excessive trace calls in shaders, and using more than one ray tracing pipeline/state object when it keeps shaders specialized.
- NVIDIA Vulkan guidance recommends asynchronous pipeline creation, pipeline caches, specialization constants, and starting with simpler generic pipelines before specializing.
- NVIDIA Vulkan driver notes for 2026 include recent compiler fixes around descriptor/BDA-adjacent cases and 64-bit indexing. Our `hash` crash is close enough that a minimal repro is worth producing, but production recovery should not depend on a driver fix.

## Information Needed

Information we can gather automatically:

- Driver version, GPU, and `nvgpucomp64.dll` version.
- CMake SHARC gates and query mode.
- Exact last Radiance marker before crash.
- SPIR-V disassembly and instruction counts for `world_rgen.spv` per query mode.
- Pipeline creation success/failure by mode: `off`, `stub`, `math`, `hash`, `radiance`.
- GPU timings for SHARC update/resolve with query off.
- Debug-menu capture bundle: feature truth, scene complexity, frame histogram, GPU profile, SHARC settings.

Information we need from user-run scenes:

- One cave/indoor scene where SHARC historically helped.
- One outdoor/control scene where SHARC is expected to help little.
- Visual notes: light leaking, stale indirect lighting, water/glass/specular instability, animated/emissive response.
- Whether testing NVIDIA Vulkan beta driver `596.60+` is acceptable later. This is optional and should not replace the refactor.

## Refactor Shape

### Stage 0: Keep Production Boot-Safe

Deploy baseline:

```text
MCVR_ENABLE_SHARC=ON
MCVR_ENABLE_SHARC_BUFFERS=ON
MCVR_ENABLE_SHARC_UPDATE_PASS=ON
MCVR_ENABLE_SHARC_RESOLVE_PASS=ON
MCVR_ENABLE_SHARC_MAIN_TRACE_QUERY=OFF
MCVR_SHARC_MAIN_TRACE_QUERY_MODE=off
```

The user can boot, render, and run SHARC probe/debug captures. Main trace gets no cache benefit yet.

### Stage 1: Make The Crash Repro Small

Build a tiny standalone pipeline or sample shader that contains only:

- SHARC hash math
- hash-entry buffer-device-address read
- optional resolved-radiance read
- no path-tracing loop
- no DLSS-RR guide writes
- no closest-hit stack
- no material system

Acceptance:

- If the tiny pipeline crashes, package it as a minimal NVIDIA repro.
- If it does not crash, the bug is an interaction with `world.rgen` size/live-state/control-flow, and we continue with query isolation.

### Stage 2: Add A Separate Query Classification Pass

Before main trace, run a small compute or raygen pass over selected pixels/history candidates:

- computes SHARC hash/query for candidate secondary-hit positions when available
- writes cache-hit radiance or miss state into compact screen-space guide buffers
- keeps all BDA/hash/radiance reads out of `world.rgen`

This pass must be optional and timed separately:

- `RT.SHARCQuery`
- query attempts
- query hits
- skipped roughness
- skipped voxel-size
- skipped missing candidate
- invalid/stale cache misses

Acceptance:

- Pipeline creates reliably.
- No boot crash.
- `RT.SHARCQuery + RT.SHARCUpdate + RT.SHARCResolve` is less than the saved `RT.MainTrace` time in the cave scene.

### Stage 3: Consume Query Results In Main Trace With Minimal Live State

Main raygen should only read a simple screen-space result:

- `hit/miss`
- cached radiance
- allowed bounce range or validity flag

It should not compute SHARC hashes, traverse hash buckets, or read SHARC cache buffers directly.

Start conservative:

- query only `bounce >= 1`
- diffuse/rough paths only
- no specular/refraction/water/glass use
- keep voxel-size leak guard
- keep full-quality path available

Acceptance:

- Caves show lower `RT.MainTrace`.
- No visible light leaking through one-block walls.
- No specular, glass, water, hand/item, particle, or emissive instability.

### Stage 4: Reduce SHARC Overhead

Only after query is stable:

- replace full-capacity resolve with touched/active-entry resolve or budgeted resolve
- lower default capacity from large custom settings to measured scene needs
- build a lightweight SHARC update hit shader that skips POM/displacement, DLSS-RR guide writes, dense direct-light extras, and unrelated main closest-hit work
- cache SBT/update resources by signature where possible

Acceptance:

- SHARC net win is positive in cave scenes after all update/resolve/query costs.
- Outdoor scenes do not regress beyond the selected quality preset budget.

## Decision Rules

- If `hash` mode crashes in `world.rgen`, do not continue editing the in-raygen query path as production work. This has already happened.
- If standalone tiny query crashes, create an NVIDIA bug package and keep production query off.
- If standalone tiny query works, continue isolation/refactor and avoid putting BDA hash traversal back into `world.rgen`.
- If net SHARC gain is under `0.5 ms` in the cave test after overhead reductions, stop and keep SHARC experimental.
- If SHARC improves caves but hurts outdoors, ship it as a scene/quality-sensitive setting, not a hidden default.

## Debug And Capture Workflow

No mid-game PowerShell. Expose through DebugBridge/debug menu:

- `Run SHARC Probe`
- `Run SHARC Query Refactor Probe`
- `captureContext`
- `featureTruth`
- `gpuProfile`
- `frameHistogram`

Each bundle should include:

- driver/GPU/compiler DLL version
- SHARC compile gates
- query architecture: `off`, `world_rgen`, `standalone_query`, or `screen_space_query`
- scene complexity
- chunk settled state
- SHARC settings and active capacity
- timings for update, resolve, query, and main trace
- query counters and hit rate
- visual checklist result

## NVIDIA Repro Package

If a small shader still crashes, collect:

- minimal GLSL/HLSL and SPIR-V
- command line used to compile SPIR-V
- exact `vkCreateRayTracingPipelinesKHR` setup
- driver/GPU info
- `hs_err_pid*.log`
- Radiance marker log
- CMake SHARC gates
- pass/fail table for query modes

This is separate from production recovery. The game should stay boot-safe while the repro exists.

## Success Criteria

- Production builds boot with SHARC compiled and main query off.
- SHARC update/resolve remain measurable and stable.
- Query is recovered through an isolated pass or tiny alternate pipeline, not by bloating `world.rgen`.
- Cave scenes gain meaningful total RT frame time without unacceptable visual loss.
- Full-quality non-SHARC path remains available.
