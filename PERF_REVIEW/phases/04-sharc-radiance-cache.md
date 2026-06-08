# Phase 4: SHARC And Radiance Cache

Goal: decide whether SHARC should be enabled, fixed, replaced, or exposed as experimental.

## Current Truth

The 11:28 run did not have compiled SHARC support:

```text
MCVR_ENABLE_SHARC:BOOL=OFF
```

The Java option reported `sharcEnabled=true`, but the shader path requires `SHARC_COMPILE_ENABLE=1`, which only happens when native CMake enables SHARC.

Update from the 2026-05-28 build probe: SHARC-enabled shader and core builds succeed on RTX 5090 driver `596.49`, including the full NRD-on build when `SHADERMAKE_FXC_PATH` points at the Windows SDK `fxc.exe`. See [../evidence/2026-05-28-sharc-build-probe.md](../evidence/2026-05-28-sharc-build-probe.md).

## First Fix

Expose these states separately:

- option requested
- native compiled support
- SHARC buffers initialized
- update pipeline initialized
- resolve pipeline initialized
- SBT valid
- main trace queries active
- update/resolve passes active

## If SHARC Is Re-enabled

Measure before tuning:

- `RT.SHARCUpdate`
- `RT.SHARCResolve`
- query attempts
- query hits
- skipped by roughness threshold
- skipped by voxel size
- active/touched entries
- update rays selected
- resolve entries processed

Do not deploy the first SHARC runtime test with persisted custom settings. The local options file currently requests `capacityExponent=26`, full-rate/downscale `1`, update block `2`, and update bounces `16`; that can turn the first test into a huge allocation/update stress test instead of a useful SHARC evaluation.

Use a safe validation profile first:

- startup SHARC off; enable only for an explicit sweep
- capacity exponent `18` or `19`
- downscale `2`
- update block size `5+`
- update bounces `4`

## Optimization Leads

- SHARC resolve is O(capacity). Replace full-capacity resolve with touched/active-entry resolve or a budgeted resolve.
- Lower default capacity until scene data proves it needs the larger cache.
- Build a lightweight SHARC update hit shader that avoids POM, guide writes, and direct-light extras.
- Ensure SHARC only handles appropriate diffuse/rough indirect paths and does not degrade specular/refraction.
- Treat SHARC as a quality/performance setting, not as a hidden default.

## Alternatives

- NVIDIA RTXGI SHaRC/NRC style cache integration.
- AMD FSR Radiance Cache style query/training split.
- ReSTIR GI or ReSTIR PT direction for indirect path reuse.

## Risks

- CMake notes a driver compiler issue with SHARC.
- Re-enabling can add update/resolve passes and hurt performance if cache hit rate is low.
- Cache leakage, roughness gating, stale lighting, and dynamic scene response must be validated visually.

## Exit Criteria

- Debug UI states are truthful.
- SHARC either has measured benefit and stays behind a quality setting, or remains compiled off with a clear reason and no misleading runtime state.
