# 2026-05-28 SHARC Build Probe

## Result

SHARC is not blocked at shader/core build time on the current machine.

- GPU: NVIDIA GeForce RTX 5090
- Driver: 596.49
- Current deployed/normal MCVR build: `MCVR_ENABLE_SHARC:BOOL=OFF`
- Probe build, NRD off: configure, `shaders`, and `core` succeeded
- Probe build, NRD on with explicit `fxc.exe`: configure, `shaders`, and `core` succeeded

No SHARC-enabled artifacts were deployed during this probe.

## Probe Artifacts

- `C:\RadSER\results\sharc_probe\20260528_162936_nrd_off`
- `C:\RadSER\results\sharc_probe\20260528_163542_nrd_on_fxc`

The fresh NRD-on configure initially failed only because `fxc.exe` was not on `PATH`. Passing:

```text
-DSHADERMAKE_FXC_PATH=C:\Program Files (x86)\Windows Kits\10\bin\10.0.26100.0\x64\fxc.exe
```

allowed the full SHARC + NRD probe build to configure and compile.

## Current Runtime Risk

The saved game options currently request an aggressive SHARC configuration:

```text
sharcEnabled=true
sharcCapacityExponent=26
sharcDownscale=1
sharcUpdateBlockSize=2
sharcUpdateBounces=16
sharcSceneScaleTenths=10
sharcRoughnessThresholdPercent=71
```

If SHARC is compiled into the deployed build, those settings can allocate a very large cache and run an expensive update pass immediately on world load. That is not a fair first validation of SHARC benefit.

## Runtime Gates

With `MCVR_ENABLE_SHARC=ON`, SHARC still only runs when:

- Java/native `sharcEnabled` is true
- offline accumulation is not active
- SHARC buffers are allocated
- SHARC update and resolve pipelines are created
- a SHARC update SBT exists for the active context

Main trace SHARC queries also require:

- push-constant flag bit 32
- bounce index `b >= 1`
- valid hit
- path and current roughness above threshold
- ray segment length at least the SHARC query voxel size

## Existing Measurement

Already present:

- `featureTruth` reports SHARC option, compiled support, buffer/pipeline/SBT readiness, query/update possibility, capacity, and frame index.
- GPU profiler scopes include `RT.SHARCUpdate`, `RT.SHARCResolve`, and `RT.MainTrace`.
- `captureContext` records options, `featureTruth`, GPU profile, frame histogram, VMA stats, scene complexity, and logs.

Missing before performance conclusions:

- query attempts, hits, misses
- skip-by-roughness count
- skip-by-voxel-size count
- early path terminations from cache hits
- SHARC update selected rays, hit/miss updates, and terminated update paths
- touched/active entry count for judging resolve waste

## Next Validation Shape

Do not globally deploy SHARC with the current saved options.

First runtime validation should use a deliberately safe SHARC profile:

- `sharcEnabled=false` at startup, then enable only for an explicit debug sweep
- `sharcCapacityExponent=18` or `19`
- `sharcDownscale=2`
- `sharcUpdateBlockSize=5` or higher
- `sharcUpdateBounces=4`
- keep profiler/captureContext enabled

Acceptance for moving beyond probe:

- world load does not crash
- `featureTruth` shows `sharcCompiled=1`, buffers/pipelines/SBT ready, and update/resolve possible
- GPU profile shows separate `RT.SHARCUpdate`, `RT.SHARCResolve`, and reduced `RT.MainTrace`
- net `RayTracing` or total GPU improves in the same scene, not just `RT.MainTrace`
- no obvious cache leakage, stale lighting, or roughness/refraction artifacts
