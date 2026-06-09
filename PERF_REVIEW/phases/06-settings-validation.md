# Phase 6: Settings, Validation, And Deployment

Goal: convert wins into reliable user-facing quality controls and avoid misleading perf claims.

## Quality Ladders

Create quality tiers for:

- bounce count
- reflection probe density
- guide-buffer fidelity
- displacement/POM distance and steps
- radiance cache usage
- OMM usage
- SER usage
- area-light/ReSTIR mode

High-end/reference paths stay available. Fast paths should be explicit.

## Validation Matrix

Each candidate change should be tested with:

- normal Release build
- deployed jar and native artifacts
- GPU Trace/native profiler, not Graphics Capture for timing
- warmup before capture
- `frameHistogram`
- `gpuProfile frames=120`
- capture context bundle
- at least one visual pass over water, glass, foliage, emissive blocks, animated textures, and dense chunks

## Deployment Discipline

The user should not have to build manually. Agent workflow should build, install, rebuild jar, and deploy when implementation is requested.

Keep separate:

- normal Release timing build
- Nsight source-attribution build with debug shader info

Use normal Release for final perf claims.

## Perf Report Template

Each phase should end with:

```text
Change:
Build:
Scene:
Options:
Before:
After:
Delta:
Visual risk:
Revert plan:
Commit:
```

## Exit Criteria

- The best measured changes are available through settings or adaptive policies.
- The default profile is faster without silently deleting the physically accurate path.
- Every accepted optimization has a deploy commit and before/after evidence.

