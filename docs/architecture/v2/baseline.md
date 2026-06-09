# V2 Architecture Rewrite — Baseline Snapshot

Captured: 2026-04-03
MCVR HEAD: `729ed97` (main)
Radiance HEAD: `064e370` (main)
Driver: NVIDIA 595.97, RTX 5090

## Codebase Size

### MCVR (C++ Vulkan RT Engine)

| Category | Files | Lines |
|----------|-------|-------|
| src/core/**/*.cpp | 91 | 35,456 |
| src/core/**/*.hpp | 80 | 8,850 |
| src/shader/**/* | 147 | 18,587 |
| **Total** | **318** | **62,893** |

External dependencies: 13 (extern/)
CMakeLists.txt files: 164 (includes extern subtrees)

### Radiance (Java Fabric Mod)

| Category | Files | Lines |
|----------|-------|-------|
| src/**/*.java | 227 | 39,669 |
| Options.java alone | 1 | 5,247 |
| YAML pipeline configs | 8 | — |

JNI native declarations: 260 across 16 files
  - Options.java alone: 157 native methods

### Mixin Surface

| Category | Files |
|----------|-------|
| vulkan_render_integration | 52 |
| vanilla_resource_tracker | 17 |
| vulkan_options | 2 |
| **Total mixins** | **71** |

## Current Module Map

### src/core/ directory structure

```
src/core/
├── middleware/          16 JNI bridge files
├── render/
│   ├── anvil/          (unused/legacy)
│   └── modules/world/
│       ├── cloud/
│       ├── dlss/
│       ├── frame_gen/
│       ├── fsr_upscaler/
│       ├── nrd/
│       ├── post_render/
│       ├── ray_tracing/
│       ├── svgf/
│       ├── template/
│       ├── temporal_accumulation/
│       └── tone_mapping/
└── vulkan/             41 files (wrappers)
```

### Key God-Files (candidates for decomposition)

| File | Lines | Responsibility |
|------|-------|----------------|
| chunks.cpp | ~1800 | BLAS build, compaction, scheduling, geometry upload, reset |
| ray_tracing_module.cpp | ~2100 | Descriptors, SBT, pipelines, SHARC, images, render dispatch |
| world_prepare.cpp | ~950 | Cache, TLAS build/update, instance building, entity handling |
| render_framework.cpp | ~600 | Acquire, submit, present, recreate, GC sync, overlay |
| Options.java | 5,247 | All settings, JNI propagation, UI metadata, persistence |

### External Dependencies (MCVR/extern/)

| Dependency | Type | Notes |
|------------|------|-------|
| aftermath | SDK | Crash dump decoder |
| DLSS | SDK | Ray Reconstruction + upscaling |
| FidelityFX-SDK | SDK | FSR3 upscaler + frame gen |
| glfw | header-only | Window/input |
| glm | header-only | Math |
| nrd | SDK | NRD denoiser |
| omm | SDK | Opacity Micro-Maps |
| sharc | header-only | Hash grid radiance cache |
| stb | header-only | Image I/O |
| streamline | SDK | Reflex |
| vma | header-only | Vulkan Memory Allocator |
| volk | header-only | Vulkan meta-loader |
| vulkan_headers | headers | Vulkan API |

**spdlog**: NOT present. Will be added in PR3.

## Known Stability Issues at Baseline

1. **BLAS compaction disabled** — Blackwell/595.97 driver crash. Runtime toggle deferred.
2. **SHARC disabled** — nvgpucomp64.dll crash. Compile-gated via MCVR_ENABLE_SHARC=OFF.
3. **SER hurts performance (-17%)** — Consider disabling.
4. **PTLAS dead** — Driver bug on Blackwell. Standard TLAS only.
5. **OMM disabled by default** — Micromap lifetime bug (dormant).

## Parity Metrics

V2 must match or exceed these before legacy deletion:

### Functional Parity

- [ ] Startup to title screen (no crash)
- [ ] World load to first rendered frame
- [ ] Render distance change 2 → 12 → 32 (no crash, no stale cache)
- [ ] Window resize / recreate / minimize / restore
- [ ] Clean shutdown (no leaked resources, no hung threads)
- [ ] Chunk load/unload cycle (enter new chunks, return)
- [ ] Entity rendering (mobs, players, items)
- [ ] Glass/water transparency (stable planes)
- [ ] DLSS-RR denoising + upscaling
- [ ] Tone mapping (PsychoV, auto-exposure)
- [ ] Area lights (ReSTIR DI)
- [ ] Offline rendering (3 presets)
- [ ] All 260 JNI methods functional

### Performance Parity

Baseline to be captured via `radiance_benchmark.ps1`:
- [ ] Overworld FPS (avg, p1, p99)
- [ ] Cave FPS
- [ ] Heavy scene FPS
- [ ] VRAM usage at RD 12 / 32
- [ ] Chunk load time (first batch, steady state)
- [ ] GPU per-module timing (RayTracing, DLSS, ToneMapping, etc.)

### Crash Parity

- [ ] No new crash paths introduced
- [ ] Aftermath/diagnostics still functional
- [ ] Validation layers clean at startup + steady state
