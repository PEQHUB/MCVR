# Frame Generation Architecture — MCVR Integration Plan

## Current State Analysis

### What MCVR Already Has
- **Streamline 2.10.3** integrated via interposer model (`sl.interposer.dll`)
- **Reflex** fully working (sleep + PCL markers every frame)
- **PCL** frame lifecycle markers (SimStart→SimEnd→RenderSubmitStart→End→PresentStart→End)
- **Frame tokens** per-frame via `slGetNewFrameToken()`
- **DLSS-RR** via NGX (21 resource bindings, full guide buffer support)
- **Swapchain**: Triple-buffered, HDR10 (A2B10G10R10_UNORM), MAILBOX with Reflex
- **HdrCompositePass**: Composites world + UI overlay → swapchain image

### What's Missing for DLSS-G
1. `sl::kFeatureDLSS_G` not loaded in feature list
2. No `slDLSSGSetOptions` / `slDLSSGGetState` function pointers
3. No `slSetConstants()` call (camera matrices for frame gen)
4. No resource tagging (`slSetTagForFrame()` for depth, MVs, HUD-less, UI)
5. No HUD-less buffer capture point
6. No UI-only buffer capture
7. No swapchain recreation logic for DLSS-G toggle

---

## Proposed Architecture

### Pipeline Position
```
RT Module (render res)
    ↓
DLSS-RR / NRD / SVGF (denoise + upscale to display res)
    ↓
PostRender Module (bloom, CAS)
    ↓
ToneMapping Module (auto-exposure, SDR/HDR)
    ↓
[NEW] FrameGenModule — captures HUD-less buffer, tags resources
    ↓
HdrCompositePass (world + UI → swapchain)
    ↓
[NEW] FrameGenModule::tagUI() — captures UI-only buffer
    ↓
vkQueuePresentKHR (Streamline intercepts, generates frames)
```

### New Module: FrameGenModule (WorldModule)

```
FrameGenModule
├── Inputs:
│   ├── [0] Tone-mapped HDR output (display res) — becomes HUD-less buffer
│   ├── [1] Depth (render res) — from RT module
│   └── [2] Motion Vectors (render res) — from RT module
├── Outputs:
│   └── [0] Pass-through (same as input[0], unmodified)
├── Per-frame work:
│   ├── Copy tone-mapped output → HUD-less buffer (before UI composite)
│   ├── slSetConstants() — camera matrices, MV scale, jitter, reset flags
│   ├── slSetTagForFrame() — depth at render res
│   ├── slSetTagForFrame() — MVs at render res
│   ├── slSetTagForFrame() — HUD-less at display res
│   └── slSetTagForFrame() — UI at display res (after composite)
└── Lifecycle:
    ├── init() — query DLSS-G support, load function pointers
    ├── build() — allocate HUD-less + UI capture images
    ├── setMode() — enable/disable, set numFramesToGenerate
    └── preClose() — cleanup
```

### HUD-less Buffer Strategy

**Option A (Recommended)**: Copy the tone-mapped output before `fuseFinal()` composites the UI.
- The output of `ToneMappingModule` is the last stage before UI composition
- Copy this to a persistent image that survives until present
- Tag with `sl::kBufferTypeHUDLessColor` + `eValidUntilPresent`

**Option B**: Render the world composite pass without UI, then render UI separately.
- Requires refactoring `HdrCompositePass` to output two targets
- More complex but provides true separation

### UI Buffer Strategy

**Option A (Simple)**: Tag the overlay image from `RenderFramework::fuseOverlay()` directly.
- The overlay is already rendered to a separate image before compositing
- Just need to ensure alpha channel is correct (0 = no UI)

**Option B**: Render UI to an offscreen target with alpha, tag it.
- More control over what constitutes "UI"

### Swapchain Integration

When DLSS-G is toggled ON:
1. `vkDeviceWaitIdle()`
2. Mark `needRecreate` flag on renderer
3. During recreation: `slSetFeatureLoaded(kFeatureDLSS_G, true)` BEFORE swapchain creation
4. Streamline's interposer will manage the new swapchain

When DLSS-G is toggled OFF:
1. `slDLSSGSetOptions(viewport, {mode: eOff})`
2. `vkDeviceWaitIdle()`
3. Release swapchain-dependent resources
4. `slSetFeatureLoaded(kFeatureDLSS_G, false)`
5. Recreate swapchain without DLSS-G management

### Present Mode Interaction
- Current: MAILBOX with Reflex, FIFO with VSync, IMMEDIATE without VSync
- DLSS-G active: Keep MAILBOX (Reflex controls timing, DLSS-G pacer handles generated frames)
- No change needed to present mode selection logic

---

## Implementation Phases

### Phase 1: StreamlineContext Extension
- Add `kFeatureDLSS_G` to feature loading
- Load `slDLSSGSetOptions`, `slDLSSGGetState` function pointers
- Add `slSetConstants()` call with camera matrices per frame
- Query DLSS-G hardware support
- Add enable/disable API

### Phase 2: Resource Tagging Infrastructure
- Create `sl::Resource` wrappers for depth, MVs, HUD-less, UI
- Implement `slSetTagForFrame()` calls at correct pipeline points
- Verify image layouts are correct at tagging time

### Phase 3: HUD-less Buffer Capture
- Add copy pass after tone mapping, before UI composite
- Persistent image per swapchain context
- Correct format (must match swapchain format for DLSS-G)

### Phase 4: UI Buffer Capture
- Capture overlay image with alpha channel
- Tag with `kBufferTypeUIColorAndAlpha`

### Phase 5: Swapchain Recreation
- Toggle logic with `slSetFeatureLoaded()`
- Coordinate with existing `RenderFramework::reconstructResources()`
- Handle edge cases (window resize during FG active, etc.)

### Phase 6: Java UI Integration
- Add Frame Generation option to Options.java
- Add quality modes (Off, On, Auto)
- Add MFG multiplier (2x, 3x, 4x) — auto-detect hardware capability
- Add to Pipeline.yaml module configs
- JNI bridge for setOption()

### Phase 7: FSR 3 Frame Generation (Cross-Vendor Fallback)
- Populate FidelityFX-SDK extern/ with FSR3 FG SDK
- Implement FsrFrameGenModule as alternative to DLSS-G
- Runtime GPU detection selects DLSS-G (NVIDIA) vs FSR3-FG (AMD/Intel)
- Same resource tagging pattern but via FFX API instead of Streamline

---

## Interaction with Existing Features

### Offline Rendering
- **DLSS-G must be disabled during offline accumulation** (state 2)
- Generated frames would corrupt the Welford accumulator
- Auto-disable when `offlineState != 0`

### Motion Blur
- When DLSS-G is active, halve application motion blur strength
- Generated frames already contain interpolated motion

### VSync / Frame Limiter
- DLSS-G's pacer handles frame timing — disable application-side frame limiters
- Reflex sleep still operates (latency reduction)

### HDR
- DLSS-G requires RGB10/UINT10 format — MCVR's HDR10 swapchain is compatible
- scRGB/FP16 NOT supported by DLSS-G

### Debug Bridge
- Add frame gen status to debug JSON output
- Report: mode (off/on/auto), multiplier, generated FPS, status

---

## File Layout (Proposed)

```
src/core/render/modules/world/frame_gen/
├── frame_gen_module.hpp          — FrameGenModule WorldModule class
├── frame_gen_module.cpp          — Module lifecycle, resource tagging
├── dlssg_wrapper.hpp             — DLSS-G Streamline wrapper
├── dlssg_wrapper.cpp             — Init, options, state queries
├── fsr_frame_gen_wrapper.hpp     — FSR3 FG wrapper (Phase 7)
└── fsr_frame_gen_wrapper.cpp     — FSR3 FG implementation (Phase 7)
```
