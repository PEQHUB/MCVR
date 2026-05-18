# DLSS-G Frame Generation — Implementation Reference

## Merge Instructions

Both repos have a `FrameGen` branch with clean, squashed commits ready to merge.

**MCVR** (`C:\RadSER\MCVR-FrameGen`, branch `FrameGen`):
```
e80ce0e  Upgrade G-buffers to FP32, fix DLSS-RR exposure and metal albedo
c24c5ec  Add DLSS-G frame generation via Streamline
9af6be5  Add frame generation research reports (reference only)  ← optional
```

**Radiance** (`C:\RadSER\Radiance-FrameGen`, branch `FrameGen`):
```
d41d944  Update pipeline YAML configs and VulkanConstants for FP32 buffers
31fd640  Add DLSS-G frame generation UI and sl.dlss_g.dll plugin
```

**Merge order:** Commit 1 in each repo (FP32) is independent and can merge to main separately. Commit 2 (DLSS-G) depends on commit 1. Research docs (MCVR commit 3) are `.gitignore`'d on main — skip unless you want them.

**IMPORTANT:** Main has the same FP32 changes as uncommitted modifications. Either commit them on main first and rebase FrameGen, or merge FrameGen (which includes them) and discard main's working copy changes.

---

## Architecture Overview

DLSS-G operates at the **swapchain/present level**, not as a pipeline WorldModule. Streamline's interposer intercepts `vkQueuePresentKHR` and inserts AI-generated frames.

```
RT Module (render res, FP32)
    ↓
DLSS-RR / NRD / SVGF (denoise + upscale)
    ↓
PostRender (bloom, CAS)
    ↓
ToneMapping (auto-exposure, SDR/HDR)
    ↓
[FrameGenManager::tagFrame]  ← tags depth, MVs, HUD-less, UI for SL
    ↓
HdrCompositePass (world + UI → swapchain)
    ↓
vkQueuePresentKHR  ← Streamline intercepts, runs AI model, presents real + generated frames
```

---

## Files Added/Modified

### New Files (MCVR)
| File | Purpose |
|------|---------|
| `src/core/render/modules/world/frame_gen/frame_gen_manager.hpp` | Static manager class API |
| `src/core/render/modules/world/frame_gen/frame_gen_manager.cpp` | Constants, resource tagging, swapchain lifecycle |
| `src/shader/world/ray_tracing/emission_compose.comp` | Subtract/add emission around DLSS-RR |

### Modified Files (MCVR)
| File | Changes |
|------|---------|
| `streamline_context.hpp/cpp` | kFeatureDLSS_G, slSetConstants, slSetTagForFrame, slSetFeatureLoaded, slDLSSGSetOptions/GetState, eUseFrameBasedResourceTagging, requiredTags logging |
| `renderer.hpp/cpp` | frameGenEnabled/Mode/Multiplier options, frameGenDepthImages/frameGenMotionVectorImages statics |
| `render_framework.cpp` | FrameGenManager::init/tagFrame/before+afterSwapchainRecreate integration |
| `ray_tracing_module.cpp` | Publishes depth/MV images to Renderer statics |
| `image.hpp` | vkDeviceMemory() accessor |
| `com_radiance_client_option_Options.cpp` | 4 JNI methods |
| `dlss_module.cpp` | Pre-exposure comment cleanup |
| `dlss_wrapper.cpp` | InExposureScale=1.0 |
| `tone_mapping_module.cpp` | Pre-exposure comment cleanup |
| 40+ shaders | FP16 → FP32 format declarations |

### New Files (Radiance)
| File | Purpose |
|------|---------|
| `src/main/resources/sl.dlss_g.dll` | Streamline DLSS-G plugin (v2.10.3) |

### Modified Files (Radiance)
| File | Changes |
|------|---------|
| `Options.java` | frameGenMode/Multiplier fields, native methods, load/save, Reflex auto-dependency |
| `RadianceSettingsScreen.java` | Frame Generation UI row |
| `RadianceClient.java` | sl.dlss_g.dll extraction (separate from core SL plugins) |
| `en_us.json` | Translation keys |
| `VulkanConstants.java` | R32_SFLOAT, R32G32_SFLOAT, R32G32B32A32_SFLOAT |
| 6 YAML module configs | FP32 format strings |

---

## Buffer Tagging (Per DLSS-G Guide Section 5.0)

| Buffer | Tag | Resolution | Format | Source |
|--------|-----|-----------|--------|--------|
| Linear Depth | `kBufferTypeLinearDepth` (49) | Render | R32_SFLOAT | RT module output [5] |
| Motion Vectors | `kBufferTypeMotionVectors` (1) | Render | R32G32_SFLOAT | RT module output [4] |
| HUD-less Color | `kBufferTypeHUDLessColor` (2) | Display | Swapchain format | Pipeline outputImage |
| UI Color+Alpha | `kBufferTypeUIColorAndAlpha` (23) | Display | R8G8B8A8_SRGB | UIModule overlayDrawColorImage |

All tagged with `eValidUntilPresent` lifecycle. No command buffer needed.

**Not tagged (correctly excluded per guide):**
- `kBufferTypeBidirectionalDistortionField` — no post-process distortion
- `kBufferTypeNoWarpMask` — no Reflex 2
- `kBufferTypeBackbuffer` — full-frame (no subrect)

---

## SL Common Constants (Per Frame)

Set via `slSetConstants()` in `FrameGenManager::tagFrame()`:

| Field | Source | Notes |
|-------|--------|-------|
| `cameraViewToClip` | `cameraProjMat` | Row-major (transpose of GLM column-major) |
| `clipToCameraView` | `cameraProjMatInv` | Same transpose |
| `clipToPrevClip` | `prevProj * prevView * viewInv * projInv` | Cached prev inverses (no per-frame glm::inverse) |
| `prevClipToClip` | `proj * view * prevViewInv * prevProjInv` | Same |
| `jitterOffset` | `worldUBO->cameraJitter` | Pixel space |
| `mvecScale` | `{1/renderWidth, 1/renderHeight}` | Pixel-space MVs → [-1,1] |
| `cameraPos/Up/Right/Fwd` | From `cameraEffectedViewMatInv` | World space |
| `cameraNear/Far` | Extracted from projection matrix | Handles infinite far (p22≈0) |
| `cameraFOV` | `2*atan(1/p11)` | Guarded against zero |
| `depthInverted` | `eFalse` | Linear depth (larger = farther) |
| `cameraMotionIncluded` | `eTrue` | MVs include camera motion |
| `reset` | `eTrue` on teleport/first frame | Uses `resetExposureAdaptation` |

---

## Swapchain Lifecycle

**On any swapchain recreation (resize, toggle, etc.):**
1. `beforeSwapchainRecreate()`: Always sets DLSS-G to `eOff` (prevents vkQueuePresentKHR deadlock). If user disabled FG, also calls `slSetFeatureLoaded(false)`.
2. Swapchain tears down and rebuilds normally.
3. `afterSwapchainRecreate()`: If user wants FG, calls `slSetFeatureLoaded(true)` (if first enable) then `slDLSSGSetOptions(mode, multiplier)`.

**Thread safety:** All SL API calls happen on the render thread during recreation. JNI `nativeSetFrameGenMode` only writes to `Renderer::options` and sets `needRecreate=true`.

---

## Reflex Dependency

DLSS-G requires Reflex. Handled in Java:
- `setFrameGenMode(on)` → auto-enables Reflex if off
- `setReflexEnabled(false)` → auto-disables Frame Gen if on

---

## Offline Mode

`FrameGenManager::tagFrame()` returns early when `offlineState == 2` (accumulating). Generated frames would corrupt the Welford accumulator.

---

## Build Notes

MCVR-FrameGen worktree CMake requires:
```
-DSHADERMAKE_FXC_PATH="C:/Program Files (x86)/Windows Kits/10/bin/10.0.26100.0/x64/fxc.exe"
```

Full build order: shaders → core.dll → install → JAR (generates JNI headers) → deploy. Use `/build` skill.

---

## Runtime Diagnostics

Check `streamline.log` next to `core.dll` for:
- DLSS-G feature loading status
- `requiredTags` per feature (logged by `queryFeatureRequirements`)
- `slDLSSGGetState` status, maxFramesToGenerate, VRAM estimate
- Per-frame SL errors (logged every 300 frames if failing)

---

## What's NOT Implemented

1. **FSR 3 Frame Generation** — cross-vendor fallback (FidelityFX-SDK extern/ is empty)
2. **Motion blur halving** when FG active (cosmetic)
3. **Reflex 2 / NoWarpMask** — requires Reflex 2 SDK integration
4. **DLSSGFlags::eRetainResourcesWhenOff** — could reduce stutter on toggle
5. **DLSSGFlags::eEnableFullscreenMenuDetection** — auto-disable in menus
6. **DLSS 4.5 Dynamic MFG** — `eAuto` mode is wired but needs spring 2026 plugin
