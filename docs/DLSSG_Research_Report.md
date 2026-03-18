# DLSS Frame Generation (DLSSG) — Vulkan Integration Research Report

## Executive Summary

DLSS Frame Generation (DLSSG) generates interpolated frames between real rendered frames using NVIDIA's AI model and dedicated Optical Flow Accelerator (OFA) hardware. It operates at the swapchain/present level — completely independent of the render pipeline. **Streamline is the mandatory integration path** (NGX alone cannot do frame generation). MCVR already has Streamline integrated with Reflex + PCL, making this the natural next step.

---

## 1. Streamline vs NGX: Streamline is Mandatory

- **No standalone DLSS-G SDK exists.** NVIDIA distributes `sl.dlss_g.dll` only as a Streamline plugin.
- Frame generation works by intercepting `vkQueuePresentKHR` and `vkAcquireNextImageKHR` — this mechanism lives in Streamline's interposer, not NGX.
- Internally, Streamline's `sl.common` plugin manages the NGX lifecycle (`NVSDK_NGX_Feature_FrameGeneration`, feature ID 11), but this is an implementation detail.
- MCVR already loads `sl.interposer.dll` before Volk init and routes all Vulkan calls through it. Adding DLSS-G means loading one more feature.

## 2. Architecture

### Swapchain Interposition

When DLSS-G is active, Streamline intercepts these Vulkan functions:
- `vkQueuePresentKHR` — intercepts present, runs AI model, presents real + generated frames
- `vkAcquireNextImageKHR` — manages backbuffer acquisition
- `vkCreateSwapchainKHR` / `vkDestroySwapchainKHR` — manages proxy swapchain
- `vkGetSwapchainImagesKHR` — returns proxy images
- `vkDeviceWaitIdle` — synchronization
- `vkCreateWin32SurfaceKHR` / `vkDestroySurfaceKHR` — surface management

### Per-Frame Flow
1. App calls `vkQueuePresentKHR` with the real frame
2. Streamline intercepts, reads tagged depth/MVs/HUD-less buffers
3. AI model generates 1-3 interpolated frames (OFA + neural network)
4. Streamline presents all frames (real + generated) via multiple actual presents
5. Frame pacer schedules presents for smooth timing

### Queue Parallelism (Vulkan-specific)
- `eBlockPresentingClientQueue` (default): Client queue blocked until DLSS-G completes
- `eBlockNoClientQueues`: Enables queue-level parallelism. App must wait on `DLSSGState::inputsProcessingCompletionFence` before reusing tagged resources

## 3. Resource Requirements

### Mandatory
| Buffer | Resolution | Format | Notes |
|--------|-----------|--------|-------|
| Depth | Render res | Any depth format | Must match MV source |
| Motion Vectors | Render res | R16G16_FLOAT typical | Dense field, camera+dynamic, [-1,1] or scaled via mvecScale |
| Backbuffer | Display res | (implicit via swapchain) | Automatically captured |

### Critical for Quality
| Buffer | Resolution | Notes |
|--------|-----------|-------|
| HUD-less Color | Display res | Scene without UI, after tone mapping, before UI composite |
| UI Color+Alpha | Display res | Alpha=0 = no UI; must exactly match backbuffer size |

### HDR Constraint
**DLSS-G does NOT support FP16/scRGB.** Must use UINT10/RGB10 with HDR10/BT.2100. MCVR's swapchain already uses `VK_FORMAT_A2B10G10R10_UNORM_PACK32` for HDR10 — this is compatible.

### VRAM Cost
| Resolution | VRAM (MB) |
|-----------|----------|
| 1080p | 272 |
| 1440p | 489 |
| 4K | 725 |

## 4. Integration Steps for MCVR

### 4.1 Initialization (StreamlineContext)

```cpp
// Add to features list in StreamlineContext::init()
sl::Feature features[] = {sl::kFeatureReflex, sl::kFeaturePCL, sl::kFeatureDLSS_G};
pref.featuresToLoad = features;
pref.numFeaturesToLoad = 3;

// After device creation, query requirements
sl::FeatureRequirements reqs{};
slGetFeatureRequirements(sl::kFeatureDLSS_G, reqs);
// May need additional graphics queues, extensions

// Load function pointers
slGetFeatureFunction(sl::kFeatureDLSS_G, "slDLSSGSetOptions", (void*&)pfnDLSSGSetOptions);
slGetFeatureFunction(sl::kFeatureDLSS_G, "slDLSSGGetState", (void*&)pfnDLSSGGetState);
```

### 4.2 Common Constants (Every Frame)

```cpp
sl::Constants consts{};
consts.cameraViewToClip = projection;    // Row-major, NO jitter
consts.clipToCameraView = invProjection;
consts.clipToPrevClip = prevReprojection;
consts.prevClipToClip = invPrevReprojection;
consts.mvecScale = {1.0f, 1.0f};         // If MVs already in [-1,1]
consts.jitterOffset = {jitterX, jitterY};
consts.cameraPos = camPos;
consts.cameraNear = nearPlane;
consts.cameraFar = farPlane;
consts.depthInverted = true;              // Reversed-Z
consts.cameraMotionIncluded = true;       // MVs include camera motion
consts.reset = isSceneChange;             // Teleport, etc.
slSetConstants(consts, *currentFrameToken_, viewport);
```

### 4.3 Resource Tagging (After Render, Before Present)

```cpp
// Tag depth + MVs at render resolution
sl::ResourceTag tags[] = {
    {&depthResource, sl::kBufferTypeDepth, sl::ResourceLifecycle::eValidUntilPresent, &renderExtent},
    {&mvecResource, sl::kBufferTypeMvec, sl::ResourceLifecycle::eValidUntilPresent, &renderExtent},
};
slSetTagForFrame(*currentFrameToken_, viewport, tags, 2, cmdBuffer);

// Tag HUD-less at display resolution (after tone mapping, before UI)
sl::ResourceTag hudlessTag = {&hudlessResource, sl::kBufferTypeHUDLessColor,
                               sl::ResourceLifecycle::eValidUntilPresent, &displayExtent};
slSetTagForFrame(*currentFrameToken_, viewport, &hudlessTag, 1, cmdBuffer);

// Tag UI (if available)
sl::ResourceTag uiTag = {&uiResource, sl::kBufferTypeUIColorAndAlpha,
                          sl::ResourceLifecycle::eValidUntilPresent, &displayExtent};
slSetTagForFrame(*currentFrameToken_, viewport, &uiTag, 1, cmdBuffer);
```

### 4.4 Enable/Disable

```cpp
sl::DLSSGOptions options{};
options.mode = sl::DLSSGMode::eOn;
options.numFramesToGenerate = 1;  // 1=2x, 2=3x, 3=4x (MFG, RTX 50 only)
slDLSSGSetOptions(viewport, options);
```

### 4.5 Swapchain Recreation on Toggle

```cpp
// When turning DLSS-G OFF:
vkDeviceWaitIdle(device);
// Release all swapchain-dependent resources
slSetFeatureLoaded(sl::kFeatureDLSS_G, false);
// Recreate swapchain without interposer management

// When turning DLSS-G ON:
slSetFeatureLoaded(sl::kFeatureDLSS_G, true);
// Recreate swapchain (interposer will manage it)
```

## 5. Reflex Requirement

**Reflex is mandatory for DLSS-G.** MCVR already has Reflex integrated. Key requirements:
- `slReflexSleep()` and `slPCLSetMarker()` must be called every frame (already done)
- Frame indices in PCL markers must match `slSetConstants` frame tokens (already done)
- Mismatched indices cause magenta tint or stutter
- `DLSSGStatus::eFailReflexNotDetectedAtRuntime` fires if Reflex is broken

## 6. DLSS-G Alongside DLSS-RR

They compose perfectly:
```
RT (render res) → DLSS-RR (denoise+upscale to display res) → Post-processing → DLSS-G (frame gen at present)
```
- DLSS-RR operates at render submission time (denoising + upscaling)
- DLSS-G operates at present time (frame interpolation)
- Both share the same Streamline infrastructure (constants, tokens, Reflex markers)

## 7. DLSS 4 Multi Frame Generation (RTX 50 Only)

- `numFramesToGenerate=1` → 2x (1 generated + 1 real)
- `numFramesToGenerate=2` → 3x
- `numFramesToGenerate=3` → 4x
- Check `DLSSGState::numFramesToGenerateMax` for hardware capability
- 40% faster, 30% less VRAM than DLSS 3 single-frame model
- Each additional generated frame costs ~1ms on RTX 5090

### DLSS 4.5 Dynamic MFG (Spring 2026)
- Up to 6x mode (5 generated frames per real frame)
- `DLSSGMode::eAuto` — automatically adjusts multiplier to match display refresh rate
- Hardware flip metering on Blackwell for sub-ms timing precision
- Plugin releasing spring 2026

## 8. Reference Implementations

### Indiana Jones: The Great Circle (id Tech 7)
- Vulkan-only engine
- Shipped with DLSS 3.5 (SR + RR + FG), upgraded to DLSS 4 MFG
- Known launch issues: DLSS-G not activating with HDR enabled; Reflex weaker than D3D12

### Doom: The Dark Ages (id Tech 8)
- Vulkan-only engine
- Launched with DLSS 4 MFG + DLSS-RR + path tracing
- 6.8x performance multiplier at 4K with MFG 4x + SR + RR on RTX 5090

## 9. Key Engineering Challenges for MCVR

1. **HUD-less Buffer**: Need a copy of the tone-mapped output BEFORE Minecraft UI is composited. This maps to the output of `ToneMappingModule` / after `HdrCompositePass` but before Java-side HUD rendering.

2. **UI Buffer**: Need the Minecraft UI rendered to a separate RGBA texture with alpha transparency. Currently the UI is composited in `fuseFinal()` — would need to capture the overlay separately.

3. **Swapchain Coordination**: Swapchain teardown/recreation when toggling DLSS-G must coordinate across the JNI boundary since Minecraft's window is Java-managed.

4. **Motion Blur Halving**: When DLSS-G is active, application-side motion blur should be halved (generated frames already contain interpolated motion).

5. **Offline Mode Interaction**: DLSS-G should be disabled during offline accumulation (it would generate incorrect interpolated frames from converging data).

## 10. Hardware Requirements

| Feature | Minimum GPU |
|---------|-------------|
| DLSS-G (1x FG) | RTX 40 Series (Ada) |
| DLSS MFG (2x-4x) | RTX 50 Series (Blackwell) |
| Dynamic MFG (6x) | RTX 50 Series (Blackwell) |
| Hardware Flip Metering | RTX 50 Series (Blackwell) |

System: Windows 10 20H1+, Hardware-Accelerated GPU Scheduling enabled, driver 527.64+.
