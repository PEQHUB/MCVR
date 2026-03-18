# Intel XeSS Frame Generation (XeFG) — Research Report

## Executive Summary

**XeSS Frame Generation (XeSS-FG) does NOT support Vulkan.** The entire stack (XeSS-FG + XeLL latency reduction) is DX12-only. There are no Vulkan headers, no Vulkan libraries, no Vulkan samples, and no public roadmap for Vulkan support. This makes XeSS-FG **incompatible with MCVR's Vulkan renderer** without a fundamental API translation layer.

---

## 1. Overview

XeSS-FG is part of the **XeSS 3** technology suite:
- **XeSS-SR** — Super Resolution (AI upscaling) — **has Vulkan support**
- **XeSS-FG** — Frame Generation (AI frame interpolation) — **DX12 only**
- **XeLL** — Xe Low Latency (frame pacing) — **DX12 only**

SDK: https://github.com/intel/xess (public, Intel license)
Current version: 3.0.0 (March 2025)

## 2. Why XeSS-FG Cannot Be Used in MCVR

| Component | Vulkan Support |
|-----------|---------------|
| XeSS-SR (upscaling) | Yes — `xess_vk.h` exists |
| XeSS-FG (frame gen) | **No** — only `xefg_swapchain_d3d12.h` exists |
| XeLL (latency) | **No** — only `xellD3D12CreateContext()` exists |
| Sample apps | No Vulkan FG sample ships |
| Developer guide | Exclusively documents DX12 |

XeSS-FG's proxy swapchain wraps `IDXGISwapChain4` and DXGI Present calls. There is no `VkSwapchainKHR` proxy, no Vulkan command buffer integration, and no Vulkan function interception.

XeLL (mandatory dependency — XeSS-FG refuses to init without it) is also DX12-only.

## 3. Architecture (DX12 — For Reference)

### Proxy Swapchain Model
1. Application creates or provides a DXGI swapchain
2. XeSS-FG takes ownership (ref count must be exactly 1)
3. Returns a proxy `IDXGISwapChain4` pointer
4. On each `Present()`, XeSS-FG runs compute shader AI passes, then issues multiple actual DXGI presents

### AI Model
- Runs entirely as **compute shaders** (no dedicated hardware like NVIDIA's OFA)
- This is what enables cross-vendor support — any GPU with SM 6.4 can run the shaders
- Trade-off: limited to 1 generated frame on non-Intel GPUs

### Multi-Frame Generation (SDK 3.0)
- Up to 4x on Intel Arc GPUs
- Up to 2x on NVIDIA/AMD GPUs (1 generated frame max)
- Configurable via `xefgSwapChainSetNumInterpolatedFrames()`

## 4. Resource Requirements (DX12 — For Reference)

### Mandatory Inputs
| Resource | Notes |
|----------|-------|
| Motion Vectors | Per-pixel screen-space, R16G16_FLOAT |
| Depth | Standard depth format |
| HUD-less Color | Scene without UI (conditional based on composition mode) |

### Frame Constants
```c
typedef struct {
    float viewMatrix[4][4];       // Row-major, no jitter
    float projectionMatrix[4][4]; // Row-major, no jitter
    float jitterOffsetX, jitterOffsetY;
    float motionVectorScaleX, motionVectorScaleY;
    uint32_t resetHistory;        // Teleport/scene cut
    float frameRenderTime;        // Duration in ms
} xefg_swapchain_frame_constant_data_t;
```

### UI Composition Modes (6 total)
- AUTO, NONE, BACKBUFFER+UITEXTURE, HUDLESS+UITEXTURE, BACKBUFFER+HUDLESS, all three combined
- Similar concept to DLSS-G's HUD-less + UI mask approach

## 5. Cross-Vendor Compatibility

| GPU | Max Generated Frames | Multi-FG |
|-----|---------------------|----------|
| Intel Arc (A/B-series) | 3-4 | Yes |
| Intel Core Ultra iGPU | Hardware max | TBD |
| NVIDIA (GTX 1660+, RTX) | 1 only | No |
| AMD (RX 5000+, RDNA 1+) | 1 only | No |

Requirement: Shader Model 6.4 (DP4a) support.

## 6. Comparison with DLSS-G

| Aspect | XeSS-FG | DLSS-G (Streamline) |
|--------|---------|---------------------|
| **Vulkan** | No | Yes |
| **DX12** | Yes | Yes |
| **SDK Model** | Direct C API, link `libxess_fg.lib` | Streamline interposer DLL |
| **Optical Flow** | Compute shaders only | Dedicated OFA hardware |
| **Cross-vendor** | Yes (limited to 1 gen frame) | No (RTX only) |
| **Multi-FG** | Up to 4x (Intel Arc) | Up to 4x (RTX 50), 6x coming |
| **Latency** | XeLL (mandatory, bundled) | Reflex (mandatory, separate) |
| **Open source** | SDK public, DLL closed | SL SDK open, DLSS-G plugin closed |
| **HDR** | R10G10B10A2 only | R10G10B10A2 only |
| **Fullscreen** | Borderless/windowed only | Borderless/windowed only |

## 7. Game Implementations

### Indiana Jones: The Great Circle
- Supports XeSS-FG via its DX12 renderer path
- id Tech engine historically Vulkan, but XeSS-FG integration uses DX12

### Doom: The Dark Ages
- Vulkan-only engine — XeSS-FG status unconfirmed
- If the game has no DX12 path, XeSS-FG would not be applicable

### Other Titles
- F1 24, Dying Light 2, A Quiet Place, Unknown 9: Awakening

## 8. Recommendation for MCVR

**XeSS-FG is not viable for MCVR** in its current form. The entire frame generation + latency reduction stack is DX12-only with no Vulkan API surface.

### Alternatives for Cross-Vendor Frame Generation
1. **FSR 3 Frame Generation** (AMD FidelityFX SDK) — Vulkan supported, open source, cross-vendor
2. Wait for Intel to ship Vulkan XeSS-FG headers (no timeline)
3. Implement a Vulkan→DX12 translation layer (extreme complexity, not recommended)

### If Cross-Vendor Frame Gen is Important
FSR 3 Frame Generation is the only production-ready, cross-vendor, Vulkan-compatible frame generation technology available today. It would complement DLSS-G nicely:
- NVIDIA RTX users → DLSS-G (better quality via OFA hardware)
- AMD/Intel users → FSR 3 FG (compute shader based, works everywhere)

This mirrors the existing DLSS-RR / FSR3-upscaler / NRD fallback pattern already in MCVR's pipeline.
