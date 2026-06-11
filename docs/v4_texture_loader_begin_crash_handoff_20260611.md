# V4 Texture Loader Begin Crash Handoff - 2026-06-11

## Incident

User run crashed during block-atlas resource reload immediately after:

```text
[13:56:51] [Render thread/INFO]: [TextureSystem] Sprite lookup refreshed: 1810 entries, renderable capacity 4096
EXCEPTION_UNCAUGHT_CXX_EXCEPTION (0xe06d7363)
Java frame: com.radiance.client.proxy.vulkan.TextureArrayBridgeV4.nativeBeginTextureLoaderV4(JJI)Z
```

Crash report:

```text
C:\Users\Administrator\AppData\Roaming\PrismLauncher\instances\1.21.4\minecraft\hs_err_pid38396.log
```

PDB symbolization of `core.dll+0x136a09` against the deployed `core.pdb` resolved to:

```text
TextureLoaderV4::beginGeneration(unsigned __int64)
C:\RadSER\MCVR\src\core\render\texture_loader_v4.cpp:105
```

At that revision, line 105 was the `std::lock_guard<std::mutex>` in `TextureLoaderV4::beginGeneration`.

## Root Cause

The V4 texture architecture had already moved shader-visible material pages to `TextureSystem` descriptor pages, but `TextureLoaderV4` still behaved like the old sampled `TexturePagePool` owner at transaction begin:

- `TextureLoaderV4::initialize` eagerly initialized `GpuUploadService`.
- `TextureLoaderV4::initialize` eagerly initialized `TexturePagePool`.
- `TextureLoaderV4::beginGeneration` took the loader mutex and reset the page pool.
- JNI lifecycle functions allowed C++ exceptions to cross into HotSpot.

The crash itself occurred before `TextureLoaderV4` logged `Begin generation`, so generation publication never reached page upload or commit. The concrete throw site was the loader transaction mutex in `beginGeneration`; the broader workflow flaw was keeping throw-prone legacy lifecycle work in the strict V4 first-frame path.

## Fix Applied

MCVR changes:

- `TextureLoaderV4` is now a descriptor-backed generation/status tracker.
- `initialize` no longer initializes `GpuUploadService` or `TexturePagePool`.
- `beginGeneration` uses atomic generation state only and does not lock or reset page-pool state.
- `commitGeneration` uses atomic committed state only.
- `cancelGeneration`, `pump`, `pollCompletions`, and `generationIdle` no longer route through dormant legacy upload/page-pool queues.
- V4 JNI boolean entry points are guarded with `try/catch` and return `JNI_FALSE` after logging native exceptions.
- V4 JNI status entry points are guarded and return `{"error":"native_exception","method":"..."}` instead of crashing diagnostics.

Touched files:

```text
src/core/render/texture_loader_v4.cpp
src/core/render/texture_loader_v4.hpp
src/core/middleware/texture_loader_v4_jni.cpp
```

## Validation Done

Build/install validation:

```text
cmake --build C:\RadSER\MCVR\build --config Release --target core
cmake --install C:\RadSER\MCVR\build --config Release --prefix C:\RadSER\Radiance
```

Both completed successfully before this handoff was written.

Minecraft was not launched by Codex.

## Runtime Confirmation Needed

On the next user run, check:

- No new `hs_err_pid*.log` at `nativeBeginTextureLoaderV4`.
- `latest.log` contains:
  - `[TextureLoaderV4] Initialized descriptor-backed tracker`
  - `[TextureLoaderV4] Begin generation 1`
  - `[TextureLoaderV4] Committed generation 1`
- No `TextureLoaderV4JNI ... caught native exception` lines.
- No `nativeBeginTextureLoaderV4 returned false`.
- `readyMaterialPages > 0`.
- `visibleResidentMaterialCount > 0`.
- Visible fallback count trends toward zero.
- `firstFrameReadiness.ready=true` only after V4 descriptor residency is ready.

## Remaining Risk

This fixes the fatal begin crash and removes the obsolete begin-time upload/page-pool dependency. It does not prove the full gray-world issue is resolved until a real resource reload and world load show descriptor pages, material table, sprite registry, and runtime material residency all agreeing.
