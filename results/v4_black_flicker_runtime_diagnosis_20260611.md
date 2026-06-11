# V4 Black/Flicker Runtime Diagnosis - 2026-06-11

## What the screenshot means

The HUD and chat render while the world is black, so Minecraft and the overlay
path are alive. The V4 world path is not producing stable submitted world frames.

`render_diag.log` confirms the renderer stayed gated:

```text
submitCommand shouldRender=0 overlayActive=0
```

The visible sky flicker is consistent with swapchain frames occasionally showing
old/partial sky or clear content while strict first-frame V4 readiness never
opens world rendering.

## Immediate root cause: stale deployed artifacts

The 16:20 run did not use the committed CTM residency fix.

Runtime log:

```text
[16:20:44] [Radiance] Build 0.1.3-alpha commit=d6ce57dd1171ac25cead455b6a87bb16928b2c27 jarSha256=2632708D1638070FBF5A0DE3A5B65E2DB165D0EB552CE0CB7C99E3EFBFC4DE71
[16:21:42] [MaterialCompat] Material page upload starting: ... layerSize=128, pageCapacity=512 ...
```

The fixed Radiance commit is `62e03a3`, and the fixed CTM native/shader commit is
`b44f027`/later. A correct runtime should not report `pageCapacity=512` for the
128px CTM residency path. It should use native V4 tier capacity, which is 256
layers for 128px.

Local artifact comparison before redeploy:

- Deployed Prism jar: SHA256 `2632708D1638070FBF5A0DE3A5B65E2DB165D0EB552CE0CB7C99E3EFBFC4DE71`
- Rebuilt jar before native install: SHA256 `8A143711A9CC3CBA79D966413391DC2918353581EFFCA830FB2A97E18509569D`
- Deployed runtime `radiance/core.dll`: SHA256 `4C8B4539A4F132EA1BF16A9FDBD129E549B4350A9DE5C3BC7E906466227C76F3`
- Fixed native build `core.dll`: SHA256 `7B76DC0C2996CD9CC0D06C058C88796E9DDFDE64C77DBC3D9E10F3F0C42F5DA7`

The earlier Java jar build had packaged the old native DLL because native install
was skipped while Minecraft was running.

## Readiness failure

Material status froze with visible CTM fallbacks still present:

```json
{
  "status": "residencyStarted",
  "visibleResidentMaterialCount": 255,
  "visibleFallbackMaterialCount": 28,
  "failedUniqueMaterialCount": 2,
  "retryableFailedVisibleMaterialCount": 2,
  "uploadInFlight": true,
  "event": {
    "layerSize": 128,
    "pageCapacity": 512,
    "ctmResidentCapacity": 61440
  }
}
```

That keeps strict V4 first-frame readiness false, so `shouldRender` remains 0.

## Separate native crash: BLAS device lost

The same run then hit a native GPU fault:

```text
Device state: Error_DMA_PageFault
Address[0]: 0x0 type=2 (WRITE_AFTER_DESTROY)
BLAS build submit DEVICE_LOST
```

`blas_diag.log` shows the transition:

```text
COMPLETE iter=9617 tv=174 type=build chunks=1 inFlight=1 gpuVal=18446744073709551615
BUILD_SUBMIT iter=9647 tv=175 vk=-4 chunks=1 inFlight=0
DEVICE_LOST_BUILD
```

`18446744073709551615` is `UINT64_MAX`. The BLAS loop was treating a failed
timeline semaphore query as "all work complete", then releasing resources and
submitting another build. This was hardened so failed timeline queries are
recorded and stop the BLAS path instead of masquerading as completion.

This hardening improves crash fidelity. It does not prove the original
WRITE_AFTER_DESTROY source is fully fixed; if the next correctly deployed run
still reaches device lost, the next target is BLAS/TLAS resource lifetime around
chunk integration and cross-queue GC.

## Deployment performed after diagnosis

After Minecraft exited:

- Installed native artifacts into `C:\RadSER\Radiance`.
- Rebuilt the Radiance jar.
- Deployed the rebuilt jar to the Prism instance.

The final deployed jar after this diagnosis is SHA256
`20383660AFA8777A3CC67E378DFAEC6C92362A9A542BCB2666928C1C09AB820A` and contains:

- `core.dll` SHA256 `474C65D558DD4C6AA6B3B5FAE37395612BB3112D4FE65BA2F59389CBB2132500`
- `shaders/util/sprite_fetch.glsl` with CTM mapping `64u + page`
