# V4 CTM Residency Live Evidence - 2026-06-11

This evidence was captured from the user-started Minecraft process
`javaw.exe` PID 41376, started at 2026-06-11 14:14:11 local time.

## Atlas and bootstrap success

The earlier `nativeBeginTextureLoaderV4` crash did not recur in this run. The
game reached the world after the vanilla atlas and CTM bootstrap completed.

```text
[14:14:57] [Render thread/INFO]: [TextureSystem] Sprite lookup refreshed: 1810 entries, renderable capacity 4096
[14:14:59] [TextureLoaderV4] Uploaded vanilla texture tier page=1 size=16 layers=910
[14:14:59] [TextureLoaderV4] Uploaded vanilla texture tier page=2 size=32 layers=101
[14:14:59] [TextureLoaderV4] Uploaded vanilla texture tier page=3 size=64 layers=44
[14:14:59] [TextureLoaderV4] Uploaded vanilla texture tier page=4 size=128 layers=1089
[14:14:59] [TextureLoaderV4] Uploaded vanilla texture tier page=5 size=256 layers=83
[14:14:59] [TextureLoaderV4] Uploaded vanilla texture tier page=6 size=512 layers=1
[14:14:59] [MaterialCompat] Runtime material bootstrap published 9357 CTM materials from 1190 property files for generation 1
```

## Last successful/failed residency markers

The smoking gun is the 128px CTM page upload using a 512-layer Java page
capacity, then failing on material page 8.

```text
[14:16:00] [RadSER Material Residency Upload/INFO]: [MaterialCompat] Material page upload starting: 1 candidate materials, layerSize=128, pageCapacity=512, pageBudget=120, decodeThreads=4, generation=1, visibleOnly=true
[14:16:00] [RadSER Material Residency Upload/INFO]: [MaterialCompat] Runtime material residency complete for generation 1: 1 materials resident, semanticUploaded=true, tableUploaded=false, nativePageFailures=0, totalMs=21.08
[14:16:02] [RadSER Material Residency Upload/INFO]: [MaterialCompat] Material page upload starting: 5 candidate materials, layerSize=128, pageCapacity=512, pageBudget=120, decodeThreads=4, generation=1, visibleOnly=true
[14:16:02] [RadSER Material Residency Upload/WARN]: [MaterialCompat] Material page 8 upload failed: 5 layers
[14:16:02] [RadSER Material Residency Upload/INFO]: [MaterialCompat] Runtime material residency complete for generation 1: 0 materials resident, semanticUploaded=false, tableUploaded=false, nativePageFailures=1, totalMs=22.35
[14:16:02] [RadSER Material Residency Upload/WARN]: [MaterialCompat] Residency descriptor-only success blocked for generation 1: uploadedMaterials=0, nativePageFailures=1, nativePageFailedMaterials=5, materialTableUploadFailures=0, visibleFallbacks=31
[14:16:02] [RadSER Material Residency Upload/INFO]: [MaterialCompat] Material page upload starting: 5 candidate materials, layerSize=128, pageCapacity=512, pageBudget=120, decodeThreads=4, generation=1, visibleOnly=true
```

No completion line followed the final `Material page upload starting` marker.

## Frozen status JSON

`radser-material-runtime-status.json` stopped updating at 2026-06-11 14:16:02
local time with:

```json
{
  "status": "residencyStarted",
  "generation": 1,
  "gpuResidentMaterialCount": 2064,
  "pendingResidencyMaterialCount": 7786,
  "fallbackMaterialCount": 9103,
  "visibleResidentMaterialCount": 254,
  "visibleFallbackMaterialCount": 31,
  "failedUniqueMaterialCount": 5,
  "retryableFailedVisibleMaterialCount": 5,
  "uploadInFlight": true,
  "pendingQueueSize": 31,
  "event": {
    "candidateMaterialCount": 5,
    "layerSize": 128,
    "pageCapacity": 512,
    "ctmFirstMaterialPage": 8,
    "ctmResidentCapacity": 61440,
    "queuedMaterialCount": 5,
    "visibleOnly": true
  }
}
```

## Thread dump

The residency upload thread was still runnable in the native upload path after
the status file stopped moving.

```text
"RadSER Material Residency Upload" daemon RUNNABLE
  at com.radiance.client.proxy.vulkan.TextureArrayBridgeV4.nativeUploadTexturePageV4(Native Method)
  at com.radiance.client.texture.material.ResourceMaterialResidencyUploader.uploadFromCompatReport(ResourceMaterialResidencyUploader.java:310)
  at com.radiance.client.texture.compat.ResourcePackRuntimeMaterialBootstrap.lambda$startResidencyUpload$7(ResourcePackRuntimeMaterialBootstrap.java:594)
```

## Render gate and chunk state

`render_diag.log` continued to present with strict RT rendering closed:

```text
14:34:42.882 submitCommand shouldRender=0 overlayActive=0
14:34:42.883 submitCommand shouldRender=0 overlayActive=0
14:34:42.885 submitCommand shouldRender=0 overlayActive=0
14:34:42.889 submitCommand shouldRender=0 overlayActive=0
```

DebugBridge status while monitoring:

```json
{"inWorld":true,"inGame":false,"screen":"class_481","x":712.1,"y":105.0,"z":248.9,"yaw":-94.1,"pitch":16.8,"measuring":false}
```

DebugBridge chunk status:

```json
{
  "completedChunks": 906,
  "totalChunks": 1944.0,
  "settled": true,
  "javaRebuildQueueSize": 96,
  "pendingImportantTasks": 0,
  "nativeInputQueueSize": 0,
  "nativeReadyChunkCount": 0,
  "tlasReadyChunkCount": 0,
  "importantWaitAverageMs": 4.617447,
  "importantWaitMaxMs": 5.9107
}
```

## Interpretation

The monitored runtime is still using the old deployed artifacts. The committed
fix changes Java CTM allocation to native tier capacity and aligns CTM material
handle/descriptor mapping, but it still needs native install, jar deploy, and a
fresh run after PID 41376 exits.
