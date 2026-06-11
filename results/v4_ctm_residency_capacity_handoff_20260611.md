# V4 CTM Residency Capacity Handoff - 2026-06-11

## Live finding

The 14:14 game run no longer reproduced the earlier `nativeBeginTextureLoaderV4`
crash. It reached the world, but rendering stayed gated:

- `render_diag.log` continued presenting frames with `submitCommand shouldRender=0`.
- `radser-material-runtime-status.json` stopped updating at 14:16:02 with
  `status=residencyStarted`, `uploadInFlight=true`, `visibleFallbackMaterialCount=31`,
  and the last event describing 5 candidate 128px CTM materials.
- The Java thread dump showed `RadSER Material Residency Upload` stuck in
  `TextureArrayBridgeV4.nativeUploadTexturePageV4`.
- `latest.log` showed `Material page 8 upload failed: 5 layers`, followed by a
  second `Material page upload starting: 5 candidate materials...` with no
  completion line.

## Root cause

Java CTM residency allocation used a 512-layer page for 128px materials, derived
from the old 128 MiB page budget. Native V4 uses
`TexturePagePool::pageLayerCapacityStatic(128px) == 256`.

Once Java reached page 8, startLayer 256, native descriptor publication could no
longer address the requested range in descriptor page 64. The material table then
kept visible CTM materials retryable, first-frame readiness stayed false, and RT
submission stayed closed.

There was also a semantic mismatch between packed CTM material handles and shader
descriptor lookup. V4 CTM material handles should pack namespace CTM page 0 as
descriptor page 64, page 1 as descriptor page 65, etc.

## Source changes

- Radiance `ResourceMaterialResidencyUploader` now asks native for the V4 tier
  page capacity and uses that for CTM page allocation.
- Radiance `ResourceMaterialRegistry.ResidencyHandle.sameLayer` now packs CTM
  material handles with namespace-local page indices starting at 0.
- MCVR JNI descriptor mapping now normalizes CTM uploads across native page
  boundaries using `startLayer / nativeCapacity`.
- MCVR `TextureLoaderV4::enqueueUpload` mirrors that CTM normalization and
  reports CTM cross-native-page-boundary rejection telemetry.
- MCVR shader `textureDescriptorIndexV4` now maps CTM as `64 + page`.
- MCVR `TextureSystem` logs concrete invalid page/range/input reasons for future
  descriptor publish failures.

## Validation

Passed:

- `cmake --build C:\RadSER\MCVR\build --config Release --target shaders`
- `cmake --build C:\RadSER\MCVR\build --config Release --target core`
- `cmd.exe /c "set TEMP=C:\Users\Administrator\AppData\Local\Temp && set TMP=C:\Users\Administrator\AppData\Local\Temp && cd /d C:\RadSER\Radiance && C:\RadSER\Radiance\gradlew.bat clean build 2>&1"`

Not performed:

- Native install into `C:\RadSER\Radiance`
- Prism mod jar deployment
- Fresh runtime verification

Those were skipped because Minecraft was still running as PID 41376. Do not
overwrite the deployed jar or installed runtime DLLs until that JVM exits.

## Next run acceptance

After installing native artifacts and deploying the rebuilt Radiance jar, the
next run should show:

- No `Material page 8 upload failed` when CTM residency crosses the 256-layer
  boundary.
- `radser-material-runtime-status.json` continues updating after CTM upload.
- `visibleResidentMaterialCount` rises and `visibleFallbackMaterialCount` trends
  down from the current 31.
- `render_diag.log` eventually changes from persistent `shouldRender=0` to RT
  submission once strict first-frame readiness is satisfied.

Chunk loading still needs a separate pass. The live run showed the old knob
change (`maxImportantTasksPerFrame=2`) is present, so the remaining issue is not
that limit. Earlier thread dumps showed the expensive section build path
serializing around `ChunkBuilder.class` while CTM/overlay variant resolution runs.
