# 2026-05-29 Direct-Light Boundary Checkpoint

## Latest Probe

Latest stable probe before the ablation gate:

- `C:\RadSER\results\rtxdi\20260529_182127\rt_direct_light_probe.json`
- Backend: `UpstreamReSTIR`
- `directLightPipelineActive=1`
- `directLightPrimaryPipelineReady=1`
- `directLightInitialPipelineReady=1`
- `directLightUtilityPipelineReady=1`
- `directLightLightCount=113`
- Reservoir pixels: `888167`
- Primary surface pixels: `761646`
- Valid reservoirs: `761646`
- Temporal reused: `378878`
- Spatial taps: `1515512`
- Visibility candidates: `478869`

GPU timings from that scene:

- `RT.MainTrace`: `2.044 ms`
- `RT.Primary`: `0.022 ms`
- `RT.DirectLight.Initial`: `0.027 ms`
- `RT.DirectLight.Temporal`: `0.014 ms`
- `RT.DirectLight.Spatial`: `0.016 ms`
- `RT.DirectLight.Visibility`: `0.016 ms`
- `RT.DirectLight.Shade`: `0.018 ms`
- Direct-light compute scaffold total: about `0.113 ms`
- `RayTracing`: `4.308 ms`
- Total GPU: `5.594 ms`

## Interpretation

The direct-light pass boundary is stable and cheap, but it is still an observe/mirror path. It does not yet remove direct-light work from `RT.MainTrace`.

The next useful test is a debug-only ablation that disables sampled direct lighting inside `world_solid_transparent.rchit`. This intentionally darkens the image and is not a production setting. It measures the upper bound of time that a real external direct-light visibility/shading path could recover.

## New Gate

Added `RT_DEBUG_DISABLE_DIRECT_LIGHTING` as transient `rtDebugFlags` bit `16`.

Added Debug menu button:

- `Run Direct-Light Ablation`

The DebugBridge `rtDirectLightProbe` command now accepts:

```json
{"cmd":"rtDirectLightProbe","ablateDirectLight":true}
```

It restores the original `rtDebugFlags` after the probe.

## Decision Rule

If the ablation reduces `RT.MainTrace` by less than about `0.4 ms`, do not continue splitting direct lighting right now.

If it saves around `0.5 ms` or more, the next production step is not more scaffolding. It is real compute visibility/shading against the direct-light reservoirs, then final compose integration.

## Externalized Direct-Light Implementation

Follow-up ablation pair:

- `20260529_182951`: normal `UpstreamReSTIR`, `RT.MainTrace=2.135 ms`, total GPU `5.713 ms`
- `20260529_182944`: direct-light ablation, `RT.MainTrace=1.174 ms`, total GPU `4.815 ms`

This measured about `0.96 ms` of removable main-trace direct-light work in that scene.

Implemented first active path:

- `UpstreamReSTIR` now sets the transient main-trace direct-light disable flag only when external direct-light resources are ready.
- `direct_light_initial.comp` reconstructs first-hit surface position from the view-space position buffer.
- It evaluates sun/moon direct lighting with compute `rayQuery` shadows against the TLAS.
- It adds a tile-limited area-light estimate using the existing clustered light buffer.
- The utility shade pass writes the final direct-light value back into `firstHitDiffuseDirectLightImage`, so SVGF/compose consume the externalized direct-light result.
- `Legacy` remains the full old path and fallback.

Known limits for the first active path:

- Directional lighting uses a single hard-ish visibility query instead of the full VMF soft sun sample.
- Area lights are tile-limited and simplified compared with the closest-hit ReSTIR path.
- Cloud transmittance and advanced material details are not fully matched yet.
- This should be treated as a performance/visual candidate, not the final quality answer, until same-scene probes compare cost and artifacts.
