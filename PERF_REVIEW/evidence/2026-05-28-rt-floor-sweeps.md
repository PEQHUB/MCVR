# 2026-05-28 RT MainTrace And Floor Sweeps

Two scene pairs were captured with the normal RT sweep and the RT floor sweep.

## Reports

- Open/plains scene normal sweep: `C:\RadSER\results\rt_sweeps\20260528_150603_rt_main_trace_sweep.json`
- Open/plains scene floor sweep: `C:\RadSER\results\rt_sweeps\20260528_150615_rt_main_trace_floor_sweep.json`
- Cave-facing/occluded scene normal sweep: `C:\RadSER\results\rt_sweeps\20260528_151214_rt_main_trace_sweep.json`
- Cave-facing/occluded scene floor sweep: `C:\RadSER\results\rt_sweeps\20260528_151237_rt_main_trace_floor_sweep.json`

## Validation

- Both new scene captures report `chunkStatus.settled=true`.
- The floor sweep restored `rtDebugFlags=0`.
- The floor sweep restored the original options:
  - `rayBounces=12`
  - `simplifiedIndirect=false`
  - `multiScatterGGX=true`
  - `eonDiffuse=true`
  - `serEnabled=false`
  - `serHintsEnabled=true`
- Graphics Capture remains avoided; these findings come from DebugBridge/GPU timestamp sweeps.

## Open/Plains Scene

- Normal baseline `RT.MainTrace`: `7.542 ms`
- 12 to 8 bounces: `0.163 ms` saved
- 12 to 6 bounces: `0.632 ms` saved
- 12 to 4 bounces: `1.692 ms` saved
- `simplifiedIndirect`: `0.184 ms` saved
- legacy secondary BRDF: `0.186 ms` saved
- SER: `1.773 ms` worse

Floor sweep:

- Baseline floor: `6.860 ms`
- Disable reflection probe: `0.070 ms` saved
- Disable secondary cloud shadow only: `0.125 ms` saved
- Disable hand pass: `2.526 ms` saved
- Disable secondary sun/direct block: `3.008 ms` saved

## Cave-Facing/Occluded Scene

User clarification: the sky was above and contributing significantly to lighting, but was not visible on screen. Do not describe this as a no-sky scene.

- Normal baseline `RT.MainTrace`: `16.302 ms`
- 12 to 8 bounces: `2.445 ms` saved
- 12 to 6 bounces: `4.753 ms` saved
- 12 to 4 bounces: `7.859 ms` saved
- `simplifiedIndirect`: `0.660 ms` worse
- legacy secondary BRDF: `0.665 ms` worse
- SER: `3.525 ms` worse

Floor sweep:

- Baseline floor: `17.040 ms`
- Disable reflection probe: `0.151 ms` saved
- Disable secondary cloud shadow only: `0.419 ms` saved
- Disable hand pass: `7.291 ms` saved
- Disable secondary sun/direct block: `9.380 ms` saved
- 4-bounce floor: `8.533 ms`
- 4-bounce plus no hand pass: `4.457 ms`
- 4-bounce plus no secondary sun/cloud: `4.705 ms`

## Conclusions To Carry Forward

- Reflection probe is not the current priority. It saved `0.070-0.151 ms` in these scenes.
- SER should remain off. It was `1.773-3.525 ms` worse.
- Simplified indirect and legacy BRDF are not robust wins; both regressed in the heavier scene.
- Bounce count is a real quality/performance lever in occluded, sky-lit views. It should not be the first default reduction, but it deserves a later preset/termination phase.
- Secondary direct lighting/shadowing after early bounces is a major bucket. Because the sky was contributing real light, this is not dead work; it needs a quality setting or importance heuristic, not a blind deletion.
- Hand pass is unexpectedly expensive and is now the active investigation. The feature-truth dumps did not show `entityFlag_hand_count`, yet disabling the hand pass saved `2.526-7.291 ms`, suggesting the shader pays a full-screen hand query even when no HAND instances are present.

## Next Active Lead

Start with the safest hand fix: skip the shader hand pass when WorldPrepare reports zero HAND TLAS instances. More aggressive hand scissoring or hand-path bounce simplification should only follow after this no-hand-instance case is measured.
