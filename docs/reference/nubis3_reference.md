# Nubis³ Reference — Code Snippets & Algorithms

Extracted from: **Nubis, Cubed: Methods (and madness) to model and render immersive real-time voxel-based clouds**
Andrew Schneider, Atmospherics Lead, Guerrilla — SIGGRAPH 2023 Advances in Real-Time Rendering

Also incorporates algorithms from **Nubis Evolved** (SIGGRAPH 2022) where Nubis³ references them.

---

## Three Cloud Methods in Nubis

Nubis³ presents three progressive methods (p187/204):

| Feature | Vertical Profile (2015) | Envelope (2022) | Voxel (2023) |
|---------|------------------------|-----------------|--------------|
| Evolution | Yes | Pseudomotion Only | Pseudomotion Only |
| Time of Day | Yes | Yes | Yes |
| Lightning | Yes | No | Yes |
| Terrain Shadows | No | Yes | Yes |
| High Frame-rates | Yes | Yes | Yes |
| **Flight-Capable** | **No** | **Yes** | **Yes** |
| Freeform Modeling | No | No | Yes |
| Memory/Cloudscape | 0.541 Mb | 9.437 Mb | 25.166 Mb |

**For RadSER**: Use **Envelope Method** as base (flight-capable, procedural, no offline data needed),
with Nubis³ improvements to noise, density, and lighting shared across all methods.

The Voxel Method requires offline Houdini fluid simulation → voxel grids (512x512x64 NVDF).
Not feasible for procedural Minecraft clouds.

---

## 1. Nubis Data Fields (NDF)

### Vertical Profile Method NDF (p19)
Per-texel fields in the weather/coverage map:
- **Min Height** — cloud bottom altitude
- **Max Height** — cloud top altitude
- **Coverage** — cloud coverage [0,1]
- **Bottom Type** — cloud type at bottom (wispy→billowy)
- **Top Type** — cloud type at top

### Envelope Method NDF (p20, p85)
Per-texel fields:
- **Dimensional Profile** — combined height×coverage envelope
- **Detail Type** — wispy/billowy blend control
- **Density Scale** — per-cell density multiplier

---

## 2. Dimensional Profile

### Vertical Profile Method (p19, p29)
```hlsl
float dimensional_profile = vertical_profile * cloud_coverage;
```
Where `vertical_profile` is sampled from a height gradient texture indexed by cloud type.

### Envelope Method (p20, p97 from Evolved)
```hlsl
float height_fraction = Remap(height, min_height, max_height, 0.0, 1.0);
float bottom_gradient = pow(height_fraction, 2.0);
float top_gradient    = pow(1.0 - height_fraction, 1.5);
float edge_gradient   = Remap(sample_height, 0.0, 35.0, 1.0, 0.0);

float dimensional_profile = bottom_gradient * top_gradient * edge_gradient;
```
- `bottom_gradient`: quadratic ramp from cloud base
- `top_gradient`: ^1.5 falloff from cloud top (creates rounded dome)
- `edge_gradient`: horizontal edge fade at cloud boundaries (35m fade zone)

---

## 3. Noise

### 3D Noise Texture (p22, p93-94)
- **Format**: 4 channels, 128×128×128 voxels, 2 bytes/texel (4.194 MB)
- **Nubis (2015)**: Perlin-Worley (R), Worley low (G), Worley med (B), Worley high (A)
- **Nubis³ (2023)**: Curly-Alligator low (R), Curly-Alligator high (G), Alligator low (B), Alligator high (A)
  - Wispy = Curly-Alligator composites (layered wisps)
  - Billowy = Alligator composites (puffy billows)

### Noise Sampling with Distance-Based LOD (p96)
```hlsl
float mipmap_level = log2(1.0 + abs(inRaymarchInfo.mDistance * cVoxelFineDetailMipMapDistanceScale));
float4 noise = Cloud3DNoiseTextureC.SampleLOD(Cloud3DNoiseSamplerC, inSamplePosition * 0.01, mipmap_level);
```

### Wind Offset (p26, p95)
```hlsl
// 2015:
float3 noise_sample_pos = sample_pos - wind_direction * scroll_offset;

// Nubis³:
inSamplePosition -= float3(cCloudWindOffset.x, cCloudWindOffset.y, 0.0) * voxel_cloud_animation_speed;
```

---

## 4. Noise Compositing

### Wispy Noise (p99)
```hlsl
// Blend low-freq and high-freq wisps using dimensional_profile as interpolant.
// Low density (thin edges) → more low-freq wisps, high density (core) → more high-freq wisps.
float wispy_noise = lerp(noise.r, noise.g, inDimensionalProfile);
```

### Billowy Noise (p104)
```hlsl
// Billowy type gradient: dimensional_profile^0.25 — spreads billows across most of the profile.
float billowy_type_gradient = pow(inDimensionalProfile, 0.25);
float billowy_noise = lerp(noise.b * 0.3, noise.a * 0.3, billowy_type_gradient);
```

### Noise Composite (p104, p108)
```hlsl
// Final composite: detail_type blends between wispy and billowy character.
// detail_type=0 → all wispy (thin, curly edges), detail_type=1 → all billowy (puffy, round)
float noise_composite = lerp(wispy_noise, billowy_noise, detail_type);
```

### Height-Based Type Blend (Evolved p99)
```hlsl
// Alternative: blend noise character based on height within cloud + cloud_type parameter.
float noise_height_blend = Remap(height_fraction, cloud_type + 0.1, cloud_type - 0.1);
float composite = lerp(wispy_noise, billowy_noise, noise_height_blend);
```

### High-High Frequency Detail (p113)
```hlsl
// Extra-fine detail for close-up viewing. Applied on top of base noise composite.
float hhf_wisps   = 1.0 - pow(abs(abs(noise.g * 2.0 - 1.0) * 2.0 - 1.0), 4.0);
float hhf_billows = pow(abs(abs(noise.a * 2.0 - 1.0) * 2.0 - 1.0), 2.0);
```

---

## 5. Cloud Density

### Core Density Remap (p24-25) — ALL methods use this
```hlsl
// THE fundamental Nubis density equation.
// dimensional_profile acts as threshold: higher profile → more noise passes through.
// Result: cloud density emerges where noise > (1 - dimensional_profile).
cloud_density = saturate(noise - (1.0 - dimensional_profile));
```

### ValueErosion + Post-Processing (p118) — Nubis³ voxel density
```hlsl
// Composite noises and use as value erosion against the dimensional profile.
float uprezzed_density = ValueErosion(inDimensionalProfile, noise_composite);

// Apply per-cell density scale from modeling data.
uprezzed_density *= inDensityScale;

// Density-adaptive sharpening: low density → sharper (pow 0.3), high density → softer (pow 0.6).
// Also reduces undersampling noise near camera by lowering density.
uprezzed_density = pow(uprezzed_density, lerp(0.3, 0.6, max(EPSILON, pow(saturate(inDensityScale), 4.0))));

return uprezzed_density;
```

### Envelope Method Density (Evolved p105)
```hlsl
// Height fraction multiplier after the remap creates dome shaping.
float cloud_density_sample = height_fraction * pow(saturate(noise_composite - (1.0 - dimensional_profile)), 0.27);

// Inverse edge signal powered by 3 fades cloud edges in several places.
float inv_edge_signal_pow_3 = pow(inv_edge_signal, 3.0);

float cloud_density = cloud_density_sample;

// Coarse density for lighting (cheaper, no fine detail).
float cloud_coarse_density = pow(ValueErosion(dimensional_profile, 0.04), 0.5) * inv_edge_signal_pow_3 * 5.0;
```

### 2.5D Coverage-Based Density (Evolved p65)
```hlsl
// Three coverage remaps based on cloud shape character:
// cr_streaky, cr_wispy, cr_round are coverage-weighted noise composites.
float density = ValueRemap(cloud_type, 0.5, 1.0,
                    ValueRemap(cloud_type, 0.0, 0.5, cr_streaky, cr_wispy),
                    cr_round);

// Coverage shapes density via power function.
density = pow(density, 1.0 - ValueRemap(cloud_coverage, 0.0, 1.0, -0.9, 0.9));

// Coverage threshold: only visible where coverage^3 > 0.
density *= ValueRemap(pow(cloud_coverage, 3.0), 0.0, 0.5, 0.0, 1.0);
```

---

## 6. Lighting

### Light Energy Model (p28, p65, p121)
```
Light Energy = Direct Scattering + Ambient Scattering + Secondary Scattering
```

### Direct Scattering (p29)
```
Direct Scattering = (Transmittance × Primary Phase) + (Multiple Scattering × Secondary Phase)
```
- **Transmittance**: Beer-Lambert law `T = exp(-optical_depth)`
- **Phase Function**: Henyey-Greenstein (dual-lobe: forward + back scatter)

### Multi-Scatter Approximation (p29, p136)
```hlsl
// Nubis multi-scatter approximation (replaces octave-based Wrenninge approach).
ms_volume = dimensional_profile;
cloud_distance = GetVoxelCloudDistance(inSamplePosition);

// Multi-scatter volume is modulated by sun-facing exposure and cloud distance.
ms_volume *= exp(-inSunLightSummedDensitySamples *
    Remap(sun_dot, 0.0, 0.9, 0.25,
        ValueRemap(cloud_distance, -128.0, 0.0, 0.05, 0.25)));
```

### Multi-Scatter with Decay (Evolved p54)
```hlsl
ms_volume = Remap(dimensional_profile * step_size, 0.1, 1.0, 0.0, 1.0)
          * pow(cloud_coverage * cloud_type, 0.25);
ms_volume *= pow(attenuated_light, cMultipleScatteringDepthPower);
ms_volume *= pow(height_fraction, cMultipleScatteringHeightPower);
```

### Ambient Scattering (p32, p141, p144)
```hlsl
// Basic: inverse of dimensional profile, square-rooted.
// Bright at cloud top (profile→0), dark at cloud core (profile→1).
float ambient_scattering = pow(1.0 - dimensional_profile, 0.5);

// Enhanced with summed ambient density (voxel-based precomputed per 8 frames):
float ambient_scattering = pow(1.0 - dimensional_profile, 0.5) * exp(-summed_ambient_density);
```

### Ambient with Height (Evolved p113)
```hlsl
// Height fraction reduces ambient influence at cloud bottoms.
float height_fraction = ValueRemap(inSamplePosition.z, min_height, max_height, 0.0, 1.0);
float ambient_scattering = pow(1.0 - saturate(cloud_coarse_density), 0.25) * height_fraction;
```

### Lightning/Glow Energy (p147)
```hlsl
// Point light glow inside clouds (for lightning).
potential_energy  = pow(1.0 - (d1 / radius), 12.0);
pseudo_attenuation = (1.0 - saturate(density * 5.0));
glow_energy = potential_energy * pseudo_attenuation;
```

---

## 7. Ray Marching

### Ray-March Procedure (p13, p27, p39, p48)
```
For each Pixel:
  Start a march along a ray:
  For each step along the ray:
    Sample Density
    Sample Light Energy
    Integrate into Pixel data
    Determine step size and take the next step
```

### Distance-Based Step Size (p40)
```hlsl
float near_step_size = 3.0;
float far_step_size_offset = 60.0;
float step_adjustment_distance = 16384.0;

float step_size = near_step_size + ((far_step_size_offset * distance_from_camera) / step_adjustment_distance);
```
- Near camera: 3m steps
- At 16384m: 63m steps
- Linear interpolation between

### SDF-Accelerated Stepping (p163) — Nubis³
```hlsl
// SDF gives signed distance to nearest cloud surface. Positive = outside, negative = inside.
sdf_cloud_distance = GetVoxelCloudDistance(sample_pos);

// Adaptive: scales with sqrt(distance) for graceful quality degradation.
adaptive_step_size = max(1.0, max(sqrt(distance_from_camera), EPSILON) * 0.08);

// Take the larger of SDF skip and adaptive step, plus jitter.
step_size = max(sdf_cloud_distance, adaptive_step_size) + jitter_offset;

// Jitter: animated near camera (< 250m) for temporal accumulation, static hash far away.
jitter_offset = distance_from_camera < 250.0 ? animated_hash : static_hash;

sample_pos = distance_from_camera + view_direction * step_size;
```

### Density Check (p164)
```hlsl
// Only sample density when SDF says we're INSIDE a cloud (negative distance).
if (sdf_cloud_distance < 0.0) {
    voxel_cloud_sample_data = GetVoxelCloudSampleData();
    IntegrateCloudSampleData(voxel_cloud_sample_data, pixel_data);
}
```

---

## 8. Rendering

### Split-Distance Rendering (p171-174)
- < 200 meters from camera: render at 480×270 (quarter res)
- > 200 meters: render at 960×540 (half res)
- Saves ~50% render cost

### Performance (p166-168)
| Optimization | Cloud Cost (ms) | Geometry Cost (ms) |
|---|---|---|
| Base | 10-12 | 4-7 |
| + Voxel-Based Lighting | 6.1-8.2 | same |
| + Adaptive & SDF Ray-March | 2.2-4.0 | same |

### Density Sampler Cost (p119)
- Current: 10ms @ 960×540
- Dense grid sampling: 30+ms
- "0.5 meter precision, balanced between memory and instructions"

---

## 9. Utility Functions

### ValueErosion (used throughout)
```hlsl
// Erodes a profile using a noise value. Equivalent to:
// saturate(profile - noise) / max(1.0 - noise, EPSILON)
// When noise is high, more of the profile is eroded away.
float ValueErosion(float profile, float noise) {
    return saturate(remap(profile, noise, 1.0, 0.0, 1.0));
}
```

### ValueRemap / Remap (used throughout)
```hlsl
float Remap(float value, float low1, float high1, float low2, float high2) {
    return low2 + (value - low1) / (high1 - low1) * (high2 - low2);
}

// Clamped version:
float ValueRemap(float value, float low1, float high1, float low2, float high2) {
    return low2 + saturate((value - low1) / (high1 - low1)) * (high2 - low2);
}
```

---

## 10. Key Design Principles

1. **Dimensional profile is everything** — it encodes both height shape and coverage into a single [0,1] value that drives density, noise blending, ambient scattering, and multi-scatter.

2. **One density equation** — `saturate(noise - (1.0 - dimensional_profile))` is the core of all Nubis methods. Everything else is about computing `dimensional_profile` and `noise_composite`.

3. **Density-adaptive sharpening** — `pow(density, lerp(0.3, 0.6, ...))` — thin edges are sharpened more (0.3), dense cores less (0.6). Prevents mushy low-density regions.

4. **Ambient = inverse profile** — `pow(1.0 - dimensional_profile, 0.5)` is simple, physical, and avoids expensive secondary ray marches. Bright at tops/edges, dark in cores.

5. **No user-facing fade distance** — atmospheric perspective is handled by the atmosphere system, not the cloud renderer. March distance is controlled by geometry (slab/SDF), not artistic fade.

6. **Distance-based quality** — step size, noise LOD, and render resolution all degrade with distance. Near-camera quality is maximized; far clouds use less budget.

7. **SDF for empty space skipping** — negative SDF = inside cloud (sample density), positive SDF = outside (skip). Dramatically reduces wasted samples.
