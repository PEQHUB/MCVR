# Nubis Cubed — Authoritative Technical Reference

> Andrew Schneider, Guerrilla Games. SIGGRAPH 2023, "Advances in Real-Time Rendering in Games."
> "Nubis, Cubed: Methods (and madness) to model and render immersive real-time voxel-based clouds."
>
> This document is the single source of truth for our cloud implementation.
> All code should match what is described here. If our code disagrees with this document, the code is wrong.

---

## 1. Architecture Overview

Nubis³ is a **voxel-based** volumetric cloud renderer. The key paradigm shift from Nubis 2015/2017 (procedural noise + weather map) is that cloud shapes are now authored via **fluid simulation** and stored as voxel grids (NVDF format). Procedural 3D noise is used only for **up-rezzing** — adding high-frequency detail to the low-resolution voxel data during the ray march.

### 1.1 Pipeline Stages

```
1. Light Grid Precompute  (compute)  — accumulate density toward sun into 3D voxel grid
2. Cloud Ray March         (compute)  — primary march through voxel volume, sample density + lighting
3. Temporal Reprojection   (compute)  — blend with previous frame (optional fractional rendering)
4. Composite               (graphics) — blend cloud layer over scene
```

### 1.2 Key Design Principles

- **Voxel clouds as volumetric skyboxes**: Support time-of-day from ground level
- **Immersive fly-through**: High frame rates while flying through clouds on mounts
- **SDF acceleration**: Compressed signed distance fields skip empty space during march
- **Memory-efficient up-rezzing**: Procedural noise adds detail without storing high-res voxel grids

---

## 2. Coordinate Spaces

### 2.1 Voxel Bounding Box (World Space)

Unlike Nubis 2015 which used a spherical shell, Nubis³ operates in a **flat axis-aligned bounding box** in world space:

```glsl
#define VOXEL_BOUND_MIN  vec3(-1024.0, -1024.0, -128.0)
#define VOXEL_BOUND_MAX  vec3( 1024.0,  1024.0,  128.0)
```

- **XY plane**: horizontal extent (2048 units wide)
- **Z axis**: vertical / altitude (256 units tall)
- **Units**: world-space units (game-dependent, not necessarily meters)
- **No planet curvature** — flat bounding box, not concentric spheres

### 2.2 Normalized Sample Coordinates

World positions are converted to [0,1]³ for texture sampling:

```glsl
vec3 GetSampleCoord(vec3 inSamplePosition) {
    vec3 sample_coord = (inSamplePosition - VOXEL_BOUND_MIN)
                      / (VOXEL_BOUND_MAX - VOXEL_BOUND_MIN);
    sample_coord = vec3(1.0) - sample_coord;  // flip all axes
    return sample_coord;
}
```

The **axis flip** (`1.0 - coord`) is required because the NVDF textures are stored with inverted orientation relative to world space.

### 2.3 Height Fraction

Normalized [0,1] vertical position within the cloud volume:

```glsl
float GetHeightFractionForPoint(vec3 inPosition, float cloudMin, float cloudMax) {
    float height_fraction = (inPosition.z - cloudMin) / (cloudMax - cloudMin);
    return clamp(height_fraction, 0, 1);
}
```

Where `cloudMin = VOXEL_BOUND_MIN.z` and `cloudMax = VOXEL_BOUND_MAX.z`.

### 2.4 Ray Generation

Rays are generated from camera parameters in view space, then marched in world space:

```glsl
Ray GenerateRay(vec2 uv) {
    vec3 camLook  = normalize(vec3(view[0][2], view[1][2], view[2][2]));
    vec3 camRight = normalize(vec3(view[0][0], view[1][0], view[2][0]));
    vec3 camUp    = normalize(vec3(view[0][1], view[1][1], view[2][1]));

    vec2 screenPoint = uv * 2.0 - 1.0;

    vec3 refPoint = cameraPos - camLook;
    vec3 p = refPoint
           + aspectRatio * screenPoint.x * halfTanFOV * camRight
           - screenPoint.y * halfTanFOV * camUp;

    ray.mOrigin    = cameraPos;
    ray.mDirection = normalize(p - cameraPos);
}
```

---

## 3. NVDF Voxel Data (Modeling Textures)

### 3.1 Format

**NVDF (Nubis Voxel Data Format)** — 3D textures storing fluid-simulated cloud shapes.

| Property | Value |
|----------|-------|
| Resolution | 512 x 512 x 64 |
| Format | RGBA (4 channels) |
| Source | VDB fluid simulation exports |
| Storage | 3D texture, trilinear filtered |

### 3.2 Channel Layout

| Channel | Name | Range | Description |
|---------|------|-------|-------------|
| **R** | Dimensional Profile | [0, 1] | Overall cloud shape / density envelope. 0 = empty, >0 = cloud present. Also provides gradient information for up-rezzing. |
| **G** | Detail Type | [0, 1] | Controls billow vs wispy noise blending. 0 = fully wispy, 1 = fully billowy. |
| **B** | Density Scale | [0, 1] | Per-voxel density multiplier. Controls how thick/opaque that region is. |
| **A** | SDF (packed) | [0, 1] → [-256, 4096] | Signed distance to nearest cloud surface. Packed to [0,1] for storage, unpacked at sample time. |

### 3.3 SDF Unpacking

The SDF channel is stored normalized in [0,1] and remapped to world-space distance at sample time:

```glsl
modeling_data.mSdf = ValueRemap(Modeling_NVDF.a, 0.0, 1.0, -256.0, 4096.0);
```

- **Negative SDF** → inside cloud (should sample density)
- **Positive SDF** → outside cloud (skip forward by this distance)
- Range: -256 (deep inside) to +4096 (far from any cloud)

### 3.4 Multiple Cloud Types

Different NVDF textures can represent different cloud formations:

```glsl
// Frostnova reference uses two presets:
if (cloud_type == 0)
    Modeling_NVDF = texture(modelingParkourTexture, coord).rgba;   // cumulus-like
else
    Modeling_NVDF = texture(modelingStormBirdTexture, coord).rgba; // storm/cumulonimbus
```

---

## 4. Detail Noise (Up-Rezzing)

### 4.1 Noise Texture

The 3D detail noise texture adds high-frequency structure to the low-res voxel data:

| Property | Value |
|----------|-------|
| Resolution | 128 x 128 x 128 |
| Format | RGBA (4 channels) |
| Tiling | Seamlessly repeating |

### 4.2 Channel Layout (Nubis³-specific)

This is **different** from Nubis 2015. Nubis³ uses billow/wispy pairs:

| Channel | Name | Description |
|---------|------|-------------|
| **R** | Wispy Low-Freq | Low-frequency wispy noise (cloud body emanation, flocculent structures) |
| **G** | Wispy High-Freq | High-frequency wispy noise |
| **B** | Billow Low-Freq | Low-frequency billow noise (upward uplift, wave structures). Scaled 0.3x |
| **A** | Billow High-Freq | High-frequency billow noise. Scaled 0.3x |

### 4.3 Noise Composition

The `mDetailType` channel from the NVDF controls the blend between wispy and billowy noise:

```glsl
// Wispy: blend low/high freq based on dimensional profile
float wispy_noise = mix(noise.r, noise.g, inDimensionalProfile);

// Billowy: blend low/high freq based on profile^0.25 (gentler gradient)
float billowy_type_gradient = pow(inDimensionalProfile, 0.25);
float billowy_noise = mix(noise.b * 0.3, noise.a * 0.3, billowy_type_gradient);

// Final composite: detail type drives wispy↔billowy blend
float noise_composite = mix(wispy_noise, billowy_noise, inType);
```

Key: the **billow channels are scaled by 0.3** to prevent over-erosion of billowy forms.

### 4.4 Hyper-High-Frequency (HHF) Detail

Close-range fly-through requires extra detail. HHF noise is derived from the existing channels:

```glsl
// HHF wispy: double-abs-remap creates sharp, intricate wispy structures
float hhf_wisps = 1.0 - pow(abs(abs(noise.g * 2.0 - 1.0) * 2.0 - 1.0), 4.0);

// HHF billowy: double-abs-remap creates round, puffy structures
float hhf_billows = pow(abs(abs(noise.a * 2.0 - 1.0) * 2.0 - 1.0), 2.0);

// Blend by type
float hhf_noise = clamp(mix(hhf_wisps, hhf_billows, inType), 0, 1);

// Fade HHF with distance (only visible close-up, 50-150 unit range)
float hhf_blend = ValueRemap(distance, 50.0, 150.0, 0.9, 1.0);
noise_composite = mix(hhf_noise, noise_composite, hhf_blend);
```

The double `abs(x * 2.0 - 1.0)` transform creates **frequency doubling** without additional texture reads.

### 4.5 Noise Tiling Frequency

The tiling frequency of detail noise against the voxel volume is a user parameter:

```glsl
vec4 noise = texture(cloudDetailNoiseTexture, inSamplePos * tiling_freq);
```

This controls how many times the 128³ noise tile repeats across the cloud volume. Higher values = finer detail.

---

## 5. Density Function

### 5.1 Complete Pipeline

```
NVDF Voxel Sample
    ├─ Dimensional Profile (R) ─── shape envelope
    ├─ Detail Type (G) ──────────── noise blend control
    ├─ Density Scale (B) ────────── thickness multiplier
    └─ SDF (A) ──────────────────── ray march acceleration

    IF profile > 0:
        Sample 3D detail noise at animated position
        Compose wispy/billowy noise based on Detail Type
        Add HHF detail if close to camera
        Erode profile by noise composite → uprezzed density
        Apply density scale with pow(scale, 4.0) shaping
        Apply distance-based power and scale adjustments
```

### 5.2 Full Density Function (GetUprezzedVoxelCloudDensity)

```glsl
float GetUprezzedVoxelCloudDensity(
    RaymarchInfo info, vec3 samplePos,
    float dimensionalProfile, float type, float densityScale,
    float mipLevel, bool hfDetails)
{
    // 1. Wind animation offset
    samplePos -= vec3(WIND_OFFSET.x, WIND_OFFSET.y, 0.0) * animateSpeed * totalTime;

    // 2. Sample 128³ detail noise
    vec4 noise = texture(detailNoise3D, samplePos * tilingFreq);

    // 3. Compose wispy/billowy based on NVDF detail type
    float wispy  = mix(noise.r, noise.g, dimensionalProfile);
    float bilGrad = pow(dimensionalProfile, 0.25);
    float billowy = mix(noise.b * 0.3, noise.a * 0.3, bilGrad);
    float composite = mix(wispy, billowy, type);

    // 4. HHF detail (close-range only)
    if (hfDetails) {
        float hhfW = 1.0 - pow(abs(abs(noise.g * 2.0 - 1.0) * 2.0 - 1.0), 4.0);
        float hhfB = pow(abs(abs(noise.a * 2.0 - 1.0) * 2.0 - 1.0), 2.0);
        float hhf  = clamp(mix(hhfW, hhfB, type), 0, 1);
        float fade = ValueRemap(info.distance, 50.0, 150.0, 0.9, 1.0);
        composite  = mix(hhf, composite, fade);
    }

    // 5. Erosion: profile eroded by noise composite
    float density = ValueErosion(dimensionalProfile, composite);
    // ValueErosion(v, m) = clamp((v - m) / (1.0 - m), 0, 1)

    // 6. Density scale shaping (pow4 curve makes scale very non-linear)
    float powScale = pow(clamp(densityScale, 0, 1), 4.0);
    density *= powScale;

    // 7. Power curve based on density scale (0.3 to 0.6)
    density = pow(density, mix(0.3, 0.6, max(0.1, powScale)));

    // 8. Distance-based density adjustment (close-range boost)
    if (hfDetails) {
        float distBlend = GetFractionFromValue(info.distance, 50.0, 150.0);
        density = pow(density, mix(0.5, 1.0, distBlend))
                * mix(0.666, 1.0, distBlend);
    }

    return density;
}
```

### 5.3 Profile-Level Density Samples

The two-tier density is computed as:

```glsl
VoxelCloudDensitySamples GetVoxelCloudDensitySamples(...) {
    DensitySamples result = {0, 0};

    if (dimensionalProfile > 0.0) {
        // Cheap: profile * scale (for light march, early tests)
        result.profile = dimensionalProfile * densityScale;

        // Full: uprezzed density with distance fade-in
        result.full = GetUprezzedVoxelCloudDensity(...)
                    * ValueRemap(distance, 10.0, 120.0, 0.25, 1.0);
    }
    return result;
}
```

The **distance fade-in** (`ValueRemap(dist, 10.0, 120.0, 0.25, 1.0)`) prevents sharp density pop-in when flying into clouds — density ramps from 25% to 100% over the 10-120 unit range.

### 5.4 Key Utility Functions

```glsl
// Remap value from one range to another (unclamped)
float Remap(float value, float oldMin, float oldMax, float newMin, float newMax) {
    return newMin + ((value - oldMin) / (oldMax - oldMin)) * (newMax - newMin);
}

// Remap with clamped normalization (clamped to [0,1] before scaling)
float ValueRemap(float value, float oldMin, float oldMax, float newMin, float newMax) {
    float norm = clamp((value - oldMin) / (oldMax - oldMin), 0, 1);
    return newMin + norm * (newMax - newMin);
}

// Erosion: how much of 'value' survives after subtracting 'threshold'
float ValueErosion(float value, float threshold) {
    return clamp((value - threshold) / (1.0 - threshold), 0, 1);
}
```

**ValueErosion** is the core of up-rezzing: the dimensional profile (cloud envelope) is eroded by the noise composite. Where noise is high, the profile is eaten away; where noise is low, the profile survives. This creates detailed edges while preserving the overall shape.

---

## 6. SDF Acceleration

### 6.1 How It Works

Each NVDF voxel stores a signed distance to the nearest cloud surface in the A channel. During the ray march:

1. Sample NVDF at current position
2. Unpack SDF: `sdf = Remap(A, 0, 1, -256, 4096)` — world-space distance
3. If SDF > 0 (outside cloud): advance ray by SDF distance (skip empty space)
4. If SDF < 0 (inside cloud): sample full density, use adaptive stepping

```glsl
float adaptive_step = max(1.0, max(sqrt(distance), 0.1) * 0.08);

raymarch_info.mStepSize = max(modeling_data.mSdf, adaptive_step);

if (modeling_data.mSdf < 0.0) {
    // Inside cloud — sample density and light
    VoxelCloudDensitySamples samples = GetVoxelCloudDensitySamples(...);
    if (samples.mProfile > 0.0) {
        // Accumulate density, integrate lighting
    }
}

distance += stepSize;
```

### 6.2 Adaptive Step Size

Inside clouds (SDF < 0), step size is adaptive based on camera distance:

```glsl
float adaptive_step_size = max(1.0, max(sqrt(distance), EPSILON) * 0.08);
```

- **Close to camera** (distance small): step size ≈ 1.0 (fine detail)
- **Far from camera** (distance large): step size grows with √distance (coarser, faster)
- **Minimum step**: 1.0 unit (prevents infinite loops)

### 6.3 Performance Impact

SDF acceleration provides the most significant performance gain for voxel clouds. Without it, every step must sample the NVDF regardless of whether clouds are present. With SDF, rays jump directly to the next cloud surface, reducing wasted samples dramatically.

---

## 7. Light Grid (Precomputed Light Density)

### 7.1 Format

| Property | Value |
|----------|-------|
| Resolution | 256 x 256 x 32 (1/8 of NVDF bounding box in each dimension) |
| Format | RG (2 channels minimum) |
| Updated | Per-frame compute pass |

### 7.2 Channel Layout

| Channel | Content |
|---------|---------|
| **R** | Accumulated density from this voxel toward the sun (for shadow/transmittance) |
| **G** | Local density at this voxel (low-LOD profile density, for depth probability) |

### 7.3 Computation

A compute shader marches from each voxel toward the sun direction, accumulating profile density:

```glsl
// Per voxel:
float density = GetVoxelCloudProfileDensity(coord);
localDensity = density;     // store in G

vec3 nextCoord = coord + sunDir;
while (InBoundary(nextCoord)) {
    density += GetVoxelCloudProfileDensity(nextCoord);
    nextCoord += sunDir;
}
accumulatedDensity = density; // store in R
```

This is **not** a full lighting calculation — just raw density accumulation. The actual lighting math happens in the per-sample `IntegrateLightEnergy` function using this precomputed data.

### 7.4 Usage in Lighting

The light grid serves two purposes:
1. **Far-field sun occlusion**: The accumulated density (R channel) provides the bulk of the sun shadow computation, replacing many of the light march steps
2. **Depth probability**: The local density (G channel) feeds the in-scatter probability function

---

## 8. Lighting Model

### 8.1 Light Energy Computation

Per-sample light energy combines four components:

```
Light Energy = Attenuation Probability
             * In-Scatter Probability
             * Phase Probability
             * Brightness
```

Then converted to final value via: `light_energy = exp(-light_energy * 5.0)`

### 8.2 Sun Density (GetDensityToSun)

A hybrid approach: 2 short-range samples + light grid for far field:

```glsl
float GetDensityToSun(RaymarchInfo info, ModelingData data, vec3 pos, vec3 lightDir) {
    float totalDensity = 0.0;

    // 2 close-range steps with full density sampling
    for (int i = 0; i < 2; i++) {
        pos += lightDir * info.stepSize;
        vec3 coord = GetSampleCoord(pos);
        totalDensity += GetVoxelCloudDensitySamples(info, data, coord, 1.0, false).full;
    }

    // 1 far-field step from precomputed light grid
    pos += lightDir * info.stepSize;
    vec3 coord = GetSampleCoord(pos);
    totalDensity += texture(lightGrid, coord).r;

    return totalDensity;
}
```

This is much cheaper than the Nubis 2015 6-sample cone march. The light grid replaces all distant samples.

### 8.3 Attenuation (Beer-Lambert, Dual-Term)

Two Beer-Lambert terms (Schneider's "Beer-Powder" heritage):

```glsl
float primary_attenuation   = exp(-density_to_sun);
float secondary_attenuation = exp(-density_to_sun * 0.25) * 0.7;

float attenuation_probability = max(
    Remap(cos_angle, 0.7, 1.0, secondary_attenuation, secondary_attenuation * 0.25),
    primary_attenuation
);
```

- **Primary**: standard Beer-Lambert extinction
- **Secondary**: reduced extinction (0.25x) scaled by 0.7 — pushes light deeper into clouds
- The `max()` ensures at least the primary attenuation applies
- When looking toward the sun (cos_angle > 0.7), secondary attenuation is reduced further, preventing over-brightening of directly back-lit clouds

### 8.4 In-Scatter Probability

Models the probability of light scattering toward the camera from within the cloud:

```glsl
float depth_probability = mix(
    0.05 + pow(density_Lod, Remap(0.5, 0.3, 0.85, 0.5, 2.0)),
    1.0,
    clamp(density_to_sun / 0.5, 0, 1)
);

float vertical_probability = pow(Remap(0.5, 0.07, 0.14, 0.1, 1.0), 0.8);

float in_scatter_probability = clamp(depth_probability * vertical_probability, 0, 1);
```

- **depth_probability**: low-LOD density from light grid controls how deep light penetrates
- **vertical_probability**: height-based modulation
- Product gives the overall in-scatter likelihood

### 8.5 Phase Function (Henyey-Greenstein)

```glsl
float HG(float cos_angle, float eccentricity) {
    float g2 = eccentricity * eccentricity;
    return ((1.0 - g2) / pow(1.0 + g2 - 2.0 * eccentricity * cos_angle, 1.5))
         * ONE_OVER_FOURPI;  // 1 / (4 * PI)
}
```

Dual-lobe with silver lining effect:

```glsl
float silver_spread    = 1.32;
float silver_intensity = 1.27;
float eccentricity     = 1.0;  // extreme forward scatter

float phase_probability = max(
    HG(cos_angle, eccentricity),
    silver_intensity * HG(cos_angle, 0.99 - silver_spread)
);
```

- **Primary lobe**: eccentricity=1.0 (very strong forward scatter)
- **Silver lining lobe**: `g = 0.99 - 1.32 = -0.33` (mild back-scatter), intensity boosted 1.27x
- `max()` of both lobes — each lobe contributes where it dominates

### 8.6 Ambient Scattering

Based on profile density and sun transmittance:

```glsl
float ambient_scattering = pow(1.0 - profile, 0.5) * exp(-density_to_sun);
```

- `pow(1.0 - profile, 0.5)`: thin cloud regions get more ambient (sqrt falloff)
- `exp(-density_to_sun)`: ambient is attenuated by sun occlusion

Applied to cloud color:

```glsl
if (density <= 1.0) {
    cloudColor = (1.0 - ambient_scattering) * cloudColor
               + ambient_scattering * ambientColor;
}
```

### 8.7 Cloud Coloring

Cloud color is derived from sky color and light energy:

```glsl
vec3 warmColor = mix(skyColorNoSun, vec3(1.0, 0.87, 0.65), clamp(-lightDir.z + 0.2, 0, 1));
vec3 coolColor = mix(skyColorNoSun, vec3(0.23, 0.36, 0.47), clamp(-lightDir.z + 0.5, 0, 1));

float colorOffset1 = 0.16;
float colorOffset2 = 12.6;

vec3 cloudColor = mix(warmColor, white, clamp(light_energy * colorOffset1, 0, 1));
cloudColor = mix(coolColor, cloudColor, clamp(pow(light_energy * colorOffset2, 3), 0, 1));
```

- **warmColor**: golden-warm tint for sunlit surfaces, modulated by sun elevation
- **coolColor**: blue-grey tint for shadowed/ambient surfaces
- **colorOffset1 (0.16)**: how quickly sunlit areas wash to white
- **colorOffset2 (12.6)**: how quickly shadowed areas transition to lit (cubic falloff)
- Low sun (lightDir.z close to 0) → more sky color; high sun → more warm/cool tints

### 8.8 Front-to-Back Compositing

Per-step accumulation:

```glsl
// Accumulate transmittance-weighted radiance
pixelData.transmittance += sample.full * light_energy * pixelData.alpha;
pixelData.cloudColor    += sample.full * cloudColor   * pixelData.alpha;

// Update alpha (Beer-Lambert per step)
pixelData.alpha *= exp(-sample.full * 1.0);
```

Final blend with background:

```glsl
vec3 finalColor = mix(bgColor, nightFactor * cloudColor * max(0.0, transmittance),
                      1.0 - alpha);
```

---

## 9. Ray Marching

### 9.1 AABB Intersection

Rays are intersected with the voxel bounding box using slab testing per axis:

```glsl
void SetRaymarchLimit(Ray ray, inout RaymarchInfo info) {
    float tmin = 4096.0, tmax = -4096.0;

    // If camera is inside the box, tmin = 0
    if (insideBox(ray.origin)) tmin = 0.0;

    // Per-axis slab intersection (X, Y, Z)
    for each axis:
        t1 = abs((BOUND_MAX[axis] - origin[axis]) / direction[axis]);
        t2 = abs((BOUND_MIN[axis] - origin[axis]) / direction[axis]);
        // If intersection point is within other axes' bounds:
        tmin = min(tmin, t);
        tmax = max(tmax, t);

    info.limit = vec2(tmin, tmax);
    info.distance = tmin;  // start marching here
}
```

### 9.2 March Loop

```glsl
void RaymarchVoxelClouds(Ray ray, vec3 lightDir, inout PixelData pixel) {
    RaymarchInfo info;
    SetRaymarchLimit(ray, info);
    float cos_angle = dot(ray.direction, lightDir);

    while (pixel.transmittance > transmittance_limit
        && info.distance < min(farclip, info.limit.y))
    {
        vec3 samplePos  = ray.origin + ray.direction * info.distance;
        vec3 sampleCoord = GetSampleCoord(samplePos);

        if (inBounds(sampleCoord)) {
            ModelingData data = GetVoxelCloudModelingData(sampleCoord, 0.0);

            // Adaptive step: sqrt(distance) * 0.08, min 1.0
            float adaptiveStep = max(1.0, max(sqrt(info.distance), 0.1) * 0.08);

            info.cloudDistance = data.sdf;
            info.stepSize = max(data.sdf, adaptiveStep);

            if (data.sdf < 0.0) {  // Inside cloud
                DensitySamples samples = GetVoxelCloudDensitySamples(
                    info, data, samplePos, 1.0, true);

                if (samples.profile > 0.0) {
                    pixel.density += samples.full;
                    IntegrateLightEnergy(info, data, samples,
                        samplePos, sampleCoord, lightDir, cos_angle, pixel);
                }
            }
        }
        info.distance += info.stepSize;
    }
}
```

### 9.3 Termination Conditions

| Condition | Default | Description |
|-----------|---------|-------------|
| Transmittance limit | configurable (e.g. 0.01) | Cloud fully opaque, no more light gets through |
| Far clip | configurable | Maximum march distance from camera |
| Box exit | `info.distance >= info.limit.y` | Ray has left the voxel bounding box |

### 9.4 Step Size Summary

| Situation | Step Size | Why |
|-----------|-----------|-----|
| Outside cloud (SDF > 0) | SDF distance (up to 4096) | Jump to nearest cloud surface |
| Inside cloud, close to camera | ~1.0 | Fine detail sampling |
| Inside cloud, far from camera | √distance * 0.08 | Coarser, fewer samples needed |
| Minimum | 1.0 | Prevent infinite loops |

### 9.5 Near/Far Split (Temporal Upscaling)

For performance, the march can be split into two passes at different resolutions:

| Pass | Resolution | Distance Range | Purpose |
|------|-----------|----------------|---------|
| Near | 1/4 resolution | 0 to threshold (200-500 units) | Close clouds, temporal upscaled |
| Far | Full resolution | threshold to farclip | Distant clouds, integrated directly |

This provides 30-70% FPS improvement with acceptable quality.

---

## 10. Wind and Animation

### 10.1 Wind Offset

Wind animates cloud detail noise by offsetting the sample position:

```glsl
#define CLOUD_WIND_OFFSET vec2(0.1, 0.1)

samplePos -= vec3(WIND_OFFSET.x, WIND_OFFSET.y, 0.0) * animateSpeed * totalTime;
```

- Wind is applied **only to the detail noise sampling position**, not to the NVDF voxel lookup
- This means the macro cloud shapes stay fixed while the detail noise flows through them
- Wind is in the **XY plane** (horizontal), Z component is 0

### 10.2 Wind Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| CLOUD_WIND_OFFSET | vec2(0.1, 0.1) | Direction and magnitude of wind in XY |
| animate_speed | UI-controlled | Speed multiplier for wind animation |

### 10.3 No Height-Dependent Shearing in Nubis³

Unlike Nubis 2015 which had explicit height-dependent wind shearing (`heightFraction * windDir * topOffset`), the Nubis³ voxel approach handles this differently:
- The NVDF voxels are pre-shaped by fluid simulation (which includes wind/shear in the sim)
- Only the detail noise is animated at runtime
- This simplifies the runtime code significantly

---

## 11. Controls and Parameters

### 11.1 Core Cloud Parameters

| Parameter | Type | Range | Description |
|-----------|------|-------|-------------|
| cloud_type | int | 0, 1, ... | Select NVDF preset (parkour, stormbird, etc.) |
| tiling_freq | float | 0.1 - 10.0 | Detail noise tiling frequency. Higher = finer detail noise |
| DENSITY_SCALE | float | (hardcoded 0.01) | Global density multiplier |

### 11.2 Ray March Parameters

| Parameter | Type | Range | Description |
|-----------|------|-------|-------------|
| farclip | float | 100 - 10000 | Maximum march distance from camera |
| transmittance_limit | float | 0.001 - 0.1 | Early termination threshold (lower = more opaque before stopping) |

### 11.3 Animation Parameters

| Parameter | Type | Range | Description |
|-----------|------|-------|-------------|
| animate_speed | float | 0 - 100 | Wind speed multiplier |
| animate_offset | vec3 | arbitrary | Additional wind direction offset |

### 11.4 Post-Processing Parameters

| Parameter | Type | Range | Description |
|-----------|------|-------|-------------|
| enable_godray | float | 0 or 1 | Enable/disable god ray effect |
| godray_exposure | float | 0.0 - 5.0 | God ray intensity |
| sky_turbidity | float | 1.0 - 10.0 | Atmospheric turbidity for Preetham sky model |

### 11.5 Internal Constants

| Constant | Value | Description |
|----------|-------|-------------|
| VOXEL_BOUND_MIN | (-1024, -1024, -128) | Cloud volume lower bound |
| VOXEL_BOUND_MAX | (1024, 1024, 128) | Cloud volume upper bound |
| CLOUD_WIND_OFFSET | (0.1, 0.1) | Wind direction |
| EPSILON | 0.1 | Minimum threshold for divisions |
| ONE_OVER_FOURPI | 0.0796 | 1/(4*PI) for phase function normalization |

### 11.6 Lighting Constants (hardcoded)

| Constant | Value | Description |
|----------|-------|-------------|
| silver_spread | 1.32 | HG back-lobe eccentricity offset (g = 0.99 - 1.32 = -0.33) |
| silver_intensity | 1.27 | Silver lining brightness boost |
| brightness | 0.5 | Overall light energy scalar |
| eccentricity | 1.0 | Primary HG forward-scatter g value |
| colorOffset1 | 0.16 | Sunlit-to-white transition speed |
| colorOffset2 | 12.6 | Shadow-to-lit cubic transition speed |

---

## 12. Temporal Reprojection

### 12.1 Strategy (Nubis 2015/2017 — carried forward)

Only a fraction of pixels are ray-marched per frame. The rest are filled by reprojecting from the previous frame:

- **4x4 Bayer pattern**: 1/16 pixels marched per frame (each pixel updated every 16 frames)
- **Halton sequence jitter**: sub-pixel temporal jitter for anti-aliasing
- **Blend factor**: ~75-95% weight on reprojected history

### 12.2 Reprojection Approach

```
1. Generate camera ray for current pixel
2. Find world-space hit position (position of maximum density along ray)
3. Project hit position through previous frame's view-projection matrix
4. Sample previous frame at reprojected UV
5. Blend: result = mix(current, reprojected, blendFactor)
```

### 12.3 Ghosting Mitigation

- **Neighborhood clamping**: clamp reprojected color to min/max of local neighborhood (AABB clamp)
- **Motion-adaptive blend**: reduce blend factor during fast camera motion
- **Depth similarity**: reject history where reprojected depth differs significantly from current

### 12.4 Nubis³ Specific: Near/Far Split

Instead of (or in addition to) traditional 1/16 reprojection, Nubis³ uses a **two-pass near/far split**:
- Near clouds at 1/4 resolution (temporal upscaled across frames)
- Far clouds at full resolution (integrated directly)
- The distance threshold (200-500 units) is configurable

---

## 13. Memory Budget

| Resource | Dimensions | Channels | Approx Size |
|----------|-----------|----------|-------------|
| NVDF Modeling (per cloud type) | 512 x 512 x 64 | RGBA | ~64 MB |
| Detail Noise | 128 x 128 x 128 | RGBA | ~32 MB |
| Light Grid | 256 x 256 x 32 | RG | ~8 MB |
| Near/Far Temporal Buffers | 1/4 screen res | RGBA | ~10 MB |
| **Total (2 NVDF presets)** | | | **~240 MB** |

---

## 14. Differences from Nubis 2015/2017

| Aspect | Nubis 2015/2017 | Nubis Cubed (2023) |
|--------|-----------------|-------------------|
| **Modeling** | Procedural noise + 2D weather map | 3D voxel grids from fluid sim (NVDF) |
| **Cloud types** | Weather map B channel → height gradient blend (stratus/stratocumulus/cumulus) | Pre-authored NVDF presets |
| **Noise** | 128³ Perlin-Worley + 32³ Worley detail | 128³ billow/wispy RGBA for up-rezzing |
| **Noise composition** | FBM weights 0.625/0.25/0.125, detail erosion with height inversion | Billow/wispy blend driven by NVDF Detail Type channel |
| **Coverage** | `remap(base, coverage, 1, 0, 1) * coverage` | Implicit in NVDF Dimensional Profile |
| **Height gradients** | vec4 trapezoid smoothstep per type (STRATUS, STRATOCUMULUS, CUMULUS) | Embedded in NVDF Dimensional Profile |
| **Cloud shell** | Concentric spheres (planet curvature) | Flat AABB |
| **SDF** | None | Compressed SDF in NVDF A channel |
| **Light march** | 6 cone samples + 1 far sample | 2 near samples + precomputed light grid |
| **Detail** | Fixed LOD | HHF double-abs frequency doubling for close-range |
| **Wind** | Offsets both shape + detail, height shearing | Offsets detail noise only (shapes are pre-authored) |
| **Phase function** | Dual HG (g=0.6, g=-0.2) | Dual HG (g=1.0, silver g=-0.33) |

---

## 15. Sources

### Primary

- **Nubis Cubed (2023)**: [Guerrilla Games page](https://www.guerrilla-games.com/read/nubis-cubed) | [SIGGRAPH 2023 Advances](https://advances.realtimerendering.com/s2023/)
- **Nubis Evolved (2022)**: [Guerrilla Games page](https://www.guerrilla-games.com/read/nubis-evolved)
- **Nubis (2015)**: "The Real-time Volumetric Cloudscapes of Horizon: Zero Dawn", GPU Pro 7 pp. 97-127
- **Nubis Voxel Cloud Pack**: [bit.ly/NubisVoxelCloudPack](http://bit.ly/NubisVoxelCloudPack) — NVDF samples + noise generator

### Reference Implementations

- **Frostnova** (Vulkan, Nubis³): [github.com/YueZhang1027/CIS5650-Final-Project-Frostnova](https://github.com/YueZhang1027/CIS5650-Final-Project-Frostnova) — Primary reference for this document's shader code
- **Meteoros** (Vulkan, Nubis 2015): [github.com/AmanSachan1/Meteoros](https://github.com/AmanSachan1/Meteoros)
- **Godot Cloud Demo** (Nubis 2015): [github.com/clayjohn/godot-volumetric-cloud-demo](https://github.com/clayjohn/godot-volumetric-cloud-demo)

### Related Techniques

- **Hillaire, Frostbite (2016)**: Multi-scattering approximation (octave-based Wrenninge method)
- **Bridson (2007)**: Curl noise for divergence-free turbulence
- **Preetham (1999)**: Analytic sky model used for cloud ambient/sky color
