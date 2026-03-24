#ifndef CLOUD_DENSITY_GLSL
#define CLOUD_DENSITY_GLSL

// Schneider 2015 procedural cloud density and lighting model.
// Matched to Frostnova compute.comp reference implementation.
//
// Two-stage density pipeline:
//   1. getBaseDensity() — height gradient × base noise, coverage erosion
//   2. getDetailDensity() — curl displacement + high-freq erosion
//
// References:
//   [Schneider15]   Schneider, "Real-time Volumetric Cloudscapes of HZD", SIGGRAPH 2015
//   [Frostnova]     github.com/Frostnova — compute.comp reference
//   [Wrenninge13]   Multi-scatter octave approximation
//   [Hillaire16]    Frostbite decay parameters
//
// Used by: cloud_raymarch.comp, cloud_shadow.comp

const float PI = 3.14159265358979;

// --- Utility ---

// Unclamped remap [Frostnova ValueRemap] — for height gradients
float valueRemap(float value, float low1, float high1, float low2, float high2) {
    return low2 + (value - low1) / max(high1 - low1, 0.0001) * (high2 - low2);
}

// Clamped remap [Frostnova ValueRemapClamped] — the core density operation.
float valueRemapClamped(float value, float low1, float high1, float low2, float high2) {
    return clamp(valueRemap(value, low1, high1, low2, high2), min(low2, high2), max(low2, high2));
}

// --- Height Gradients [Frostnova GetCloudLayerDensity] ---
//
// Linear ramps via ValueRemap (NOT smoothstep). Creates trapezoidal profiles.
// Frostnova uses 3 types: stratus(0), stratocumulus(0.5), cumulus(1.0).
// We add cumulonimbus for thunderstorms.

float gradientStratus(float h) {
    // Frostnova: ramp 0.0→0.1 up, 0.2→0.3 down
    return max(0.0, valueRemap(h, 0.0, 0.1, 0.0, 1.0) * valueRemap(h, 0.2, 0.3, 1.0, 0.0));
}

float gradientStratocumulus(float h) {
    // Frostnova: ramp 0.0→0.2 up, 0.2→0.7 down
    return max(0.0, valueRemap(h, 0.0, 0.2, 0.0, 1.0) * valueRemap(h, 0.2, 0.7, 1.0, 0.0));
}

float gradientCumulus(float h) {
    // Frostnova: ramp 0.0→0.2 up, 0.7→0.9 down
    return max(0.0, valueRemap(h, 0.0, 0.2, 0.0, 1.0) * valueRemap(h, 0.7, 0.9, 1.0, 0.0));
}

float gradientCumulonimbus(float h) {
    // Tall tower for thunderstorms — extends nearly to top
    return max(0.0, valueRemap(h, 0.0, 0.1, 0.0, 1.0) * valueRemap(h, 0.8, 0.95, 1.0, 0.0));
}

// Blend types [Frostnova GetCloudLayerDensity blend formula]
// type 0=St, 0.5=Sc, 1.0=Cu (Frostnova 3-type system)
// Extended: type > 1.0 blends toward Cb for thunderstorms
float heightProfile(float h, float type) {
    float st = gradientStratus(h);
    float sc = gradientStratocumulus(h);
    float cu = gradientCumulus(h);
    float cb = gradientCumulonimbus(h);

    // Frostnova blend: d1 = mix(St, Sc, type*2), d2 = mix(Sc, Cu, (type-0.5)*2), out = mix(d1, d2, type)
    if (type <= 1.0) {
        float d1 = mix(st, sc, clamp(type * 2.0, 0.0, 1.0));
        float d2 = mix(sc, cu, clamp((type - 0.5) * 2.0, 0.0, 1.0));
        return mix(d1, d2, type);
    }
    // Extended: type 1.0→2.0 blends Cu→Cb (thunderstorm towers)
    return mix(cu, cb, clamp(type - 1.0, 0.0, 1.0));
}

// --- Base Density [Frostnova GetBaseDensity] ---
//
// Noise channel layout (Schneider 2015):
//   R = Perlin-Worley blend (connected shape)
//   G = Worley F1 2x (erosion octave 1)
//   B = Worley F1 4x (erosion octave 2)
//   A = Worley F1 8x (erosion octave 3)

float getBaseDensity(vec3 pos, float coverage, float type,
                     vec4 noise, float cloudBase, float cloudThickness,
                     float densityMul,
                     out float outHeight) {
    float h = (pos.y - cloudBase) / cloudThickness;
    outHeight = h;
    if (h < 0.0 || h > 1.0) return 0.0;

    // Height gradient [Frostnova GetCloudLayerDensity]
    float hProfile = heightProfile(h, type);
    if (hProfile < 0.001) return 0.0;

    // Base noise remapped [Frostnova: layerDensity * ValueRemapClamped(noise.r, 0.3, 1.0, 0.0, 1.0)]
    float baseNoise = valueRemapClamped(noise.r, 0.3, 1.0, 0.0, 1.0);
    float density = hProfile * baseNoise;
    if (density < 0.0001) return 0.0;

    // Coverage with anvil bias [Frostnova: pow(coverage, ValueRemap(h, 0.7, 0.8, 1.0, 0.8))]
    float anvilBias = valueRemap(h, 0.7, 0.8, 1.0, 0.8);
    float adjCoverage = pow(max(coverage, 0.001), clamp(anvilBias, 0.5, 1.0));

    // FBM erosion from G/B/A channels [Frostnova: 0.625*g + 0.25*b + 0.125*w]
    float erosion = 0.625 * noise.g + 0.25 * noise.b + 0.125 * noise.a;

    // Coverage erosion [Frostnova: ValueRemapClamped(erosion, coverage, 1.0, 0.0, 1.0)]
    erosion = valueRemapClamped(erosion, adjCoverage, 1.0, 0.0, 1.0);

    // Apply erosion to density [Frostnova: ValueRemapClamped(density, erosion, 1.0, 0.0, 1.0)]
    density = valueRemapClamped(density, erosion, 1.0, 0.0, 1.0);

    // Density multiplier (user control)
    density *= densityMul;

    return density;
}

// Overload without outHeight for shadow/light march
float getBaseDensity(vec3 pos, float coverage, float type,
                     vec4 noise, float cloudBase, float cloudThickness,
                     float densityMul) {
    float unused;
    return getBaseDensity(pos, coverage, type, noise, cloudBase, cloudThickness,
                          densityMul, unused);
}

// --- Detail Density [Frostnova GetDetailDensity] ---
//
// Applies high-frequency erosion to base density.
// Uses same noise texture at higher frequency (DETAIL_FREQ).
// Height-inverted: wispy bottoms, smooth cauliflower tops.

float getDetailDensity(float baseDensity, vec4 detailNoise, float height, float detailStr) {
    if (baseDensity < 0.0001 || detailStr < 0.001) return baseDensity;

    // FBM from detail noise G/B/A [Frostnova: 0.625*r + 0.25*g + 0.125*b — uses r/g/b of separate texture]
    // We reuse same texture at higher freq, so erosion octaves are in G/B/A
    float erosion = 0.625 * detailNoise.g + 0.25 * detailNoise.b + 0.125 * detailNoise.a;

    // Height-invert [Frostnova: mix(erosion, 1-erosion, clamp(height*10, 0, 1))]
    // Bottom of cloud (h≈0): use raw erosion → wispy tendrils
    // Top of cloud (h>0.1): use inverted erosion → round cauliflower shapes
    erosion = mix(erosion, 1.0 - erosion, clamp(height * 10.0, 0.0, 1.0));

    // Apply detail erosion [Frostnova: ValueRemapClamped(density, erosion, 1.0, 0.0, 1.0)]
    return valueRemapClamped(baseDensity, erosion * detailStr, 1.0, 0.0, 1.0);
}

// --- Phase Functions ---

float phaseHG(float cosTheta, float g) {
    float g2 = g * g;
    float denom = max(1.0 + g2 - 2.0 * g * cosTheta, 1e-6);
    return (1.0 - g2) / (4.0 * PI * denom * sqrt(denom));
}

// Dual-lobe HG [Nubis3 §8.5: max() of forward + silver-lining back lobe]
// g1 = forward eccentricity (0.8 for first octave, decays per octave)
// Silver lobe: g = 0.99 - 1.32 = -0.33 (mild back-scatter), boosted 1.27x
float phaseDualHG(float cosTheta, float g1) {
    const float SILVER_INTENSITY = 1.27;
    const float SILVER_SPREAD    = 1.32;
    float forward = phaseHG(cosTheta, g1);
    float silver  = SILVER_INTENSITY * phaseHG(cosTheta, 0.99 - SILVER_SPREAD);
    return max(forward, silver);
}

// --- Multi-scatter octave approximation [Wrenninge13, Hillaire16] ---

float multiScatterEnergy(float opticalDepth, float cosTheta, uint octaves) {
    const float ISOTROPIC_PHASE = 1.0 / (4.0 * PI);
    const float A_DECAY = 0.5;
    const float B_DECAY = 0.5;
    const float C_DECAY = 0.5;

    float energy = 0.0;
    float a = 1.0, b = 1.0, c = 1.0;

    for (uint n = 0; n < octaves; n++) {
        float phase = phaseDualHG(cosTheta, 0.8 * c);
        phase = mix(ISOTROPIC_PHASE, phase, c);

        float od = opticalDepth * a;

        // Dual-Beer attenuation [Nubis3 §8.3]
        float primary   = exp(-od);
        float secondary = exp(-od * 0.25) * 0.7;
        // Angle-dependent: reduce secondary when looking toward sun (cos > 0.7)
        float attenuation = max(
            valueRemap(cosTheta, 0.7, 1.0, secondary, secondary * 0.25),
            primary
        );

        energy += b * phase * attenuation;
        a *= A_DECAY;
        b *= B_DECAY;
        c *= C_DECAY;
    }
    return energy;
}

#endif // CLOUD_DENSITY_GLSL
