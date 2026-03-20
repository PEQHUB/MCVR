#ifndef CLOUD_DENSITY_GLSL
#define CLOUD_DENSITY_GLSL

// Nubis³ cloud density and lighting model.
//
// Core equation: density = ValueErosion(dimensional_profile, noise_composite)
//
// dimensional_profile encodes height shape × coverage into [0,1].
// It drives density thresholding, noise blending, and ambient scattering.
//
// References:
//   [Nubis³]       Schneider, "Nubis, Cubed", SIGGRAPH 2023
//   [NubisEvolved]  Schneider, "Nubis, Evolved", SIGGRAPH 2022
//   [Wrenninge13]   Multi-scatter octave approximation
//   [Hillaire16]    Frostbite decay parameters (a=0.2, b=0.5, c=0.5)
//
// Used by: cloud_raymarch.comp, cloud_shadow.comp

const float PI = 3.14159265358979;

// --- Utility ---

// Unclamped remap [Nubis³ p9 "Remap"]
float cloudRemap(float value, float low1, float high1, float low2, float high2) {
    return low2 + (value - low1) / (high1 - low1) * (high2 - low2);
}

// Nubis ValueErosion [Nubis³ p118]: erodes profile by noise.
// Where noise is high, more profile is removed → creates detail holes.
// saturate((profile - noise) / (1 - noise))
float valueErosion(float profile, float noise) {
    return clamp((profile - noise) / max(1.0 - noise, 0.0001), 0.0, 1.0);
}

// --- Dimensional Profile [NubisEvolved p97, Nubis³ p82-86] ---
//
// Height envelope × coverage. Type controls vertical extent:
//   type=0   (stratus):        thin layer, cloud lives in h=[0, ~0.25]
//   type=0.33 (stratocumulus):  medium,    h=[0, ~0.5]
//   type=0.67 (cumulus):        tall,      h=[0, ~0.8]
//   type=1.0  (cumulonimbus):   full,      h=[0, 1.0]
//
// Coverage scales the profile, controlling the noise threshold:
//   low coverage  → sparse isolated clouds (only strongest noise features)
//   high coverage → dense overcast (most noise produces density)

float dimensionalProfile(float h, float coverage, float type) {
    // Base: sharp condensation level at cloud bottom [NubisEvolved p97]
    float base = smoothstep(0.0, 0.1, h);

    // Top: type-dependent fade. Stratus fades early, Cb uses full height.
    float topFade = mix(0.25, 1.0, type);
    float top = smoothstep(topFade, topFade * 0.35, h);

    return base * top * coverage;
}

// --- Noise Composite [Nubis³ p99, p104, p108] ---
//
// Noise channels (from cloud_noise_gen.comp):
//   R = Perlin-Worley  → wispy low-freq  (connected shapes)
//   G = Worley freq 4  → billowy low-freq (cell boundaries)
//   B = Worley freq 8  → billowy high-freq (fine cells)
//   A = Curl noise     → wispy high-freq  (tendrils)
//
// Wispy noise: blends low→high freq wisps using dimensional_profile.
//   Low density edges → more base PW, dense core → more curl detail. [p99]
//
// Billowy noise: blends low→high freq Worley using profile^0.25.
//   The 0.3 scale matches Nubis³ — billowy channels are stronger,
//   so less erosion → denser, puffier shapes. [p104]
//
// Cloud type blends wispy↔billowy character. [p108]
//   type=0 (stratus) → wispy (thin, curly), type=1 (cb) → billowy (round, puffy)

float noiseComposite(vec4 noise, float dimProfile, float type) {
    // Wispy: PW base + curl detail [Nubis³ p99]
    float wispy = mix(noise.r, noise.a, dimProfile);

    // Billowy: Worley cells, profile^0.25 gradient [Nubis³ p104]
    float billowyGrad = pow(dimProfile, 0.25);
    float billowy = mix(noise.g * 0.3, noise.b * 0.3, billowyGrad);

    // Type blend [Nubis³ p108]
    return mix(wispy, billowy, type);
}

// --- Main Cloud Density [Nubis³ p24, p118] ---
//
// Returns density and writes out dimensional_profile for ambient scattering.
// Pipeline: dimProfile → noiseComposite → valueErosion → scale → sharpen

float cloudDensity(vec3 pos, float coverage, float type,
                   vec4 noise, float cloudBase, float cloudThickness,
                   float densityMul, float detailStr,
                   out float outDimProfile) {
    float h = (pos.y - cloudBase) / cloudThickness;
    if (h < 0.0 || h > 1.0) { outDimProfile = 0.0; return 0.0; }

    // Dimensional profile [NubisEvolved p97]
    float dimProfile = dimensionalProfile(h, coverage, type);
    outDimProfile = dimProfile;
    if (dimProfile < 0.001) return 0.0;

    // Noise composite [Nubis³ p99-108]
    float nc = noiseComposite(noise, dimProfile, type);

    // ValueErosion [Nubis³ p118]: noise erodes the profile
    // detailStr scales erosion: 0=solid, 1.0=standard, 2.0=extra detail
    float density = valueErosion(dimProfile, nc * clamp(detailStr, 0.0, 2.0));

    // Density multiplier
    density *= densityMul;

    // Sharpening [Nubis³ p118]: pow(density, exponent)
    float sharp = clamp(pc.sharpening, 0.1, 1.0);
    return pow(max(density, 0.0), sharp);
}

// Overload without outDimProfile for shadow/light march
float cloudDensity(vec3 pos, float coverage, float type,
                   vec4 noise, float cloudBase, float cloudThickness,
                   float densityMul, float detailStr) {
    float unused;
    return cloudDensity(pos, coverage, type, noise, cloudBase, cloudThickness,
                        densityMul, detailStr, unused);
}

// --- Phase Functions ---

// Henyey-Greenstein phase function [Henyey41]
float phaseHG(float cosTheta, float g) {
    float g2 = g * g;
    float denom = 1.0 + g2 - 2.0 * g * cosTheta;
    return (1.0 - g2) / (4.0 * PI * denom * sqrt(denom));
}

// Dual-lobe HG phase (forward + back scatter blend)
float phaseDualHG(float cosTheta, float g1, float g2, float blend) {
    return mix(phaseHG(cosTheta, g1), phaseHG(cosTheta, g2), blend);
}

// --- Multi-scatter octave approximation [Wrenninge13 §4, Hillaire16 §4.6.3] ---
//
// Each bounce: extinction decays by a, contribution by b, anisotropy by c.
// Produces soft bright interior ("silver lining") that single-scatter Beer's law misses.

float multiScatterEnergy(float opticalDepth, float cosTheta, uint octaves, float powderStr) {
    const float ISOTROPIC_PHASE = 1.0 / (4.0 * PI);
    const float A_DECAY = 0.2;
    const float B_DECAY = 0.5;
    const float C_DECAY = 0.5;

    float powderBlend = clamp(-cosTheta * 0.5 + 0.5, 0.0, 1.0) * powderStr;

    float energy = 0.0;
    float a = 1.0, b = 1.0, c = 1.0;

    for (uint n = 0; n < octaves; n++) {
        float phase = phaseDualHG(cosTheta, 0.8 * c, -0.3 * c, 0.2);
        phase = mix(ISOTROPIC_PHASE, phase, c);

        float od = opticalDepth * a;
        float beer = exp(-od);
        float powder = 1.0 - exp(-od * 2.0);
        float t = beer * mix(1.0, powder * 2.0, powderBlend);

        energy += b * phase * t;
        a *= A_DECAY;
        b *= B_DECAY;
        c *= C_DECAY;
    }
    return energy;
}

#endif // CLOUD_DENSITY_GLSL
