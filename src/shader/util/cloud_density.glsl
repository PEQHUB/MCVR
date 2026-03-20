#ifndef CLOUD_DENSITY_GLSL
#define CLOUD_DENSITY_GLSL

// Nubis cloud density and lighting model.
//
// Core equation [Nubis³ p24]:
//   cloud_density = saturate(noise_composite - (1.0 - dimensional_profile))
//
// Noise is the shape source; dimensional_profile sets the threshold.
// High profile + high noise = density. Low profile = sparse/no clouds.
//
// References:
//   [Nubis³]        Schneider, "Nubis, Cubed", SIGGRAPH 2023
//   [NubisEvolved]  Schneider, "Nubis, Evolved", SIGGRAPH 2022
//   [Wrenninge13]   Multi-scatter octave approximation
//   [Hillaire16]    Frostbite decay parameters
//
// Used by: cloud_raymarch.comp, cloud_shadow.comp

const float PI = 3.14159265358979;

// --- Utility ---

float cloudRemap(float value, float low1, float high1, float low2, float high2) {
    return low2 + (value - low1) / (high1 - low1) * (high2 - low2);
}

// --- Dimensional Profile [NubisEvolved p97, Nubis³ p29] ---
//
// Height profile × coverage. The profile controls cloud vertical extent
// per type; coverage controls what fraction of the noise produces density.
//
// type [0,1] controls vertical extent:
//   0 = stratus (thin, h=[0, ~0.25])
//   0.33 = stratocumulus (medium, h=[0, ~0.5])
//   0.67 = cumulus (tall, h=[0, ~0.8])
//   1.0 = cumulonimbus (full height)

// Height profile: vertical shape of the cloud. Peaks at ~1.0 in the cloud body,
// tapers at base (condensation level) and top (type-dependent).
// Coverage is NOT included here — it controls the threshold separately.
float heightProfile(float h, float type) {
    // Base: quick ramp from condensation level [NubisEvolved p97]
    float base = smoothstep(0.0, 0.1, h);

    // Top: type-dependent fade height
    float topFade = mix(0.25, 1.0, type);
    float top = smoothstep(topFade, topFade * 0.35, h);

    return base * top;
}

// --- Noise Composite [Nubis³ p99, p104, p108] ---
//
// Noise channels (from cloud_noise_gen.comp):
//   R = Perlin-Worley  (connected, flowing shapes)
//   G = Worley freq 4  (medium cellular)
//   B = Worley freq 8  (fine cellular)
//   A = Curl noise     (wispy tendrils)
//
// Wispy (type=0): PW base → curl detail. Connected flowing shapes.
// Billowy (type=1): PW base → Worley detail. Rounded cellular shapes.
// Both channels use full [0,1] range (no 0.3 scaling — that was calibrated
// for Nubis's Alligator noise, not our Worley).

float noiseComposite(vec4 noise, float hProfile, float type) {
    // Wispy: curl-dominated with PW connectivity [Nubis³ p99]
    // Curl provides thin flowing tendrils; PW adds large-scale connectivity.
    float wispy = noise.r * 0.4 + noise.a * 0.6;

    // Billowy: Worley-dominated with PW connectivity [Nubis³ p104]
    // Worley provides round puffy cells; PW fills gaps for connected shapes.
    // Height-dependent blend: more PW at base, more Worley at top [p104]
    float billowyGrad = pow(max(hProfile, 0.001), 0.25);
    float billowy = mix(noise.r * 0.5 + noise.g * 0.5,
                        noise.g * 0.7 + noise.b * 0.3, billowyGrad);

    // Type blend [Nubis³ p108]: stratus=wispy, cumulus/cb=billowy
    return mix(wispy, billowy, type);
}

// --- Main Cloud Density [Nubis³ p24, p118] ---
//
// Pipeline: dimProfile → noiseComposite → density threshold → detail erosion → sharpen
// Returns density; writes dimensional_profile for ambient scattering.

float cloudDensity(vec3 pos, float coverage, float type,
                   vec4 noise, float cloudBase, float cloudThickness,
                   float densityMul, float detailStr,
                   out float outDimProfile) {
    float h = (pos.y - cloudBase) / cloudThickness;
    if (h < 0.0 || h > 1.0) { outDimProfile = 0.0; return 0.0; }

    // Height profile: vertical cloud shape [NubisEvolved p97]
    float hProfile = heightProfile(h, type);
    if (hProfile < 0.001) { outDimProfile = 0.0; return 0.0; }

    // Dimensional profile for ambient scattering: height × coverage [Nubis³ p29]
    float dimProfile = hProfile * coverage;
    outDimProfile = dimProfile;

    // Noise composite: type blends wispy↔billowy [Nubis³ p99-108]
    float nc = noiseComposite(noise, hProfile, type);

    // Coverage threshold [Nubis³ p24, Schneider15 §3.3]:
    // Coverage DIRECTLY sets the noise threshold — NOT multiplied by height.
    // This ensures visible clouds at moderate coverage (0.3-0.5).
    // Height profile shapes the cloud AFTER thresholding.
    float threshold = 1.0 - coverage;
    float density = clamp(cloudRemap(nc, threshold, 1.0, 0.0, 1.0), 0.0, 1.0);
    density *= coverage;    // Energy conservation [Schneider15]
    density *= hProfile;    // Height shaping AFTER threshold — creates dome tops

    // Detail erosion: secondary noise breaks up edges [Nubis³ p104]
    float detailNoise = mix(noise.a, noise.b, type);
    density = max(density - detailNoise * 0.3 * detailStr, 0.0);

    // Density multiplier
    density *= densityMul;

    // Sharpening [Nubis³ p118]: pow pushes low densities toward zero
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

float phaseHG(float cosTheta, float g) {
    float g2 = g * g;
    float denom = 1.0 + g2 - 2.0 * g * cosTheta;
    return (1.0 - g2) / (4.0 * PI * denom * sqrt(denom));
}

float phaseDualHG(float cosTheta, float g1, float g2, float blend) {
    return mix(phaseHG(cosTheta, g1), phaseHG(cosTheta, g2), blend);
}

// --- Multi-scatter octave approximation [Wrenninge13, Hillaire16] ---

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
