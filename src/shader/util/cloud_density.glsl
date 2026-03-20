#ifndef CLOUD_DENSITY_GLSL
#define CLOUD_DENSITY_GLSL

// Physically-based cloud density model.
//
// References:
//   [Schneider15] §3: 4 cloud type gradients, Perlin-Worley base shape,
//                 height-inverted erosion, coverage remap with coverage^2 boost
//   [Wrenninge13] §4: Multi-scatter octave approximation
//   [Hillaire16]  §4.6.3: Frostbite decay parameters (a=0.2, b=0.5, c=0.5)
//
// Used by: cloud_raymarch.comp, cloud_shadow.comp

const float PI = 3.14159265358979;

// Remap utility: maps value from [low1,high1] to [low2,high2]
float cloudRemap(float value, float low1, float high1, float low2, float high2) {
    return low2 + (value - low1) / (high1 - low1) * (high2 - low2);
}

// --- Two-Step Height Gradients [Schneider15 §3.1, Nubis³] ---
//
// Split into ENVELOPE (wide plateau) and DOME (shape modulation):
//   Envelope: applied BEFORE coverage remap — keeps baseShape above threshold
//             across most of the cloud height, preventing flat-top clipping.
//   Dome:     applied AFTER coverage remap — sculpts rounded tops without
//             creating hard density cutoffs at the remap threshold.
//
// type [0,1] maps: 0=stratus → 0.33=stratocumulus → 0.67=cumulus → 1.0=cumulonimbus

// --- Envelope gradients (wide plateaus, survive remap threshold) ---

float stratusEnvelope(float h) {
    return smoothstep(0.0, 0.05, h) * smoothstep(0.25, 0.15, h);
}

float stratocumulusEnvelope(float h) {
    return smoothstep(0.0, 0.10, h) * smoothstep(0.65, 0.45, h);
}

float cumulusEnvelope(float h) {
    return smoothstep(0.0, 0.07, h) * smoothstep(0.95, 0.75, h);
}

float cumulonimbusEnvelope(float h) {
    return smoothstep(0.0, 0.05, h) * smoothstep(1.0, 0.85, h);
}

// --- Dome gradients (shape modulation, applied after remap) ---

float stratusDome(float h) {
    return 1.0; // Flat — shape comes entirely from tight envelope
}

float stratocumulusDome(float h) {
    return 1.0 - 0.6 * pow(smoothstep(0.10, 0.55, h), 1.5);
}

float cumulusDome(float h) {
    return 1.0 - pow(smoothstep(0.25, 0.95, h), 2.0);
}

float cumulonimbusDome(float h) {
    // Dense column to 0.7, anvil flare above
    float column = 1.0 - 0.3 * smoothstep(0.3, 0.7, h);
    float anvil = smoothstep(0.65, 0.8, h) * 0.4;
    return column + anvil;
}

// Interpolate envelope across 4 cloud types
float cloudEnvelope(float h, float type) {
    float t = type * 3.0;
    int lo = int(floor(t));
    float blend = fract(t);

    float g0, g1;
    if (lo == 0) {
        g0 = stratusEnvelope(h);
        g1 = stratocumulusEnvelope(h);
    } else if (lo == 1) {
        g0 = stratocumulusEnvelope(h);
        g1 = cumulusEnvelope(h);
    } else {
        g0 = cumulusEnvelope(h);
        g1 = cumulonimbusEnvelope(h);
    }
    return mix(g0, g1, blend);
}

// Interpolate dome across 4 cloud types
float cloudDome(float h, float type) {
    float t = type * 3.0;
    int lo = int(floor(t));
    float blend = fract(t);

    float g0, g1;
    if (lo == 0) {
        g0 = stratusDome(h);
        g1 = stratocumulusDome(h);
    } else if (lo == 1) {
        g0 = stratocumulusDome(h);
        g1 = cumulusDome(h);
    } else {
        g0 = cumulusDome(h);
        g1 = cumulonimbusDome(h);
    }
    return mix(g0, g1, blend);
}

// Main cloud density function. [Schneider15 §3.2]
//
// noise: 128^3 RGBA8 texture sample (see cloud_noise_gen.comp for channel layout)
//   R = Perlin-Worley (base shape — connected yet billowy)
//   G = Single-octave Worley freq 4 (medium erosion, sharp cell edges)
//   B = Single-octave Worley freq 8 (fine erosion)
//   A = Curl noise (wispy tendrils for cloud edges)
float cloudDensity(vec3 pos, float coverage, float type,
                   vec4 noise, float cloudBase, float cloudThickness,
                   float densityMul, float detailStr) {
    // Normalized height within cloud layer [0,1]
    float h = (pos.y - cloudBase) / cloudThickness;
    if (h < 0.0 || h > 1.0) return 0.0;

    // --- Base shape [Schneider15 §3.2] ---
    // Multi-frequency erosion: Schneider 3-octave Worley FBM
    float baseErosion = noise.g * 0.625 + noise.b * 0.25 + noise.a * 0.125;

    // Erode base shape: positive threshold creates actual holes (Schneider remap trick)
    float baseShape = cloudRemap(noise.r, baseErosion * 0.5, 1.0, 0.0, 1.0);
    baseShape = max(baseShape, 0.0);

    // Envelope: wide plateau that keeps baseShape above remap threshold.
    // Applied BEFORE remap so the cloud extends to full height without flat-top clipping.
    float envelope = cloudEnvelope(h, type);
    baseShape *= envelope;

    // --- Schneider coverage remap [Schneider15 §3.3] ---
    float threshold = 1.0 - sqrt(coverage);
    float coverageShape = cloudRemap(baseShape, threshold, 1.0, 0.0, 1.0);
    coverageShape *= coverage;               // Energy conservation [Schneider15]

    // Dome: sculpts rounded tops AFTER remap — no hard density cutoffs.
    float dome = cloudDome(h, type);
    coverageShape *= dome;
    if (coverageShape <= 0.0) return 0.0;

    // --- Dual-character detail erosion [Nubis³] ---
    // Height-dependent transition: wispy at base, billowy at top.
    float heightFraction = smoothstep(0.0, 0.5, h);

    // Wispy (curl noise) vs billowy (single-octave Worley)
    float wispy = noise.a;                            // Curl noise: tendrils
    float billowy = noise.b;                          // Fine Worley only (G consumed by base erosion)
    float detailNoise = mix(wispy, billowy, heightFraction);

    // Height-inverted erosion: turbulent breakup at base, smooth billows at top
    float erosion = mix(1.0 - detailNoise, detailNoise, heightFraction);

    // Erode edges: 50% erosion strength (Schneider/Nubis use 50-80%)
    float finalDensity = cloudRemap(coverageShape, erosion * 0.5 * detailStr, 1.0, 0.0, 1.0);

    // Density sharpening — pushes low densities toward zero for crisp edges [Nubis³]
    finalDensity = max(finalDensity, 0.0) * densityMul;
    float sharp = clamp(pc.sharpening, 0.1, 1.0);
    return pow(finalDensity, sharp);
}

// Beer-Lambert transmittance
vec3 beerTransmittance(float opticalDepth, vec3 extinction) {
    return exp(-extinction * opticalDepth);
}

// Henyey-Greenstein phase function [Henyey41]
// g ∈ (-1,1): positive = forward scatter, negative = back scatter, 0 = isotropic
float phaseHG(float cosTheta, float g) {
    float g2 = g * g;
    float denom = 1.0 + g2 - 2.0 * g * cosTheta;
    return (1.0 - g2) / (4.0 * PI * denom * sqrt(denom));
}

// Dual-lobe HG phase (forward + back scatter blend)
// g1 > 0 (forward), g2 < 0 (back), blend ∈ [0,1] weight on back lobe
float phaseDualHG(float cosTheta, float g1, float g2, float blend) {
    return mix(phaseHG(cosTheta, g1), phaseHG(cosTheta, g2), blend);
}

// --- Multi-scatter octave approximation [Wrenninge13 §4, Hillaire16 §4.6.3] ---
//
// Each scattering bounce through the cloud volume:
//   - Sees less extinction (photons that bounced already travel through
//     sparser effective medium) → attenuation decays by factor 'a'
//   - Contributes less energy → contribution decays by factor 'b'
//   - Becomes more isotropic (phase → uniform) → eccentricity decays by 'c'
//
// Decay parameters per Frostbite [Hillaire16]:
//   a = 0.2 (extinction multiplier — fast decay, bounced light escapes easily)
//   b = 0.5 (contribution multiplier — halved per bounce)
//   c = 0.5 (eccentricity multiplier — phase approaches isotropic)
//
// These produce the characteristic soft, bright interior ("silver lining")
// that single-scatter Beer's law cannot achieve.
float multiScatterEnergy(float opticalDepth, float cosTheta, uint octaves, float powderStr) {
    // Isotropic phase (uniform sphere): 1/(4π)
    const float ISOTROPIC_PHASE = 1.0 / (4.0 * PI);

    // Frostbite decay parameters [Hillaire16 §4.6.3]
    const float A_DECAY = 0.2;  // extinction per bounce
    const float B_DECAY = 0.5;  // contribution per bounce
    const float C_DECAY = 0.5;  // eccentricity per bounce

    // Beer-powder balance: dark edges when sun is behind viewer,
    // bright silver lining when facing sun [Schneider15 §3.4, Nubis³ p136]
    // cosTheta > 0 = looking toward sun → less powder (silver lining)
    // cosTheta < 0 = sun behind camera → more powder (dark edges)
    float powderBlend = clamp(-cosTheta * 0.5 + 0.5, 0.0, 1.0) * powderStr;

    float energy = 0.0;
    float a = 1.0;  // attenuation multiplier (decreases per bounce)
    float b = 1.0;  // contribution multiplier (decreases per bounce)
    float c = 1.0;  // eccentricity multiplier (phase → isotropic)

    for (uint n = 0; n < octaves; n++) {
        // Phase interpolates from anisotropic (dual-lobe HG) toward isotropic
        // as eccentricity c decays. g parameters scale with c:
        //   n=0: g1=0.8, g2=-0.3 (strong forward + weak back lobe)
        //   n=1: g1=0.4, g2=-0.15 (reduced anisotropy)
        //   n→∞: g1→0, g2→0 (isotropic)
        float phase = phaseDualHG(cosTheta, 0.8 * c, -0.3 * c, 0.2);
        // Blend toward isotropic as c decays
        phase = mix(ISOTROPIC_PHASE, phase, c);

        // Beer-powder: replaces plain Beer's law exp(-od) [Schneider15]
        // powder ≈ 0 at cloud surface (od→0) → dark edge
        // powder ≈ 1 deep inside → normal Beer attenuation
        float od = opticalDepth * a;
        float beer = exp(-od);
        float powder = 1.0 - exp(-od * 2.0);
        // Blend: full powder for back-lit, pure beer for forward-lit
        float transmittance = beer * mix(1.0, powder * 2.0, powderBlend);

        energy += b * phase * transmittance;
        a *= A_DECAY;
        b *= B_DECAY;
        c *= C_DECAY;
    }
    return energy;
}

#endif // CLOUD_DENSITY_GLSL
