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

// --- 4 Cloud Type Height Gradients [Schneider15 §3.1] ---
// Trapezoid profiles defining density vs normalized height [0,1].
// Ordered by increasing meteorological altitude:
//   0 = stratus (low, flat)
//   1 = stratocumulus (low-mid, lumpy)
//   2 = cumulus (mid, billowing towers)
//   3 = cumulonimbus (full-height storm towers)

// Stratus: thin flat layer, very low (fog-like ceiling)
float stratusGradient(float h) {
    return smoothstep(0.0, 0.07, h) * smoothstep(0.15, 0.08, h);
}

// Stratocumulus: moderate lumpy layer (overcast with texture)
float stratocumulusGradient(float h) {
    return smoothstep(0.0, 0.2, h) * smoothstep(0.6, 0.42, h);
}

// Cumulus: tall billowing towers (fair weather puffy clouds)
float cumulusGradient(float h) {
    return smoothstep(0.0, 0.08, h) * smoothstep(0.98, 0.75, h);
}

// Cumulonimbus: full-height storm towers (anvil-shaped thunderheads)
float cumulonimbusGradient(float h) {
    return smoothstep(0.0, 0.10, h) * smoothstep(1.0, 0.7, h);
}

// Evaluate height gradient by interpolating between 4 cloud types.
// type [0,1] maps: 0=stratus → 0.33=stratocumulus → 0.67=cumulus → 1.0=cumulonimbus
// This ordering follows meteorological altitude (low → high).
float cloudHeightGradient(float h, float type) {
    float t = type * 3.0;
    int lo = int(floor(t));
    float blend = fract(t);

    float g0, g1;
    if (lo == 0) {
        g0 = stratusGradient(h);
        g1 = stratocumulusGradient(h);
    } else if (lo == 1) {
        g0 = stratocumulusGradient(h);
        g1 = cumulusGradient(h);
    } else {
        g0 = cumulusGradient(h);
        g1 = cumulonimbusGradient(h);
    }
    return mix(g0, g1, blend);
}

// Main cloud density function. [Schneider15 §3.2]
//
// noise: 128^3 RGBA8 texture sample (see cloud_noise_gen.comp for channel layout)
//   R = Perlin-Worley (base shape — connected yet billowy)
//   G = Worley FBM {2,4,8} (medium erosion)
//   B = Worley FBM {4,8,16} (fine erosion)
//   A = Worley FBM {8,16,32} (micro detail)
float cloudDensity(vec3 pos, float coverage, float type,
                   vec4 noise, float cloudBase, float cloudThickness,
                   float densityMul, float detailStr) {
    // Normalized height within cloud layer [0,1]
    float h = (pos.y - cloudBase) / cloudThickness;
    if (h < 0.0 || h > 1.0) return 0.0;

    // 4-type height gradient [Schneider15 §3.1]
    float grad = cloudHeightGradient(h, type);

    // --- Schneider base shape [Schneider15 §3.2] ---
    // Build low-frequency FBM from G,B,A channels for base erosion
    // Weights: 2^-1 = 0.625 (normalized), 2^-2 = 0.25, 2^-3 = 0.125
    float lowFreqFBM = noise.g * 0.625 + noise.b * 0.25 + noise.a * 0.125;

    // Erode base shape with low-freq FBM (Schneider remap trick)
    float baseShape = cloudRemap(noise.r, lowFreqFBM - 1.0, 1.0, 0.0, 1.0);
    baseShape = max(baseShape, 0.0);

    // Apply height gradient
    baseShape *= grad;

    // --- Schneider coverage remap with coverage^2 boost [Schneider15 §3.3] ---
    // Threshold by coverage: low coverage → small isolated clouds, high → continuous
    float coverageShape = cloudRemap(baseShape, 1.0 - coverage, 1.0, 0.0, 1.0);
    coverageShape *= coverage;
    if (coverageShape <= 0.0) return 0.0;

    // --- Height-inverted erosion [Schneider15 §3.2] ---
    // Wispy/shredded at cloud base, billowy/puffy at top.
    // smoothstep(0,0.3,h): gradual transition across bottom 30% of cloud layer.
    // At base (h≈0): use inverted detail (1 - worley) → turbulent breakup
    // At top (h≈1): use direct detail → smooth rounded billows
    float heightFraction = smoothstep(0.0, 0.3, h);

    // Build detail erosion from B + A channels (fine + micro)
    float detailFBM = noise.b * 0.5 + noise.a * 0.5;
    float erosion = mix(1.0 - detailFBM, detailFBM, heightFraction);

    // Erode edges: detail only affects low-density regions (preserves dense core)
    float finalDensity = cloudRemap(coverageShape, erosion * 0.2 * detailStr, 1.0, 0.0, 1.0);

    return max(finalDensity, 0.0) * densityMul;
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
