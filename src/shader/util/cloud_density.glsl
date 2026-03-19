#ifndef CLOUD_DENSITY_GLSL
#define CLOUD_DENSITY_GLSL

// Shared density model for volumetric clouds.
// Used by cloud_raymarch.comp, cloud_shadow.comp, and cloud_weather.comp.

// Remap utility: maps value from [low1,high1] to [low2,high2]
float cloudRemap(float value, float low1, float high1, float low2, float high2) {
    return low2 + (value - low1) / (high1 - low1) * (high2 - low2);
}

// Height gradient for cumulus clouds (puffy, round)
float cumulusGradient(float h) {
    return smoothstep(0.0, 0.15, h) * smoothstep(1.0, 0.5, h);
}

// Height gradient for stratus clouds (flat, layered)
float stratusGradient(float h) {
    return smoothstep(0.0, 0.05, h) * smoothstep(1.0, 0.8, h);
}

// Detail erosion scale (sample B channel at higher frequency for turbulent edges)
const float DETAIL_NOISE_SCALE = 4.0;

// Main cloud density function (Schneider model with detail erosion).
// pos:       world-space position
// coverage:  base coverage [0,1] (from weather map)
// type:      0=cumulus, 1=stratus (blend)
// noise:     pre-sampled 64^3 noise texture (RGBA)
//            R = Perlin-Worley blend (base shape)
//            G = Worley F1 at 2x freq (medium erosion)
//            B = Worley F1 at 4x freq (fine detail)
//            A = value noise (wind distortion)
// cloudBase: bottom of cloud layer (world Y)
// cloudThickness: height of cloud layer
// densityMul: density multiplier
float cloudDensity(vec3 pos, float coverage, float type,
                   vec4 noise, float cloudBase, float cloudThickness,
                   float densityMul) {
    // Normalized height within cloud layer [0,1]
    float h = (pos.y - cloudBase) / cloudThickness;
    if (h < 0.0 || h > 1.0) return 0.0;

    // Height gradient: blend cumulus/stratus by type
    float grad = mix(cumulusGradient(h), stratusGradient(h), type);

    // Shape from base noise (R channel = Perlin-Worley blend)
    float shape = cloudRemap(noise.r, 1.0 - coverage, 1.0, 0.0, 1.0) * grad;
    if (shape <= 0.0) return 0.0;

    // Medium erosion from G channel (Worley F1 at 2x freq)
    float erode = noise.g * 0.25 * (1.0 - shape);

    // Fine detail erosion from B channel (Worley F1 at 4x freq)
    // Stronger at cloud edges (where shape is small), weaker in core
    float edgeFactor = 1.0 - smoothstep(0.0, 0.5, shape);
    float detail = noise.b * 0.15 * edgeFactor;

    // Height-dependent detail: more breakup at cloud base and top
    float heightDetail = 1.0 - smoothstep(0.0, 0.3, abs(h - 0.5) * 2.0 - 0.4);
    detail *= mix(1.0, 0.5, heightDetail);

    return max(shape - erode - detail, 0.0) * densityMul;
}

// Beer-Lambert transmittance
vec3 beerTransmittance(float opticalDepth, vec3 extinction) {
    return exp(-extinction * opticalDepth);
}

// Beer-Powder approximation for multi-scatter (Schneider 2015)
float beerPowder(float opticalDepth) {
    float beer = exp(-opticalDepth);
    float powder = 1.0 - exp(-opticalDepth * 2.0);
    return beer * mix(1.0, powder, 0.5);
}

// Henyey-Greenstein phase function
float phaseHG(float cosTheta, float g) {
    float g2 = g * g;
    float denom = 1.0 + g2 - 2.0 * g * cosTheta;
    return (1.0 - g2) / (4.0 * 3.14159265 * denom * sqrt(denom));
}

// Dual-lobe HG phase (forward + back scatter blend)
float phaseDualHG(float cosTheta, float g1, float g2, float blend) {
    return mix(phaseHG(cosTheta, g1), phaseHG(cosTheta, g2), blend);
}

#endif // CLOUD_DENSITY_GLSL
