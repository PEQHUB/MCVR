#ifndef BLUE_NOISE_GLSL
#define BLUE_NOISE_GLSL

// Blue Noise Sampling using FidelityFX Owen-scrambled Sobol Sequence
//
// Temporal variation: use frameIndex as the Sobol sampleIndex (0-255).
// Each frame gets a truly independent Owen-scrambled sample point per pixel,
// which is what DLSS-RR expects for temporal accumulation.

layout(set = 2, binding = 9) readonly buffer SobolBuffer {
    uint data[];
} sobolBuffer;

layout(set = 2, binding = 10) readonly buffer ScramblingBuffer {
    uint data[];
} scramblingBuffer;

// Core sample function: Owen-scrambled Sobol with spatial decorrelation.
// sampleIndex = frameIndex for temporal variation (0-255 range, wraps).
float blueNoiseSample(uvec2 pixel, uint sampleIndex, uint dimension) {
    uint px = pixel.x & 127u;
    uint py = pixel.y & 127u;
    uint idx = sampleIndex & 255u;
    uint dim = dimension & 255u;

    uint sobolIdx = dim + idx * 256u;
    uint value = sobolBuffer.data[sobolIdx];

    uint scramblingIdx = (dim % 8u) + (px + py * 128u) * 8u;
    value = value ^ scramblingBuffer.data[scramblingIdx];

    return (float(value) + 0.5) / 256.0;
}

// 2D sample using frameIndex as temporal Sobol index (dimensions 0-1).
vec2 blueNoise2D(uvec2 pixel, uint frameIndex) {
    float u1 = blueNoiseSample(pixel, frameIndex, 0u);
    float u2 = blueNoiseSample(pixel, frameIndex, 1u);
    return vec2(u1, u2);
}

// 2D sample using specific dimensions (for multiple independent 2D samples per pixel).
vec2 blueNoise2DEx(uvec2 pixel, uint frameIndex, uint dimensionOffset) {
    float u1 = blueNoiseSample(pixel, frameIndex, dimensionOffset);
    float u2 = blueNoiseSample(pixel, frameIndex, dimensionOffset + 1u);
    return vec2(u1, u2);
}

// 1D sample using specific dimension.
float blueNoise1D(uvec2 pixel, uint frameIndex, uint dimension) {
    return blueNoiseSample(pixel, frameIndex, dimension);
}

#endif // BLUE_NOISE_GLSL
