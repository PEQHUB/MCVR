#ifndef BLUE_NOISE_GLSL
#define BLUE_NOISE_GLSL

// Blue Noise Sampling using FidelityFX Owen-scrambled Sobol Sequence
//
// This implementation uses precomputed data from FidelityFX SDK:
// - Sobol sequence: 256 samples x 256 dimensions
// - Scrambling tile: 128x128 pixels x 8 dimensions
//
// The algorithm provides excellent spatial distribution for denoising.

layout(set = 1, binding = 9) readonly buffer SobolBuffer {
    uint data[];
} sobolBuffer;

layout(set = 1, binding = 10) readonly buffer ScramblingBuffer {
    uint data[];
} scramblingBuffer;

// FFX Blue Noise Sample
// Returns a value in [0, 1) using Owen-scrambled Sobol sequence with spatial decorrelation
//
// pixel: screen pixel coordinate
// sampleIndex: sample index (0-255)
// dimension: dimension index (0-255)
float blueNoiseSample(uvec2 pixel, uint sampleIndex, uint dimension) {
    // Wrap indices to valid ranges
    uint px = pixel.x & 127u;       // 0-127
    uint py = pixel.y & 127u;       // 0-127
    uint idx = sampleIndex & 255u;  // 0-255
    uint dim = dimension & 255u;    // 0-255

    // Fetch Sobol sequence value
    // Layout: sobol_256spp_256d[dimension + sampleIndex * 256]
    uint sobolIdx = dim + idx * 256u;
    uint value = sobolBuffer.data[sobolIdx];

    // Apply Owen scrambling using per-pixel tile
    // Layout: scramblingTile[(dimension % 8) + (px + py * 128) * 8]
    uint scramblingIdx = (dim % 8u) + (px + py * 128u) * 8u;
    value = value ^ scramblingBuffer.data[scramblingIdx];

    // Convert to [0, 1) with proper normalization
    // Add 0.5 to center the value in its bin
    return (float(value) + 0.5) / 256.0;
}

// Generate 2D blue noise sample with temporal golden ratio offset
// This provides smooth animation between frames while maintaining spatial quality
//
// pixel: screen pixel coordinate
// frameIndex: current frame number for temporal variation
vec2 blueNoise2D(uvec2 pixel, uint frameIndex) {
    // Sample two orthogonal dimensions
    float u1 = blueNoiseSample(pixel, 0u, 0u);
    float u2 = blueNoiseSample(pixel, 0u, 1u);

    // Apply golden ratio temporal offset for smooth frame-to-frame variation
    // This provides low-discrepancy temporal sequence
    const float GOLDEN_RATIO = 1.61803398875;
    float offset = float(frameIndex & 255u) * GOLDEN_RATIO;

    return fract(vec2(u1, u2) + offset);
}

// Generate 2D blue noise sample using specific dimensions
// Use this when you need multiple independent 2D samples per pixel
//
// pixel: screen pixel coordinate
// frameIndex: current frame number
// dimensionOffset: starting dimension (0, 2, 4, etc. for pairs)
vec2 blueNoise2DEx(uvec2 pixel, uint frameIndex, uint dimensionOffset) {
    float u1 = blueNoiseSample(pixel, 0u, dimensionOffset);
    float u2 = blueNoiseSample(pixel, 0u, dimensionOffset + 1u);

    const float GOLDEN_RATIO = 1.61803398875;
    float offset = float(frameIndex & 255u) * GOLDEN_RATIO;

    return fract(vec2(u1, u2) + offset);
}

// Generate a single blue noise value with temporal variation
//
// pixel: screen pixel coordinate
// frameIndex: current frame number
// dimension: which dimension to sample
float blueNoise1D(uvec2 pixel, uint frameIndex, uint dimension) {
    float u = blueNoiseSample(pixel, 0u, dimension);

    const float GOLDEN_RATIO = 1.61803398875;
    float offset = float(frameIndex & 255u) * GOLDEN_RATIO;

    return fract(u + offset);
}

#endif // BLUE_NOISE_GLSL
