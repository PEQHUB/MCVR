#version 460

// Reads the RT first-hit-depth storage image (R32F, linear distance)
// and writes gl_FragDepth to produce a proper D32_SFLOAT depth attachment.
// This depth buffer is consumed by entity post-render rasterization (C2)
// and star field rendering (C3) for correct depth testing.

layout(set = 0, binding = 0, r32f) uniform readonly image2D firstHitDepthImage;

void main() {
    ivec2 pixelCoord = ivec2(gl_FragCoord.xy);
    float linearDepth = imageLoad(firstHitDepthImage, pixelCoord).r;

    // Convert linear depth (ray hit distance) to normalized [0,1] range.
    // 1000.0 = max primary ray length (consistent with V1 convention).
    gl_FragDepth = clamp(linearDepth / 1000.0, 0.0, 1.0);
}
