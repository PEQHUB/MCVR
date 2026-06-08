#version 460

layout(set = 0, binding = 2, r16f) uniform readonly image2D firstHitDepthImage;

void main() {
    ivec2 pixelCoord = ivec2(gl_FragCoord.xy);
    float linearDepth = imageLoad(firstHitDepthImage, pixelCoord).r;

    gl_FragDepth = clamp(linearDepth / 1000.0, 0.0, 1.0); // length of first ray is 1000
}
