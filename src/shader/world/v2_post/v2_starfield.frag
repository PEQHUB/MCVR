#version 460

// Star field fragment shader (V2 engine).
// Stars are rendered with additive blending on top of the tone-mapped image.
// Depth is fixed at 0.95 (far, behind everything except sky).
//
// Discards fragments if:
//   - skyType != 1 (not overworld — no stars in nether/end)
//   - visibility < 0.001 (daytime)
// Attenuates brightness by rain gradient.

layout(location = 0) in vec4 inColor;
layout(location = 1) in float inRainGradient;
layout(location = 2) flat in uint inSkyType;

layout(location = 0) out vec4 fragColor;

void main() {
    // Only render stars in the overworld (skyType == 1)
    if (inSkyType != 1u) discard;

    // Discard invisible stars (alpha from visibility modulation)
    if (inColor.a < 0.001) discard;

    // Rain attenuation: stars fade as rain increases
    float rainFade = 1.0 - inRainGradient;
    if (rainFade < 0.001) discard;

    // Base color with floor brightness for dim stars
    vec3 color = max(inColor.rgb, vec3(0.10 * inColor.a));

    // Apply rain attenuation
    color *= rainFade;

    // Brightness boost (matches V1 convention)
    color *= 1.40;

    fragColor = vec4(color, 1.0);

    // Fixed far depth — behind all world geometry, in front of sky
    gl_FragDepth = 0.95;
}
