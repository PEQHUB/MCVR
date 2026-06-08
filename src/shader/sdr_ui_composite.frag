#version 460

// SDR world/UI composite.
//
// World image (binding 0) is the tone-mapped SDR output stored in UNORM.
// Those values are already display-referred and are treated as sRGB-encoded
// when blending (matching the previous SDR path which copied into an sRGB
// overlay render target before UI blending).
//
// UI overlay (binding 1) is R8G8B8A8_SRGB; the sampler automatically decodes
// to linear.

layout(set = 0, binding = 0) uniform sampler2D worldLDR;
layout(set = 0, binding = 1) uniform sampler2D overlayUI;

layout(push_constant) uniform PC {
    float uiBrightnessNits; // unused in SDR
} pc;

layout(location = 0) in vec2 texCoord;
layout(location = 0) out vec4 fragColor;

// sRGB EOTF (IEC 61966-2-1)
vec3 srgbToLinear(vec3 c) {
    vec3 lo = c / 12.92;
    vec3 hi = pow((c + 0.055) / 1.055, vec3(2.4));
    return mix(lo, hi, step(vec3(0.04045), c));
}

// sRGB OETF (IEC 61966-2-1)
vec3 linearToSrgb(vec3 c) {
    vec3 lo = c * 12.92;
    vec3 hi = 1.055 * pow(c, vec3(1.0 / 2.4)) - 0.055;
    return mix(lo, hi, step(vec3(0.0031308), c));
}

void main() {
    vec4 ui = texture(overlayUI, texCoord); // linearized by SRGB sampler
    float a = ui.a;

    // Fast path: no UI pixel, passthrough world (already encoded for swapchain).
    vec3 worldEncoded = texture(worldLDR, texCoord).rgb;
    if (a < 0.001) {
        fragColor = vec4(worldEncoded, 1.0);
        return;
    }

    // Blend in linear light (straight alpha).
    vec3 worldLinear = srgbToLinear(clamp(worldEncoded, 0.0, 1.0));
    vec3 uiLinear = clamp(ui.rgb, 0.0, 1.0);
    vec3 outLinear = mix(worldLinear, uiLinear, a);

    // Encode to sRGB before writing into UNORM swapchain.
    vec3 outEncoded = linearToSrgb(clamp(outLinear, 0.0, 1.0));
    fragColor = vec4(outEncoded, 1.0);
}
