#ifndef COLOR_SPACE_GLSL
#define COLOR_SPACE_GLSL

vec3 srgbToLinear(vec3 srgbColor) {
    vec3 low = srgbColor / 12.92;
    vec3 high = pow((srgbColor + 0.055) / 1.055, vec3(2.4));
    bvec3 useLow = lessThanEqual(srgbColor, vec3(0.04045));
    return mix(high, low, useLow);
}

vec4 srgbToLinear(vec4 srgbColor) {
    return vec4(srgbToLinear(srgbColor.rgb), srgbColor.a);
}

vec4 applySrgbToLinear(vec4 sampledColor, bool isSRGB) {
    if (isSRGB) { return srgbToLinear(sampledColor); }
    return sampledColor;
}

vec4 sampleTexture(sampler2D tex, vec2 uv, bool isSRGB) {
    return applySrgbToLinear(texture(tex, uv), isSRGB);
}

vec4 sampleTexture(sampler2D tex, vec2 uv, float lod, bool isSRGB) {
    return applySrgbToLinear(textureLod(tex, uv, lod), isSRGB);
}

vec4 sampleTexture(sampler2D tex, ivec2 coord, int lod, bool isSRGB) {
    return applySrgbToLinear(texelFetch(tex, coord, lod), isSRGB);
}

vec4 sampleTexture(samplerCube tex, vec3 dir, bool isSRGB) {
    return applySrgbToLinear(texture(tex, dir), isSRGB);
}

#endif
