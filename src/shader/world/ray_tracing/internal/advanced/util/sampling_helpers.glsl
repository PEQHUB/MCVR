#ifndef SAMPLING_HELPERS_GLSL
#define SAMPLING_HELPERS_GLSL

#include "disney.glsl"
#include "color_space.glsl"

ivec2 clampTexelCoord(ivec2 coord, ivec2 size) {
    return clamp(coord, ivec2(0), max(size - ivec2(1), ivec2(0)));
}

vec2 clampUvToRect(vec2 uv, vec2 uvMin, vec2 uvMax, ivec2 size) {
    vec2 minUv = min(uvMin, uvMax);
    vec2 maxUv = max(uvMin, uvMax);
    vec2 halfTexel = 0.5 / vec2(size);

    minUv += halfTexel;
    maxUv -= halfTexel;
    maxUv = max(maxUv, minUv);

    return clamp(uv, minUv, maxUv);
}

vec4 sampleNearest(sampler2D tex, vec2 uv, int lod, bool isSRGB) {
    ivec2 size = textureSize(tex, lod);
    if (size.x <= 0 || size.y <= 0) { return vec4(0.0); }
    ivec2 texel = ivec2(uv * vec2(size));
    return sampleTexture(tex, clampTexelCoord(texel, size), lod, isSRGB);
}

vec4 sampleBilinear(sampler2D tex, vec2 uv, int lod, bool isSRGB) {
    ivec2 size = textureSize(tex, lod);
    if (size.x <= 0 || size.y <= 0) { return vec4(0.0); }

    vec2 pixelCoord = uv * vec2(size) - vec2(0.5);
    ivec2 p0 = ivec2(floor(pixelCoord));
    ivec2 p1 = p0 + ivec2(1);
    vec2 fracPart = fract(pixelCoord);

    ivec2 t00 = clampTexelCoord(ivec2(p0.x, p0.y), size);
    ivec2 t10 = clampTexelCoord(ivec2(p1.x, p0.y), size);
    ivec2 t01 = clampTexelCoord(ivec2(p0.x, p1.y), size);
    ivec2 t11 = clampTexelCoord(ivec2(p1.x, p1.y), size);

    vec4 c00 = sampleTexture(tex, t00, lod, isSRGB);
    vec4 c10 = sampleTexture(tex, t10, lod, isSRGB);
    vec4 c01 = sampleTexture(tex, t01, lod, isSRGB);
    vec4 c11 = sampleTexture(tex, t11, lod, isSRGB);

    vec4 c0 = mix(c00, c10, fracPart.x);
    vec4 c1 = mix(c01, c11, fracPart.x);
    return mix(c0, c1, fracPart.y);
}

vec4 samplePBRTexture(sampler2D tex,
                      vec2 uv,
                      vec2 atlasUvMin,
                      vec2 atlasUvMax,
                      float lod,
                      uint samplingMode) {
    int lodLevel = clamp(int(floor(lod)), 0, max(textureQueryLevels(tex) - 1, 0));
    ivec2 size = textureSize(tex, lodLevel);
    if (size.x <= 0 || size.y <= 0) { return vec4(0.0); }

    vec2 clampedUv = clampUvToRect(uv, atlasUvMin, atlasUvMax, size);
    if (samplingMode == 0u) {
        return sampleNearest(tex, clampedUv, lodLevel, false);
    } else {
        return sampleBilinear(tex, clampedUv, lodLevel, false);
    }
}

#endif
