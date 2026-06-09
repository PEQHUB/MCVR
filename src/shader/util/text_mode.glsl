#ifndef TEXT_MODE_GLSL
#define TEXT_MODE_GLSL

const uint POST_TEXT_MODE_BACKGROUND = 1u;
const uint POST_TEXT_MODE_INTENSITY = 2u;
const uint POST_TEXT_MODE_RGBA = 3u;
const uint POST_TEXT_MODE_BACKGROUND_SEE_THROUGH = 4u;
const uint POST_TEXT_MODE_INTENSITY_SEE_THROUGH = 5u;
const uint POST_TEXT_MODE_RGBA_SEE_THROUGH = 6u;
const uint POST_TEXT_MODE_INTENSITY_POLYGON_OFFSET = 7u;
const uint POST_TEXT_MODE_RGBA_POLYGON_OFFSET = 8u;

bool isTextBackgroundMode(uint textMode) {
    return textMode == POST_TEXT_MODE_BACKGROUND || textMode == POST_TEXT_MODE_BACKGROUND_SEE_THROUGH;
}

bool isTextIntensityMode(uint textMode) {
    return textMode == POST_TEXT_MODE_INTENSITY || textMode == POST_TEXT_MODE_INTENSITY_SEE_THROUGH ||
           textMode == POST_TEXT_MODE_INTENSITY_POLYGON_OFFSET;
}

bool isTextRgbaMode(uint textMode) {
    return textMode == POST_TEXT_MODE_RGBA || textMode == POST_TEXT_MODE_RGBA_SEE_THROUGH ||
           textMode == POST_TEXT_MODE_RGBA_POLYGON_OFFSET;
}

vec4 resolveTextTextureColor(vec4 textureColor, bool useTexture, uint textMode) {
    if (isTextBackgroundMode(textMode)) { return vec4(1.0); }
    if (isTextIntensityMode(textMode)) { return textureColor.rrrr; }
    if (isTextRgbaMode(textMode)) { return textureColor; }
    return useTexture ? textureColor : vec4(1.0);
}

float resolveTextCoverage(vec4 textureColor, bool useTexture, float colorLayerAlpha, uint textMode) {
    return clamp(resolveTextTextureColor(textureColor, useTexture, textMode).a * colorLayerAlpha, 0.0, 1.0);
}

#endif
