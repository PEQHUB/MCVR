#ifndef ADV_RUNTIME_TEXTURE_ARRAYS_GLSL
#define ADV_RUNTIME_TEXTURE_ARRAYS_GLSL

ivec3 rtArrayCoord(ivec2 pixel, int layer) {
    return ivec3(pixel, layer);
}

layout(set = 5, binding = 8, rgba32f) uniform image2DArray restirLightReservoirImage;
layout(set = 5, binding = 9, rgba32ui) uniform uimage2DArray restirLightReservoirSourceImage;
layout(set = 5, binding = 10, rgba32f) uniform image2DArray restirTemporalHistoryImage;
layout(set = 5, binding = 30, rgba16f) uniform image2DArray volumetricLightHistoryImage;
layout(set = 5, binding = 35, rgba16f) uniform image2DArray volumetricCloudHistoryImage;

vec4 loadRestirLightReservoirLayer(ivec2 pixel, bool pingSet, int layer) {
    return imageLoad(restirLightReservoirImage, rtArrayCoord(pixel, pingSet ? layer : (3 + layer)));
}

void storeRestirLightReservoirLayer(ivec2 pixel, bool pingSet, int layer, vec4 value) {
    imageStore(restirLightReservoirImage, rtArrayCoord(pixel, pingSet ? layer : (3 + layer)), value);
}

uvec4 loadRestirLightReservoirSource(ivec2 pixel, bool pingSet) {
    return imageLoad(restirLightReservoirSourceImage, rtArrayCoord(pixel, pingSet ? 0 : 1));
}

void storeRestirLightReservoirSource(ivec2 pixel, bool pingSet, uvec4 value) {
    imageStore(restirLightReservoirSourceImage, rtArrayCoord(pixel, pingSet ? 0 : 1), value);
}

vec4 loadRestirTemporalHistoryLayer(ivec2 pixel, bool pingSet, int layer) {
    return imageLoad(restirTemporalHistoryImage, rtArrayCoord(pixel, pingSet ? layer : (2 + layer)));
}

void storeRestirTemporalHistoryLayer(ivec2 pixel, bool pingSet, int layer, vec4 value) {
    imageStore(restirTemporalHistoryImage, rtArrayCoord(pixel, pingSet ? layer : (2 + layer)), value);
}

vec4 loadVolumetricCloudHistory(ivec2 pixel, int layer) {
    return imageLoad(volumetricCloudHistoryImage, rtArrayCoord(pixel, layer));
}

void storeVolumetricCloudHistory(ivec2 pixel, int layer, vec4 value) {
    imageStore(volumetricCloudHistoryImage, rtArrayCoord(pixel, layer), value);
}

#endif
