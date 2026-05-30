#ifndef ADV_LIGHT_GLSL
#define ADV_LIGHT_GLSL

#include "common/shared.hpp"
#include "common/chunk_lookup.glsl"
#include "common/constants.glsl"

#ifndef ADV_INITIAL_SAMPLES
#    define ADV_INITIAL_SAMPLES 1
#endif

struct Light {
    vec4 p0Area;
    vec4 p1;
    vec4 p2;
    vec4 p3;
    vec4 emission;
    vec4 sampleProb;
};

struct ChunkLightData {
    int x;
    int y;
    int z;
    uint geometryCount;
    uint lightCount;
    uint64_t lightBufferAddress;
};

struct ChunkPackedLight {
    vec4 p0Area;
    vec4 p1;
    vec4 p2;
    vec4 p3;
    vec4 normal;
    vec4 radiance;
    vec4 sourceIDData;
};

uvec2 chunkPackedLightSourceID(ChunkPackedLight chunkLight) {
    return uvec2(floatBitsToUint(chunkLight.sourceIDData.x), floatBitsToUint(chunkLight.sourceIDData.y));
}

layout(std430, set = 1, binding = 9) readonly buffer ChunkPackedDataBuffer {
    ChunkLightData chunkPackedData[];
};

layout(std430, buffer_reference, buffer_reference_align = 16) readonly buffer ChunkPackedLightBuffer {
    ChunkPackedLight lights[];
};

bool chunkPackedDataMatchesOrigin(ChunkLightData chunkData, ivec3 chunkOrigin) {
    return chunkData.x == chunkOrigin.x && chunkData.y == chunkOrigin.y && chunkData.z == chunkOrigin.z;
}

bool chunkPackedDataHasLights(ChunkLightData chunkData) {
    return chunkData.lightCount > 0u && chunkData.lightBufferAddress != 0ul;
}

#endif
