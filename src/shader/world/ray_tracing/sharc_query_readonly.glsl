// Tiny read-only SHARC query helper for world.rgen.
// This intentionally avoids SHARC update/resolve SDK includes and 64-bit atomics.
//
// SHARC_MAIN_TRACE_QUERY_MODE is used to bisect NVIDIA driver compiler crashes:
//   1 = stub call site only
//   2 = hash math, no buffer-reference reads
//   3 = hash-entry BDA read, no resolved radiance read
//   4 = full read-only radiance query

#ifndef RADIANCE_SHARC_QUERY_READONLY_GLSL
#define RADIANCE_SHARC_QUERY_READONLY_GLSL

#ifndef SHARC_MAIN_TRACE_QUERY_MODE
#define SHARC_MAIN_TRACE_QUERY_MODE 4
#endif

#define RADIANCE_SHARC_POSITION_BIT_NUM 17
#define RADIANCE_SHARC_POSITION_BIT_MASK ((1u << RADIANCE_SHARC_POSITION_BIT_NUM) - 1)
#define RADIANCE_SHARC_LEVEL_BIT_NUM 10
#define RADIANCE_SHARC_LEVEL_BIT_MASK ((1u << RADIANCE_SHARC_LEVEL_BIT_NUM) - 1)
#define RADIANCE_SHARC_BUCKET_SIZE 16
#define RADIANCE_SHARC_POSITION_BIAS 1e-4
#define RADIANCE_SHARC_NORMAL_BIAS 1e-3
#define RADIANCE_SHARC_GRID_LOGARITHM_BASE 2.0
#define RADIANCE_SHARC_GRID_LEVEL_BIAS 0.0
#define RADIANCE_SHARC_SAMPLE_NUM_THRESHOLD 0.0

#if SHARC_MAIN_TRACE_QUERY_MODE >= 3
layout(buffer_reference, std430, buffer_reference_align = 8) readonly buffer SharcReadonlyHashEntries {
    uint64_t data[];
};
#endif

// SHARC stores each resolved entry as 16 bytes:
//   half2 radiance.xy, half2 radiance.z/sampleCount, uint sampleData, uint sampleDataExt.
// Read it as uvec4 here instead of f16vec4 to avoid NVIDIA driver compiler crashes in world.rgen.
#if SHARC_MAIN_TRACE_QUERY_MODE >= 4
layout(buffer_reference, std430, buffer_reference_align = 16) readonly buffer SharcReadonlyPackedBuffer {
    uvec4 data[];
};
#endif

float sharcQueryLogBase(float x, float base) {
    return log(x) / log(base);
}

uint sharcQueryGetLevel(vec3 samplePosition, vec3 cameraPosition) {
    float distance2 = dot(cameraPosition - samplePosition, cameraPosition - samplePosition);
    return uint(clamp(0.5 * sharcQueryLogBase(distance2, RADIANCE_SHARC_GRID_LOGARITHM_BASE)
        + RADIANCE_SHARC_GRID_LEVEL_BIAS, 1.0, float(RADIANCE_SHARC_LEVEL_BIT_MASK)));
}

float sharcQueryGetVoxelSize(uint gridLevel, float sceneScale) {
    return pow(RADIANCE_SHARC_GRID_LOGARITHM_BASE, float(gridLevel))
        / (sceneScale * pow(RADIANCE_SHARC_GRID_LOGARITHM_BASE, RADIANCE_SHARC_GRID_LEVEL_BIAS));
}

#if SHARC_MAIN_TRACE_QUERY_MODE >= 2
uint sharcQueryHashJenkins32(uint a) {
    a = (a + 0x7ed55d16u) + (a << 12u);
    a = (a ^ 0xc761c23cu) ^ (a >> 19u);
    a = (a + 0x165667b1u) + (a << 5u);
    a = (a + 0xd3a2646cu) ^ (a << 9u);
    a = (a + 0xfd7046c5u) + (a << 3u);
    a = (a ^ 0xb55a4f09u) ^ (a >> 16u);
    return a;
}

uint sharcQueryHash32(uint64_t hashKey) {
    return sharcQueryHashJenkins32(uint((hashKey >> 0) & uint64_t(0xFFFFFFFFu)))
        ^ sharcQueryHashJenkins32(uint((hashKey >> 32) & uint64_t(0xFFFFFFFFu)));
}

ivec4 sharcQueryCalculatePositionLog(vec3 samplePosition, vec3 cameraPosition, float sceneScale) {
    samplePosition += vec3(RADIANCE_SHARC_POSITION_BIAS);
    uint gridLevel = sharcQueryGetLevel(samplePosition, cameraPosition);
    float voxelSize = sharcQueryGetVoxelSize(gridLevel, sceneScale);
    ivec3 gridPosition = ivec3(floor(samplePosition / voxelSize));
    return ivec4(gridPosition.xyz, int(gridLevel));
}

uint64_t sharcQueryComputeSpatialHash(vec3 samplePosition, vec3 sampleNormal, vec3 cameraPosition, float sceneScale) {
    uvec4 gridPosition = uvec4(sharcQueryCalculatePositionLog(samplePosition, cameraPosition, sceneScale));
    uint64_t hashKey =
        ((uint64_t(gridPosition.x) & uint64_t(RADIANCE_SHARC_POSITION_BIT_MASK)) << (RADIANCE_SHARC_POSITION_BIT_NUM * 0)) |
        ((uint64_t(gridPosition.y) & uint64_t(RADIANCE_SHARC_POSITION_BIT_MASK)) << (RADIANCE_SHARC_POSITION_BIT_NUM * 1)) |
        ((uint64_t(gridPosition.z) & uint64_t(RADIANCE_SHARC_POSITION_BIT_MASK)) << (RADIANCE_SHARC_POSITION_BIT_NUM * 2)) |
        ((uint64_t(gridPosition.w) & uint64_t(RADIANCE_SHARC_LEVEL_BIT_MASK)) << (RADIANCE_SHARC_POSITION_BIT_NUM * 3));

    uint normalBits =
        (sampleNormal.x + RADIANCE_SHARC_NORMAL_BIAS >= 0.0 ? 0u : 1u) +
        (sampleNormal.y + RADIANCE_SHARC_NORMAL_BIAS >= 0.0 ? 0u : 2u) +
        (sampleNormal.z + RADIANCE_SHARC_NORMAL_BIAS >= 0.0 ? 0u : 4u);
    hashKey |= (uint64_t(normalBits) << (RADIANCE_SHARC_POSITION_BIT_NUM * 3 + RADIANCE_SHARC_LEVEL_BIT_NUM));

    return hashKey;
}
#endif

#if SHARC_MAIN_TRACE_QUERY_MODE >= 3
bool sharcQueryFind(uint64_t hashEntriesBDA, uint capacity, uint64_t hashKey, out uint cacheIndex) {
    SharcReadonlyHashEntries hashEntries = SharcReadonlyHashEntries(hashEntriesBDA);
    uint baseSlot = sharcQueryHash32(hashKey) % capacity;
    [[dont_unroll]]
    for (uint bucketOffset = 0u; bucketOffset < RADIANCE_SHARC_BUCKET_SIZE; ++bucketOffset) {
        uint slotIndex = (baseSlot + bucketOffset) % capacity;
        uint64_t storedHashKey = hashEntries.data[slotIndex];
        if (storedHashKey == hashKey) {
            cacheIndex = slotIndex;
            return true;
        }
    }
    cacheIndex = 0xFFFFFFFFu;
    return false;
}
#endif

bool sharcQueryCachedRadiance(
    uint64_t hashEntriesBDA,
    uint64_t resolvedBDA,
    uint capacity,
    vec3 cameraPosition,
    float sceneScale,
    vec3 positionWorld,
    vec3 normalWorld,
    out vec3 radiance,
    out float voxelSize) {
    radiance = vec3(0.0);
    uint level = sharcQueryGetLevel(positionWorld, cameraPosition);
    voxelSize = sharcQueryGetVoxelSize(level, sceneScale);

#if SHARC_MAIN_TRACE_QUERY_MODE <= 1
    return capacity == 0u && hashEntriesBDA == uint64_t(0) && resolvedBDA == uint64_t(0);
#elif SHARC_MAIN_TRACE_QUERY_MODE == 2
    uint64_t hashKey = sharcQueryComputeSpatialHash(positionWorld, normalWorld, cameraPosition, sceneScale);
    uint foldedHash = sharcQueryHash32(hashKey) ^ capacity;
    return foldedHash == 0xFFFFFFFFu;
#elif SHARC_MAIN_TRACE_QUERY_MODE == 3
    uint64_t hashKey = sharcQueryComputeSpatialHash(positionWorld, normalWorld, cameraPosition, sceneScale);
    uint cacheIndex;
    bool found = sharcQueryFind(hashEntriesBDA, capacity, hashKey, cacheIndex);
    return found && cacheIndex == 0xFFFFFFFFu;
#else
    uint64_t hashKey = sharcQueryComputeSpatialHash(positionWorld, normalWorld, cameraPosition, sceneScale);
    uint cacheIndex;
    if (!sharcQueryFind(hashEntriesBDA, capacity, hashKey, cacheIndex)) {
        return false;
    }

    SharcReadonlyPackedBuffer resolved = SharcReadonlyPackedBuffer(resolvedBDA);
    uvec4 packedData = resolved.data[cacheIndex];
    vec2 radianceXY = unpackHalf2x16(packedData.x);
    vec2 radianceZSample = unpackHalf2x16(packedData.y);
    float sampleNum = radianceZSample.y;
    if (sampleNum <= RADIANCE_SHARC_SAMPLE_NUM_THRESHOLD) {
        return false;
    }

    radiance = vec3(radianceXY.xy, radianceZSample.x);
    return !any(isnan(radiance)) && !any(isinf(radiance));
#endif
}

#endif
