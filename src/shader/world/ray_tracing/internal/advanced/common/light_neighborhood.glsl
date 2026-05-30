#ifndef ADV_LIGHT_NEIGHBORHOOD_GLSL
#define ADV_LIGHT_NEIGHBORHOOD_GLSL

struct ChunkLightNeighborhoodEntry {
    uint chunkIndex;
    uint aliasIndex;
    float probability;
    float aliasProbability;
};

struct ChunkLightNeighborhood {
    uvec4 header;
    ChunkLightNeighborhoodEntry entries[ADV_CHUNK_NEIGHBORHOOD_CAPACITY];
};

layout(std430, set = 5, binding = 14) buffer ChunkLightNeighborhoodBuffer {
    ChunkLightNeighborhood chunkLightNeighborhoods[];
};

uint chunkLightNeighborhoodCount(ChunkLightNeighborhood neighborhood) {
    return neighborhood.header.x;
}

int sampleChunkLightNeighborhoodEntry(ChunkLightNeighborhood neighborhood,
                                      float slotXi,
                                      float aliasXi,
                                      out float chunkProbability) {
    chunkProbability = 0.0;

    uint count = chunkLightNeighborhoodCount(neighborhood);
    if (count == 0u) { return -1; }

    float scaled = clamp(slotXi, 0.0, ADV_UNIT_OPEN_UPPER_BOUND) * float(count);
    int slot = min(int(floor(scaled)), int(count) - 1);
    ChunkLightNeighborhoodEntry slotEntry = neighborhood.entries[slot];
    int sampledSlot = aliasXi < slotEntry.aliasProbability ? slot : int(slotEntry.aliasIndex);
    chunkProbability = neighborhood.entries[sampledSlot].probability;
    return sampledSlot;
}

#endif
