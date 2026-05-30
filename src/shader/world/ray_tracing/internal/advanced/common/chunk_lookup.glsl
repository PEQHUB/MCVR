#ifndef ADV_CHUNK_LOOKUP_GLSL
#define ADV_CHUNK_LOOKUP_GLSL

int floorMod(int value, int divisor) {
    if (divisor <= 0) { return 0; }
    return value - divisor * int(floor(float(value) / float(divisor)));
}

ivec3 getSectionOriginFromSectionCoordinate(ivec3 sectionCoordinate) {
    return sectionCoordinate * 16;
}

ivec3 getChunkStorageSectionCoordinate(WorldUBO worldUBO) {
    return worldUBO.chunkStorageSectionPos.xyz;
}

bool tryGetChunkIndexFromSectionCoordinate(ivec3 sectionCoordinate,
                                           WorldUBO worldUBO,
                                           out int chunkIndex,
                                           out ivec3 chunkOrigin) {
    int sectionX = sectionCoordinate.x;
    int sectionY = sectionCoordinate.y;
    int sectionZ = sectionCoordinate.z;
    int sizeX = worldUBO.chunkGridInfo.x;
    int sizeY = worldUBO.chunkGridInfo.y;
    int sizeZ = worldUBO.chunkGridInfo.z;
    int bottomSectionCoord = worldUBO.chunkGridInfo.w;

    chunkIndex = -1;
    chunkOrigin = ivec3(sectionX * 16, sectionY * 16, sectionZ * 16);

    if (sizeX <= 0 || sizeY <= 0 || sizeZ <= 0) { return false; }

    ivec3 cameraSectionCoord = getChunkStorageSectionCoordinate(worldUBO);
    int viewDistance = (sizeX - 1) / 2;
    if (sectionY < bottomSectionCoord || sectionY >= bottomSectionCoord + sizeY) { return false; }
    if (abs(sectionX - cameraSectionCoord.x) > viewDistance || abs(sectionZ - cameraSectionCoord.z) > viewDistance) {
        return false;
    }

    int gridX = floorMod(sectionX, sizeX);
    int gridY = sectionY - bottomSectionCoord;
    int gridZ = floorMod(sectionZ, sizeZ);
    chunkIndex = (gridZ * sizeY + gridY) * sizeX + gridX;
    return true;
}

bool tryGetSectionCoordinateFromChunkIndex(uint chunkIndex,
                                           WorldUBO worldUBO,
                                           out ivec3 sectionCoordinate,
                                           out ivec3 chunkOrigin) {
    int sizeX = worldUBO.chunkGridInfo.x;
    int sizeY = worldUBO.chunkGridInfo.y;
    int sizeZ = worldUBO.chunkGridInfo.z;
    int bottomSectionCoord = worldUBO.chunkGridInfo.w;

    sectionCoordinate = ivec3(0);
    chunkOrigin = ivec3(0);

    if (sizeX <= 0 || sizeY <= 0 || sizeZ <= 0) { return false; }

    uint sizeXU = uint(sizeX);
    uint sizeYU = uint(sizeY);
    uint sizeZU = uint(sizeZ);
    uint chunkCount = sizeXU * sizeYU * sizeZU;
    if (chunkIndex >= chunkCount) { return false; }

    int gridX = int(chunkIndex % sizeXU);
    int gridY = int((chunkIndex / sizeXU) % sizeYU);
    int gridZ = int(chunkIndex / (sizeXU * sizeYU));

    ivec3 cameraSectionCoord = getChunkStorageSectionCoordinate(worldUBO);
    int viewDistance = (sizeX - 1) / 2;
    int baseSectionX = cameraSectionCoord.x - viewDistance;
    int baseSectionZ = cameraSectionCoord.z - viewDistance;

    int sectionX = baseSectionX + floorMod(gridX - baseSectionX, sizeX);
    int sectionY = bottomSectionCoord + gridY;
    int sectionZ = baseSectionZ + floorMod(gridZ - baseSectionZ, sizeZ);

    sectionCoordinate = ivec3(sectionX, sectionY, sectionZ);
    chunkOrigin = getSectionOriginFromSectionCoordinate(sectionCoordinate);
    return true;
}

#endif
