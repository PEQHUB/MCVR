#include "core/render/extended_chunk_manager.hpp"
#include "core/render/chunks.hpp"
#include "core/render/renderer.hpp"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <iostream>

static auto& extCout() {
    static auto& s = std::cout;
    return s;
}

ExtendedChunkManager::~ExtendedChunkManager() {
    stop();
}

void ExtendedChunkManager::start(const std::filesystem::path& regionDir,
                                  uint32_t javaRenderDistance,
                                  uint32_t javaChunkCount,
                                  Chunks* chunks) {
    stop(); // Ensure any previous worker is stopped

    regionDir_ = regionDir;
    javaRenderDistance_ = javaRenderDistance;
    javaChunkCount_ = javaChunkCount;
    chunks_ = chunks;
    nextExtendedId_.store(javaChunkCount);
    loadedCount_.store(0);
    loadedColumns_.clear();
    freeIds_.clear();
    {
        std::lock_guard<std::mutex> lock(evictionMtx_);
        pendingEvictions_.clear();
    }
    lastCamChunkX_ = INT32_MAX;
    lastCamChunkZ_ = INT32_MAX;
    stopRequested_.store(false);
    running_.store(true);

    workerThread_ = std::thread(&ExtendedChunkManager::workerLoop, this);

    extCout() << "[ExtendedRD] Started: javaRD=" << javaRenderDistance
              << " extendedRD=" << Renderer::options.extendedRenderDistance
              << " regionDir=" << regionDir_.string() << std::endl;
}

void ExtendedChunkManager::stop() {
    if (!running_.load()) return;
    stopRequested_.store(true);
    if (workerThread_.joinable()) workerThread_.join();
    running_.store(false);
    extCout() << "[ExtendedRD] Stopped. Loaded " << loadedCount_.load() << " chunk columns." << std::endl;
}

void ExtendedChunkManager::updateCamera(glm::dvec3 cameraPos) {
    std::lock_guard<std::mutex> lock(cameraMtx_);
    cameraPos_ = cameraPos;
}

void ExtendedChunkManager::processPendingEvictions() {
    // Called from render thread — safe to call invalidateChunk here
    std::vector<uint32_t> toEvict;
    {
        std::lock_guard<std::mutex> lock(evictionMtx_);
        if (pendingEvictions_.empty()) return;
        toEvict.swap(pendingEvictions_);
    }

    for (uint32_t id : toEvict) {
        chunks_->invalidateChunk(static_cast<int>(id));
    }

    if (!toEvict.empty()) {
        extCout() << "[ExtendedRD] Evicted " << toEvict.size() << " chunk sections (render thread)" << std::endl;
    }
}

void ExtendedChunkManager::markOutOfRangeForEviction(int32_t camChunkX, int32_t camChunkZ, uint32_t totalRD) {
    // Evict columns whose center is beyond totalRD + 2 chunks (hysteresis to avoid thrashing)
    int32_t evictRD = static_cast<int32_t>(totalRD) + 2;
    int64_t evictDist2 = static_cast<int64_t>(evictRD) * evictRD;

    std::vector<ChunkColumnKey> toEvict;
    for (auto& [key, ids] : loadedColumns_) {
        int64_t dx = static_cast<int64_t>(key.x) - camChunkX;
        int64_t dz = static_cast<int64_t>(key.z) - camChunkZ;
        if (dx * dx + dz * dz > evictDist2) {
            toEvict.push_back(key);
        }
    }

    if (toEvict.empty()) return;

    int evictedSections = 0;
    {
        std::lock_guard<std::mutex> lock(evictionMtx_);
        for (auto& key : toEvict) {
            auto it = loadedColumns_.find(key);
            if (it == loadedColumns_.end()) continue;

            // Queue IDs for render-thread invalidation and recycle
            for (uint32_t id : it->second) {
                pendingEvictions_.push_back(id);
                freeIds_.push_back(id);
                evictedSections++;
            }
            loadedColumns_.erase(it);
        }
    }

    extCout() << "[ExtendedRD] Marked " << toEvict.size() << " columns ("
              << evictedSections << " sections) for eviction, "
              << freeIds_.size() << " IDs recyclable" << std::endl;
}

static uint32_t allocExtendedId(std::vector<uint32_t>& freeIds, std::atomic<uint32_t>& nextId) {
    if (!freeIds.empty()) {
        uint32_t id = freeIds.back();
        freeIds.pop_back();
        return id;
    }
    return nextId.fetch_add(1);
}

void ExtendedChunkManager::workerLoop() {
    auto& registry = Renderer::blockStateRegistry;
    if (!registry.isLoaded()) {
        extCout() << "[ExtendedRD] BlockStateRegistry not loaded — aborting" << std::endl;
        running_.store(false);
        return;
    }

    if (!Renderer::blockModelTable.isLoaded()) {
        extCout() << "[ExtendedRD] BlockModelTable not loaded — aborting" << std::endl;
        running_.store(false);
        return;
    }

    uint32_t extRD = Renderer::options.extendedRenderDistance;
    uint32_t totalRD = javaRenderDistance_ + extRD;

    // Maximum chunk array slots we're allowed to use
    uint32_t maxExtendedSlots = static_cast<uint32_t>(
        (2 * totalRD + 1) * (2 * totalRD + 1) * 24);

    extCout() << "[ExtendedRD] Worker started: loading chunks from RD "
              << javaRenderDistance_ << " to " << totalRD
              << " (max " << maxExtendedSlots << " section slots)" << std::endl;

    while (!stopRequested_.load()) {
        glm::dvec3 camPos;
        {
            std::lock_guard<std::mutex> lock(cameraMtx_);
            camPos = cameraPos_;
        }

        // Camera chunk position
        int32_t camChunkX = static_cast<int32_t>(std::floor(camPos.x / 16.0));
        int32_t camChunkZ = static_cast<int32_t>(std::floor(camPos.z / 16.0));

        // Detect camera movement (full chunk boundary crossing)
        bool cameraMoved = (camChunkX != lastCamChunkX_ || camChunkZ != lastCamChunkZ_);
        if (cameraMoved && lastCamChunkX_ != INT32_MAX) {
            // Mark columns outside the ring for eviction (render thread will actually invalidate)
            markOutOfRangeForEviction(camChunkX, camChunkZ, totalRD);
        }
        lastCamChunkX_ = camChunkX;
        lastCamChunkZ_ = camChunkZ;

        // Throttle: don't submit faster than the BLAS thread can process.
        // Check input queue depth — if it's backed up, wait.
        if (chunks_->getInputQueueSize() > 64) {
            std::this_thread::sleep_for(std::chrono::milliseconds(50));
            continue;
        }

        // Generate positions for the extended ring
        auto positions = generateSpiralPositions(camChunkX, camChunkZ,
                                                  javaRenderDistance_, totalRD);

        int loadedThisPass = 0;
        int skippedNoRegion = 0, skippedEmpty = 0;
        for (auto& [cx, cz] : positions) {
            if (stopRequested_.load()) break;

            ChunkColumnKey key{cx, cz};
            if (loadedColumns_.count(key)) continue;

            // Throttle within the pass too
            if (chunks_->getInputQueueSize() > 64) break;

            // Check if region file exists for this chunk
            if (!anvil::RegionReader::regionExists(regionDir_, cx, cz)) {
                skippedNoRegion++;
                continue;
            }

            // Read chunk from disk
            auto nbtData = anvil::RegionReader::readChunk(regionDir_, cx, cz);
            if (nbtData.empty()) {
                skippedEmpty++;
                continue;
            }

            // Decode
            auto decoded = anvil::ChunkDecoder::decode(
                nbtData.data(), nbtData.size(), cx, cz, registry);
            if (decoded.sections.empty()) continue;

            // Track section IDs for this column (for eviction later)
            std::vector<uint32_t> columnIds;

            // For each non-empty section, mesh and submit
            for (auto& section : decoded.sections) {
                if (section.empty) continue;
                if (stopRequested_.load()) break;

                uint32_t extId = allocExtendedId(freeIds_, nextExtendedId_);

                // Bounds check — don't exceed allocated chunk array
                if (extId >= javaChunkCount_ + maxExtendedSlots) {
                    extCout() << "[ExtendedRD] Chunk array full at id " << extId << std::endl;
                    freeIds_.push_back(extId);
                    break;
                }

                // Build neighbor states from adjacent sections within this column
                uint32_t neighborStates[6][256];
                std::memset(neighborStates, 0, sizeof(neighborStates));

                // DOWN (direction 0): section below
                auto* below = decoded.getSection(section.sectionY - 1);
                if (below) {
                    for (int z = 0; z < 16; z++)
                        for (int x = 0; x < 16; x++)
                            neighborStates[0][z * 16 + x] = below->blockStates[15 * 256 + z * 16 + x];
                }

                // UP (direction 1): section above
                auto* above = decoded.getSection(section.sectionY + 1);
                if (above) {
                    for (int z = 0; z < 16; z++)
                        for (int x = 0; x < 16; x++)
                            neighborStates[1][z * 16 + x] = above->blockStates[0 * 256 + z * 16 + x];
                }

                // Build mesher input
                BlockMesher::SectionInput input;
                std::memcpy(input.blockStates, section.blockStates, sizeof(input.blockStates));
                std::memcpy(input.biomes, section.biomes, sizeof(input.biomes));
                std::memcpy(input.neighborStates, neighborStates, sizeof(input.neighborStates));
                input.originX = cx * 16;
                input.originY = section.sectionY * 16;
                input.originZ = cz * 16;
                input.blockAtlasTextureId = 0;

                // Mesh
                auto meshOutput = BlockMesher::mesh(input, Renderer::blockModelTable);

                // Build geometry arrays
                std::vector<World::GeometryTypes> geometryTypes;
                std::vector<std::vector<vk::VertexFormat::PBRTriangle>> vertices;
                std::vector<std::vector<uint32_t>> indices;
                uint32_t allVertexCount = 0, allIndexCount = 0;

                if (!meshOutput.solidVertices.empty()) {
                    geometryTypes.push_back(World::WORLD_SOLID);
                    allVertexCount += static_cast<uint32_t>(meshOutput.solidVertices.size());
                    allIndexCount += static_cast<uint32_t>(meshOutput.solidIndices.size());
                    vertices.push_back(std::move(meshOutput.solidVertices));
                    indices.push_back(std::move(meshOutput.solidIndices));
                }
                if (!meshOutput.cutoutVertices.empty()) {
                    geometryTypes.push_back(World::WORLD_TRANSPARENT);
                    allVertexCount += static_cast<uint32_t>(meshOutput.cutoutVertices.size());
                    allIndexCount += static_cast<uint32_t>(meshOutput.cutoutIndices.size());
                    vertices.push_back(std::move(meshOutput.cutoutVertices));
                    indices.push_back(std::move(meshOutput.cutoutIndices));
                }

                if (geometryTypes.empty()) continue;

                int geomCount = static_cast<int>(geometryTypes.size());
                auto chunkBuildData = ChunkBuildData::create(
                    extId, cx * 16, section.sectionY * 16, cz * 16,
                    1, allVertexCount, allIndexCount, geomCount,
                    std::move(geometryTypes), std::move(vertices), std::move(indices));

                chunkBuildData->biomeGrassColor = 0x91BD59;
                chunkBuildData->biomeFoliageColor = 0x77AB2F;
                chunkBuildData->biomeWaterColor = 0x3F76E4;

                chunks_->submitExtendedBuild(extId, chunkBuildData);
                columnIds.push_back(extId);
            }

            if (!columnIds.empty()) {
                loadedColumns_[key] = std::move(columnIds);
                loadedThisPass++;
            }

            // Yield periodically to not starve other threads
            if (loadedThisPass % 4 == 0) {
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
            }
        }

        loadedCount_.store(static_cast<uint32_t>(loadedColumns_.size()));

        if (loadedThisPass > 0) {
            extCout() << "[ExtendedRD] Pass done: loaded " << loadedThisPass
                      << " columns (total " << loadedColumns_.size() << "), skipped "
                      << skippedNoRegion << " no-region, " << skippedEmpty << " empty"
                      << ", free IDs: " << freeIds_.size() << std::endl;
        }

        if (loadedThisPass == 0) {
            // All visible chunks loaded — sleep and check for camera movement
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
        }
    }

    running_.store(false);
}

std::vector<std::pair<int32_t, int32_t>> ExtendedChunkManager::generateSpiralPositions(
    int32_t centerX, int32_t centerZ,
    uint32_t innerRD, uint32_t outerRD) {

    std::vector<std::pair<int32_t, int32_t>> positions;
    int32_t inner = static_cast<int32_t>(innerRD);
    int32_t outer = static_cast<int32_t>(outerRD);

    // Generate positions ring by ring, closest first
    for (int32_t ring = inner + 1; ring <= outer; ring++) {
        for (int32_t dx = -ring; dx <= ring; dx++) {
            positions.push_back({centerX + dx, centerZ - ring});
            positions.push_back({centerX + dx, centerZ + ring});
        }
        for (int32_t dz = -ring + 1; dz < ring; dz++) {
            positions.push_back({centerX - ring, centerZ + dz});
            positions.push_back({centerX + ring, centerZ + dz});
        }
    }

    // Sort by distance from center for better visual loading
    std::sort(positions.begin(), positions.end(),
              [centerX, centerZ](auto& a, auto& b) {
                  int64_t da = (int64_t)(a.first - centerX) * (a.first - centerX) +
                               (int64_t)(a.second - centerZ) * (a.second - centerZ);
                  int64_t db = (int64_t)(b.first - centerX) * (b.first - centerX) +
                               (int64_t)(b.second - centerZ) * (b.second - centerZ);
                  return da < db;
              });

    return positions;
}
