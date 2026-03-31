#pragma once

#include "core/render/anvil/region_reader.hpp"
#include "core/render/anvil/chunk_decoder.hpp"
#include "core/render/block_state_registry.hpp"
#include "core/render/block_mesher.hpp"

#include <atomic>
#include <cstdint>
#include <filesystem>
#include <mutex>
#include <set>
#include <thread>
#include <unordered_set>
#include <glm/glm.hpp>

class Chunks;

/// Loads chunks from Anvil region files beyond Java's render distance.
/// Runs a background thread that reads → decodes → meshes → submits to the BLAS pipeline.
/// Chunks are processed in spiral order from the camera position.
class ExtendedChunkManager {
  public:
    ExtendedChunkManager() = default;
    ~ExtendedChunkManager();

    /// Start the background loader. Called when extendedRenderDistance > 0.
    /// javaRenderDistance = Java's RD in chunks (e.g., 32).
    /// javaChunkCount = number of Java chunk slots in the chunks_ array.
    void start(const std::filesystem::path& regionDir,
               uint32_t javaRenderDistance,
               uint32_t javaChunkCount,
               Chunks* chunks);

    /// Stop the background loader. Called when extendedRenderDistance = 0 or world unloads.
    void stop();

    /// Update camera position (called from render thread, thread-safe).
    void updateCamera(glm::dvec3 cameraPos);

    /// Is the manager actively loading?
    bool isRunning() const { return running_.load(); }

    /// Number of extended chunks loaded so far.
    uint32_t loadedCount() const { return loadedCount_.load(); }

  private:
    void workerLoop();

    /// Generate chunk positions in spiral order from center, ring by ring.
    /// Returns column positions (chunkX, chunkZ) for the annular ring between
    /// innerRD and outerRD (in chunk coordinates).
    std::vector<std::pair<int32_t, int32_t>> generateSpiralPositions(
        int32_t centerChunkX, int32_t centerChunkZ,
        uint32_t innerRD, uint32_t outerRD);

    std::filesystem::path regionDir_;
    uint32_t javaRenderDistance_ = 0;
    uint32_t javaChunkCount_ = 0;
    Chunks* chunks_ = nullptr;

    std::thread workerThread_;
    std::atomic<bool> running_{false};
    std::atomic<bool> stopRequested_{false};

    // Camera position (updated from render thread, read by worker)
    std::mutex cameraMtx_;
    glm::dvec3 cameraPos_{0};

    // Track which chunk columns we've already loaded (world coords)
    struct ChunkColumnKey {
        int32_t x, z;
        bool operator==(const ChunkColumnKey& o) const { return x == o.x && z == o.z; }
    };
    struct ChunkColumnHash {
        size_t operator()(const ChunkColumnKey& k) const {
            return std::hash<int64_t>()(static_cast<int64_t>(k.x) << 32 | (static_cast<uint32_t>(k.z)));
        }
    };
    std::unordered_set<ChunkColumnKey, ChunkColumnHash> loadedColumns_;

    // Next available extended chunk ID
    std::atomic<uint32_t> nextExtendedId_{0};
    std::atomic<uint32_t> loadedCount_{0};
};
