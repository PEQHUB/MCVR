#pragma once

// Ownership: EngineServices (1:1).
// Thread: Bridge thread writes (via dispatch). RT thread reads via extractedFrame().
// Invariant: ExtractedScene is an immutable snapshot — safe to read from any thread.

#include "chunk_registry.hpp"
#include "entity_registry.hpp"

#include <memory>
#include <mutex>
#include <vector>

namespace engine {

// Immutable scene snapshot for one frame. Created by SceneService,
// consumed by RT services (BLAS/TLAS build).
struct ExtractedScene {
    // Chunks with pending geometry changes (need BLAS rebuild)
    std::vector<ChunkId> dirtyChunks;
    // All chunk states (for TLAS instance building)
    std::vector<ChunkState> allChunks;
    // Entities with pending changes
    std::vector<EntityId> dirtyEntities;
    // All entity states (for TLAS instance building)
    std::vector<EntityState> allEntities;

    uint64_t frameNumber = 0;
};

class SceneService {
public:
    SceneService() = default;

    ChunkRegistry& chunks() { return chunks_; }
    EntityRegistry& entities() { return entities_; }

    // Create an immutable snapshot of the current scene state.
    // Clears dirty flags after extraction.
    // Call once per frame from the main thread.
    std::shared_ptr<const ExtractedScene> extractFrame(uint64_t frameNumber);

    // Clear all scene data (world unload).
    void clear();

private:
    ChunkRegistry chunks_;
    EntityRegistry entities_;
};

} // namespace engine
