#pragma once

// Ownership: EngineServices (1:1).
// Thread: Bridge thread writes (via dispatch). RT thread reads via extractedFrame().
// Invariant: ExtractedScene is an immutable snapshot — safe to read from any thread.
//            It OWNS all data it references. No pointers back into mutable registry state.

#include "chunk_registry.hpp"
#include "entity_registry.hpp"

#include <memory>
#include <vector>

namespace engine {

// Immutable scene snapshot for one frame. Created by SceneService,
// consumed by RT services (BLAS/TLAS build). Owns all data by value.
struct ExtractedScene {
    // Geometry for dirty chunks — BLAS workers consume these directly.
    // Each entry is a full copy; the worker can move it into GPU upload.
    std::vector<ChunkGeometry> dirtyChunkGeometries;

    // All chunk states (for TLAS instance building — transforms, bounds, IDs).
    std::vector<ChunkState> allChunks;

    // Entities with pending changes (geometry for dirty entities).
    std::vector<EntityState> dirtyEntities;

    // All entity states (for TLAS instance building — transforms, flags).
    std::vector<EntityState> allEntities;

    uint64_t frameNumber = 0;
};

class SceneService {
public:
    SceneService() = default;

    ChunkRegistry& chunks() { return chunks_; }
    EntityRegistry& entities() { return entities_; }

    // Create an immutable snapshot of the current scene state.
    // Dirty chunk geometries are MOVED into the snapshot (registry copies cleared).
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
