#pragma once

// Ownership: SceneService (1:1).
// Thread: Bridge thread writes via dispatch. RT thread reads via extractedFrame().
// Invariant: ChunkId is stable across frames. RevisionId is monotonic per chunk.

#include "scene_types.hpp"

#include <cstdint>
#include <mutex>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace engine {

// Extracted chunk geometry data (CPU-side, before GPU upload).
// This is the v2 equivalent of ChunkBuildData — but without GPU resources.
// GPU resources (BLAS, buffers) are owned by BlasService.
struct ChunkGeometry {
    ChunkId id;
    RevisionId revision;

    // Vertex data (PBR triangles, 96-byte stride)
    std::vector<uint8_t> vertexData;
    // Index data (uint32_t indices)
    std::vector<uint32_t> indexData;

    // Geometry metadata
    uint32_t triangleCount = 0;
    float boundingRadius = 0.0f;  // For distance culling

    // World-space position of chunk origin
    float originX = 0.0f;
    float originY = 0.0f;
    float originZ = 0.0f;

    bool empty() const { return triangleCount == 0; }
};

// Per-chunk state in the registry.
struct ChunkState {
    ChunkId id;
    RevisionId revision;
    bool dirty = true;          // Changed since last extracted frame
    bool uploaded = false;      // GPU resources created by BlasService
    float distanceToCamera = 0.0f;
};

// Tracks all known chunks, their revisions, and dirty state.
// The bridge thread calls insert/remove/update. The scene service
// creates an ExtractedScene snapshot for the render thread.
class ChunkRegistry {
public:
    ChunkRegistry() = default;

    // Insert or update a chunk. Bumps revision and marks dirty.
    // Returns the new revision.
    RevisionId insert(ChunkId id, ChunkGeometry geometry);

    // Remove a chunk from the registry.
    void remove(ChunkId id);

    // Check if a chunk exists.
    bool contains(ChunkId id) const;

    // Get current state of a chunk.
    const ChunkState* get(ChunkId id) const;

    // Get the geometry for a chunk (returns nullptr if not found).
    const ChunkGeometry* geometry(ChunkId id) const;

    // Get all chunks that changed since the last call to clearDirty().
    std::vector<ChunkId> dirtySet() const;

    // Clear dirty flags (called after scene extraction).
    void clearDirty();

    // Mark a chunk as uploaded (GPU resources ready).
    void markUploaded(ChunkId id);

    // Total chunk count.
    uint32_t size() const { return static_cast<uint32_t>(states_.size()); }

    // Clear everything (world unload).
    void clear();

private:
    std::unordered_map<ChunkId, ChunkState> states_;
    std::unordered_map<ChunkId, ChunkGeometry> geometries_;
    std::unordered_set<ChunkId> dirty_;
    RevisionId nextRevision_{1};
};

} // namespace engine
