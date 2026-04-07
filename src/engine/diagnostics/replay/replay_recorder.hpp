#pragma once

// Ownership: Created by diagnostics on demand. One instance per recording session.
// Thread: Record calls are serialized by the caller (bridge dispatch thread).
// Dependencies: scene_types.hpp for ChunkId/EntityId.

#include "scene/scene_types.hpp"

#include <cstdint>
#include <fstream>
#include <string>
#include <vector>

namespace engine {

// Binary replay file format:
//   Header: magic(4) + version(4) + frameCount(8)
//   Frames: [frameHeader + events...]
//   Events: [type(1) + payload...]
//
// Event types:
//   0x01 ChunkInsert: chunkX(4) + chunkY(4) + chunkZ(4) + vertexSize(4) + indexSize(4) + data
//   0x02 ChunkRemove: chunkX(4) + chunkY(4) + chunkZ(4)
//   0x03 EntityInsert: entityId(4) + posXYZ(12) + rotXYZW(16)
//   0x04 EntityRemove: entityId(4)
//   0xFF FrameEnd

class ReplayRecorder {
public:
    static constexpr uint32_t MAGIC = 0x52455052;   // "REPR"
    // Version 2 (2026-04-07): ChunkId gained a Y dimension so sections of a
    // column no longer collide. v1 replay files are no longer readable.
    static constexpr uint32_t VERSION = 2;

    explicit ReplayRecorder(const std::string& path);
    ~ReplayRecorder();

    bool isOpen() const { return file_.is_open(); }

    // Record a chunk insertion with geometry data.
    void recordChunkInsert(ChunkId id, const void* vertexData, uint32_t vertexSize,
                           const void* indexData, uint32_t indexSize);

    // Record a chunk removal.
    void recordChunkRemove(ChunkId id);

    // Record an entity insertion.
    void recordEntityInsert(EntityId id, float px, float py, float pz,
                            float rx, float ry, float rz, float rw);

    // Record an entity removal.
    void recordEntityRemove(EntityId id);

    // Mark end of frame. Increments frame counter.
    void endFrame();

    uint64_t frameCount() const { return frameCount_; }

private:
    std::ofstream file_;
    uint64_t frameCount_ = 0;
    std::streampos frameCountOffset_;  // For patching header on close

    void writeU8(uint8_t v);
    void writeU32(uint32_t v);
    void writeI32(int32_t v);
    void writeF32(float v);
    void writeBytes(const void* data, size_t size);
};

// Replays a recorded session by calling callbacks for each event.
class ReplayPlayer {
public:
    struct Callbacks {
        // onChunkInsert receives owned vectors — safe to move into async upload.
        // The vectors are valid for the lifetime of the callback and may be moved from.
        std::function<void(ChunkId, std::vector<uint8_t> vertexData, std::vector<uint8_t> indexData)> onChunkInsert;
        std::function<void(ChunkId)> onChunkRemove;
        std::function<void(EntityId, float, float, float, float, float, float, float)> onEntityInsert;
        std::function<void(EntityId)> onEntityRemove;
        std::function<void(uint64_t)> onFrameEnd;  // frame number
    };

    explicit ReplayPlayer(const std::string& path);

    bool isOpen() const { return file_.is_open(); }
    uint64_t totalFrames() const { return totalFrames_; }

    // Play all frames, calling callbacks for each event.
    // Returns number of frames played.
    uint64_t playAll(const Callbacks& cb);

    // Play one frame. Returns false when no more frames.
    bool playFrame(const Callbacks& cb);

private:
    std::ifstream file_;
    uint64_t totalFrames_ = 0;
    uint64_t currentFrame_ = 0;

    uint8_t readU8();
    uint32_t readU32();
    int32_t readI32();
    float readF32();
    bool readBytes(void* data, size_t size);
};

} // namespace engine
