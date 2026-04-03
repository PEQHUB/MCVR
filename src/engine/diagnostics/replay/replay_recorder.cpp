#include "replay_recorder.hpp"
#include "diagnostics/log.hpp"

#include <cstring>

namespace engine {

// --- ReplayRecorder ---

ReplayRecorder::ReplayRecorder(const std::string& path) {
    file_.open(path, std::ios::binary);
    if (!file_.is_open()) {
        if (log::isInitialized()) {
            log::error("replay", "Failed to open replay file for writing: " + path);
        }
        return;
    }
    // Write header (frameCount patched on close)
    writeU32(MAGIC);
    writeU32(VERSION);
    frameCountOffset_ = file_.tellp();
    uint64_t zero = 0;
    file_.write(reinterpret_cast<const char*>(&zero), 8);
}

ReplayRecorder::~ReplayRecorder() {
    if (file_.is_open()) {
        // Patch frame count in header
        file_.seekp(frameCountOffset_);
        file_.write(reinterpret_cast<const char*>(&frameCount_), 8);
        file_.close();
    }
}

void ReplayRecorder::recordChunkInsert(ChunkId id, const void* vertexData, uint32_t vertexSize,
                                        const void* indexData, uint32_t indexSize) {
    if (!file_.is_open()) return;
    writeU8(0x01);
    writeI32(id.x);
    writeI32(id.z);
    writeU32(vertexSize);
    writeU32(indexSize);
    if (vertexSize > 0) writeBytes(vertexData, vertexSize);
    if (indexSize > 0) writeBytes(indexData, indexSize);
}

void ReplayRecorder::recordChunkRemove(ChunkId id) {
    if (!file_.is_open()) return;
    writeU8(0x02);
    writeI32(id.x);
    writeI32(id.z);
}

void ReplayRecorder::recordEntityInsert(EntityId id, float px, float py, float pz,
                                         float rx, float ry, float rz, float rw) {
    if (!file_.is_open()) return;
    writeU8(0x03);
    writeI32(id.id);
    writeF32(px); writeF32(py); writeF32(pz);
    writeF32(rx); writeF32(ry); writeF32(rz); writeF32(rw);
}

void ReplayRecorder::recordEntityRemove(EntityId id) {
    if (!file_.is_open()) return;
    writeU8(0x04);
    writeI32(id.id);
}

void ReplayRecorder::endFrame() {
    if (!file_.is_open()) return;
    writeU8(0xFF);
    ++frameCount_;
}

void ReplayRecorder::writeU8(uint8_t v) { file_.write(reinterpret_cast<const char*>(&v), 1); }
void ReplayRecorder::writeU32(uint32_t v) { file_.write(reinterpret_cast<const char*>(&v), 4); }
void ReplayRecorder::writeI32(int32_t v) { file_.write(reinterpret_cast<const char*>(&v), 4); }
void ReplayRecorder::writeF32(float v) { file_.write(reinterpret_cast<const char*>(&v), 4); }
void ReplayRecorder::writeBytes(const void* data, size_t size) { file_.write(static_cast<const char*>(data), size); }

// --- ReplayPlayer ---

ReplayPlayer::ReplayPlayer(const std::string& path) {
    file_.open(path, std::ios::binary);
    if (!file_.is_open()) return;

    uint32_t magic = readU32();
    uint32_t version = readU32();
    if (magic != ReplayRecorder::MAGIC || version != ReplayRecorder::VERSION) {
        if (log::isInitialized()) {
            log::error("replay", "Invalid replay file: bad magic/version");
        }
        file_.close();
        return;
    }
    file_.read(reinterpret_cast<char*>(&totalFrames_), 8);
}

uint64_t ReplayPlayer::playAll(const Callbacks& cb) {
    uint64_t count = 0;
    while (playFrame(cb)) ++count;
    return count;
}

bool ReplayPlayer::playFrame(const Callbacks& cb) {
    if (!file_.is_open() || file_.eof()) return false;

    while (true) {
        uint8_t type = readU8();
        if (file_.eof()) return false;

        switch (type) {
            case 0x01: { // ChunkInsert
                ChunkId id{readI32(), readI32()};
                uint32_t vSize = readU32();
                uint32_t iSize = readU32();
                std::vector<uint8_t> vData(vSize), iData(iSize);
                if (vSize > 0) readBytes(vData.data(), vSize);
                if (iSize > 0) readBytes(iData.data(), iSize);
                // Pass owned vectors — callback can move them into async upload.
                if (cb.onChunkInsert) cb.onChunkInsert(id, std::move(vData), std::move(iData));
                break;
            }
            case 0x02: { // ChunkRemove
                ChunkId id{readI32(), readI32()};
                if (cb.onChunkRemove) cb.onChunkRemove(id);
                break;
            }
            case 0x03: { // EntityInsert
                EntityId id{readI32()};
                float px = readF32(), py = readF32(), pz = readF32();
                float rx = readF32(), ry = readF32(), rz = readF32(), rw = readF32();
                if (cb.onEntityInsert) cb.onEntityInsert(id, px, py, pz, rx, ry, rz, rw);
                break;
            }
            case 0x04: { // EntityRemove
                EntityId id{readI32()};
                if (cb.onEntityRemove) cb.onEntityRemove(id);
                break;
            }
            case 0xFF: { // FrameEnd
                if (cb.onFrameEnd) cb.onFrameEnd(currentFrame_);
                ++currentFrame_;
                return true;
            }
            default:
                if (log::isInitialized()) {
                    log::error("replay", "Unknown event type: " + std::to_string(type));
                }
                return false;
        }
    }
}

uint8_t ReplayPlayer::readU8() { uint8_t v = 0; file_.read(reinterpret_cast<char*>(&v), 1); return v; }
uint32_t ReplayPlayer::readU32() { uint32_t v = 0; file_.read(reinterpret_cast<char*>(&v), 4); return v; }
int32_t ReplayPlayer::readI32() { int32_t v = 0; file_.read(reinterpret_cast<char*>(&v), 4); return v; }
float ReplayPlayer::readF32() { float v = 0; file_.read(reinterpret_cast<char*>(&v), 4); return v; }
bool ReplayPlayer::readBytes(void* data, size_t size) {
    file_.read(static_cast<char*>(data), size);
    return file_.good();
}

} // namespace engine
