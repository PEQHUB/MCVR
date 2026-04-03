#include "chunk_registry.hpp"

namespace engine {

RevisionId ChunkRegistry::insert(ChunkId id, ChunkGeometry geometry) {
    auto rev = nextRevision_;
    nextRevision_ = nextRevision_.next();

    geometry.id = id;
    geometry.revision = rev;

    auto& state = states_[id];
    state.id = id;
    state.revision = rev;
    state.dirty = true;
    state.uploaded = false;

    geometries_[id] = std::move(geometry);
    dirty_.insert(id);

    return rev;
}

void ChunkRegistry::remove(ChunkId id) {
    states_.erase(id);
    geometries_.erase(id);
    dirty_.erase(id);
}

bool ChunkRegistry::contains(ChunkId id) const {
    return states_.count(id) > 0;
}

const ChunkState* ChunkRegistry::get(ChunkId id) const {
    auto it = states_.find(id);
    return it != states_.end() ? &it->second : nullptr;
}

const ChunkGeometry* ChunkRegistry::geometry(ChunkId id) const {
    auto it = geometries_.find(id);
    return it != geometries_.end() ? &it->second : nullptr;
}

std::vector<ChunkId> ChunkRegistry::dirtySet() const {
    return {dirty_.begin(), dirty_.end()};
}

void ChunkRegistry::clearDirty() {
    for (auto& id : dirty_) {
        auto it = states_.find(id);
        if (it != states_.end()) {
            it->second.dirty = false;
        }
    }
    dirty_.clear();
}

void ChunkRegistry::markUploaded(ChunkId id) {
    auto it = states_.find(id);
    if (it != states_.end()) {
        it->second.uploaded = true;
    }
}

void ChunkRegistry::clear() {
    states_.clear();
    geometries_.clear();
    dirty_.clear();
}

} // namespace engine
