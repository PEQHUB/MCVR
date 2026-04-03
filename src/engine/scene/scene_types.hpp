#pragma once

// Scene identification types. Stable across frames, monotonic revisions.

#include <cstdint>
#include <functional>

namespace engine {

// Chunk identity — derived from Minecraft chunk coordinates (x, z).
// Stable across frames and render distance changes.
struct ChunkId {
    int32_t x = 0;
    int32_t z = 0;

    bool operator==(const ChunkId& o) const { return x == o.x && z == o.z; }
    bool operator!=(const ChunkId& o) const { return !(*this == o); }
};

// Entity identity — Minecraft entity ID (globally unique per world).
struct EntityId {
    int32_t id = -1;

    bool operator==(const EntityId& o) const { return id == o.id; }
    bool operator!=(const EntityId& o) const { return !(*this == o); }
    bool valid() const { return id >= 0; }
};

// Monotonically increasing revision counter. Bumped on any data change.
struct RevisionId {
    uint64_t value = 0;

    bool operator==(const RevisionId& o) const { return value == o.value; }
    bool operator!=(const RevisionId& o) const { return value != o.value; }
    bool operator<(const RevisionId& o) const { return value < o.value; }

    RevisionId next() const { return {value + 1}; }
};

} // namespace engine

// Hash support for use in unordered containers
template<> struct std::hash<engine::ChunkId> {
    size_t operator()(const engine::ChunkId& id) const {
        // Combine x and z with a mixing function
        size_t h = std::hash<int32_t>{}(id.x);
        h ^= std::hash<int32_t>{}(id.z) + 0x9e3779b9 + (h << 6) + (h >> 2);
        return h;
    }
};

template<> struct std::hash<engine::EntityId> {
    size_t operator()(const engine::EntityId& id) const {
        return std::hash<int32_t>{}(id.id);
    }
};
