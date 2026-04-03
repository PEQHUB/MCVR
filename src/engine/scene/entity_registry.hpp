#pragma once

// Ownership: SceneService (1:1).
// Thread: Bridge thread writes via dispatch. RT thread reads via extractedFrame().
// Invariant: EntityId is stable across frames. RevisionId is monotonic per entity.

#include "scene_types.hpp"

#include <cstdint>
#include <optional>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include <glm/glm.hpp>
#include <glm/gtc/quaternion.hpp>

namespace engine {

// Extracted per-entity state.
struct EntityState {
    EntityId id;
    RevisionId revision;

    glm::vec3 position{0.0f};
    glm::vec3 prevPosition{0.0f};  // For motion vectors
    glm::quat rotation{1.0f, 0.0f, 0.0f, 0.0f};
    glm::vec3 scale{1.0f};

    uint32_t entityType = 0;
    uint32_t rtFlags = 0;          // Custom RT instance flags
    bool visible = true;
};

// Delta update — only changed fields are set.
struct EntityDelta {
    EntityId id;
    std::optional<glm::vec3> position;
    std::optional<glm::vec3> prevPosition;
    std::optional<glm::quat> rotation;
    std::optional<glm::vec3> scale;
    std::optional<bool> visible;
};

class EntityRegistry {
public:
    EntityRegistry() = default;

    // Insert a new entity. Returns its first revision.
    RevisionId insert(EntityId id, EntityState state);

    // Remove an entity.
    void remove(EntityId id);

    // Apply a partial update. Bumps revision, marks dirty.
    RevisionId update(EntityId id, const EntityDelta& delta);

    // Get current state.
    const EntityState* get(EntityId id) const;

    // Get all entities that changed since last clearDirty().
    std::vector<EntityId> dirtySet() const;
    void clearDirty();

    uint32_t size() const { return static_cast<uint32_t>(states_.size()); }
    void clear();

private:
    std::unordered_map<EntityId, EntityState> states_;
    std::unordered_set<EntityId> dirty_;
    RevisionId nextRevision_{1};
};

} // namespace engine
