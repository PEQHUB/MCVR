#include "entity_registry.hpp"

namespace engine {

RevisionId EntityRegistry::insert(EntityId id, EntityState state) {
    auto rev = nextRevision_;
    nextRevision_ = nextRevision_.next();

    state.id = id;
    state.revision = rev;

    states_[id] = std::move(state);
    dirty_.insert(id);

    return rev;
}

void EntityRegistry::remove(EntityId id) {
    states_.erase(id);
    dirty_.erase(id);
}

RevisionId EntityRegistry::update(EntityId id, const EntityDelta& delta) {
    auto it = states_.find(id);
    if (it == states_.end()) return {};

    auto& state = it->second;
    auto rev = nextRevision_;
    nextRevision_ = nextRevision_.next();
    state.revision = rev;

    if (delta.position) state.position = *delta.position;
    if (delta.prevPosition) state.prevPosition = *delta.prevPosition;
    if (delta.rotation) state.rotation = *delta.rotation;
    if (delta.scale) state.scale = *delta.scale;
    if (delta.visible) state.visible = *delta.visible;

    dirty_.insert(id);
    return rev;
}

const EntityState* EntityRegistry::get(EntityId id) const {
    auto it = states_.find(id);
    return it != states_.end() ? &it->second : nullptr;
}

std::vector<EntityId> EntityRegistry::dirtySet() const {
    return {dirty_.begin(), dirty_.end()};
}

void EntityRegistry::clearDirty() {
    dirty_.clear();
}

void EntityRegistry::clear() {
    states_.clear();
    dirty_.clear();
}

} // namespace engine
