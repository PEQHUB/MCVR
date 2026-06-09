#include "scene_service.hpp"

namespace engine {

std::shared_ptr<const ExtractedScene> SceneService::extractFrame(uint64_t frameNumber) {
    auto scene = std::make_shared<ExtractedScene>();
    scene->frameNumber = frameNumber;

    // Copy geometry for dirty chunks (BLAS rebuild input).
    // The BLAS worker owns this data — no pointer back into the registry.
    auto dirtyIds = chunks_.dirtySet();
    scene->dirtyChunkGeometries.reserve(dirtyIds.size());
    for (const auto& id : dirtyIds) {
        const auto* geo = chunks_.geometry(id);
        if (geo) {
            scene->dirtyChunkGeometries.push_back(*geo);
        }
    }

    // Snapshot ALL chunk states (TLAS needs every instance, not just dirty ones).
    scene->allChunks.reserve(chunks_.size());
    chunks_.forEachState([&](const ChunkState& state) {
        scene->allChunks.push_back(state);
    });

    // Dirty entities (full state for those that changed).
    auto dirtyEntityIds = entities_.dirtySet();
    scene->dirtyEntities.reserve(dirtyEntityIds.size());
    for (const auto& id : dirtyEntityIds) {
        const auto* state = entities_.get(id);
        if (state) {
            scene->dirtyEntities.push_back(*state);
        }
    }

    // Snapshot ALL entity states (TLAS needs every instance).
    scene->allEntities.reserve(entities_.size());
    entities_.forEachState([&](const EntityState& state) {
        scene->allEntities.push_back(state);
    });

    // Clear dirty flags after extraction
    chunks_.clearDirty();
    entities_.clearDirty();

    return scene;
}

void SceneService::clear() {
    chunks_.clear();
    entities_.clear();
}

} // namespace engine
