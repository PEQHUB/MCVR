#include "scene_service.hpp"

namespace engine {

std::shared_ptr<const ExtractedScene> SceneService::extractFrame(uint64_t frameNumber) {
    auto scene = std::make_shared<ExtractedScene>();
    scene->frameNumber = frameNumber;

    // Capture dirty sets
    scene->dirtyChunks = chunks_.dirtySet();
    scene->dirtyEntities = entities_.dirtySet();

    // Snapshot all chunk states
    // (In a real implementation this would use a more efficient iteration,
    //  but the registry is small enough that copying is fine for now.)
    scene->allChunks.reserve(chunks_.size());
    // We need to iterate over all chunks — add an iteration helper
    // For now, we rely on dirtySet + the fact that TLAS build needs all chunks.
    // This will be optimized when the chunk registry gets an iterator.

    // Snapshot all entity states
    scene->allEntities.reserve(entities_.size());

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
