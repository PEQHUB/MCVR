#include "core/render/world.hpp"

#include <glm/gtc/type_ptr.hpp>

#include "core/render/buffers.hpp"
#include "core/render/chunks.hpp"
#include "core/render/entities.hpp"
#include "core/render/extended_chunk_manager.hpp"
#include "core/render/render_framework.hpp"
#include "core/render/renderer.hpp"

World::World(std::shared_ptr<Framework> framework)
    : chunks_(Chunks::create(framework)), entities_(Entities::create(framework)) {}

void World::resetFrame() {
    // Process deferred chunk evictions from ExtendedChunkManager (render thread only)
    if (extendedChunkMgr_) {
        extendedChunkMgr_->processPendingEvictions();
    }
}

bool &World::shouldRender() {
    return shouldRenderWorld_;
}

std::shared_ptr<Chunks> World::chunks() {
    return chunks_;
}

std::shared_ptr<Entities> World::entities() {
    return entities_;
}

void World::setCameraPos(glm::dvec3 cameraPos) {
    cameraPos_ = cameraPos;
    // Forward camera to extended chunk manager (thread-safe)
    if (extendedChunkMgr_) {
        extendedChunkMgr_->updateCamera(cameraPos);
    }
}

glm::dvec3 World::getCameraPos() {
    return cameraPos_;
}

void World::startExtendedChunkLoading(uint32_t javaRenderDistance, uint32_t javaChunkCount) {
    uint32_t extRD = Renderer::options.extendedRenderDistance;
    std::cout << "[ExtendedRD] startExtendedChunkLoading: extRD=" << extRD
              << " javaRD=" << javaRenderDistance
              << " javaChunks=" << javaChunkCount
              << " regionPath=" << (Renderer::worldRegionPath.empty() ? "(empty)" : Renderer::worldRegionPath)
              << " registry=" << (Renderer::blockStateRegistry.isLoaded() ? "loaded" : "NOT LOADED")
              << " modelTable=" << (Renderer::blockModelTable.isLoaded() ? "loaded" : "NOT LOADED")
              << std::endl;
    if (extRD == 0) { std::cout << "[ExtendedRD] SKIPPED: extendedRenderDistance=0" << std::endl; return; }
    if (Renderer::worldRegionPath.empty()) { std::cout << "[ExtendedRD] SKIPPED: region path empty" << std::endl; return; }
    if (!Renderer::blockStateRegistry.isLoaded()) { std::cout << "[ExtendedRD] SKIPPED: registry not loaded" << std::endl; return; }

    extendedChunkMgr_ = std::make_unique<ExtendedChunkManager>();
    extendedChunkMgr_->start(
        Renderer::worldRegionPath,
        javaRenderDistance,
        javaChunkCount,
        chunks_.get());
}

void World::close() {
    shouldRenderWorld_ = false;
    if (extendedChunkMgr_) {
        extendedChunkMgr_->stop();
        extendedChunkMgr_.reset();
    }
    chunks_->close();
}