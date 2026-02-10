#include "core/render/world.hpp"

#include <glm/gtc/type_ptr.hpp>

#include "core/render/buffers.hpp"
#include "core/render/chunks.hpp"
#include "core/render/entities.hpp"
#include "core/render/render_framework.hpp"
#include "core/render/renderer.hpp"

World::World(std::shared_ptr<Framework> framework)
    : chunks_(Chunks::create(framework)), entities_(Entities::create(framework)) {}

void World::resetFrame() {}

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
}

glm::dvec3 World::getCameraPos() {
    return cameraPos_;
}

void World::close() {
    shouldRenderWorld_ = false;
    chunks_->close();
}