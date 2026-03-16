#include "core/render/renderer.hpp"

#include "core/render/buffers.hpp"
#include "core/render/render_framework.hpp"
#include "core/render/textures.hpp"
#include "core/render/world.hpp"

std::filesystem::path Renderer::folderPath{};
Options Renderer::options{};
float Renderer::preExposure = 0.00002f;  // Init for physical sun ~100k lux (avoids first-frame FP16 overflow)
bool Renderer::resetExposureAdaptation = false;
std::vector<std::shared_ptr<vk::DeviceLocalImage>> Renderer::emissionImages;
std::vector<std::shared_ptr<vk::DeviceLocalImage>> Renderer::renderResHdrImages;
GpuProfiler Renderer::gpuProfiler;

Renderer::Renderer(GLFWwindow *window)
    : framework_(Framework::create(window)),
      textures_(Textures::create(framework_)),
      buffers_(Buffers::create(framework_)),
      world_(World::create(framework_)) {}

Renderer::~Renderer() {}

std::shared_ptr<Framework> Renderer::framework() {
    return framework_;
}

std::shared_ptr<Textures> Renderer::textures() {
    return textures_;
}

std::shared_ptr<Buffers> Renderer::buffers() {
    return buffers_;
}

std::shared_ptr<World> Renderer::world() {
    return world_;
}

void Renderer::close() {
    if (closed_) return;

    if (framework_ != nullptr) framework_->waitDeviceIdle();

    if (world_ != nullptr) world_->close();
    if (framework_ != nullptr) framework_->close();
    closed_ = true;

#ifdef DEBUG
    std::cout << "Renderer closed" << std::endl;
#endif
}
