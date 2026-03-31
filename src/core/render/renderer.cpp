#include "core/render/renderer.hpp"

#include "core/render/buffers.hpp"
#include "core/render/render_framework.hpp"
#include "core/render/textures.hpp"
#include "core/render/world.hpp"

std::filesystem::path Renderer::folderPath{};
Options Renderer::options{};
float Renderer::preExposure = 0.1f;  // Constant pre-exposure for DLSS-RR (never varies per-frame)
bool Renderer::resetExposureAdaptation = false;
uint32_t Renderer::accumFrameCount = 0;
uint32_t Renderer::dlssEpochFrame = 0;
uint32_t Renderer::dlssEpochCount = 0;
std::shared_ptr<vk::DeviceLocalImage> Renderer::accumOutputImage;
VkPipeline Renderer::accumPipeline = VK_NULL_HANDLE;
VkPipelineLayout Renderer::accumPipelineLayout = VK_NULL_HANDLE;
std::shared_ptr<vk::DeviceLocalImage> Renderer::accumBufferImage = nullptr;
VkDescriptorPool Renderer::accumDescPool = VK_NULL_HANDLE;
VkDescriptorSetLayout Renderer::accumDescSetLayout = VK_NULL_HANDLE;
std::vector<VkDescriptorSet> Renderer::accumDescSets;
bool Renderer::accumPipelineReady = false;
VkPipeline Renderer::emissionComposePipeline = VK_NULL_HANDLE;
VkPipelineLayout Renderer::emissionComposePipelineLayout = VK_NULL_HANDLE;
VkDescriptorPool Renderer::emissionComposeDescPool = VK_NULL_HANDLE;
VkDescriptorSetLayout Renderer::emissionComposeDescSetLayout = VK_NULL_HANDLE;
std::vector<VkDescriptorSet> Renderer::emissionComposeDescSets;
bool Renderer::emissionComposePipelineReady = false;
std::vector<std::shared_ptr<vk::DeviceLocalImage>> Renderer::emissionImages;
std::vector<std::shared_ptr<vk::DeviceLocalImage>> Renderer::renderResHdrImages;
GpuProfiler Renderer::gpuProfiler;
ThreadPool Renderer::threadPool;
BlockModelTable Renderer::blockModelTable;
BlockStateRegistry Renderer::blockStateRegistry;
std::string Renderer::worldRegionPath;
TextureSystem Renderer::textureSystem;
std::vector<std::shared_ptr<vk::DeviceLocalImage>> Renderer::frameGenDepthImages;
std::vector<std::shared_ptr<vk::DeviceLocalImage>> Renderer::frameGenMotionVectorImages;

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
