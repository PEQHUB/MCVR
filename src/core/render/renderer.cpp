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
uint32_t Renderer::javaChunkCount = 0;
uint32_t Renderer::javaRenderDistance = 0;
TextureSystem Renderer::textureSystem;
std::vector<std::shared_ptr<vk::DeviceLocalImage>> Renderer::frameGenDepthImages;
std::vector<std::shared_ptr<vk::DeviceLocalImage>> Renderer::frameGenMotionVectorImages;
std::shared_ptr<vk::BufferPool> Renderer::vertexPool;
std::shared_ptr<vk::BufferPool> Renderer::indexPool;
std::shared_ptr<vk::BufferPool> Renderer::blasPool;

Renderer::Renderer(GLFWwindow *window)
    : framework_(Framework::create(window)),
      textures_(Textures::create(framework_)),
      buffers_(Buffers::create(framework_)),
      world_(World::create(framework_)) {
    // Create buffer pools for chunk vertex/index suballocation.
    // Usage flags match what uploadGPU() needs for BLAS build input + shader BDA access.
    constexpr VkDeviceSize POOL_BLOCK_SIZE = 8 * 1024 * 1024; // 8 MB per backing buffer
    VkBufferUsageFlags geoUsage = VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT |
                                  VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR |
                                  VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
    vertexPool = vk::BufferPool::create(framework_->vma(), framework_->device(), geoUsage, POOL_BLOCK_SIZE);
    indexPool = vk::BufferPool::create(framework_->vma(), framework_->device(), geoUsage, POOL_BLOCK_SIZE);

    VkBufferUsageFlags blasUsage = VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_STORAGE_BIT_KHR |
                                   VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT;
    blasPool = vk::BufferPool::create(framework_->vma(), framework_->device(), blasUsage, POOL_BLOCK_SIZE);
}

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

    // Release buffer pools after world (chunks) are destroyed but before framework
    vertexPool.reset();
    indexPool.reset();
    blasPool.reset();

    if (framework_ != nullptr) framework_->close();
    closed_ = true;

#ifdef DEBUG
    std::cout << "Renderer closed" << std::endl;
#endif
}
