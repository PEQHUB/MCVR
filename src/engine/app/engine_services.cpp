#include "engine_services.hpp"
#include "config/config_service.hpp"
#include "bridge/bridge_service.hpp"
#include "frame/frame_scheduler.hpp"
#include "diagnostics/metrics_service.hpp"
#include "scene/scene_service.hpp"
#include "scene/gpu_upload_service.hpp"
#include "scene/blas_service.hpp"
#include "scene/entity_blas_service.hpp"
#include "scene/tlas_service.hpp"
#include "scene/scene_resource_service.hpp"
#include "scene/texture_service.hpp"
#include "frame/offscreen_target.hpp"
#include "platform/vulkan/vk2_device.hpp"
#include "platform/vulkan/vk2_swapchain.hpp"

namespace engine {

EngineServices::EngineServices()
    : config_(std::make_unique<ConfigService>()),
      bridge_(std::make_unique<BridgeService>()),
      device_(std::make_unique<vk2::DeviceService>()),
      swapchain_(std::make_unique<vk2::SwapchainService>()),
      frame_(std::make_unique<FrameScheduler>(*this)),
      metrics_(std::make_unique<MetricsService>()),
      scene_(std::make_unique<SceneService>()),
      offscreen_(std::make_unique<OffscreenTarget>()),
      gpuUpload_(std::make_unique<GpuUploadService>()),
      blas_(std::make_unique<BlasService>()),
      entityBlas_(std::make_unique<EntityBlasService>()),
      tlas_(std::make_unique<TlasService>()),
      sceneRes_(std::make_unique<SceneResourceService>()),
      texture_(std::make_unique<TextureService>()) {}

EngineServices::~EngineServices() = default;

} // namespace engine
