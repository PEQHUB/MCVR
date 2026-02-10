#include "core/render/modules/world/world_module.hpp"

#include "core/render/render_framework.hpp"
#include "core/render/renderer.hpp"

WorldModule::WorldModule() {}

void WorldModule::init(std::shared_ptr<Framework> framework, std::shared_ptr<WorldPipeline> worldPipeline) {
    framework_ = framework;
    worldPipeline_ = worldPipeline;
}

WorldModuleContext::WorldModuleContext(std::shared_ptr<FrameworkContext> frameworkContext,
                                       std::shared_ptr<WorldPipelineContext> worldPipelineContext)
    : frameworkContext(frameworkContext), worldPipelineContext(worldPipelineContext) {}