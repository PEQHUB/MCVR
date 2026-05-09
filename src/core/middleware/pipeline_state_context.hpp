#pragma once

#include "core/render/pipeline.hpp"
#include "core/render/render_framework.hpp"
#include "core/render/renderer.hpp"

#include <memory>

namespace radiance::middleware {

inline std::shared_ptr<PipelineContext> currentPipelineContext() {
    if (!Renderer::is_initialized()) return nullptr;

    auto framework = Renderer::instance().framework();
    if (framework == nullptr || !framework->isRunning()) return nullptr;

    auto context = framework->currentContext();
    if (context == nullptr) return nullptr;

    auto pipeline = framework->pipeline();
    if (pipeline == nullptr) return nullptr;

    return pipeline->acquirePipelineContext(context);
}

}
