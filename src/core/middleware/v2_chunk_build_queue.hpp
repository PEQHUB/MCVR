#pragma once

#include "core/render/block_mesher.hpp"

#include <cstdint>

namespace engine {
class EngineApp;
}

namespace mcvr {
namespace v2chunk {

void queueDeferredSectionBuild(const BlockMesher::SectionInput& input);
uint32_t flushDeferredSectionBuilds(engine::EngineApp& app);
void submitSectionBuild(engine::EngineApp& app, const BlockMesher::SectionInput& input);

} // namespace v2chunk
} // namespace mcvr
