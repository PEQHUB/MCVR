#include "frame_scheduler.hpp"
#include "app/engine_services.hpp"
#include "config/config_service.hpp"
#include "bridge/bridge_service.hpp"
#include "diagnostics/log.hpp"

namespace engine {

FrameScheduler::FrameScheduler(EngineServices& services)
    : services_(services) {
    lastFrameTime_ = Clock::now();
}

FrameScheduler::~FrameScheduler() {
    gc_.flush();
}

FrameContext FrameScheduler::beginFrame() {
    // 1. Measure frame time
    auto now = Clock::now();
    if (!firstFrame_) {
        lastFrameTimeMs_ = std::chrono::duration<float, std::milli>(now - lastFrameTime_).count();
    }
    firstFrame_ = false;
    lastFrameTime_ = now;

    // 2. Flush bridge commands (JNI → main thread)
    services_.bridge().flush();

    // 3. Snapshot config (immutable for this frame)
    auto configSnap = services_.config().snapshot();

    // 4. Tick GC (free resources deferred N frames ago)
    gc_.tick();

    // 5. Build frame context
    FrameContext ctx;
    ctx.frameIndex = frameIndex_;
    ctx.frameNumber = frameNumber_;
    ctx.deltaTime = lastFrameTimeMs_ / 1000.0f;
    ctx.config = std::move(configSnap);

    return ctx;
}

void FrameScheduler::endFrame(const FrameContext& ctx) {
    // Advance counters
    ++frameNumber_;
    frameIndex_ = (frameIndex_ + 1) % imageCount_;

    // Emit frame complete event
    services_.bridge().emit(EvtFrameComplete{ctx.frameNumber, lastFrameTimeMs_});
}

} // namespace engine
