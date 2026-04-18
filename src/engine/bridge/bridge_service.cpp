#include "bridge_service.hpp"
#include "diagnostics/boot_trace.hpp"
#include "diagnostics/log.hpp"

namespace engine {

namespace {

// One name per variant alternative — keep in lockstep with BridgeCommand's
// std::variant<> declaration in bridge_service.hpp.
constexpr const char* kCommandNames[] = {
    "Ping",
    "WindowResize",
    "WorldLoad",
    "WorldUnload",
    "Shutdown",
    "ConfigPatch",
    "ChunkSubmit",
    "ChunkRemove",
    "CameraUpdate",
    "SkyUpdate",
    "TextureMappingUpdate",
    "SpriteTableUpload",
    "SpritePixelsUpload",
    "SpriteAuxPixelsUpload",
    "AnimationFramesUpload",
    "TextureFinalize",
    "AreaLightUpload",
    "EmissionDataUpload",
    "EntityBatchSubmit",
    "EntityPostDraw",
    "ResetAccumulation",
};
constexpr size_t kCommandNameCount = sizeof(kCommandNames) / sizeof(kCommandNames[0]);
static_assert(kCommandNameCount == std::variant_size_v<BridgeCommand>,
              "kCommandNames must mirror BridgeCommand variant alternatives 1:1");

const char* commandName(const BridgeCommand& cmd) {
    size_t idx = cmd.index();
    return idx < kCommandNameCount ? kCommandNames[idx] : "<unknown>";
}

} // anonymous namespace

void BridgeService::post(BridgeCommand cmd) {
    std::lock_guard lock(queueMutex_);
    incoming_.push_back(std::move(cmd));
}

uint32_t BridgeService::flush() {
    // Swap incoming queue under lock, then process without holding the lock.
    // This minimizes contention: JNI thread can keep posting while we drain.
    {
        std::lock_guard lock(queueMutex_);
        std::swap(incoming_, processing_);
    }

    uint32_t count = static_cast<uint32_t>(processing_.size());
    if (count > 0 && handler_) {
        for (const auto& cmd : processing_) {
            // Record before dispatch so the crash classifier knows which
            // command the handler was processing if it crashes mid-call.
            boot_trace::recordBridgeCommand(commandName(cmd), ++commandSeq_);
            handler_(cmd);
        }
        totalProcessed_ += count;
    }
    processing_.clear();
    return count;
}

void BridgeService::setHandler(CommandHandler handler) {
    handler_ = std::move(handler);
}

void BridgeService::emit(BridgeEvent event) {
    // Copy listeners under lock, invoke outside — prevents deadlock if
    // a listener calls post() or addListener() during the callback.
    std::vector<EventListener> snapshot;
    {
        std::lock_guard lock(listenerMutex_);
        snapshot.reserve(listeners_.size());
        for (const auto& entry : listeners_) {
            snapshot.push_back(entry.listener);
        }
    }
    for (const auto& listener : snapshot) {
        listener(event);
    }
}

uint32_t BridgeService::addListener(EventListener listener) {
    std::lock_guard lock(listenerMutex_);
    uint32_t id = nextListenerId_++;
    listeners_.push_back({id, std::move(listener)});
    return id;
}

void BridgeService::removeListener(uint32_t id) {
    std::lock_guard lock(listenerMutex_);
    listeners_.erase(
        std::remove_if(listeners_.begin(), listeners_.end(),
            [id](const auto& e) { return e.id == id; }),
        listeners_.end());
}

uint32_t BridgeService::pendingCommandCount() const {
    std::lock_guard lock(const_cast<std::mutex&>(queueMutex_));
    return static_cast<uint32_t>(incoming_.size());
}

} // namespace engine
