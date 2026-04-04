#include "bridge_service.hpp"
#include "diagnostics/log.hpp"

namespace engine {

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
