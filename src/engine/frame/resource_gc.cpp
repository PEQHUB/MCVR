#include "resource_gc.hpp"

namespace engine {

ResourceGC::ResourceGC(uint32_t ringSize)
    : ringSize_(ringSize), ring_(ringSize) {}

ResourceGC::~ResourceGC() {
    flush();
}

void ResourceGC::defer(Destructor dtor) {
    std::lock_guard lock(incomingMutex_);
    incoming_.push_back(std::move(dtor));
    ++totalDeferred_;
}

void ResourceGC::tick() {
    // Move incoming destructors into the current ring slot.
    {
        std::lock_guard lock(incomingMutex_);
        auto& slot = ring_[currentSlot_];
        slot.insert(slot.end(),
                    std::make_move_iterator(incoming_.begin()),
                    std::make_move_iterator(incoming_.end()));
        incoming_.clear();
    }

    // Advance to next slot — destroy whatever was there (N frames old).
    currentSlot_ = (currentSlot_ + 1) % ringSize_;
    auto& oldSlot = ring_[currentSlot_];
    for (auto& dtor : oldSlot) {
        dtor();
        ++totalDestroyed_;
    }
    oldSlot.clear();
}

void ResourceGC::flush() {
    // Drain incoming first
    {
        std::lock_guard lock(incomingMutex_);
        auto& slot = ring_[currentSlot_];
        slot.insert(slot.end(),
                    std::make_move_iterator(incoming_.begin()),
                    std::make_move_iterator(incoming_.end()));
        incoming_.clear();
    }

    // Destroy everything in all slots
    for (auto& slot : ring_) {
        for (auto& dtor : slot) {
            dtor();
            ++totalDestroyed_;
        }
        slot.clear();
    }
    currentSlot_ = 0;
}

} // namespace engine
