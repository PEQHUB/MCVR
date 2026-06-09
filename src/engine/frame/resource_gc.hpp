#pragma once

// Ownership: EngineServices (1:1).
// Thread: defer() is thread-safe. tick() and flush() are main-thread only.
// Invariant: No resource is freed while any queue might reference it.

#include <cstdint>
#include <functional>
#include <mutex>
#include <vector>

namespace engine {

// Deferred resource destruction. Resources are kept alive for N frames
// after the last active reference drops, ensuring GPU work completes.
class ResourceGC {
public:
    // Destructor callback — called when the resource should be freed.
    using Destructor = std::function<void()>;

    explicit ResourceGC(uint32_t ringSize = 32);
    ~ResourceGC();

    ResourceGC(const ResourceGC&) = delete;
    ResourceGC& operator=(const ResourceGC&) = delete;

    // Defer a destructor call for at least `ringSize` frames.
    // Thread-safe: can be called from any thread.
    void defer(Destructor dtor);

    // Advance one frame and destroy resources from the oldest slot.
    // Main thread only. Call once per frame after all GPU work is submitted.
    void tick();

    // Wait for all GPU work and destroy all deferred resources immediately.
    // Main thread only. Call at shutdown.
    void flush();

    uint32_t ringSize() const { return ringSize_; }
    uint64_t totalDeferred() const { return totalDeferred_; }
    uint64_t totalDestroyed() const { return totalDestroyed_; }

private:
    uint32_t ringSize_;
    uint32_t currentSlot_ = 0;

    // Ring of destructor lists. Each slot holds destructors deferred ~ringSize frames ago.
    std::vector<std::vector<Destructor>> ring_;

    // Incoming destructors from any thread, moved to ring during tick().
    std::mutex incomingMutex_;
    std::vector<Destructor> incoming_;

    uint64_t totalDeferred_ = 0;
    uint64_t totalDestroyed_ = 0;
};

} // namespace engine
