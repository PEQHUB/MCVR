#pragma once

// Ownership: EngineApp (1:1 lifetime).
// Thread: Main thread only.
// Dependencies: EngineServices.

#include <cstdint>

namespace engine {

class EngineServices;

enum class SessionState {
    Startup,
    Running,
    Paused,
    ShuttingDown
};

class EngineSession {
public:
    explicit EngineSession(EngineServices& services);
    ~EngineSession();

    SessionState state() const { return state_; }
    void setState(SessionState s) { state_ = s; }

    uint64_t frameNumber() const { return frameNumber_; }
    void advanceFrame() { ++frameNumber_; }

    EngineServices& services() { return services_; }

private:
    EngineServices& services_;
    SessionState state_ = SessionState::Startup;
    uint64_t frameNumber_ = 0;
};

} // namespace engine
