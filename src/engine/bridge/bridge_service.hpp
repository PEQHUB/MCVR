#pragma once

// Ownership: EngineServices (1:1).
// Thread: JNI thread enqueues via post(). Main thread processes via flush().
// Invariant: No command directly mutates engine state. All mutations
//            happen during flush() on the main thread.

#include <cstdint>
#include <functional>
#include <mutex>
#include <string>
#include <variant>
#include <vector>

namespace engine {

// --- Command types (Java → C++) ---

// Each command is a struct. Add new ones as the bridge grows.
// Commands are value types — small, copyable, queued by value.

struct CmdPing {
    uint64_t timestamp = 0;
};

struct CmdWindowResize {
    uint32_t width = 0;
    uint32_t height = 0;
};

struct CmdWorldLoad {
    std::string regionPath;
};

struct CmdWorldUnload {};

struct CmdShutdown {};

using BridgeCommand = std::variant<
    CmdPing,
    CmdWindowResize,
    CmdWorldLoad,
    CmdWorldUnload,
    CmdShutdown
>;

// --- Event types (C++ → Java, fire-and-forget) ---

struct EvtFrameComplete {
    uint64_t frameNumber = 0;
    float frameTimeMs = 0.0f;
};

struct EvtDeviceLost {
    std::string reason;
};

using BridgeEvent = std::variant<
    EvtFrameComplete,
    EvtDeviceLost
>;

// --- Bridge Service ---

// Handler callback receives the full variant. Use std::visit to dispatch.
using CommandHandler = std::function<void(const BridgeCommand&)>;
using EventListener = std::function<void(const BridgeEvent&)>;

class BridgeService {
public:
    BridgeService() = default;
    ~BridgeService() = default;

    BridgeService(const BridgeService&) = delete;
    BridgeService& operator=(const BridgeService&) = delete;

    // --- Command queue (JNI thread → main thread) ---

    // Enqueue a command. Thread-safe (called from JNI thread).
    void post(BridgeCommand cmd);

    // Process all queued commands on the main thread.
    // Calls the registered handler for each command in FIFO order.
    // Returns the number of commands processed.
    uint32_t flush();

    // Register the command handler (replaces any previous handler).
    void setHandler(CommandHandler handler);

    // --- Event dispatch (main thread → listeners) ---

    // Fire an event to all registered listeners. Called from main thread.
    void emit(BridgeEvent event);

    // Register an event listener. Returns an ID for removal.
    uint32_t addListener(EventListener listener);
    void removeListener(uint32_t id);

    // --- Stats ---

    uint64_t totalCommandsProcessed() const { return totalProcessed_; }
    uint32_t pendingCommandCount() const;

private:
    // Double-buffered command queue: JNI writes to incoming_, flush() swaps and drains.
    std::mutex queueMutex_;
    std::vector<BridgeCommand> incoming_;
    std::vector<BridgeCommand> processing_;  // Only touched by main thread during flush()

    CommandHandler handler_;

    struct ListenerEntry {
        uint32_t id;
        EventListener listener;
    };
    std::vector<ListenerEntry> listeners_;
    uint32_t nextListenerId_ = 1;
    std::mutex listenerMutex_;

    uint64_t totalProcessed_ = 0;
};

} // namespace engine
