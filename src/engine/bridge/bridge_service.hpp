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

struct CmdChunkSubmit {
    int32_t chunkX = 0;
    int32_t chunkZ = 0;
    int32_t originX = 0, originY = 0, originZ = 0;
    std::vector<uint8_t> vertexData;   // 96-byte PBRTriangle vertices
    std::vector<uint32_t> indexData;
    uint32_t triangleCount = 0;
};

struct CmdChunkRemove {
    int32_t chunkX = 0;
    int32_t chunkZ = 0;
};

struct CmdCameraUpdate {
    // View matrix (column-major, 4x4)
    float view[16] = {};
    // Projection matrix (column-major, 4x4)
    float projection[16] = {};
    // Camera position in world space
    float posX = 0.0f, posY = 0.0f, posZ = 0.0f;
    // Camera direction
    float dirX = 0.0f, dirY = 0.0f, dirZ = -1.0f;
    // Near/far planes
    float nearPlane = 0.05f;
    float farPlane = 1024.0f;
    // Frame tick (to detect stale data)
    uint64_t tick = 0;
};

// Config patch: a type-erased mutation applied on the main thread.
// The JNI bridge captures the field write + side effects into a lambda.
struct CmdConfigPatch {
    std::function<void()> apply;
};

// Sky/atmosphere parameters bridged from Java each frame.
// Subset of V1's SkyUBO — the parts Java actually computes (sun/moon dir, fog, weather).
// Atmosphere LUT params and cloud volumetrics are added later.
struct CmdSkyUpdate {
    float baseColor[3] = {};
    float horizonColor[4] = {};
    float sunDirection[3] = {};
    float moonDirection[3] = {};
    uint32_t skyType = 0;
    uint32_t sunRisingOrSetting = 0;
    uint32_t skyDark = 0;
    uint32_t hasBlindnessOrDarkness = 0;
    uint32_t cameraSubmersionType = 0;
    uint32_t moonPhase = 0;
    float rainGradient = 0.0f;
    float thunderGradient = 0.0f;
    uint32_t sunTextureID = 0;
    uint32_t moonTextureID = 0;
};

// Replace the texture mapping SSBO contents with raw bytes from Java.
// Layout matches V1's TextureMapping table (TEX_ENTRY_COUNT * TEX_ENTRY_INTS * 4 bytes).
// Owning copy — JNI thread allocates, main thread consumes.
struct CmdTextureMappingUpdate {
    std::vector<uint8_t> data;
};

using BridgeCommand = std::variant<
    CmdPing,
    CmdWindowResize,
    CmdWorldLoad,
    CmdWorldUnload,
    CmdShutdown,
    CmdConfigPatch,
    CmdChunkSubmit,
    CmdChunkRemove,
    CmdCameraUpdate,
    CmdSkyUpdate,
    CmdTextureMappingUpdate
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
