#pragma once

// Ownership: EngineServices (1:1).
// Thread: Main thread writes. Any thread reads via snapshot().
// Invariant: snapshot() returns a consistent, immutable copy.
//            Frame code never reads live config directly.

#include "engine_config.hpp"
#include "config_defaults.hpp"

#include <atomic>
#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

namespace engine {

using ConfigSubscriptionId = uint32_t;
using ConfigCallback = std::function<void(ConfigKey)>;

// Immutable config snapshot — created once per frame, read by any thread.
struct ConfigSnapshot {
    EngineConfig data;

    // Typed accessors (templates specialized per-key in config_defaults.hpp later).
    // For now, direct field access: snapshot.data.rayBounces, etc.
};

class ConfigService {
public:
    ConfigService();
    ~ConfigService();

    // Load from options.properties. Returns false on I/O error (defaults applied).
    bool load(const std::string& path);

    // Save to options.properties. Returns false on I/O error.
    bool save(const std::string& path) const;

    // Direct mutable access (main thread only, between frames).
    EngineConfig& live() { return live_; }
    const EngineConfig& live() const { return live_; }

    // Freeze current config into an immutable snapshot for this frame.
    // Call once at frame start. The returned shared_ptr is safe to read from any thread.
    std::shared_ptr<const ConfigSnapshot> snapshot();

    // Subscribe to changes on a specific key. Returns ID for unsubscribe.
    ConfigSubscriptionId subscribe(ConfigKey key, ConfigCallback callback);
    void unsubscribe(ConfigSubscriptionId id);

    // Notify subscribers after a config change (called by bridge setters).
    void notifyChange(ConfigKey key);

    // Apply range validation to all fields.
    void validate();

    // Get restart scope for a key.
    RestartScope restartScope(ConfigKey key) const { return getRestartScope(key); }

private:
    EngineConfig live_;
    std::shared_ptr<const ConfigSnapshot> currentSnapshot_;
    mutable std::mutex snapshotMutex_;

    struct Subscription {
        ConfigSubscriptionId id;
        ConfigKey key;
        ConfigCallback callback;
    };
    std::vector<Subscription> subscriptions_;
    ConfigSubscriptionId nextSubId_ = 1;
    std::mutex subMutex_;
};

} // namespace engine
