#pragma once

#include <algorithm>
#include <atomic>
#include <cstdlib>
#include <exception>
#include <mutex>
#include <thread>
#include <vector>

namespace mcvr {

inline uint32_t parallelThreadCount(const char *envName = "MCVR_SHADER_PACK_BUILD_THREADS") {
    const uint32_t hardwareThreads = std::max(1u, std::thread::hardware_concurrency());
    const char *rawValue = std::getenv(envName);
    if (rawValue != nullptr && rawValue[0] != '\0') {
        char *end = nullptr;
        long parsed = std::strtol(rawValue, &end, 10);
        if (end != rawValue && parsed > 0) {
            return static_cast<uint32_t>(std::min<long>(parsed, hardwareThreads));
        }
    }
    return std::min(8u, hardwareThreads);
}

template <typename Fn>
void parallelFor(size_t count, Fn &&fn, const char *envName = "MCVR_SHADER_PACK_BUILD_THREADS") {
    if (count == 0) { return; }

    uint32_t threadCount = std::min<uint32_t>(parallelThreadCount(envName), static_cast<uint32_t>(count));
    if (threadCount <= 1 || count == 1) {
        for (size_t i = 0; i < count; i++) { fn(i); }
        return;
    }

    std::atomic<size_t> nextIndex{0};
    std::exception_ptr firstException = nullptr;
    std::mutex exceptionMutex;
    std::vector<std::thread> workers;
    workers.reserve(threadCount);

    auto worker = [&]() {
        while (true) {
            size_t index = nextIndex.fetch_add(1, std::memory_order_relaxed);
            if (index >= count) { break; }
            try {
                fn(index);
            } catch (...) {
                std::lock_guard<std::mutex> lock(exceptionMutex);
                if (firstException == nullptr) { firstException = std::current_exception(); }
            }
        }
    };

    for (uint32_t i = 0; i < threadCount; i++) { workers.emplace_back(worker); }
    for (auto &thread : workers) { thread.join(); }
    if (firstException != nullptr) { std::rethrow_exception(firstException); }
}

} // namespace mcvr
