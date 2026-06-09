#pragma once

#include <atomic>
#include <condition_variable>
#include <exception>
#include <functional>
#include <future>
#include <memory>
#include <mutex>
#include <queue>
#include <string>
#include <thread>
#include <vector>

#include "core/vulkan/debug_utils.hpp"

/**
 * Simple thread pool for CPU-bound work.
 * Workers stay alive for the lifetime of the pool. Work is submitted via submit() or
 * parallelFor(). Thread count defaults to hardware_concurrency() - 2 (leave headroom
 * for the render thread and Java/Minecraft thread).
 */
class ThreadPool {
  public:
    explicit ThreadPool(uint32_t numThreads = 0) {
        if (numThreads == 0) {
            uint32_t hw = std::thread::hardware_concurrency();
            numThreads = (hw > 4) ? hw - 2 : std::max(hw, 1u);
        }
        for (uint32_t i = 0; i < numThreads; i++) {
            workers_.emplace_back([this, i] {
                vk::DebugUtils::setCurrentThreadName("Radiance Worker " + std::to_string(i));
                workerLoop();
            });
        }
    }

    ~ThreadPool() {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            stop_ = true;
        }
        cv_.notify_all();
        for (auto &w : workers_) {
            if (w.joinable()) w.join();
        }
    }

    ThreadPool(const ThreadPool &) = delete;
    ThreadPool &operator=(const ThreadPool &) = delete;

    /// Submit a callable and get a future for its result.
    template <class F>
    auto submit(F &&f) -> std::future<decltype(f())> {
        using ReturnType = decltype(f());
        auto task = std::make_shared<std::packaged_task<ReturnType()>>(std::forward<F>(f));
        std::future<ReturnType> result = task->get_future();
        {
            std::lock_guard<std::mutex> lock(mutex_);
            tasks_.emplace([task]() { (*task)(); });
        }
        cv_.notify_one();
        return result;
    }

    /// Execute f(i) for i in [0, count) across all worker threads, blocking until done.
    /// All shared state lives on the heap (SharedState) so workers can safely access it
    /// even after parallelFor returns — prevents use-after-free on stack locals.
    void parallelFor(uint32_t count, const std::function<void(uint32_t)> &f) {
        if (count == 0) return;
        if (count == 1) {
            f(0);
            return;
        }

        // Shared state on the heap — outlives parallelFor's stack frame.
        // Workers hold shared_ptr copies, so state survives until all workers finish.
        struct SharedState {
            std::atomic<uint32_t> nextIndex{0};
            std::atomic<uint32_t> completed{0};
            std::mutex doneMtx;
            std::condition_variable doneCv;
            std::mutex exceptionMtx;
            std::exception_ptr exception;
            uint32_t count;
        };
        auto state = std::make_shared<SharedState>();
        state->count = count;

        auto runLoop = [state, &f]() {
            while (true) {
                uint32_t idx = state->nextIndex.fetch_add(1, std::memory_order_relaxed);
                if (idx >= state->count) break;

                try {
                    f(idx);
                } catch (...) {
                    std::lock_guard<std::mutex> lock(state->exceptionMtx);
                    if (!state->exception) state->exception = std::current_exception();
                }

                if (state->completed.fetch_add(1, std::memory_order_acq_rel) + 1 == state->count) {
                    state->doneCv.notify_one();
                }
            }
        };

        uint32_t numTasks = std::min(count, static_cast<uint32_t>(workers_.size()));
        for (uint32_t t = 0; t < numTasks; t++) {
            std::lock_guard<std::mutex> lock(mutex_);
            tasks_.emplace(runLoop);
        }
        cv_.notify_all();

        runLoop();

        std::unique_lock<std::mutex> doneLock(state->doneMtx);
        state->doneCv.wait(doneLock, [&state] {
            return state->completed.load(std::memory_order_acquire) >= state->count;
        });

        if (state->exception) {
            std::rethrow_exception(state->exception);
        }
    }

    uint32_t threadCount() const { return static_cast<uint32_t>(workers_.size()); }

  private:
    void workerLoop() {
        while (true) {
            std::function<void()> task;
            {
                std::unique_lock<std::mutex> lock(mutex_);
                cv_.wait(lock, [this] { return stop_ || !tasks_.empty(); });
                if (stop_ && tasks_.empty()) return;
                task = std::move(tasks_.front());
                tasks_.pop();
            }
            task();
        }
    }

    std::vector<std::thread> workers_;
    std::queue<std::function<void()>> tasks_;
    std::mutex mutex_;
    std::condition_variable cv_;
    bool stop_ = false;
};
