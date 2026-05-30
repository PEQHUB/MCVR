#pragma once
#include <atomic>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <utility>

template <typename Derived>
class Singleton {
  public:
    template <typename... Args>
    static void init(Args &&...args) {
        bool constructed = false;
        std::call_once(flag(), [&] {
            auto up = std::unique_ptr<Derived>(new Derived(std::forward<Args>(args)...));
            raw().store(up.get(), std::memory_order_release);
            storage() = std::move(up);
            constructed = true;
        });
        if (!constructed) { throw std::logic_error("Singleton<>::init(): already initialized"); }
    }

    static Derived &instance() {
        Derived *p = raw().load(std::memory_order_acquire);
        if (!p) { throw std::logic_error("Singleton<>::instance(): not initialized yet"); }
        return *p;
    }

    static bool is_initialized() noexcept {
        return raw().load(std::memory_order_acquire) != nullptr;
    }

    static Derived *try_instance() noexcept {
        return raw().load(std::memory_order_acquire);
    }

  protected:
    Singleton() = default;
    ~Singleton() = default;
    Singleton(const Singleton &) = delete;
    Singleton &operator=(const Singleton &) = delete;
    Singleton(Singleton &&) = delete;
    Singleton &operator=(Singleton &&) = delete;

  private:
    static std::unique_ptr<Derived> &storage() {
        static std::unique_ptr<Derived> ptr;
        return ptr;
    }
    static std::atomic<Derived *> &raw() {
        static std::atomic<Derived *> p{nullptr};
        return p;
    }
    static std::once_flag &flag() {
        static std::once_flag f;
        return f;
    }
};
