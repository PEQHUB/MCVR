#pragma once

// Ownership: Value type (stack-allocated, movable).
// Thread: No internal synchronization — same thread safety as the contained T.

#include <optional>
#include <string>
#include <variant>
#include <vulkan/vulkan.h>

namespace engine::vk2 {

// Error type for Vulkan operations.
struct Error {
    VkResult vkResult = VK_SUCCESS;
    std::string message;

    bool isOutOfMemory() const {
        return vkResult == VK_ERROR_OUT_OF_HOST_MEMORY ||
               vkResult == VK_ERROR_OUT_OF_DEVICE_MEMORY;
    }

    bool isDeviceLost() const {
        return vkResult == VK_ERROR_DEVICE_LOST;
    }
};

// Result<T> — either a T value or an Error.
// All vk2 creation functions return this instead of throwing or crashing.
template <typename T>
class Result {
public:
    // Success construction
    Result(T value) : data_(std::move(value)) {}

    // Error construction
    Result(Error error) : data_(std::move(error)) {}

    bool ok() const { return std::holds_alternative<T>(data_); }
    explicit operator bool() const { return ok(); }

    T& value() { return std::get<T>(data_); }
    const T& value() const { return std::get<T>(data_); }

    Error& error() { return std::get<Error>(data_); }
    const Error& error() const { return std::get<Error>(data_); }

    // Move value out (call only when ok())
    T take() { return std::move(std::get<T>(data_)); }

private:
    std::variant<T, Error> data_;
};

// Specialization for void (operations that succeed or fail with no value)
template <>
class Result<void> {
public:
    Result() : error_(std::nullopt) {}
    Result(Error error) : error_(std::move(error)) {}

    bool ok() const { return !error_.has_value(); }
    explicit operator bool() const { return ok(); }

    Error& error() { return *error_; }
    const Error& error() const { return *error_; }

private:
    std::optional<Error> error_;
};

// Helper to create error results
inline Error makeError(VkResult vkResult, std::string message) {
    return Error{vkResult, std::move(message)};
}

} // namespace engine::vk2
