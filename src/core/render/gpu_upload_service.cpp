#include "core/render/gpu_upload_service.hpp"
#include "core/render/renderer.hpp"
#include "core/render/render_framework.hpp"
#include <algorithm>
#include <cstring>
#include <iostream>
#include <sstream>
#include <functional>
#include <vector>

bool GpuUploadService::initialize(std::shared_ptr<vk::Device> device, std::shared_ptr<vk::VMA> vma, uint64_t stagingBytes) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (initialized_) return true;
    if (!device || !vma) return false;

    device_ = device;
    vma_ = vma;

    // Check for dedicated transfer queue
    auto framework = Renderer::try_instance() ? Renderer::instance().framework() : nullptr;
    if (framework && framework->physicalDevice()) {
        VkPhysicalDevice physDev = framework->physicalDevice()->vkPhysicalDevice();
        uint32_t familyCount = 0;
        vkGetPhysicalDeviceQueueFamilyProperties(physDev, &familyCount, nullptr);
        std::vector<VkQueueFamilyProperties> families(familyCount);
        vkGetPhysicalDeviceQueueFamilyProperties(physDev, &familyCount, families.data());
        // Look for a queue family with transfer but not graphics
        for (const auto& family : families) {
            if ((family.queueFlags & VK_QUEUE_TRANSFER_BIT) &&
                !(family.queueFlags & VK_QUEUE_GRAPHICS_BIT)) {
                hasDedicatedTransferQueue_ = true;
                asyncTransferQueueUpload_ = true;
                break;
            }
        }
    }

    if (!hasDedicatedTransferQueue_) {
        asyncMainQueueUpload_ = true;
        asyncTransferQueueUpload_ = false;
    }

    // Create transfer command pool
    transferCmdPool_ = vk::CommandPool::create(
        framework ? framework->physicalDevice() : nullptr, device);

    // Create timeline semaphore for upload completion tracking
    uploadTimeline_ = vk::TimelineSemaphore::create(device, 0);
    nextTimelineValue_ = 1;

    // Create staging ring buffer
    stagingRingSize_ = stagingBytes;
    stagingRingOffset_ = 0;
    stagingRing_ = std::make_shared<vk::HostVisibleBuffer>(
        vma, device, stagingBytes,
        VK_BUFFER_USAGE_TRANSFER_SRC_BIT);

    initialized_ = true;
    std::cout << "[GpuUploadService] Initialized: transferQueue="
              << (hasDedicatedTransferQueue_ ? "dedicated" : "main-queue-fallback")
              << " stagingBytes=" << stagingBytes << std::endl;
    return true;
}

void GpuUploadService::shutdown() {
    std::lock_guard<std::mutex> lock(mutex_);

    // Wait for in-flight submissions
    for (auto& flight : inFlight_) {
        if (flight.timeline && device_) {
            flight.timeline->waitValue(flight.timelineValue, UINT64_MAX);
        }
        if (flight.cmd != VK_NULL_HANDLE && flight.cmdPool != VK_NULL_HANDLE && device_) {
            vkFreeCommandBuffers(device_->vkDevice(), flight.cmdPool, 1, &flight.cmd);
        }
    }
    inFlight_.clear();

    for (int i = 0; i < static_cast<int>(Priority::Count); i++) {
        queues_[i].clear();
    }

    stagingRing_.reset();
    uploadTimeline_.reset();
    transferCmdPool_.reset();
    initialized_ = false;
}

void GpuUploadService::cancelGeneration(uint64_t generation) {
    std::lock_guard<std::mutex> lock(mutex_);
    for (int i = 0; i < static_cast<int>(Priority::Count); i++) {
        auto& q = queues_[i];
        q.erase(std::remove_if(q.begin(), q.end(),
            [generation](const Pending& p) { return p.generation == generation; }),
            q.end());
    }

    // Recalculate queued and total pending bytes. In-flight uploads cannot be
    // unsignaled safely here, so keep them counted until timeline completion.
    uint64_t queuedBytes = 0, queuedVisibleBytes = 0;
    for (int i = 0; i < static_cast<int>(Priority::Count); i++) {
        for (const auto& p : queues_[i]) {
            queuedBytes += p.payload.size();
            if (p.visible) queuedVisibleBytes += p.payload.size();
        }
    }
    uint64_t flightBytes = 0, flightVisibleBytes = 0;
    for (const auto& flight : inFlight_) {
        flightBytes += flight.bytes;
        if (flight.visible) flightVisibleBytes += flight.bytes;
    }
    pendingUploadBytes_.store(queuedBytes + flightBytes, std::memory_order_relaxed);
    pendingVisibleUploadBytes_.store(queuedVisibleBytes + flightVisibleBytes, std::memory_order_relaxed);
    queuedBytes_.store(queuedBytes, std::memory_order_relaxed);
    queuedVisibleBytes_.store(queuedVisibleBytes, std::memory_order_relaxed);
    inFlightBytes_.store(flightBytes, std::memory_order_relaxed);
    inFlightVisibleBytes_.store(flightVisibleBytes, std::memory_order_relaxed);
}

bool GpuUploadService::enqueueTextureUpload(const TextureUpload& upload) {
    if (!initialized_ || upload.data == nullptr || upload.bytes == 0) return false;
    if (upload.generation == 0) return false;
    if (upload.dstImage == VK_NULL_HANDLE) return false;
    if (upload.layerCount == 0 || upload.width == 0 || upload.height == 0) return false;
    const uint64_t expectedBytes = static_cast<uint64_t>(upload.bytesPerLayer) * upload.layerCount;
    if (upload.bytesPerLayer == 0 || expectedBytes == 0 || upload.bytes < expectedBytes) return false;

    Pending pending{};
    pending.kind = Kind::TextureSubresource;
    pending.priority = upload.priority;
    pending.generation = upload.generation;
    pending.visible = upload.visible;
    pending.namespaceId = upload.namespaceId;
    pending.tier = upload.tier;
    pending.page = upload.page;
    pending.layer = upload.layer;
    pending.layerCount = upload.layerCount;
    pending.width = upload.width;
    pending.height = upload.height;
    pending.bytesPerLayer = upload.bytesPerLayer;
    pending.format = upload.format;
    pending.dstImage = upload.dstImage;
    pending.mipLevels = std::max(1u, upload.mipLevels);
    pending.payload.assign(upload.data, upload.data + upload.bytes);
    pending.onComplete = upload.onComplete;

    std::lock_guard<std::mutex> lock(mutex_);
    int idx = static_cast<int>(upload.priority);
    if (idx < 0 || idx >= static_cast<int>(Priority::Count))
        idx = static_cast<int>(Priority::BackgroundCtm);
    queues_[idx].push_back(std::move(pending));

    pendingUploadBytes_.fetch_add(upload.bytes, std::memory_order_relaxed);
    queuedBytes_.fetch_add(upload.bytes, std::memory_order_relaxed);
    if (upload.visible) {
        pendingVisibleUploadBytes_.fetch_add(upload.bytes, std::memory_order_relaxed);
        queuedVisibleBytes_.fetch_add(upload.bytes, std::memory_order_relaxed);
    }
    return true;
}

bool GpuUploadService::enqueueBufferUpload(const BufferUpload& upload) {
    if (!initialized_ || upload.data == nullptr || upload.bytes == 0) return false;
    if (upload.generation == 0) return false;
    if (upload.dst == VK_NULL_HANDLE) return false;
    if (upload.regionCount > 0 && upload.regions == nullptr) return false;

    Pending pending{};
    pending.kind = Kind::BufferRange;
    pending.priority = upload.priority;
    pending.generation = upload.generation;
    pending.visible = upload.visible;
    pending.dstBuffer = upload.dst;
    pending.dstOffset = upload.dstOffset;
    pending.dstOwner = upload.dstOwner;
    pending.payload.assign(upload.data, upload.data + upload.bytes);
    if (upload.regionCount > 0) {
        pending.copyRegions.assign(upload.regions, upload.regions + upload.regionCount);
        for (const VkBufferCopy& region : pending.copyRegions) {
            if (region.size == 0) return false;
            if (region.srcOffset > upload.bytes || region.size > upload.bytes - region.srcOffset) {
                return false;
            }
        }
    }

    std::lock_guard<std::mutex> lock(mutex_);
    int idx = static_cast<int>(upload.priority);
    if (idx < 0 || idx >= static_cast<int>(Priority::Count))
        idx = static_cast<int>(Priority::BackgroundCtm);
    queues_[idx].push_back(std::move(pending));

    pendingUploadBytes_.fetch_add(upload.bytes, std::memory_order_relaxed);
    queuedBytes_.fetch_add(upload.bytes, std::memory_order_relaxed);
    if (upload.visible) {
        pendingVisibleUploadBytes_.fetch_add(upload.bytes, std::memory_order_relaxed);
        queuedVisibleBytes_.fetch_add(upload.bytes, std::memory_order_relaxed);
    }
    return true;
}

void GpuUploadService::pump(uint64_t frameBudgetBytes) {
    if (!initialized_) return;
    std::lock_guard<std::mutex> lock(mutex_);

    uint64_t budgetUsed = 0;
    // Process queues in priority order
    for (int i = 0; i < static_cast<int>(Priority::Count); i++) {
        auto& q = queues_[i];
        while (!q.empty() && budgetUsed < frameBudgetBytes) {
            auto& pending = q.front();
            bool ok = false;
            if (pending.kind == Kind::TextureSubresource) {
                ok = submitTextureUploadLocked(pending);
            } else if (pending.kind == Kind::BufferRange) {
                ok = submitBufferUploadLocked(pending);
            }
            if (ok) {
                budgetUsed += pending.payload.size();
                uint64_t sz = pending.payload.size();
                bool visible = pending.visible;
                q.pop_front();
                queuedBytes_.fetch_sub(sz, std::memory_order_relaxed);
                inFlightBytes_.fetch_add(sz, std::memory_order_relaxed);
                if (visible) {
                    queuedVisibleBytes_.fetch_sub(sz, std::memory_order_relaxed);
                    inFlightVisibleBytes_.fetch_add(sz, std::memory_order_relaxed);
                }
                submittedBytes_.fetch_add(sz, std::memory_order_relaxed);
            } else {
                break; // Can't submit more this frame
            }
        }
    }
}

void GpuUploadService::pollCompletions() {
    if (!initialized_) return;
    std::vector<std::function<void(uint64_t)>> completions;
    {
    std::lock_guard<std::mutex> lock(mutex_);

    while (!inFlight_.empty()) {
        auto& flight = inFlight_.front();
        if (flight.timeline) {
            VkResult result = flight.timeline->waitValue(flight.timelineValue, 0);
            if (result == VK_TIMEOUT) break; // Not done yet
        }
        completedBytes_.fetch_add(flight.bytes, std::memory_order_relaxed);
        pendingUploadBytes_.fetch_sub(flight.bytes, std::memory_order_relaxed);
        inFlightBytes_.fetch_sub(flight.bytes, std::memory_order_relaxed);
        if (flight.visible) {
            completedVisibleBytes_.fetch_add(flight.bytes, std::memory_order_relaxed);
            pendingVisibleUploadBytes_.fetch_sub(flight.bytes, std::memory_order_relaxed);
            inFlightVisibleBytes_.fetch_sub(flight.bytes, std::memory_order_relaxed);
        }
        if (flight.onComplete) {
            completions.push_back(std::move(flight.onComplete));
        }
        // Release command buffer
        if (flight.cmd != VK_NULL_HANDLE && flight.cmdPool != VK_NULL_HANDLE && device_) {
            vkFreeCommandBuffers(device_->vkDevice(), flight.cmdPool, 1, &flight.cmd);
        }
        inFlight_.pop_front();
    }

    // Reset staging ring offset when all in-flight are done
    if (inFlight_.empty()) {
        stagingRingOffset_ = 0;
    }
    }

    for (auto& completion : completions) {
        if (completion) completion(0);
    }
}

bool GpuUploadService::generationIdle(uint64_t generation, bool visibleOnly) const {
    std::lock_guard<std::mutex> lock(mutex_);
    for (int i = 0; i < static_cast<int>(Priority::Count); i++) {
        for (const auto& p : queues_[i]) {
            if (p.generation == generation && (!visibleOnly || p.visible)) {
                return false;
            }
        }
    }
    for (const auto& flight : inFlight_) {
        if (flight.generation == generation && (!visibleOnly || flight.visible)) {
            return false;
        }
    }
    return true;
}

GpuUploadService::Status GpuUploadService::status() const {
    Status s;
    s.initialized = initialized_;
    s.hasDedicatedTransferQueue = hasDedicatedTransferQueue_;
    s.asyncTransferQueueUpload = asyncTransferQueueUpload_;
    s.asyncMainQueueUpload = asyncMainQueueUpload_;
    s.blockingFenceUploads = false; // Never in v4
    s.pendingUploadBytes = pendingUploadBytes_.load(std::memory_order_relaxed);
    s.pendingVisibleUploadBytes = pendingVisibleUploadBytes_.load(std::memory_order_relaxed);
    s.queuedBytes = queuedBytes_.load(std::memory_order_relaxed);
    s.queuedVisibleBytes = queuedVisibleBytes_.load(std::memory_order_relaxed);
    s.inFlightBytes = inFlightBytes_.load(std::memory_order_relaxed);
    s.inFlightVisibleBytes = inFlightVisibleBytes_.load(std::memory_order_relaxed);
    s.submittedBytes = submittedBytes_.load(std::memory_order_relaxed);
    s.completedBytes = completedBytes_.load(std::memory_order_relaxed);
    s.completedVisibleBytes = completedVisibleBytes_.load(std::memory_order_relaxed);
    s.failedBytes = failedBytes_.load(std::memory_order_relaxed);
    s.timelineSubmissions = timelineSubmissions_.load(std::memory_order_relaxed);
    s.actualVkCopyCommands = actualVkCopyCommands_.load(std::memory_order_relaxed);
    s.actualVkCopyBufferToImageCommands = actualVkCopyBufferToImageCommands_.load(std::memory_order_relaxed);
    s.actualVkBufferCopyCommands = actualVkBufferCopyCommands_.load(std::memory_order_relaxed);
    s.vkDeviceWaitIdleDuringLoad = vkDeviceWaitIdleDuringLoad_.load(std::memory_order_relaxed);
    return s;
}

std::string GpuUploadService::statusJson() const {
    auto s = status();
    std::ostringstream out;
    out << "{"
        << "\"schema\":\"radser_gpu_upload_service_status_v4\","
        << "\"initialized\":" << (s.initialized ? "true" : "false") << ","
        << "\"hasDedicatedTransferQueue\":" << (s.hasDedicatedTransferQueue ? "true" : "false") << ","
        << "\"asyncTransferQueueUpload\":" << (s.asyncTransferQueueUpload ? "true" : "false") << ","
        << "\"asyncMainQueueUpload\":" << (s.asyncMainQueueUpload ? "true" : "false") << ","
        << "\"blockingFenceUploads\":" << (s.blockingFenceUploads ? "true" : "false") << ","
        << "\"pendingUploadBytes\":" << s.pendingUploadBytes << ","
        << "\"pendingVisibleUploadBytes\":" << s.pendingVisibleUploadBytes << ","
        << "\"queuedBytes\":" << s.queuedBytes << ","
        << "\"queuedVisibleBytes\":" << s.queuedVisibleBytes << ","
        << "\"inFlightBytes\":" << s.inFlightBytes << ","
        << "\"inFlightVisibleBytes\":" << s.inFlightVisibleBytes << ","
        << "\"submittedBytes\":" << s.submittedBytes << ","
        << "\"completedBytes\":" << s.completedBytes << ","
        << "\"completedVisibleBytes\":" << s.completedVisibleBytes << ","
        << "\"failedBytes\":" << s.failedBytes << ","
        << "\"timelineSubmissions\":" << s.timelineSubmissions << ","
        << "\"actualVkCopyCommands\":" << s.actualVkCopyCommands << ","
        << "\"v4ActualVkCopyCommands\":" << s.actualVkCopyCommands << ","
        << "\"actualVkCopyBufferToImageCommands\":" << s.actualVkCopyBufferToImageCommands << ","
        << "\"actualVkBufferCopyCommands\":" << s.actualVkBufferCopyCommands << ","
        << "\"vkDeviceWaitIdleDuringLoad\":" << s.vkDeviceWaitIdleDuringLoad
        << "}";
    return out.str();
}

bool GpuUploadService::submitTextureUploadLocked(const Pending& pending) {
    if (!device_ || !transferCmdPool_ || !uploadTimeline_ || !stagingRing_) return false;

    uint64_t payloadSize = pending.payload.size();
    if (payloadSize == 0) return false;
    if (pending.dstImage == VK_NULL_HANDLE || pending.layerCount == 0 || pending.width == 0 || pending.height == 0) {
        return false;
    }
    const uint64_t requiredBytes = static_cast<uint64_t>(pending.bytesPerLayer) * pending.layerCount;
    if (pending.bytesPerLayer == 0 || payloadSize < requiredBytes) return false;

    // Allocate staging space (simple bump allocator, wraps around)
    uint64_t alignedOffset = (stagingRingOffset_ + 255) & ~255ULL;
    if (alignedOffset + payloadSize > stagingRingSize_) {
        // Ring buffer full this frame — try next frame
        return false;
    }
    uint64_t stagingOffset = alignedOffset;
    stagingRingOffset_ = alignedOffset + payloadSize;

    // Copy payload into staging buffer
    void* mappedPtr = stagingRing_->mappedPtr();
    if (!mappedPtr) return false;
    stagingRing_->uploadToBuffer(const_cast<uint8_t*>(pending.payload.data()), payloadSize, stagingOffset);

    // Allocate and begin command buffer
    VkCommandBufferAllocateInfo allocInfo{};
    allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    allocInfo.commandPool = transferCmdPool_->vkCommandPool();
    allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    allocInfo.commandBufferCount = 1;

    VkCommandBuffer cmd = VK_NULL_HANDLE;
    if (vkAllocateCommandBuffers(device_->vkDevice(), &allocInfo, &cmd) != VK_SUCCESS || cmd == VK_NULL_HANDLE) {
        return false;
    }

    VkCommandBufferBeginInfo beginInfo{};
    beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    if (vkBeginCommandBuffer(cmd, &beginInfo) != VK_SUCCESS) {
        vkFreeCommandBuffers(device_->vkDevice(), transferCmdPool_->vkCommandPool(), 1, &cmd);
        return false;
    }

    VkImageSubresourceRange range{};
    range.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    range.baseMipLevel = 0;
    range.levelCount = std::max(1u, pending.mipLevels);
    range.baseArrayLayer = pending.layer;
    range.layerCount = pending.layerCount;

    VkImageMemoryBarrier toTransfer{};
    toTransfer.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
    toTransfer.srcAccessMask = 0;
    toTransfer.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    toTransfer.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    toTransfer.newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
    toTransfer.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    toTransfer.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    toTransfer.image = pending.dstImage;
    toTransfer.subresourceRange = range;
    vkCmdPipelineBarrier(cmd,
        VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
        VK_PIPELINE_STAGE_TRANSFER_BIT,
        0, 0, nullptr, 0, nullptr, 1, &toTransfer);

    std::vector<VkBufferImageCopy> copies;
    copies.reserve(pending.layerCount);
    for (uint32_t i = 0; i < pending.layerCount; ++i) {
        VkBufferImageCopy region{};
        region.bufferOffset = stagingOffset + static_cast<uint64_t>(pending.bytesPerLayer) * i;
        region.bufferRowLength = 0;
        region.bufferImageHeight = 0;
        region.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        region.imageSubresource.mipLevel = 0;
        region.imageSubresource.baseArrayLayer = pending.layer + i;
        region.imageSubresource.layerCount = 1;
        region.imageOffset = {0, 0, 0};
        region.imageExtent = {pending.width, pending.height, 1};
        copies.push_back(region);
    }

    vkCmdCopyBufferToImage(cmd,
        stagingRing_->vkBuffer(),
        pending.dstImage,
        VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
        static_cast<uint32_t>(copies.size()),
        copies.data());

    VkImageMemoryBarrier toShader{};
    toShader.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
    toShader.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    toShader.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
    toShader.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
    toShader.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    toShader.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    toShader.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    toShader.image = pending.dstImage;
    toShader.subresourceRange = range;
    vkCmdPipelineBarrier(cmd,
        VK_PIPELINE_STAGE_TRANSFER_BIT,
        VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT | VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_RAY_TRACING_SHADER_BIT_KHR,
        0, 0, nullptr, 0, nullptr, 1, &toShader);

    if (vkEndCommandBuffer(cmd) != VK_SUCCESS) {
        vkFreeCommandBuffers(device_->vkDevice(), transferCmdPool_->vkCommandPool(), 1, &cmd);
        failedBytes_.fetch_add(payloadSize, std::memory_order_relaxed);
        return false;
    }

    // Submit with timeline semaphore signal
    uint64_t timelineValue = nextTimelineValue_++;
    VkTimelineSemaphoreSubmitInfo timelineInfo{};
    timelineInfo.sType = VK_STRUCTURE_TYPE_TIMELINE_SEMAPHORE_SUBMIT_INFO;
    timelineInfo.signalSemaphoreValueCount = 1;
    timelineInfo.pSignalSemaphoreValues = &timelineValue;

    VkSemaphore signalSema = uploadTimeline_->vkSemaphore();
    VkSubmitInfo submitInfo{};
    submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submitInfo.pNext = &timelineInfo;
    submitInfo.commandBufferCount = 1;
    submitInfo.pCommandBuffers = &cmd;
    submitInfo.signalSemaphoreCount = 1;
    submitInfo.pSignalSemaphores = &signalSema;

    std::lock_guard<std::mutex> qLock(device_->queueMutex());
    VkResult result = vkQueueSubmit(device_->mainVkQueue(), 1, &submitInfo, VK_NULL_HANDLE);
    if (result != VK_SUCCESS) {
        vkFreeCommandBuffers(device_->vkDevice(), transferCmdPool_->vkCommandPool(), 1, &cmd);
        return false;
    }

    // Track in-flight
    InFlight flight{};
    flight.generation = pending.generation;
    flight.cmd = cmd;
    flight.cmdPool = transferCmdPool_->vkCommandPool();
    flight.stagingBuffer = stagingRing_;
    flight.timeline = uploadTimeline_;
    flight.timelineValue = timelineValue;
    flight.bytes = payloadSize;
    flight.visible = pending.visible;
    flight.onComplete = pending.onComplete;
    inFlight_.push_back(std::move(flight));

    timelineSubmissions_.fetch_add(1, std::memory_order_relaxed);
    actualVkCopyCommands_.fetch_add(copies.size(), std::memory_order_relaxed);
    actualVkCopyBufferToImageCommands_.fetch_add(copies.size(), std::memory_order_relaxed);
    return true;
}

bool GpuUploadService::submitBufferUploadLocked(const Pending& pending) {
    if (!device_ || !transferCmdPool_ || !uploadTimeline_ || !stagingRing_) return false;

    uint64_t payloadSize = pending.payload.size();
    if (payloadSize == 0) return false;

    // Allocate staging space
    uint64_t alignedOffset = (stagingRingOffset_ + 255) & ~255ULL;
    if (alignedOffset + payloadSize > stagingRingSize_) {
        return false;
    }
    uint64_t stagingOffset = alignedOffset;
    stagingRingOffset_ = alignedOffset + payloadSize;

    // Copy payload into staging buffer
    void* mappedPtr = stagingRing_->mappedPtr();
    if (!mappedPtr) return false;
    stagingRing_->uploadToBuffer(const_cast<uint8_t*>(pending.payload.data()), payloadSize, stagingOffset);

    // Allocate and begin command buffer
    VkCommandBufferAllocateInfo allocInfo{};
    allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    allocInfo.commandPool = transferCmdPool_->vkCommandPool();
    allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    allocInfo.commandBufferCount = 1;

    VkCommandBuffer cmd = VK_NULL_HANDLE;
    if (vkAllocateCommandBuffers(device_->vkDevice(), &allocInfo, &cmd) != VK_SUCCESS || cmd == VK_NULL_HANDLE) {
        return false;
    }

    VkCommandBufferBeginInfo beginInfo{};
    beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    if (vkBeginCommandBuffer(cmd, &beginInfo) != VK_SUCCESS) {
        vkFreeCommandBuffers(device_->vkDevice(), transferCmdPool_->vkCommandPool(), 1, &cmd);
        failedBytes_.fetch_add(payloadSize, std::memory_order_relaxed);
        return false;
    }

    uint32_t copyCommandCount = 0;
    if (!pending.copyRegions.empty()) {
        std::vector<VkBufferCopy> regions = pending.copyRegions;
        for (VkBufferCopy& region : regions) {
            region.srcOffset += stagingOffset;
        }
        vkCmdCopyBuffer(cmd, stagingRing_->vkBuffer(), pending.dstBuffer,
                        static_cast<uint32_t>(regions.size()), regions.data());
        copyCommandCount = static_cast<uint32_t>(regions.size());
    } else if (pending.dstBuffer != VK_NULL_HANDLE) {
        VkBufferCopy region{};
        region.srcOffset = stagingOffset;
        region.dstOffset = pending.dstOffset;
        region.size = payloadSize;
        vkCmdCopyBuffer(cmd, stagingRing_->vkBuffer(), pending.dstBuffer, 1, &region);
        copyCommandCount = 1;
    }

    if (vkEndCommandBuffer(cmd) != VK_SUCCESS) {
        vkFreeCommandBuffers(device_->vkDevice(), transferCmdPool_->vkCommandPool(), 1, &cmd);
        return false;
    }

    // Submit with timeline semaphore signal
    uint64_t timelineValue = nextTimelineValue_++;
    VkTimelineSemaphoreSubmitInfo timelineInfo{};
    timelineInfo.sType = VK_STRUCTURE_TYPE_TIMELINE_SEMAPHORE_SUBMIT_INFO;
    timelineInfo.signalSemaphoreValueCount = 1;
    timelineInfo.pSignalSemaphoreValues = &timelineValue;

    VkSemaphore signalSema = uploadTimeline_->vkSemaphore();
    VkSubmitInfo submitInfo{};
    submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submitInfo.pNext = &timelineInfo;
    submitInfo.commandBufferCount = 1;
    submitInfo.pCommandBuffers = &cmd;
    submitInfo.signalSemaphoreCount = 1;
    submitInfo.pSignalSemaphores = &signalSema;

    std::lock_guard<std::mutex> qLock(device_->queueMutex());
    VkResult result = vkQueueSubmit(device_->mainVkQueue(), 1, &submitInfo, VK_NULL_HANDLE);
    if (result != VK_SUCCESS) {
        vkFreeCommandBuffers(device_->vkDevice(), transferCmdPool_->vkCommandPool(), 1, &cmd);
        return false;
    }

    // Track in-flight
    InFlight flight{};
    flight.generation = pending.generation;
    flight.cmd = cmd;
    flight.cmdPool = transferCmdPool_->vkCommandPool();
    flight.stagingBuffer = stagingRing_;
    flight.dstOwner = pending.dstOwner;
    flight.timeline = uploadTimeline_;
    flight.timelineValue = timelineValue;
    flight.bytes = payloadSize;
    flight.visible = pending.visible;
    inFlight_.push_back(std::move(flight));

    timelineSubmissions_.fetch_add(1, std::memory_order_relaxed);
    actualVkBufferCopyCommands_.fetch_add(copyCommandCount, std::memory_order_relaxed);
    return true;
}

bool GpuUploadService::ensureStagingCapacityLocked(uint64_t bytes) {
    if (!stagingRing_) return false;
    // Check if remaining staging space is enough
    uint64_t alignedOffset = (stagingRingOffset_ + 255) & ~255ULL;
    if (alignedOffset + bytes <= stagingRingSize_) return true;
    // Not enough space — pump completions to free up staging buffers
    // (InFlight holds shared_ptr to stagingRing_, so completions don't free memory,
    //  but we reset the bump offset when all in-flight are done)
    return false;
}
