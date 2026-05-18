#ifdef _WIN32
#include "core/render/overlay_compositor.hpp"

#include "core/render/render_framework.hpp"
#include "core/render/renderer.hpp"

#include <d3d11_1.h>
#include <dcomp.h>
#include <dwmapi.h>
#include <dxgi1_2.h>

#include <cstdarg>
#include <filesystem>
#include <fstream>
#include <iostream>

static std::ostream &ocCout() { return std::cout << "[OverlayCompositor] "; }
static std::ostream &ocCerr() { return std::cerr << "[OverlayCompositor] "; }

// Window class name (registered once)
static const wchar_t *kOverlayClassName = L"RadianceOverlay";
static bool sWindowClassRegistered = false;

// Diagnostic log file (unconditional — always written during init for debugging)
static std::ofstream sDiagFile;

void OverlayCompositor::diagLog(const char *fmt, ...) {
    char buf[1024];
    va_list args;
    va_start(args, fmt);
    vsnprintf(buf, sizeof(buf), fmt, args);
    va_end(args);

    // Console
    std::cerr << "[OverlayCompositor] " << buf << std::endl;

    // File
    if (sDiagFile.is_open()) {
        // Timestamp
        SYSTEMTIME st;
        GetLocalTime(&st);
        char ts[32];
        snprintf(ts, sizeof(ts), "%02d:%02d:%02d.%03d",
                 st.wHour, st.wMinute, st.wSecond, st.wMilliseconds);
        sDiagFile << ts << " " << buf << std::endl;
        sDiagFile.flush();
    }
}

OverlayCompositor::~OverlayCompositor() {
    destroy();
}

bool OverlayCompositor::init(std::shared_ptr<Framework> framework) {
    framework_ = framework;
    initFailReason_.clear();

    // Open diagnostic log file
    auto logDir = Renderer::folderPath / "logs";
    std::filesystem::create_directories(logDir);
    sDiagFile.open((logDir / "overlay_diag.log").string(), std::ios::trunc);

    auto device = framework->device();
    auto physDev = framework->physicalDevice();
    auto swapchain = framework->swapchain();

    width_ = swapchain->vkExtent().width;
    height_ = swapchain->vkExtent().height;

    diagLog("init() start: %ux%u", width_, height_);

    // Get game HWND
    diagLog("GLFW_GetWin32Window ptr=%p", (void *)GLFW_GetWin32Window);
    gameHwnd_ = GLFW_GetWin32Window(framework->window()->window());
    diagLog("gameHwnd=%p", gameHwnd_);
    if (!gameHwnd_) {
        initFailReason_ = "GLFW_GetWin32Window returned null";
        diagLog("FAIL: %s", initFailReason_.c_str());
        return false;
    }

    diagLog("running precondition checks...");
    if (!checkPreconditions()) {
        if (initFailReason_.empty()) initFailReason_ = "preconditions failed";
        diagLog("FAIL: %s", initFailReason_.c_str());
        return false;
    }
    diagLog("preconditions PASSED");

    diagLog("creating overlay window...");
    if (!createOverlayWindow()) { initFailReason_ = "createOverlayWindow failed"; diagLog("FAIL: %s", initFailReason_.c_str()); destroy(); return false; }
    diagLog("overlay window CREATED hwnd=%p", overlayHwnd_);

    diagLog("creating D3D11 device...");
    if (!createD3D11Device()) { initFailReason_ = "createD3D11Device failed"; diagLog("FAIL: %s", initFailReason_.c_str()); destroy(); return false; }
    diagLog("D3D11 device CREATED");

    diagLog("creating DXGI swapchain...");
    if (!createDxgiSwapChain()) { initFailReason_ = "createDxgiSwapChain failed"; diagLog("FAIL: %s", initFailReason_.c_str()); destroy(); return false; }
    diagLog("DXGI swapchain CREATED");

    diagLog("creating DirectComposition...");
    if (!createDirectComposition()) { initFailReason_ = "createDirectComposition failed"; diagLog("FAIL: %s", initFailReason_.c_str()); destroy(); return false; }
    diagLog("DirectComposition CREATED");

    for (int i = 0; i < kSharedImageCount; i++) {
        diagLog("creating shared Vulkan image %d...", i);
        if (!createSharedVulkanImage(i)) { initFailReason_ = "createSharedVulkanImage failed idx=" + std::to_string(i); diagLog("FAIL: %s", initFailReason_.c_str()); destroy(); return false; }
        diagLog("shared image %d CREATED handle=%p", i, sharedHandles_[i]);
    }

    diagLog("creating premultiply pipeline...");
    if (!createPremultiplyPipeline()) { initFailReason_ = "createPremultiplyPipeline failed"; diagLog("FAIL: %s", initFailReason_.c_str()); destroy(); return false; }
    diagLog("premultiply pipeline CREATED");

    active_ = true;
    initFailReason_.clear();
    diagLog("init() SUCCESS: active=true %ux%u", width_, height_);
    return true;
}

std::string OverlayCompositor::getDiagnostics() const {
    std::string s;
    s += "active=" + std::to_string(active_ ? 1 : 0);
    s += ",width=" + std::to_string(width_);
    s += ",height=" + std::to_string(height_);
    s += ",gameHwnd=" + std::to_string(reinterpret_cast<uintptr_t>(gameHwnd_));
    s += ",overlayHwnd=" + std::to_string(reinterpret_cast<uintptr_t>(overlayHwnd_));
    s += ",d3dDevice=" + std::to_string(d3dDevice_ != nullptr ? 1 : 0);
    s += ",dxgiSwapChain=" + std::to_string(dxgiSwapChain_ != nullptr ? 1 : 0);
    s += ",dcompDevice=" + std::to_string(dcompDevice_ != nullptr ? 1 : 0);
    s += ",sharedImg0=" + std::to_string(sharedImages_[0] != VK_NULL_HANDLE ? 1 : 0);
    s += ",sharedImg1=" + std::to_string(sharedImages_[1] != VK_NULL_HANDLE ? 1 : 0);
    s += ",pipeline=" + std::to_string(premultiplyPipeline_ != nullptr ? 1 : 0);
    s += ",currentIdx=" + std::to_string(currentSharedIndex_);
    if (!initFailReason_.empty()) {
        s += ",failReason=" + initFailReason_;
    }
    return s;
}

void OverlayCompositor::destroy() {
    active_ = false;

    auto fw = framework_.lock();
    VkDevice vkDev = VK_NULL_HANDLE;
    if (fw) vkDev = fw->device()->vkDevice();

    // Vulkan premultiply pipeline
    premultiplyPipeline_.reset();
    premultiplyDescriptor_.reset();
    premultiplyRenderPass_.reset();
    premultiplySampler_.reset();
    premultiplyFrag_.reset();
    premultiplyVert_.reset();

    if (vkDev) {
        for (int i = 0; i < kSharedImageCount; i++) {
            if (premultiplyFramebuffers_[i] != VK_NULL_HANDLE) {
                vkDestroyFramebuffer(vkDev, premultiplyFramebuffers_[i], nullptr);
                premultiplyFramebuffers_[i] = VK_NULL_HANDLE;
            }
        }
    }

    // Vulkan shared images (destroy before D3D11 — Vulkan imported from D3D11)
    if (vkDev) {
        for (int i = 0; i < kSharedImageCount; i++) {
            if (sharedViews_[i] != VK_NULL_HANDLE) {
                vkDestroyImageView(vkDev, sharedViews_[i], nullptr);
                sharedViews_[i] = VK_NULL_HANDLE;
            }
            if (sharedImages_[i] != VK_NULL_HANDLE) {
                vkDestroyImage(vkDev, sharedImages_[i], nullptr);
                sharedImages_[i] = VK_NULL_HANDLE;
            }
            if (sharedMemories_[i] != VK_NULL_HANDLE) {
                vkFreeMemory(vkDev, sharedMemories_[i], nullptr);
                sharedMemories_[i] = VK_NULL_HANDLE;
            }
        }
    }

    // Keyed mutexes (release before D3D11 textures)
    for (int i = 0; i < kSharedImageCount; i++) {
        if (sharedMutexes_[i]) {
            sharedMutexes_[i]->Release();
            sharedMutexes_[i] = nullptr;
        }
    }

    // D3D11 shared textures (source — destroy after Vulkan)
    for (int i = 0; i < kSharedImageCount; i++) {
        if (d3dSharedTextures_[i]) {
            d3dSharedTextures_[i]->Release();
            d3dSharedTextures_[i] = nullptr;
        }
    }

    // Close shared handles (after both Vulkan and D3D11)
    for (int i = 0; i < kSharedImageCount; i++) {
        if (sharedHandles_[i]) {
            CloseHandle(static_cast<HANDLE>(sharedHandles_[i]));
            sharedHandles_[i] = nullptr;
        }
    }

    // DirectComposition
    if (dcompVisual_) { dcompVisual_->Release(); dcompVisual_ = nullptr; }
    if (dcompTarget_) { dcompTarget_->Release(); dcompTarget_ = nullptr; }
    if (dcompDevice_) { dcompDevice_->Release(); dcompDevice_ = nullptr; }

    // DXGI
    if (dxgiSwapChain_) { dxgiSwapChain_->Release(); dxgiSwapChain_ = nullptr; }

    // D3D11
    if (d3dContext_) { d3dContext_->Release(); d3dContext_ = nullptr; }
    if (d3dDevice_) { d3dDevice_->Release(); d3dDevice_ = nullptr; }

    // Overlay window
    destroyOverlayWindow();

    currentSharedIndex_ = 0;
}

// ─── Precondition Checks ──────────────────────────────────────────────────

bool OverlayCompositor::checkPreconditions() {
    auto fw = framework_.lock();
    if (!fw) return false;

    auto physDev = fw->physicalDevice();

    // Check 1: External image format support for D3D11 texture sharing
    VkPhysicalDeviceExternalImageFormatInfo externalInfo{};
    externalInfo.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_EXTERNAL_IMAGE_FORMAT_INFO;
    externalInfo.handleType = VK_EXTERNAL_MEMORY_HANDLE_TYPE_D3D11_TEXTURE_BIT;

    VkPhysicalDeviceImageFormatInfo2 formatInfo{};
    formatInfo.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_IMAGE_FORMAT_INFO_2;
    formatInfo.pNext = &externalInfo;
    formatInfo.format = VK_FORMAT_R8G8B8A8_SRGB;
    formatInfo.type = VK_IMAGE_TYPE_2D;
    formatInfo.tiling = VK_IMAGE_TILING_OPTIMAL;
    formatInfo.usage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;

    VkExternalImageFormatProperties externalProps{};
    externalProps.sType = VK_STRUCTURE_TYPE_EXTERNAL_IMAGE_FORMAT_PROPERTIES;

    VkImageFormatProperties2 props{};
    props.sType = VK_STRUCTURE_TYPE_IMAGE_FORMAT_PROPERTIES_2;
    props.pNext = &externalProps;

    VkResult result = vkGetPhysicalDeviceImageFormatProperties2(
        physDev->vkPhysicalDevice(), &formatInfo, &props);
    diagLog("vkGetPhysicalDeviceImageFormatProperties2 result=%d", (int)result);
    if (result != VK_SUCCESS) {
        initFailReason_ = "external image format not supported (result=" + std::to_string((int)result) + ")";
        return false;
    }

    auto features = externalProps.externalMemoryProperties.externalMemoryFeatures;
    diagLog("externalMemoryFeatures=0x%x", features);
    if (!(features & VK_EXTERNAL_MEMORY_FEATURE_IMPORTABLE_BIT)) {
        initFailReason_ = "external memory not importable (features=0x" + std::to_string(features) + ")";
        return false;
    }

    // Check 2: DWM composition active
    BOOL dwmEnabled = FALSE;
    DwmIsCompositionEnabled(&dwmEnabled);
    diagLog("DwmIsCompositionEnabled=%d", (int)dwmEnabled);
    if (!dwmEnabled) {
        initFailReason_ = "DWM composition not active";
        return false;
    }
    return true;
}

// ─── Overlay Window ───────────────────────────────────────────────────────

// WNDPROC that makes the overlay fully transparent to input.
// Returning HTTRANSPARENT from WM_NCHITTEST causes Win32 to forward all mouse
// events to the window underneath (the GLFW game window). This is safer than
// WS_EX_LAYERED which may conflict with WS_EX_NOREDIRECTIONBITMAP (DComp).
static LRESULT CALLBACK overlayWndProc(HWND hwnd, UINT msg, WPARAM wParam, LPARAM lParam) {
    if (msg == WM_NCHITTEST) return HTTRANSPARENT;
    return DefWindowProcW(hwnd, msg, wParam, lParam);
}

bool OverlayCompositor::createOverlayWindow() {
    if (!sWindowClassRegistered) {
        WNDCLASSEXW wc = {};
        wc.cbSize = sizeof(wc);
        wc.lpfnWndProc = overlayWndProc;
        wc.hInstance = GetModuleHandleW(nullptr);
        wc.lpszClassName = kOverlayClassName;
        if (!RegisterClassExW(&wc)) {
            ocCerr() << "RegisterClassExW failed: " << GetLastError() << std::endl;
            return false;
        }
        sWindowClassRegistered = true;
    }

    RECT gameRect;
    GetClientRect(static_cast<HWND>(gameHwnd_), &gameRect);
    MapWindowPoints(static_cast<HWND>(gameHwnd_), nullptr, reinterpret_cast<POINT *>(&gameRect), 2);

    overlayHwnd_ = CreateWindowExW(
        WS_EX_NOREDIRECTIONBITMAP | WS_EX_TRANSPARENT | WS_EX_NOACTIVATE | WS_EX_TOOLWINDOW,
        kOverlayClassName,
        L"",
        WS_POPUP | WS_VISIBLE,
        gameRect.left, gameRect.top,
        gameRect.right - gameRect.left,
        gameRect.bottom - gameRect.top,
        static_cast<HWND>(gameHwnd_),  // owner
        nullptr, GetModuleHandleW(nullptr), nullptr);

    if (!overlayHwnd_) {
        ocCerr() << "CreateWindowExW failed: " << GetLastError() << std::endl;
        return false;
    }

    ocCout() << "overlay window created" << std::endl;
    return true;
}

void OverlayCompositor::destroyOverlayWindow() {
    if (overlayHwnd_) {
        DestroyWindow(static_cast<HWND>(overlayHwnd_));
        overlayHwnd_ = nullptr;
    }
}

// ─── D3D11 Device ─────────────────────────────────────────────────────────

bool OverlayCompositor::createD3D11Device() {
    auto fw = framework_.lock();
    if (!fw) return false;

    // Get Vulkan device LUID for adapter matching
    VkPhysicalDeviceIDProperties idProps{};
    idProps.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ID_PROPERTIES;
    VkPhysicalDeviceProperties2 devProps2{};
    devProps2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2;
    devProps2.pNext = &idProps;
    vkGetPhysicalDeviceProperties2(fw->physicalDevice()->vkPhysicalDevice(), &devProps2);

    LUID vkLuid;
    memcpy(&vkLuid, idProps.deviceLUID, sizeof(LUID));

    // Find matching DXGI adapter
    IDXGIFactory1 *factory = nullptr;
    HRESULT hr = CreateDXGIFactory1(IID_PPV_ARGS(&factory));
    if (FAILED(hr)) {
        ocCerr() << "CreateDXGIFactory1 failed: 0x" << std::hex << hr << std::endl;
        return false;
    }

    IDXGIAdapter1 *matchedAdapter = nullptr;
    IDXGIAdapter1 *adapter = nullptr;
    for (UINT i = 0; factory->EnumAdapters1(i, &adapter) != DXGI_ERROR_NOT_FOUND; i++) {
        DXGI_ADAPTER_DESC1 desc;
        adapter->GetDesc1(&desc);
        if (memcmp(&desc.AdapterLuid, &vkLuid, sizeof(LUID)) == 0) {
            matchedAdapter = adapter;
            ocCout() << "matched DXGI adapter: " << std::hex
                     << desc.AdapterLuid.HighPart << ":" << desc.AdapterLuid.LowPart << std::endl;
            break;
        }
        adapter->Release();
    }
    factory->Release();

    if (!matchedAdapter) {
        ocCerr() << "no DXGI adapter matches Vulkan device LUID" << std::endl;
        return false;
    }

    // Create D3D11 device on matched adapter
    D3D_FEATURE_LEVEL featureLevels[] = {D3D_FEATURE_LEVEL_11_1, D3D_FEATURE_LEVEL_11_0};
    UINT flags = D3D11_CREATE_DEVICE_BGRA_SUPPORT;

    hr = D3D11CreateDevice(
        matchedAdapter,
        D3D_DRIVER_TYPE_UNKNOWN, // adapter specified
        nullptr, flags,
        featureLevels, 2,
        D3D11_SDK_VERSION,
        &d3dDevice_, nullptr, &d3dContext_);

    matchedAdapter->Release();

    if (FAILED(hr)) {
        ocCerr() << "D3D11CreateDevice failed: 0x" << std::hex << hr << std::endl;
        return false;
    }

    ocCout() << "D3D11 device created" << std::endl;
    return true;
}

// ─── DXGI Swap Chain ──────────────────────────────────────────────────────

bool OverlayCompositor::createDxgiSwapChain() {
    IDXGIDevice *dxgiDev = nullptr;
    HRESULT hr = d3dDevice_->QueryInterface(IID_PPV_ARGS(&dxgiDev));
    if (FAILED(hr)) {
        ocCerr() << "QueryInterface IDXGIDevice failed" << std::endl;
        return false;
    }

    IDXGIAdapter *adapter = nullptr;
    dxgiDev->GetAdapter(&adapter);

    IDXGIFactory2 *factory2 = nullptr;
    adapter->GetParent(IID_PPV_ARGS(&factory2));
    adapter->Release();

    DXGI_SWAP_CHAIN_DESC1 desc = {};
    desc.Width = width_;
    desc.Height = height_;
    desc.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
    desc.SampleDesc.Count = 1;
    desc.BufferUsage = DXGI_USAGE_RENDER_TARGET_OUTPUT;
    desc.BufferCount = 2;
    desc.SwapEffect = DXGI_SWAP_EFFECT_FLIP_SEQUENTIAL;
    desc.AlphaMode = DXGI_ALPHA_MODE_PREMULTIPLIED;
    desc.Flags = DXGI_SWAP_CHAIN_FLAG_FRAME_LATENCY_WAITABLE_OBJECT;

    hr = factory2->CreateSwapChainForComposition(d3dDevice_, &desc, nullptr, &dxgiSwapChain_);
    factory2->Release();
    dxgiDev->Release();

    if (FAILED(hr)) {
        ocCerr() << "CreateSwapChainForComposition failed: 0x" << std::hex << hr << std::endl;
        return false;
    }

    // Set up waitable swapchain for display-rate pacing.
    // The waitable object signals when the swapchain is ready for the next frame,
    // perfectly matching the display refresh rate (e.g., 240Hz = every ~4.17ms).
    IDXGISwapChain2 *sc2 = nullptr;
    if (SUCCEEDED(dxgiSwapChain_->QueryInterface(IID_PPV_ARGS(&sc2)))) {
        sc2->SetMaximumFrameLatency(1);
        frameLatencyWaitable_ = sc2->GetFrameLatencyWaitableObject();
        sc2->Release();
        diagLog("waitable swapchain enabled");
    } else {
        diagLog("IDXGISwapChain2 not available, falling back to timed present");
    }

    ocCout() << "DXGI swapchain created " << width_ << "x" << height_
             << " PREMULTIPLIED" << std::endl;
    return true;
}

// ─── DirectComposition ────────────────────────────────────────────────────

bool OverlayCompositor::createDirectComposition() {
    IDXGIDevice *dxgiDev = nullptr;
    HRESULT hr = d3dDevice_->QueryInterface(IID_PPV_ARGS(&dxgiDev));
    if (FAILED(hr)) {
        ocCerr() << "QueryInterface IDXGIDevice failed (dcomp)" << std::endl;
        return false;
    }

    hr = DCompositionCreateDevice(dxgiDev, IID_PPV_ARGS(&dcompDevice_));
    dxgiDev->Release();
    if (FAILED(hr)) {
        ocCerr() << "DCompositionCreateDevice failed: 0x" << std::hex << hr << std::endl;
        return false;
    }

    hr = dcompDevice_->CreateTargetForHwnd(static_cast<HWND>(overlayHwnd_), TRUE, &dcompTarget_);
    if (FAILED(hr)) {
        ocCerr() << "CreateTargetForHwnd failed: 0x" << std::hex << hr << std::endl;
        return false;
    }

    hr = dcompDevice_->CreateVisual(&dcompVisual_);
    if (FAILED(hr)) {
        ocCerr() << "CreateVisual failed: 0x" << std::hex << hr << std::endl;
        return false;
    }

    dcompVisual_->SetContent(dxgiSwapChain_);
    dcompTarget_->SetRoot(dcompVisual_);
    dcompDevice_->Commit();

    ocCout() << "DirectComposition initialized" << std::endl;
    return true;
}

// ─── Shared Image (D3D11 creates → Vulkan imports) ───────────────────────

bool OverlayCompositor::createSharedVulkanImage(int index) {
    auto fw = framework_.lock();
    if (!fw) return false;

    auto device = fw->device();
    VkDevice vkDev = device->vkDevice();

    // Step A: D3D11 creates the shared texture (D3D11 is the owner)
    D3D11_TEXTURE2D_DESC texDesc{};
    texDesc.Width = width_;
    texDesc.Height = height_;
    texDesc.MipLevels = 1;
    texDesc.ArraySize = 1;
    texDesc.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
    texDesc.SampleDesc.Count = 1;
    texDesc.Usage = D3D11_USAGE_DEFAULT;
    texDesc.BindFlags = D3D11_BIND_RENDER_TARGET;
    texDesc.MiscFlags = D3D11_RESOURCE_MISC_SHARED_NTHANDLE |
                        D3D11_RESOURCE_MISC_SHARED_KEYEDMUTEX;

    HRESULT hr = d3dDevice_->CreateTexture2D(&texDesc, nullptr,
                                              &d3dSharedTextures_[index]);
    if (FAILED(hr)) {
        diagLog("CreateTexture2D failed for shared texture %d: 0x%08x",
                index, (unsigned)hr);
        return false;
    }

    // Query keyed mutex (required for D3D11 access to shared textures)
    hr = d3dSharedTextures_[index]->QueryInterface(
        IID_PPV_ARGS(&sharedMutexes_[index]));
    if (FAILED(hr)) {
        diagLog("QueryInterface IDXGIKeyedMutex failed: 0x%08x", (unsigned)hr);
        return false;
    }

    // Step B: Export NT handle via DXGI
    IDXGIResource1 *dxgiRes = nullptr;
    hr = d3dSharedTextures_[index]->QueryInterface(IID_PPV_ARGS(&dxgiRes));
    if (FAILED(hr)) {
        diagLog("QueryInterface IDXGIResource1 failed: 0x%08x", (unsigned)hr);
        return false;
    }
    hr = dxgiRes->CreateSharedHandle(
        nullptr, GENERIC_ALL, nullptr,
        reinterpret_cast<HANDLE *>(&sharedHandles_[index]));
    dxgiRes->Release();
    if (FAILED(hr)) {
        diagLog("CreateSharedHandle failed: 0x%08x", (unsigned)hr);
        return false;
    }

    // Step C: Create VkImage with external memory (for import)
    VkExternalMemoryImageCreateInfo externalInfo{};
    externalInfo.sType = VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_IMAGE_CREATE_INFO;
    externalInfo.handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_D3D11_TEXTURE_BIT;

    VkImageCreateInfo imageInfo{};
    imageInfo.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
    imageInfo.pNext = &externalInfo;
    imageInfo.imageType = VK_IMAGE_TYPE_2D;
    imageInfo.format = VK_FORMAT_R8G8B8A8_SRGB;
    imageInfo.extent = {width_, height_, 1};
    imageInfo.mipLevels = 1;
    imageInfo.arrayLayers = 1;
    imageInfo.samples = VK_SAMPLE_COUNT_1_BIT;
    imageInfo.tiling = VK_IMAGE_TILING_OPTIMAL;
    imageInfo.usage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;
    imageInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    imageInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;

    VkResult result = vkCreateImage(vkDev, &imageInfo, nullptr, &sharedImages_[index]);
    if (result != VK_SUCCESS) {
        diagLog("vkCreateImage failed for shared image %d: %d", index, (int)result);
        return false;
    }

    // Step D: Import memory from D3D11 via NT handle
    VkMemoryRequirements memReqs;
    vkGetImageMemoryRequirements(vkDev, sharedImages_[index], &memReqs);

    VkImportMemoryWin32HandleInfoKHR importInfo{};
    importInfo.sType = VK_STRUCTURE_TYPE_IMPORT_MEMORY_WIN32_HANDLE_INFO_KHR;
    importInfo.handleType = VK_EXTERNAL_MEMORY_HANDLE_TYPE_D3D11_TEXTURE_BIT;
    importInfo.handle = static_cast<HANDLE>(sharedHandles_[index]);

    VkMemoryDedicatedAllocateInfo dedicatedInfo{};
    dedicatedInfo.sType = VK_STRUCTURE_TYPE_MEMORY_DEDICATED_ALLOCATE_INFO;
    dedicatedInfo.pNext = &importInfo;
    dedicatedInfo.image = sharedImages_[index];

    // Find device-local memory type
    VkPhysicalDeviceMemoryProperties memProps;
    vkGetPhysicalDeviceMemoryProperties(
        fw->physicalDevice()->vkPhysicalDevice(), &memProps);
    uint32_t memTypeIndex = UINT32_MAX;
    for (uint32_t i = 0; i < memProps.memoryTypeCount; i++) {
        if ((memReqs.memoryTypeBits & (1 << i)) &&
            (memProps.memoryTypes[i].propertyFlags & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT)) {
            memTypeIndex = i;
            break;
        }
    }
    if (memTypeIndex == UINT32_MAX) {
        diagLog("no suitable device-local memory type for shared image %d", index);
        return false;
    }

    VkMemoryAllocateInfo allocInfo{};
    allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    allocInfo.pNext = &dedicatedInfo;
    allocInfo.allocationSize = memReqs.size;
    allocInfo.memoryTypeIndex = memTypeIndex;

    result = vkAllocateMemory(vkDev, &allocInfo, nullptr, &sharedMemories_[index]);
    if (result != VK_SUCCESS) {
        diagLog("vkAllocateMemory (import) failed for shared image %d: %d",
                index, (int)result);
        return false;
    }

    result = vkBindImageMemory(vkDev, sharedImages_[index], sharedMemories_[index], 0);
    if (result != VK_SUCCESS) {
        diagLog("vkBindImageMemory failed for shared image %d: %d",
                index, (int)result);
        return false;
    }

    // Step E: Create VkImageView
    VkImageViewCreateInfo viewInfo{};
    viewInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
    viewInfo.image = sharedImages_[index];
    viewInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
    viewInfo.format = VK_FORMAT_R8G8B8A8_SRGB;
    viewInfo.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};

    result = vkCreateImageView(vkDev, &viewInfo, nullptr, &sharedViews_[index]);
    if (result != VK_SUCCESS) {
        diagLog("vkCreateImageView failed for shared image %d: %d",
                index, (int)result);
        return false;
    }

    ocCout() << "shared image " << index << " created " << width_ << "x" << height_
             << " (D3D11->Vulkan import)" << std::endl;
    return true;
}

// ─── Premultiply Pipeline ─────────────────────────────────────────────────

bool OverlayCompositor::createPremultiplyPipeline() {
    auto fw = framework_.lock();
    if (!fw) return false;

    auto device = fw->device();
    VkDevice vkDev = device->vkDevice();

    std::filesystem::path shaderPath = Renderer::folderPath / "shaders";

    // Shaders
    premultiplyVert_ = vk::Shader::create(
        device, (shaderPath / "full_screen_vert.spv").string());
    premultiplyFrag_ = vk::Shader::create(
        device, (shaderPath / "premultiply_alpha_frag.spv").string());

    // Sampler (nearest -- dimensions match)
    premultiplySampler_ = vk::Sampler::create(
        device,
        VK_FILTER_NEAREST,
        VK_SAMPLER_MIPMAP_MODE_NEAREST,
        VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE);

    // Descriptor set: one combined image sampler (overlay input)
    premultiplyDescriptor_ = vk::DescriptorTableBuilder{}
        .beginDescriptorLayoutSet()  // set 0
        .beginDescriptorLayoutSetBinding()
        .defineDescriptorLayoutSetBinding({
            .binding = 0,
            .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
            .descriptorCount = 1,
            .stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT,
        })
        .endDescriptorLayoutSetBinding()
        .endDescriptorLayoutSet()
        .build(device);

    // Render pass: R8G8B8A8_SRGB color attachment, DONT_CARE load, finalLayout = GENERAL
    premultiplyRenderPass_ = vk::RenderPassBuilder{}
        .beginAttachmentDescription()
        .defineAttachmentDescription({
            .format = VK_FORMAT_R8G8B8A8_SRGB,
            .samples = VK_SAMPLE_COUNT_1_BIT,
            .loadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE,
            .storeOp = VK_ATTACHMENT_STORE_OP_STORE,
            .stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE,
            .stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE,
            .initialLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
            .finalLayout = VK_IMAGE_LAYOUT_GENERAL,
        })
        .endAttachmentDescription()
        .beginAttachmentReference()
        .defineAttachmentReference({
            .attachment = 0,
            .layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
        })
        .endAttachmentReference()
        .beginSubpassDescription()
        .defineSubpassDescription({
            .pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS,
            .colorAttachmentIndices = {0},
        })
        .endSubpassDescription()
        .build(device);

    // Framebuffers (one per shared image)
    for (int i = 0; i < kSharedImageCount; i++) {
        VkFramebufferCreateInfo fbInfo{};
        fbInfo.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
        fbInfo.renderPass = premultiplyRenderPass_->vkRenderPass();
        fbInfo.attachmentCount = 1;
        fbInfo.pAttachments = &sharedViews_[i];
        fbInfo.width = width_;
        fbInfo.height = height_;
        fbInfo.layers = 1;

        VkResult result = vkCreateFramebuffer(vkDev, &fbInfo, nullptr, &premultiplyFramebuffers_[i]);
        if (result != VK_SUCCESS) {
            ocCerr() << "vkCreateFramebuffer failed for shared image " << i
                     << ": " << result << std::endl;
            return false;
        }
    }

    // Graphics pipeline
    premultiplyPipeline_ = vk::GraphicsPipelineBuilder{}
        .defineRenderPass(premultiplyRenderPass_, 0)
        .beginShaderStage()
        .defineShaderStage(premultiplyVert_, VK_SHADER_STAGE_VERTEX_BIT)
        .defineShaderStage(premultiplyFrag_, VK_SHADER_STAGE_FRAGMENT_BIT)
        .endShaderStage()
        .defineVertexInputState<void>()
        .defineViewportScissorState({
            .viewport = {
                .x = 0,
                .y = 0,
                .width = static_cast<float>(width_),
                .height = static_cast<float>(height_),
                .minDepth = 0.0,
                .maxDepth = 1.0,
            },
            .scissor = {
                .offset = {.x = 0, .y = 0},
                .extent = {width_, height_},
            },
        })
        .defineDepthStencilState({
            .depthTestEnable = VK_FALSE,
            .depthWriteEnable = VK_FALSE,
            .depthCompareOp = VK_COMPARE_OP_ALWAYS,
            .depthBoundsTestEnable = VK_FALSE,
            .stencilTestEnable = VK_FALSE,
        })
        .beginColorBlendAttachmentState()
        .defineDefaultColorBlendAttachmentState()
        .endColorBlendAttachmentState()
        .definePipelineLayout(premultiplyDescriptor_)
        .build(device);

    ocCout() << "premultiply pipeline created" << std::endl;
    return true;
}

// ─── Record Premultiply Pass ──────────────────────────────────────────────

void OverlayCompositor::recordPremultiply(std::shared_ptr<vk::CommandBuffer> cmd,
                                           std::shared_ptr<vk::DeviceLocalImage> overlayImage,
                                           uint32_t queueFamilyIndex) {
    if (!active_ || !overlayImage) return;

    int idx = currentSharedIndex_;
    currentSharedIndex_ = (currentSharedIndex_ + 1) % kSharedImageCount;

    // Bind overlay image to descriptor set
    premultiplyDescriptor_->bindSamplerImageForShader(
        premultiplySampler_, overlayImage, 0, 0);  // set 0, binding 0

    VkImageLayout oldOverlayLayout = overlayImage->imageLayout();

    // Barriers: overlay -> SHADER_READ_ONLY_OPTIMAL, shared -> COLOR_ATTACHMENT_OPTIMAL
    cmd->barriersBufferImage(
        {}, {
            {
                .srcStageMask = VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT |
                                VK_PIPELINE_STAGE_2_TRANSFER_BIT,
                .srcAccessMask = VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT,
                .dstStageMask = VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT,
                .dstAccessMask = VK_ACCESS_2_SHADER_READ_BIT,
                .oldLayout = overlayImage->imageLayout(),
                .newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                .srcQueueFamilyIndex = queueFamilyIndex,
                .dstQueueFamilyIndex = queueFamilyIndex,
                .image = overlayImage,
                .subresourceRange = vk::wholeColorSubresourceRange,
            },
        });
    overlayImage->imageLayout() = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

    // Begin render pass targeting shared image framebuffer
    VkRenderPassBeginInfo rpBegin{};
    rpBegin.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
    rpBegin.renderPass = premultiplyRenderPass_->vkRenderPass();
    rpBegin.framebuffer = premultiplyFramebuffers_[idx];
    rpBegin.renderArea = {{0, 0}, {width_, height_}};

    vkCmdBeginRenderPass(cmd->vkCommandBuffer(), &rpBegin, VK_SUBPASS_CONTENTS_INLINE);

    // Draw fullscreen triangle
    cmd->bindGraphicsPipeline(premultiplyPipeline_)
        ->bindDescriptorTable(premultiplyDescriptor_, VK_PIPELINE_BIND_POINT_GRAPHICS)
        ->draw(3, 1)
        ->endRenderPass();

    // Render pass finalLayout = GENERAL handles the shared image transition.

    // Restore overlay image layout
    if (oldOverlayLayout != VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL &&
        oldOverlayLayout != VK_IMAGE_LAYOUT_UNDEFINED) {
        cmd->barriersBufferImage(
            {}, {{
                .srcStageMask = VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT,
                .srcAccessMask = VK_ACCESS_2_SHADER_READ_BIT,
                .dstStageMask = VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT |
                                VK_PIPELINE_STAGE_2_TRANSFER_BIT,
                .dstAccessMask = VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT,
                .oldLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                .newLayout = oldOverlayLayout,
                .srcQueueFamilyIndex = queueFamilyIndex,
                .dstQueueFamilyIndex = queueFamilyIndex,
                .image = overlayImage,
                .subresourceRange = vk::wholeColorSubresourceRange,
            }});
        overlayImage->imageLayout() = oldOverlayLayout;
    }
}

// ─── Present (called by PresentThread) ────────────────────────────────────

void OverlayCompositor::present() {
    if (!active_) return;

    // Sync overlay window position to game window
    syncPosition();

    // Get the shared image index that was most recently written
    // (currentSharedIndex_ has already been advanced by recordPremultiply,
    //  so the last written index is (currentSharedIndex_ - 1 + count) % count)
    int readIdx = (currentSharedIndex_ - 1 + kSharedImageCount) % kSharedImageCount;

    // Acquire keyed mutex before D3D11 access to shared texture.
    // Bounded timeout (100ms) prevents deadlock if Vulkan/DXGI state is inconsistent.
    HRESULT hr = sharedMutexes_[readIdx]->AcquireSync(0, 100);
    if (hr == static_cast<HRESULT>(WAIT_TIMEOUT)) {
        diagLog("AcquireSync timed out for shared texture %d — skipping frame", readIdx);
        return;
    }
    if (FAILED(hr)) {
        diagLog("AcquireSync failed for shared texture %d: 0x%08x",
                readIdx, (unsigned)hr);
        return;
    }

    // D3D11: Copy shared texture -> DXGI swapchain backbuffer
    ID3D11Texture2D *backBuffer = nullptr;
    hr = dxgiSwapChain_->GetBuffer(0, IID_PPV_ARGS(&backBuffer));
    if (FAILED(hr)) {
        sharedMutexes_[readIdx]->ReleaseSync(0);
        ocCerr() << "GetBuffer failed: 0x" << std::hex << hr << std::endl;
        return;
    }

    d3dContext_->CopyResource(backBuffer, d3dSharedTextures_[readIdx]);
    backBuffer->Release();

    // Release keyed mutex after D3D11 access
    sharedMutexes_[readIdx]->ReleaseSync(0);

    // DXGI: Present without VSync — DComp+DWM composites at display refresh rate
    // regardless of SyncInterval. VSync=1 would add a redundant VBlank wait on top
    // of the Vulkan swapchain's own FIFO VSync, halving the effective UI update rate.
    DXGI_PRESENT_PARAMETERS params = {};
    hr = dxgiSwapChain_->Present1(0, 0, &params);

    if (hr == DXGI_ERROR_DEVICE_REMOVED || hr == DXGI_ERROR_DEVICE_RESET) {
        ocCerr() << "DXGI device lost, deactivating overlay compositor" << std::endl;
        active_ = false;
    }
}

bool OverlayCompositor::waitForDisplayReady(uint32_t timeoutMs) {
    if (frameLatencyWaitable_) {
        DWORD result = WaitForSingleObject(frameLatencyWaitable_, timeoutMs);
        return result == WAIT_OBJECT_0;
    }
    // Fallback: sleep ~4ms (approximate 240Hz)
    Sleep(4);
    return true;
}

// ─── Resize ───────────────────────────────────────────────────────────────

void OverlayCompositor::resize(uint32_t width, uint32_t height) {
    if (width == width_ && height == height_) return;

    auto fw = framework_.lock();
    if (!fw) return;

    VkDevice vkDev = fw->device()->vkDevice();

    width_ = width;
    height_ = height;

    // Destroy old shared resources (Vulkan first — imported from D3D11)
    for (int i = 0; i < kSharedImageCount; i++) {
        if (premultiplyFramebuffers_[i] != VK_NULL_HANDLE) {
            vkDestroyFramebuffer(vkDev, premultiplyFramebuffers_[i], nullptr);
            premultiplyFramebuffers_[i] = VK_NULL_HANDLE;
        }
        if (sharedViews_[i] != VK_NULL_HANDLE) {
            vkDestroyImageView(vkDev, sharedViews_[i], nullptr);
            sharedViews_[i] = VK_NULL_HANDLE;
        }
        if (sharedImages_[i] != VK_NULL_HANDLE) {
            vkDestroyImage(vkDev, sharedImages_[i], nullptr);
            sharedImages_[i] = VK_NULL_HANDLE;
        }
        if (sharedMemories_[i] != VK_NULL_HANDLE) {
            vkFreeMemory(vkDev, sharedMemories_[i], nullptr);
            sharedMemories_[i] = VK_NULL_HANDLE;
        }
        if (sharedMutexes_[i]) { sharedMutexes_[i]->Release(); sharedMutexes_[i] = nullptr; }
        if (d3dSharedTextures_[i]) { d3dSharedTextures_[i]->Release(); d3dSharedTextures_[i] = nullptr; }
        if (sharedHandles_[i]) {
            CloseHandle(static_cast<HANDLE>(sharedHandles_[i]));
            sharedHandles_[i] = nullptr;
        }
    }

    // Destroy old pipeline resources that depend on dimensions
    premultiplyPipeline_.reset();

    // Resize DXGI swapchain
    HRESULT hr = dxgiSwapChain_->ResizeBuffers(
        2, width_, height_, DXGI_FORMAT_R8G8B8A8_UNORM, 0);
    if (FAILED(hr)) {
        ocCerr() << "ResizeBuffers failed: 0x" << std::hex << hr << std::endl;
        active_ = false;
        return;
    }

    // Recreate shared images
    for (int i = 0; i < kSharedImageCount; i++) {
        if (!createSharedVulkanImage(i)) {
            active_ = false;
            return;
        }
    }

    // Recreate premultiply pipeline at new size
    if (!createPremultiplyPipeline()) {
        active_ = false;
        return;
    }

    currentSharedIndex_ = 0;

    // Reposition overlay window
    syncPosition();

    ocCout() << "resized to " << width_ << "x" << height_ << std::endl;
}

// ─── Position Sync ────────────────────────────────────────────────────────

void OverlayCompositor::syncPosition() {
    RECT r;
    GetClientRect(static_cast<HWND>(gameHwnd_), &r);
    MapWindowPoints(static_cast<HWND>(gameHwnd_), nullptr,
                    reinterpret_cast<POINT *>(&r), 2);
    // SWP_ASYNCWINDOWPOS: critical when called from PresentThread — the overlay window
    // is owned by the render thread, so SetWindowPos would synchronously SendMessage to
    // the render thread. If the render thread is blocked (e.g., in pause() or recreate()),
    // it can't pump messages, causing a cross-thread Win32 message deadlock.
    SetWindowPos(static_cast<HWND>(overlayHwnd_), static_cast<HWND>(gameHwnd_),
                 r.left, r.top, r.right - r.left, r.bottom - r.top,
                 SWP_NOACTIVATE | SWP_NOZORDER | SWP_ASYNCWINDOWPOS);
}

#endif // _WIN32
