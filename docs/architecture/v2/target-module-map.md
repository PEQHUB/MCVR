# V2 Target Module Map

## engine/app — Session Lifecycle

```
EngineApp
├── init(JNIEnv*, WindowHandle, DeviceConfig)
├── shutdown()
├── session() → EngineSession&
└── services() → EngineServices&

EngineSession
├── state: Startup | Running | Paused | ShuttingDown
├── frameCount: uint64_t
├── config: ConfigService&
└── mode: legacy | v2

EngineServices (service locator, no globals)
├── device() → DeviceService&
├── swapchain() → SwapchainService&
├── config() → ConfigService&
├── frame() → FrameScheduler&
├── scene() → SceneService&
├── rt() → RtService&
├── log() → LogService&
└── diagnostics() → DiagnosticsService&
```

**Ownership**: Created once at init, destroyed at shutdown.
**Thread**: Main thread creates. Individual services declare their own affinity.

## engine/config — Schema-Backed Config

```
ConfigService
├── load(path) → Result<void>
├── save(path) → Result<void>
├── get<T>(key) → T
├── set<T>(key, value) → bool (returns false if validation fails)
├── subscribe(key, callback) → SubscriptionId
├── unsubscribe(SubscriptionId)
├── restartScope(key) → RestartScope {None, Pipeline, Renderer, Game}
└── snapshot() → ConfigSnapshot (immutable copy for frame use)
```

**Ownership**: EngineServices (1:1)
**Thread**: Main thread writes. Any thread reads via snapshot.
**Invariant**: `snapshot()` returns a consistent, immutable view. Frame code never reads live config.

## engine/bridge — JNI Command/Event Transport

```
BridgeService
├── registerHandler(CommandType, Handler)
├── dispatch(CommandType, payload) → Result<Response>
├── post(EventType, payload)  // fire-and-forget, Java → C++
└── flush()  // process all queued events

CommandType: SetConfig | SubmitChunk | SubmitEntity | SubmitTexture | ...
EventType:   ChunkReady | EntityUpdated | TextureUploaded | WindowResized | ...
```

**Ownership**: EngineServices (1:1)
**Thread**: JNI thread posts. Main thread dispatches.
**Invariant**: No JNI call directly mutates engine state. All mutations go through dispatch.

## engine/platform/vulkan — vk2 Wrappers

```
DeviceService
├── physicalDevice() → VkPhysicalDevice
├── device() → VkDevice
├── mainQueue() → QueueHandle
├── secondaryQueue() → QueueHandle
├── vma() → VMA&
└── capabilities() → DeviceCaps (RT, mesh shader, etc.)

vk2::Buffer — RAII buffer, explicit error returns
vk2::Image  — RAII image, layout tracking
vk2::CommandBuffer — scoped recording
vk2::Fence, vk2::Semaphore, vk2::TimelineSemaphore
vk2::DescriptorSet — typed binding helpers
```

**Ownership**: DeviceService owns device/queues. Resources are ref-counted (shared_ptr).
**Thread**: Main thread creates. Command recording on any thread with CommandPool per thread.
**Invariant**: No creation function returns a partially valid object. All return Result<T>.

## engine/frame — Frame Scheduling + Submission

```
FrameScheduler
├── beginFrame() → FrameContext
├── endFrame(FrameContext)
├── frameIndex() → uint32_t (swapchain image index)
└── frameNumber() → uint64_t (monotonic)

FrameContext
├── commandBuffer() → vk2::CommandBuffer&
├── frameIndex: uint32_t
├── frameNumber: uint64_t
├── configSnapshot: ConfigSnapshot
└── deltaTime: float

SwapchainService
├── acquire() → Result<uint32_t>
├── present(uint32_t imageIndex, VkSemaphore waitSem) → Result<void>
├── recreate(uint32_t width, uint32_t height) → Result<void>
├── imageCount() → uint32_t
└── extent() → VkExtent2D

SubmissionService
├── submit(QueueHandle, vk2::CommandBuffer, SubmitInfo) → Result<void>
├── waitIdle(QueueHandle)
└── gc() → ResourceGC&  // per-frame garbage collection

ResourceGC
├── defer(shared_ptr<T>, uint32_t framesUntilFree)
├── tick(uint64_t currentFrame)
└── flush()  // wait + free all
```

**Ownership**: EngineServices (1:1 each)
**Thread**: FrameScheduler on main thread. SubmissionService serializes per-queue.
**Invariant**: ResourceGC never frees a resource while any queue might reference it.
Cross-queue sync via timeline semaphore wait before GC tick.

## engine/rendergraph — Graph Builder + Resource Planner

```
GraphBuilder
├── addPass(name, PassDescriptor) → PassHandle
├── addResource(name, ResourceDescriptor) → ResourceHandle
├── connect(output: ResourceHandle, input: ResourceHandle)
├── compile() → CompiledGraph
└── importYaml(path) → Result<void>  // read existing pipeline YAML

CompiledGraph
├── passes: ordered list of PassHandle
├── barriers: auto-computed image/buffer barriers
├── transients: images that can alias memory
└── execute(FrameContext)

PassDescriptor
├── inputs: vector<ResourceHandle>
├── outputs: vector<ResourceHandle>
├── queueAffinity: Main | Compute | Transfer
└── execute: function<void(FrameContext, PassResources)>
```

**Ownership**: EngineServices (1:1)
**Thread**: compile() on main thread. execute() dispatches per pass queue affinity.
**Invariant**: Barrier insertion is automatic. No manual vkCmdPipelineBarrier in pass code.

## engine/scene — Extracted World State

```
SceneService
├── chunks() → ChunkRegistry&
├── entities() → EntityRegistry&
├── textures() → TextureService&
├── lights() → LightService&
├── materials() → MaterialService&
└── extractedFrame() → ExtractedScene (immutable snapshot for RT)

ChunkRegistry
├── insert(ChunkId, ChunkGeometry) → RevisionId
├── remove(ChunkId)
├── get(ChunkId) → ChunkState&
├── revision(ChunkId) → RevisionId
└── dirtySet() → span<ChunkId>  // changed since last frame

EntityRegistry
├── insert(EntityId, EntityState) → RevisionId
├── remove(EntityId)
├── update(EntityId, EntityDelta)
└── dirtySet() → span<EntityId>
```

**Ownership**: EngineServices (1:1)
**Thread**: Bridge thread writes (via dispatch). RT thread reads via extractedFrame().
**Invariant**: ChunkId and EntityId are stable across frames. RevisionId is monotonic.
No BLAS or GPU resource in scene — that belongs to engine/rt.

## engine/rt — Geometry Cache + BLAS/TLAS Services

```
BlasService
├── getOrBuild(ChunkId, ChunkGeometry) → BlasHandle
├── invalidate(ChunkId)
├── pendingBuilds() → uint32_t
├── budgetMs: float  // per-frame build time budget
└── compactionEnabled: bool  // runtime toggle

TlasService
├── build(ExtractedScene, BlasRegistry) → TlasHandle
├── update(ExtractedScene, BlasRegistry, prevTlas) → TlasHandle
├── canUpdate(prev, curr) → bool
└── instanceCount() → uint32_t

BlasBuildScheduler
├── enqueue(vector<BlasJob>)
├── tick() → vector<CompletedBlas>
├── timelineValue() → uint64_t
└── waitAll()

SbtService
├── build(RtPipeline, BlasRegistry) → SbtHandle
└── update(SbtHandle, changes)
```

**Ownership**: EngineServices (1:1 each)
**Thread**: BlasBuildScheduler on dedicated BLAS thread. TlasService on main thread.
**Invariant**: BlasHandle is valid for its ChunkId's current RevisionId.
BLAS thread signals timeline semaphore. Main thread waits before TLAS build.

## engine/features — Vendor Adapters

```
// Common adapter interface
class FeatureAdapter {
    virtual void init(EngineServices&) = 0;
    virtual void configure(ConfigSnapshot) = 0;
    virtual void execute(FrameContext, PassResources) = 0;
    virtual void shutdown() = 0;
};

DlssAdapter : FeatureAdapter    // wraps DlssRR
FsrAdapter : FeatureAdapter     // wraps FidelityFX FSR3
NrdAdapter : FeatureAdapter     // wraps NRD
SvgfAdapter : FeatureAdapter    // built-in SVGF
TonemappingAdapter : FeatureAdapter
PostRenderAdapter : FeatureAdapter
CloudAdapter : FeatureAdapter
FramegenAdapter : FeatureAdapter
TaaAdapter : FeatureAdapter
```

**Ownership**: Render graph owns via PassDescriptor. EngineServices provides DI.
**Thread**: Execute on the queue declared by the pass.
**Invariant**: Adapters never read global state. All inputs come from PassResources.

## engine/diagnostics — Structured Logging + Crash Packs

```
LogService (engine::log facade)
├── init(LogConfig)
├── shutdown()
├── trace/debug/info/warn/error/fatal(category, msg)
└── event(category, eventId, fields)  // JSONL structured event

CrashPack
├── capture() → CrashData  // ring buffer + GPU state + VMA stats
├── write(path)
└── attach(Aftermath dump, validation errors)

GpuMarkers
├── begin(cmd, name)
├── end(cmd)
├── queryResults() → vector<TimingResult>
```

**Ownership**: EngineServices (1:1)
**Thread**: LogService is thread-safe. CrashPack captures from any thread.
**Invariant**: fatal() always flushes before abort(). No silent crashes.

## engine/streaming — Chunk I/O + Extended RD (Phase 6)

```
StreamingService
├── budget: StreamingBudget  // MB/frame upload, builds/frame
├── requestLoad(ChunkId, priority)
├── requestUnload(ChunkId)
├── tick() → vector<ChunkId>  // completed this frame
└── stats() → StreamingStats
```

Deferred to Phase 6. Placeholder documented here for completeness.
