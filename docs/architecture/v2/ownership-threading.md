# Ownership & Threading Model

## Thread Inventory

| Thread | Purpose | Legacy | V2 |
|--------|---------|--------|----|
| Main (render) | Frame loop, submit, present | Renderer::render() | FrameScheduler::beginFrame/endFrame |
| BLAS build | AS construction on secondary queue | ChunkBuildScheduler::blasThreadLoop | BlasBuildScheduler::tick |
| JNI bridge | Java→C++ calls from Minecraft thread | Direct function calls | BridgeService::dispatch |
| Java main | Minecraft tick, chunk meshing | N/A (Java-side) | N/A |

## Ownership Hierarchy

```
EngineApp (static lifetime)
└── EngineSession (per-world lifetime)
    └── EngineServices (same as session)
        ├── DeviceService        [main thread, static after init]
        ├── SwapchainService     [main thread]
        ├── ConfigService        [main thread writes, any thread reads snapshot]
        ├── FrameScheduler       [main thread]
        ├── SubmissionService    [per-queue serialization]
        ├── ResourceGC           [main thread tick, any thread defer]
        ├── BridgeService        [JNI thread posts, main thread dispatches]
        ├── SceneService         [bridge thread writes via dispatch, RT reads snapshot]
        ├── BlasService          [BLAS thread builds, main thread reads]
        ├── TlasService          [main thread]
        ├── LogService           [any thread, internally synchronized]
        └── DiagnosticsService   [any thread capture, main thread write]
```

## Synchronization Points

### 1. BLAS Thread → Main Thread (existing pattern, preserved)

```
BLAS thread                         Main thread
    │                                   │
    ├── build BLAS batch                │
    ├── vkQueueSubmit(secondary)        │
    ├── store(timelineValue)            │
    │                                   │
    │                           blasSem->waitValue(lastSubmitted)
    │                           gc.tick()
    │                           tlasService.build()
```

Timeline semaphore ensures BLAS builds complete before:
- GC frees backing buffers
- TLAS references BLAS device addresses

### 2. JNI Thread → Main Thread

```
JNI thread                          Main thread
    │                                   │
    ├── bridge.post(ChunkReady, data)   │
    │   (lock-free queue)               │
    │                               bridge.flush()
    │                               scene.insert(chunk)
    │                               blasService.enqueue(chunk)
```

No JNI call directly mutates render state. All mutations are queued.

### 3. Frame Scheduling

```
Main thread (each frame):
    swapchain.acquire()
    config.snapshot()           // freeze config for this frame
    bridge.flush()              // process all queued JNI events
    scene.extractedFrame()      // snapshot scene for RT
    blasSem.waitValue()         // sync BLAS builds
    gc.tick()                   // free deferred resources
    graph.execute(frameCtx)     // run all passes
    submission.submit()
    swapchain.present()
```

## Resource Lifetime Rules

### Shared Ownership (shared_ptr)

Used for: Buffers, Images, BLAS, TLAS, Pipelines, Descriptor Sets

Rule: The last shared_ptr drop triggers destruction. GC defers destruction
by holding a shared_ptr for N frames after the last active reference drops.

### Exclusive Ownership (unique_ptr)

Used for: Command Buffers (per-thread pools), Fences, Query Pools

Rule: Creator is sole owner. No sharing across threads.

### Snapshot Ownership

Used for: ConfigSnapshot, ExtractedScene, TlasBlasSnapshot

Rule: Immutable copy created at frame start. Lives for one frame.
Any thread can read. No thread can modify. Dropped at frame end.

## Deadlock Prevention

1. Lock ordering: bridge_mutex < scene_mutex < blas_mutex < gc_mutex
2. Timeline semaphores are one-directional: BLAS thread signals, main thread waits
3. No mutex held during Vulkan queue submit (submit serializes internally)
4. ConfigService uses atomic snapshot swap, no mutex for reads
