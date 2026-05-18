// Stubs for OMM SDK debug functions (debug_impl.cpp excluded to avoid STB conflicts)
#ifdef MCVR_ENABLE_OMM

#include <omm.h>

// Forward declarations matching debug_impl.h signatures
// These are in the omm namespace, using internal OMM types
namespace omm {
    template <typename T> class StdAllocator;
    class Logger;
}

// The symbols are declared with OMM_API (extern "C") linkage in bake.cpp's scope.
// We provide no-op stubs since we never call the debug APIs.
extern "C" {
    ommResult SaveAsImagesImpl(void*, const ommCpuBakeInputDesc*, const ommCpuBakeResultDesc*, const void*) {
        return ommResult_NOT_IMPLEMENTED;
    }
    ommResult GetStatsImpl(void*, const ommCpuBakeResultDesc*, const float*, void*) {
        return ommResult_NOT_IMPLEMENTED;
    }
    ommResult SaveBinaryToDiskImpl(const void*, const ommCpuBlobDesc*, const char*) {
        return ommResult_NOT_IMPLEMENTED;
    }
}

#endif
