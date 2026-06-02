// Minimal HIP Runtime Context for PhysX DCU Port
//
// Provides device management, memory allocation, stream/event control,
// module loading, and kernel launching — the full GPU runtime surface
// needed by PhysX.

#ifndef PXG_HIP_CONTEXT_H
#define PXG_HIP_CONTEXT_H

#include <hip/hip_runtime.h>

struct HipContext {
    hipDevice_t     device;
    hipCtx_t        context;
    int             deviceOrdinal;
    char            deviceName[256];
    int             computeUnits;
    int             maxThreadsPerBlock;
    size_t          sharedMemPerBlock;
    size_t          totalGlobalMem;
    int             warpSize;

    bool            isValid;
};

// ---- Lifecycle ----
bool hipContext_init(HipContext* ctx, int deviceOrdinal);
void hipContext_release(HipContext* ctx);

// ---- Memory ----
bool hipContext_memAlloc(HipContext* ctx, hipDeviceptr_t* dptr, size_t bytes);
bool hipContext_memFree(HipContext* ctx, hipDeviceptr_t dptr);
bool hipContext_memHostAlloc(HipContext* ctx, void** ptr, size_t bytes, unsigned int flags);
bool hipContext_memFreeHost(HipContext* ctx, void* ptr);

// ---- Copy ----
bool hipContext_memcpyHtoD(HipContext* ctx, hipDeviceptr_t dst, const void* src, size_t bytes);
bool hipContext_memcpyDtoH(HipContext* ctx, void* dst, hipDeviceptr_t src, size_t bytes);
bool hipContext_memcpyDtoD(HipContext* ctx, hipDeviceptr_t dst, hipDeviceptr_t src, size_t bytes);

// ---- Stream ----
bool hipContext_streamCreate(HipContext* ctx, hipStream_t* stream);
bool hipContext_streamDestroy(HipContext* ctx, hipStream_t stream);
bool hipContext_streamSync(HipContext* ctx, hipStream_t stream);

// ---- Event ----
bool hipContext_eventCreate(HipContext* ctx, hipEvent_t* event);
bool hipContext_eventRecord(HipContext* ctx, hipEvent_t event, hipStream_t stream);
bool hipContext_eventSync(HipContext* ctx, hipEvent_t event);
bool hipContext_eventDestroy(HipContext* ctx, hipEvent_t event);

// ---- Module ----
bool hipContext_moduleLoadData(HipContext* ctx, hipModule_t* module, const void* image);
bool hipContext_moduleGetFunc(HipContext* ctx, hipFunction_t* func, hipModule_t module, const char* name);
bool hipContext_moduleUnload(HipContext* ctx, hipModule_t module);

// ---- Kernel Launch ----
bool hipContext_launchKernel(HipContext* ctx,
    hipFunction_t func,
    unsigned int gridX, unsigned int gridY, unsigned int gridZ,
    unsigned int blockX, unsigned int blockY, unsigned int blockZ,
    unsigned int sharedMemBytes,
    hipStream_t stream,
    void** kernelParams,
    void** extra);

#endif
