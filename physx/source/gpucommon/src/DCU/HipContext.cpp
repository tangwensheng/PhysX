// HipContext Implementation — HIP Runtime wrapper for PhysX DCU Port

#include "HipContext.h"
#include <cstdio>
#include <cstring>

#define HIP_CHECK(call, msg) do { \
    hipError_t _err = call; \
    if (_err != hipSuccess) { \
        fprintf(stderr, "[HIP ERROR] %s: %s (code=%d)\n", msg, hipGetErrorString(_err), (int)_err); \
        return false; \
    } \
} while(0)

// ---- Lifecycle ----

bool hipContext_init(HipContext* ctx, int deviceOrdinal)
{
    memset(ctx, 0, sizeof(HipContext));
    ctx->isValid = false;

    HIP_CHECK(hipInit(0), "hipInit");
    HIP_CHECK(hipDeviceGet(&ctx->device, deviceOrdinal), "hipDeviceGet");

    hipDeviceProp_t props;
    HIP_CHECK(hipGetDeviceProperties(&props, ctx->device), "hipGetDeviceProperties");

    strncpy(ctx->deviceName, props.name, sizeof(ctx->deviceName) - 1);
    ctx->computeUnits       = props.multiProcessorCount;
    ctx->maxThreadsPerBlock = props.maxThreadsPerBlock;
    ctx->sharedMemPerBlock  = props.sharedMemPerBlock;
    ctx->totalGlobalMem     = props.totalGlobalMem;
    ctx->warpSize           = props.warpSize;
    ctx->deviceOrdinal      = deviceOrdinal;

    HIP_CHECK(hipCtxCreate(&ctx->context, 0, ctx->device), "hipCtxCreate");

    ctx->isValid = true;
    printf("[HipContext] Device %d: %s, %d CUs, warpSize=%d, sharedMem=%zu KB, globalMem=%.1f GB\n",
        deviceOrdinal, ctx->deviceName, ctx->computeUnits, ctx->warpSize,
        ctx->sharedMemPerBlock / 1024, (double)ctx->totalGlobalMem / (1024.0*1024.0*1024.0));
    return true;
}

void hipContext_release(HipContext* ctx)
{
    if (ctx->context) {
        hipCtxDestroy(ctx->context);
        ctx->context = nullptr;
    }
    ctx->isValid = false;
}

// ---- Memory ----

bool hipContext_memAlloc(HipContext*, hipDeviceptr_t* dptr, size_t bytes)
{
    HIP_CHECK(hipMalloc(dptr, bytes), "hipMalloc");
    return true;
}

bool hipContext_memFree(HipContext*, hipDeviceptr_t dptr)
{
    HIP_CHECK(hipFree(dptr), "hipFree");
    return true;
}

bool hipContext_memHostAlloc(HipContext*, void** ptr, size_t bytes, unsigned int)
{
    HIP_CHECK(hipHostMalloc(ptr, bytes, hipHostMallocMapped | hipHostMallocPortable), "hipHostMalloc");
    return true;
}

bool hipContext_memFreeHost(HipContext*, void* ptr)
{
    HIP_CHECK(hipHostFree(ptr), "hipHostFree");
    return true;
}

// ---- Copy ----

bool hipContext_memcpyHtoD(HipContext*, hipDeviceptr_t dst, const void* src, size_t bytes)
{
    HIP_CHECK(hipMemcpyHtoD(dst, (void*)src, bytes), "hipMemcpyHtoD");
    return true;
}

bool hipContext_memcpyDtoH(HipContext*, void* dst, hipDeviceptr_t src, size_t bytes)
{
    HIP_CHECK(hipMemcpyDtoH(dst, src, bytes), "hipMemcpyDtoH");
    return true;
}

bool hipContext_memcpyDtoD(HipContext*, hipDeviceptr_t dst, hipDeviceptr_t src, size_t bytes)
{
    HIP_CHECK(hipMemcpyDtoD(dst, src, bytes), "hipMemcpyDtoD");
    return true;
}

// ---- Stream ----

bool hipContext_streamCreate(HipContext*, hipStream_t* stream)
{
    HIP_CHECK(hipStreamCreate(stream), "hipStreamCreate");
    return true;
}

bool hipContext_streamDestroy(HipContext*, hipStream_t stream)
{
    HIP_CHECK(hipStreamDestroy(stream), "hipStreamDestroy");
    return true;
}

bool hipContext_streamSync(HipContext*, hipStream_t stream)
{
    HIP_CHECK(hipStreamSynchronize(stream), "hipStreamSynchronize");
    return true;
}

// ---- Event ----

bool hipContext_eventCreate(HipContext*, hipEvent_t* event)
{
    HIP_CHECK(hipEventCreate(event), "hipEventCreate");
    return true;
}

bool hipContext_eventRecord(HipContext*, hipEvent_t event, hipStream_t stream)
{
    HIP_CHECK(hipEventRecord(event, stream), "hipEventRecord");
    return true;
}

bool hipContext_eventSync(HipContext*, hipEvent_t event)
{
    HIP_CHECK(hipEventSynchronize(event), "hipEventSynchronize");
    return true;
}

bool hipContext_eventDestroy(HipContext*, hipEvent_t event)
{
    HIP_CHECK(hipEventDestroy(event), "hipEventDestroy");
    return true;
}

// ---- Module ----

bool hipContext_moduleLoadData(HipContext*, hipModule_t* module, const void* image)
{
    HIP_CHECK(hipModuleLoadData(module, image), "hipModuleLoadData");
    return true;
}

bool hipContext_moduleGetFunc(HipContext*, hipFunction_t* func, hipModule_t module, const char* name)
{
    HIP_CHECK(hipModuleGetFunction(func, module, name), "hipModuleGetFunction");
    return true;
}

bool hipContext_moduleUnload(HipContext*, hipModule_t module)
{
    HIP_CHECK(hipModuleUnload(module), "hipModuleUnload");
    return true;
}

// ---- Kernel Launch ----

bool hipContext_launchKernel(HipContext*,
    hipFunction_t func,
    unsigned int gridX, unsigned int gridY, unsigned int gridZ,
    unsigned int blockX, unsigned int blockY, unsigned int blockZ,
    unsigned int sharedMemBytes,
    hipStream_t stream,
    void** kernelParams,
    void**)
{
    HIP_CHECK(hipModuleLaunchKernel(func,
        gridX, gridY, gridZ,
        blockX, blockY, blockZ,
        sharedMemBytes,
        stream,
        kernelParams,
        nullptr), "hipModuleLaunchKernel");
    return true;
}
