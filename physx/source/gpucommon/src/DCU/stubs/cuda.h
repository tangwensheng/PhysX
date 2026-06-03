// HIP stub for <cuda.h> — redirects to HIP runtime + CUDA type compatibility
#pragma once
#include <hip/hip_runtime.h>

// Prevent PxCudaTypes.h from redefining types
#define CUDA_VERSION 12000

// Map CUDA host API types to HIP equivalents
// CUdeviceptr: keep as unsigned long long (integer type) for compatibility
// with PhysX host code that does arithmetic on device pointers.
// HIP API calls will cast appropriately in HipContext.cpp.
#if defined(__x86_64__) || defined(__aarch64__) || defined(__LP64__)
typedef unsigned long long CUdeviceptr;
#else
typedef unsigned int CUdeviceptr;
#endif

typedef int                      CUdevice;
typedef hipCtx_t                 CUcontext;
typedef hipModule_t              CUmodule;
typedef hipFunction_t            CUfunction;
typedef hipStream_t              CUstream;
typedef hipEvent_t               CUevent;

// CUarray and CUtexObject: only used in PxgGeometryManager for SDF
// Skip for now (compile check only) — these are host-side types
struct CUarray_st;    typedef CUarray_st*    CUarray;
struct CUtexObject_st; typedef CUtexObject_st* CUtexObject;

// CUresult values
#define CUDA_SUCCESS        hipSuccess
#define CUDA_ERROR_UNKNOWN  hipErrorUnknown
#define CUDA_ERROR_INVALID_VALUE hipErrorInvalidValue
#define CUDA_ERROR_NO_BINARY_FOR_GPU hipErrorNoBinaryForGpu
#define CUDA_ERROR_OUT_OF_MEMORY hipErrorOutOfMemory

// CUDA context flags
#define CU_CTX_LMEM_RESIZE_TO_MAX   0
#define CU_CTX_SCHED_BLOCKING_SYNC  0
#define CU_CTX_MAP_HOST             0

// CUDA host memory flags (for cuMemHostAlloc compatibility)
#define CU_MEMHOSTALLOC_PORTABLE  0x01
#define CU_MEMHOSTALLOC_DEVICEMAP 0x02

// Device attribute enum values
#define CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK        1
#define CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_MULTIPROCESSOR 2
#define CU_DEVICE_ATTRIBUTE_CLOCK_RATE                          3
#define CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR            4
#define CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR            5
#define CU_DEVICE_ATTRIBUTE_INTEGRATED                          6
#define CU_DEVICE_ATTRIBUTE_CAN_MAP_HOST_MEMORY                 7
#define CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT                8
#define CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK               9

// CUjit_option
typedef unsigned int CUjit_option;

// CUstream default
#define CU_STREAM_DEFAULT ((CUstream)0)
