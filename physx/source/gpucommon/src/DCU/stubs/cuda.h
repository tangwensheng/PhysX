// HIP stub for <cuda.h> — redirects to HIP runtime + CUDA type compatibility
//
// For hipcc (kernel .cu files): just delegate to <hip/hip_runtime.h>.
// The HIP runtime provides all CUDA types, builtins (atomicAdd etc.),
// and vector types (float4) that kernel code needs.
//
// For g++ (host-side .cpp files): do NOT include <hip/hip_runtime.h>.
// HIP's "using float4 = HIP_vector_type<...>" conflicts with PhysX GPU
// headers that forward-declare "struct float4".  Instead we define all
// CUDA types ourselves as opaque pointers.
#pragma once

#if defined(__HIPCC__)
// Kernel code compiled by hipcc — use real HIP runtime
#include <hip/hip_runtime.h>
// CUDA type aliases on top of HIP types (PhysX GPU headers use CUDA names)
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
struct CUarray_st;    typedef CUarray_st*    CUarray;
struct CUtexObject_st; typedef CUtexObject_st* CUtexObject;
#else
// Host-side code compiled by g++ — standalone CUDA type definitions
#include <stddef.h>  // for size_t

// Prevent PxCudaTypes.h from redefining types (including HIP runtime)
#define CUDA_VERSION 12000

// ---- Host-side CUDA compatibility macros ----
// PxgHIPCompat.h provides these for HIP-compiled (.cu→HIP) files but NOT for
// plain C++ host-side files compiled with g++.  We provide the essential ones
// here since the DCU stub <cuda.h> is force-included (-include) before every
// HipContext source file.

// CUDA alignment attribute → GCC/Clang
#ifndef __builtin_align__
#define __builtin_align__(N)  __attribute__((aligned(N)))
#endif

// Suppress CUDA __device__ / __host__ / __global__ decorators (host code only)
#define __device__
#define __host__
#define __global__
#define __forceinline__  inline
#define __builtin_expect(x, y)  __builtin_expect((x), (y))

// ---- CUDA type mappings ----
// CUdeviceptr: keep as unsigned long long for pointer arithmetic compatibility
#if defined(__x86_64__) || defined(__aarch64__) || defined(__LP64__)
typedef unsigned long long CUdeviceptr;
#else
typedef unsigned int CUdeviceptr;
#endif

typedef int                      CUdevice;

// Opaque struct pointers — compatible with both CUDA Driver API and our
// CudaWrappersDCU.cpp which casts them to HIP types internally.
typedef struct CUctx_st*         CUcontext;
typedef struct CUmod_st*         CUmodule;
typedef struct CUfunc_st*        CUfunction;
typedef struct CUstream_st*      CUstream;
typedef struct CUevent_st*       CUevent;
typedef struct CUarray_st*       CUarray;
typedef struct CUtexObject_st*   CUtexObject;

// CUDA array format enum (used by PxgGeometryManager for SDF textures)
typedef enum CUarray_format_enum {
	CU_AD_FORMAT_UNSIGNED_INT8  = 0x01,
	CU_AD_FORMAT_UNSIGNED_INT16 = 0x02,
	CU_AD_FORMAT_UNSIGNED_INT32 = 0x03,
	CU_AD_FORMAT_SIGNED_INT8    = 0x08,
	CU_AD_FORMAT_SIGNED_INT16   = 0x09,
	CU_AD_FORMAT_SIGNED_INT32   = 0x0a,
	CU_AD_FORMAT_HALF           = 0x10,
	CU_AD_FORMAT_FLOAT          = 0x20
} CUarray_format;

// CUDA resource descriptor types (needed by PxgGeometryManager)
typedef enum CUresourcetype_enum {
	CU_RESOURCE_TYPE_ARRAY           = 0x00,
	CU_RESOURCE_TYPE_MIPMAPPED_ARRAY = 0x02,
	CU_RESOURCE_TYPE_LINEAR          = 0x04
} CUresourcetype;

struct CUDA_RESOURCE_DESC {
	CUresourcetype resType;
	union {
		struct { CUarray hArray; } array;
	} res;
};

struct CUDA_TEXTURE_DESC {
	unsigned int addressMode[3];
	unsigned int filterMode;
	unsigned int flags;
	unsigned int maxAnisotropy;
	unsigned int minMipmapLevelClamp;
	unsigned int maxMipmapLevelClamp;
	float borderColor[4];
};
struct CUDA_RESOURCE_VIEW_DESC {};

// Texture addressing / filter constants
#define CU_TR_ADDRESS_MODE_CLAMP  1
#define CU_TR_ADDRESS_MODE_WRAP   2
#define CU_TR_FILTER_MODE_POINT   1
#define CU_TR_FILTER_MODE_LINEAR  2

// 3D array descriptor (for SDF texture upload in PxgGeometryManager)
typedef struct CUDA_ARRAY3D_DESCRIPTOR_st {
	size_t Width, Height, Depth;
	CUarray_format Format;
	unsigned int NumChannels;
	unsigned int Flags;
} CUDA_ARRAY3D_DESCRIPTOR;

// 3D memory copy descriptor
typedef enum CUmemorytype_enum {
	CU_MEMORYTYPE_HOST   = 0x01,
	CU_MEMORYTYPE_DEVICE = 0x02,
	CU_MEMORYTYPE_ARRAY  = 0x03,
	CU_MEMORYTYPE_UNIFIED = 0x04
} CUmemorytype;

typedef struct CUDA_MEMCPY3D_st {
	size_t srcXInBytes, srcY, srcZ;
	size_t srcPitch, srcHeight;
	size_t dstXInBytes, dstY, dstZ;
	size_t dstPitch, dstHeight;
	size_t WidthInBytes, Height, Depth;
	CUmemorytype srcMemoryType;
	const void* srcHost;
	CUdeviceptr srcDevice;
	CUarray srcArray;
	CUmemorytype dstMemoryType;
	void* dstHost;
	CUdeviceptr dstDevice;
	CUarray dstArray;
} CUDA_MEMCPY3D;

// CUresult — PxCudaContext.h uses PxCUenum<CUresult> on POSIX, so this
// must be a valid type.  Use int (hipError_t values fit in int).
typedef int CUresult;

// ---- CUDA error codes (map to HIP values, matched in CudaWrappersDCU.cpp) ----
#define CUDA_SUCCESS                    0   // hipSuccess
#define CUDA_ERROR_UNKNOWN              1   // hipErrorUnknown
#define CUDA_ERROR_INVALID_VALUE        2   // hipErrorInvalidValue
#define CUDA_ERROR_NO_BINARY_FOR_GPU    3   // hipErrorNoBinaryForGpu
#define CUDA_ERROR_OUT_OF_MEMORY        4   // hipErrorOutOfMemory
#define CUDA_ERROR_INVALID_IMAGE        5   // hipErrorInvalidImage
#define CUDA_ERROR_DEINITIALIZED        6   // hipErrorDeinitialized
#define CUDA_ERROR_NOT_READY            7   // hipErrorNotReady
#define CUDA_ERROR_INVALID_CONTEXT      8   // hipErrorInvalidContext

// ---- CUDA context flags ----
#define CU_CTX_LMEM_RESIZE_TO_MAX   0
#define CU_CTX_SCHED_BLOCKING_SYNC  0
#define CU_CTX_MAP_HOST             0
#define CU_CTX_SCHED_AUTO           0

// ---- CUDA stream flags ----
#define CU_STREAM_DEFAULT       ((CUstream)0)
#define CU_STREAM_NON_BLOCKING  0x01

// ---- CUDA host memory flags ----
#define CU_MEMHOSTALLOC_PORTABLE      0x01
#define CU_MEMHOSTALLOC_DEVICEMAP     0x02
#define CU_MEMHOSTALLOC_WRITECOMBINED 0x04

// ---- CUDA device attribute enum values ----
#define CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK           1
#define CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_MULTIPROCESSOR  2
#define CU_DEVICE_ATTRIBUTE_CLOCK_RATE                             3
#define CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR               4
#define CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR               5
#define CU_DEVICE_ATTRIBUTE_INTEGRATED                             6
#define CU_DEVICE_ATTRIBUTE_CAN_MAP_HOST_MEMORY                    7
#define CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT                   8
#define CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK                  9
#define CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN      10

// ---- CUDA event flags ----
#define CU_EVENT_DEFAULT          0
#define CU_EVENT_BLOCKING_SYNC    1
#define CU_EVENT_DISABLE_TIMING   2

// ---- CUjit_option ----
typedef unsigned int CUjit_option;

// ---- CUDA Driver API wrapper functions ----
// Implemented in CudaWrappersDCU.cpp (which includes hip_runtime.h and
// performs opaque-pointer casts to the real HIP types internally).
#ifdef __cplusplus
extern "C" {
#endif

// Stream management
CUresult cuStreamSynchronize(CUstream hStream);
CUresult cuStreamQuery(CUstream hStream);

// Stream priority
CUresult cuCtxGetStreamPriorityRange(int* leastPriority, int* greatestPriority);

// Synchronous memory copy
CUresult cuMemcpyDtoH(void* dstHost, CUdeviceptr srcDevice, size_t ByteCount);
CUresult cuMemcpyHtoD(CUdeviceptr dstDevice, const void* srcHost, size_t ByteCount);

// Texture / Array (SDF) — stubs, not yet implemented for DCU
CUresult cuArray3DCreate(CUarray* pHandle, const CUDA_ARRAY3D_DESCRIPTOR* pAllocateArray);
CUresult cuArrayDestroy(CUarray hArray);
CUresult cuTexObjectCreate(CUtexObject* pTexObject, const void* pResDesc, const void* pTexDesc, const void* pResViewDesc);
CUresult cuTexObjectDestroy(CUtexObject texObject);
CUresult cuMemcpy3D(const CUDA_MEMCPY3D* pCopy);
CUresult cuMemcpy3DAsync(const CUDA_MEMCPY3D* pCopy, CUstream hStream);

#ifdef __cplusplus
}
#endif

#endif // __HIPCC__  (host-side g++ path above, hipcc path delegates to <hip/hip_runtime.h>)

