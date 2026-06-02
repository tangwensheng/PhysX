// gpucommon module HIP compilation test
// Validates key gpucommon headers compile with hipcc
// Uses minimal PhysX type stubs since we compile standalone

// ---- Minimal PhysX type stubs (for standalone compilation without foundation lib) ----
namespace physx {
    typedef unsigned int  PxU32;
    typedef unsigned long long PxU64;
    typedef float PxReal;
    struct PxVec3 { float x, y, z;
        __host__ __device__ PxVec3() : x(0), y(0), z(0) {}
        __host__ __device__ PxVec3(float a, float b, float c) : x(a), y(b), z(c) {}
        __host__ __device__ float dot(const PxVec3& v) const { return x*v.x + y*v.y + z*v.z; }
        __host__ __device__ float magnitudeSquared() const { return x*x + y*y + z*z; }
    };
    struct PxVec4 { float x, y, z, w;
        __host__ __device__ PxVec4() : x(0), y(0), z(0), w(0) {}
        __host__ __device__ PxVec4(float a, float b, float c, float d_) : x(a), y(b), z(c), w(d_) {}
    };
}
// Atomic.cuh includes "foundation/PxVec3.h", "foundation/PxSimpleTypes.h", "PxgIntrinsics.h"
// These are provided by the PhysX CPU build system, not needed in standalone test.
// For now, just define the minimal types above and include atomic.cuh directly.
// The full build will use the actual PhysX headers.

#define PX_FORCE_INLINE inline __attribute__((always_inline))
#define PX_RESTRICT __restrict__
#define PX_UNUSED(x) (void)(x)

// ---- Now include the adapted gpucommon headers ----

// Core (auto-pulls PxgHIPCompat.h for HIP)
#include "PxgCommonDefines.h"

// Adapted headers
#include "shuffle.cuh"

// Test kernel
extern "C" __global__ void test_headers(float* out, int n)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n) return;
    int lane = tid & (WARP_SIZE - 1);
    float val = (float)tid;
    float bc = __shfl_sync(FULL_MASK, val, 0, WARP_SIZE);
    out[tid] = bc;
}

extern "C" __global__ void test_warp_size(unsigned int* out)
{
    int tid = threadIdx.x;
    if (tid == 0) {
        out[0] = WARP_SIZE;
        out[1] = (unsigned int)(FULL_MASK & 0xFFFFFFFFULL);
        out[2] = (unsigned int)((FULL_MASK >> 32) & 0xFFFFFFFFULL);
    }
}

#include <cstdio>
int main()
{
    const int N = 64;
    float *d_out, h_out[N];
    hipMalloc(&d_out, N * sizeof(float));

    test_headers<<<1, 64>>>(d_out, N);
    hipDeviceSynchronize();
    hipMemcpy(h_out, d_out, N * sizeof(float), hipMemcpyDeviceToHost);

    unsigned int *d_info, h_info[3];
    hipMalloc(&d_info, 3 * sizeof(unsigned int));
    test_warp_size<<<1, 1>>>(d_info);
    hipDeviceSynchronize();
    hipMemcpy(h_info, d_info, 3 * sizeof(unsigned int), hipMemcpyDeviceToHost);

    printf("=== gpucommon Header Test ===\n");
    printf("WARP_SIZE: %u (expect 64)\n", h_info[0]);
    printf("FULL_MASK lo32: 0x%08x (expect 0xffffffff)\n", h_info[1]);
    printf("FULL_MASK hi32: 0x%08x (expect 0xffffffff)\n", h_info[2]);

    bool pass = (h_info[0] == 64) && (h_info[1] == 0xffffffff) && (h_info[2] == 0xffffffff);
    printf("%s\n", pass ? "gpucommon headers: ALL PASS" : "FAIL");

    hipFree(d_out); hipFree(d_info);
    return pass ? 0 : 1;
}
