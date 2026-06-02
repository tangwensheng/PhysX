// DCU Toolchain Probe Kernel
// Validates: hipcc compilation, HIP runtime, device intrinsics, WARP_SIZE correctness
// Compatible with older HIP/ROCm versions (no _sync suffix intrinsics)

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <cstdio>

// Verify that PxgCommonDefines.h correctly detects the DCU platform
#define PX_DCU_PORT
#include "PxgCommonDefines.h"

// ---- Compile-time assertions ----
// These fire at compile time if the platform defines are wrong
#if WARP_SIZE != 64
#error "WARP_SIZE must be 64 on DCU. Check PxgCommonDefines.h platform detection."
#endif

#if FULL_MASK != 0xffffffffffffffffULL
#error "FULL_MASK must be 64-bit on DCU. Check PxgCommonDefines.h platform detection."
#endif


// ---- Probe 1: Basic vector add (validates kernel launch + memory) ----
extern "C" __global__ void probe_vector_add(const float* a, const float* b, float* c, int n)
{
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    if (tid < n)
        c[tid] = a[tid] + b[tid];
}


// ---- Probe 2: Warp shuffle (validates shuffle with wavefront=64) ----
// Old HIP uses __shfl / __shfl_xor without _sync suffix, no mask argument
extern "C" __global__ void probe_shuffle(float* out, int n)
{
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    if (tid >= n) return;

    int lane = tid & (WARP_SIZE - 1);

    // Broadcast from lane 0 in each wavefront
    float val = (float)(tid + 1);
    float bc = __shfl(val, 0, WARP_SIZE);

    // Min reduction via xor shuffle (butterfly pattern)
    float minVal = val;
#pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1)
    {
        float other = __shfl_xor(minVal, offset, WARP_SIZE);
        if (other < minVal) minVal = other;
    }

    out[tid] = bc + minVal;
}


// ---- Probe 3: Ballot + popc (validates warp vote and bit ops with 64-bit mask) ----
extern "C" __global__ void probe_ballot_popc(unsigned int* out, int n)
{
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    if (tid >= n) return;

    int lane = tid & (WARP_SIZE - 1);

    // Old HIP: __ballot takes a predicate only, returns 64-bit mask
    unsigned long long mask = __ballot((lane % 2) == 0);

    // popc on 64-bit mask
    int count = __popcll(mask);

    // Full-mask: all lanes active in this simple probe
    unsigned long long active = FULL_MASK;

    out[tid] = (unsigned int)(count | ((unsigned int)(active & 1) << 16));
}


// ---- Probe 4: Atomic operations ----
extern "C" __global__ void probe_atomics(float* atomicAddOut, float* atomicMinOut, float* atomicMaxOut, int n)
{
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    if (tid >= n) return;

    float val = (float)(tid + 1);

    atomicAdd(atomicAddOut, val);

    // Float atomicMin via atomicCAS (same pattern as PhysX atomic.cuh)
    int* addr_i = (int*)atomicMinOut;
    int old = *addr_i;
    int assumed;
    while (val < __int_as_float(old))
    {
        assumed = old;
        old = atomicCAS(addr_i, assumed, __float_as_int(val));
    }
}


// ---- Probe 5: Shared memory + sync ----
extern "C" __global__ void probe_shared_sync(float* out, int n)
{
    extern __shared__ float smem[];
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    int lane = threadIdx.x;

    // Each thread writes its lane index to shared memory
    smem[lane] = (float)lane;

    __syncthreads();

    // Read neighbor's value (ring shift)
    int neighbor = (lane + 1) % blockDim.x;
    float neighborVal = smem[neighbor];

    __syncthreads();

    if (tid < n)
        out[tid] = neighborVal;
}


// ---- Probe 6: __launch_bounds__ equivalent ----
extern "C" __global__ __launch_bounds__(256, 4)
void probe_launch_bounds(float* out, int n)
{
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    if (tid < n)
        out[tid] = (float)tid;
}
