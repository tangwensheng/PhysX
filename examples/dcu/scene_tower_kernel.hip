// GPU kernel for tower collapse scene
#include "foundation/PxVec3.h"
#include <hip/hip_runtime.h>

using namespace physx;

extern "C" __global__ void gpu_contact_filter(
    const PxVec3* positions, int n, float groundY, int* belowCount)
{
    __shared__ int smem[256];
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int lane = threadIdx.x;
    smem[lane] = (i < n && positions[i].y < groundY) ? 1 : 0;
    __syncthreads();

    for (int s = 128; s > 0; s >>= 1) {
        if (lane < s) smem[lane] += smem[lane + s];
        __syncthreads();
    }
    if (lane == 0) atomicAdd(belowCount, smem[0]);
}

// Host-side helper: launch the kernel
extern "C" void launch_contact_filter(const PxVec3* d_pos, int n, float groundY, int* d_count)
{
    gpu_contact_filter<<<(n+255)/256, 256>>>(d_pos, n, groundY, d_count);
}
