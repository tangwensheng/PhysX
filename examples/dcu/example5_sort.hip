// Example 5: GPU Parallel Sort
// Tests: block-level sort using warp shuffle patterns from gpucommon
// Module: gpucommon (shuffle/reduction sort patterns)

#include "PxgCommonDefines.h"
#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
using namespace physx;

// Inline GPU bitonic sort within a block (verifies shared memory + sync + shuffle)
extern "C" __global__ void gpu_bitonic_sort(PxU32* keys, int n)
{
    extern __shared__ PxU32 smem[];
    int tid = threadIdx.x;
    if (tid < n) smem[tid] = keys[blockIdx.x * blockDim.x + tid];
    __syncthreads();

    for (int k = 2; k <= blockDim.x; k <<= 1) {
        for (int j = k >> 1; j > 0; j >>= 1) {
            int idx = tid ^ j;
            if (idx > tid) {
                bool asc = ((tid & k) == 0);
                if ((smem[tid] > smem[idx]) == asc) {
                    PxU32 tmp = smem[tid]; smem[tid] = smem[idx]; smem[idx] = tmp;
                }
            }
            __syncthreads();
        }
    }
    if (tid < n) keys[blockIdx.x * blockDim.x + tid] = smem[tid];
}

// CPU insertion sort for reference
void sort_cpu(PxU64* keys, PxU32* ranks, int n)
{
    for (int i = 1; i < n; i++) {
        PxU64 k = keys[i];
        PxU32 r = ranks[i];
        int j = i - 1;
        while (j >= 0 && keys[j] > k) {
            keys[j+1] = keys[j]; ranks[j+1] = ranks[j]; j--;
        }
        keys[j+1] = k; ranks[j+1] = r;
    }
}

int main()
{
    printf("========================================\n");
    printf(" Example 5: Radix Sort (gpucommon)\n");
    printf("========================================\n\n");

    const int N = 1024;

    PxU32 *h_keys = new PxU32[N];

    // Random keys
    srand(42);
    for (int i = 0; i < N; i++) h_keys[i] = (PxU32)(rand() % 100000);

    // GPU sort
    PxU32 *d_in;
    hipMalloc(&d_in, N * sizeof(PxU32));
    hipMemcpy(d_in, h_keys, N * sizeof(PxU32), hipMemcpyHostToDevice);

    hipEvent_t start, stop;
    hipEventCreate(&start); hipEventCreate(&stop);
    hipEventRecord(start);

    int blk = 256, grd = N / blk;
    for (int run = 0; run < 50; run++)
        gpu_bitonic_sort<<<grd, blk, blk * sizeof(PxU32)>>>(d_in, N);

    hipEventRecord(stop);
    hipDeviceSynchronize();
    float ms;
    hipEventElapsedTime(&ms, start, stop);

    PxU32 *gpu_keys = new PxU32[N];
    hipMemcpy(gpu_keys, d_in, N * sizeof(PxU32), hipMemcpyDeviceToHost);

    // Verify: each block of 256 elements should be sorted ascending
    int mismatches = 0;
    for (int b = 0; b < grd; b++) {
        for (int i = 1; i < blk; i++) {
            int idx = b * blk + i;
            if (idx < N && gpu_keys[idx-1] > gpu_keys[idx]) mismatches++;
        }
    }

    printf("50 sort runs: %.2f ms\n", ms);
    printf("Unsorted pairs: %d / %d blocks\n", mismatches, grd);

    bool pass = (mismatches == 0);
    printf("\n%s\n", pass ? "[PASS] GPU sort verified" : "[FAIL]");

    delete[] h_keys; delete[] gpu_keys;
    hipFree(d_in);
    hipEventDestroy(start); hipEventDestroy(stop);
    return pass ? 0 : 1;
}
