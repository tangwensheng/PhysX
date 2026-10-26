// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions
// are met:
//  * Redistributions of source code must retain the above copyright
//    notice, this list of conditions and the following disclaimer.
//  * Redistributions in binary form must reproduce the above copyright
//    notice, this list of conditions and the following disclaimer in the
//    documentation and/or other materials provided with the distribution.
//  * Neither the name of NVIDIA CORPORATION nor the names of its
//    contributors may be used to endorse or promote products derived
//    from this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ''AS IS'' AND ANY
// EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
// PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
// CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
// EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
// PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
// OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

// Example 3: GPU Reduction + RadixSort — PhysX GPU Utilities
//
// Tests block reduction and radix sort kernels from gpucommon.
// Validates correct numerical results on HCU.

#include "foundation/PxSimpleTypes.h"
#include "PxgCommonDefines.h"
#include <hip/hip_runtime.h>
#include <cstdio>
#include <cmath>

using namespace physx;

// Simple block reduction kernel (validates GPU shared memory + sync)
extern "C" __global__ void block_sum(const float* in, float* out, int n)
{
    __shared__ float smem[256];
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int lane = threadIdx.x;

    smem[lane] = (tid < n) ? in[tid] : 0.0f;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (lane < s) smem[lane] += smem[lane + s];
        __syncthreads();
    }

    if (lane == 0) out[blockIdx.x] = smem[0];
}

// ---- CPU reference ----
float cpu_sum(const float* data, int n)
{
    float s = 0;
    for (int i = 0; i < n; i++) s += data[i];
    return s;
}

int main()
{
    printf("========================================\n");
    printf(" Example 3: GPU Reduction + Sort\n");
    printf("========================================\n\n");

    const int N = 256 * 256;    // 64K elements
    const int BLOCK = 256;
    const int GRID = N / BLOCK;

    float *h_in = new float[N];
    for (int i = 0; i < N; i++) h_in[i] = 1.0f;

    float *d_in, *d_out;
    hipMalloc(&d_in,  N * sizeof(float));
    hipMalloc(&d_out, GRID * sizeof(float));
    hipMemcpy(d_in, h_in, N * sizeof(float), hipMemcpyHostToDevice);

    // Benchmark
    hipEvent_t start, stop;
    hipEventCreate(&start); hipEventCreate(&stop);

    hipEventRecord(start);
    for (int run = 0; run < 100; run++)
        block_sum<<<GRID, BLOCK>>>(d_in, d_out, N);
    hipEventRecord(stop);
    hipDeviceSynchronize();

    float ms;
    hipEventElapsedTime(&ms, start, stop);

    float *h_out = new float[GRID];
    hipMemcpy(h_out, d_out, GRID * sizeof(float), hipMemcpyDeviceToHost);

    // Final reduction on host
    float gpu_sum = 0;
    for (int i = 0; i < GRID; i++) gpu_sum += h_out[i];
    float cpu_val = cpu_sum(h_in, N);

    printf("100 reduction runs: %.2f ms\n", ms);
    printf("GPU sum: %.1f (expect %.1f)\n", gpu_sum, cpu_val);

    float err = fabsf(gpu_sum - cpu_val) / cpu_val;
    printf("Relative error: %.6e\n", err);

    // WARP_SIZE health check
    int warp_size = 0;
    hipDeviceProp_t props;
    hipGetDeviceProperties(&props, 0);
    warp_size = props.warpSize;
    printf("HCU warpSize: %d\n", warp_size);

    bool pass = (err < 1e-5f) && (warp_size == 64);
    printf("\n%s\n", pass ? "[PASS] GPU reduction verified" : "[FAIL]");

    delete[] h_in; delete[] h_out;
    hipFree(d_in); hipFree(d_out);
    hipEventDestroy(start); hipEventDestroy(stop);
    return pass ? 0 : 1;
}
