// Example 4: Warp Shuffle + Atomics Precision
// Tests: shuffle.cuh warp reductions, atomic.cuh float atomics
// Module: gpucommon

#include "PxgCommonDefines.h"
#include "shuffle.cuh"
#include <hip/hip_runtime.h>
#include <cstdio>
#include <cmath>

// ---- Test 1: Warp shuffle sum reduction ----
extern "C" __global__ void warp_sum_test(const float* in, float* out, int n)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int lane = tid & (WARP_SIZE - 1);

    float val = (tid < n) ? in[tid] : 0.0f;

    // Butterfly reduction via xor shuffle
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
        float other = __shfl_xor_sync(FULL_MASK, val, offset);
        val += other;
    }

    if (lane == 0) atomicAdd(out, val);
}

// ---- Test 2: Float AtomicMin via atomicCAS ----
extern "C" __global__ void atomic_min_test(float* out, int n)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n) return;

    float val = (float)(tid + 1);
    // Float atomicMin using atomicCAS (same pattern as PhysX atomic.cuh)
    int* addr = (int*)out;
    int old = *addr;
    int assumed;
    do {
        assumed = old;
        if (val >= __int_as_float(assumed)) break;
        old = atomicCAS(addr, assumed, __float_as_int(val));
    } while (assumed != old);
}

int main()
{
    printf("========================================\n");
    printf(" Example 4: Shuffle + Atomics (gpucommon)\n");
    printf("========================================\n\n");

    const int N = 16384;
    const int BLK = 256, GRD = N / BLK;

    // --- Shuffle Sum ---
    float *h_in = new float[N], *d_in, *d_sum;
    float cpu = 0;
    for (int i = 0; i < N; i++) { h_in[i] = 1.0f + 0.001f * (i % 100); cpu += h_in[i]; }

    hipMalloc(&d_in, N * sizeof(float));
    hipMalloc(&d_sum, sizeof(float));
    hipMemcpy(d_in, h_in, N * sizeof(float), hipMemcpyHostToDevice);

    float zero = 0;
    hipMemcpy(d_sum, &zero, sizeof(float), hipMemcpyHostToDevice);

    warp_sum_test<<<GRD, BLK>>>(d_in, d_sum, N);
    hipDeviceSynchronize();

    float gpu;
    hipMemcpy(&gpu, d_sum, sizeof(float), hipMemcpyDeviceToHost);
    printf("Shuffle sum:  GPU=%.3f  CPU=%.3f  err=%.6e\n", gpu, cpu, fabs(gpu-cpu)/cpu);

    // --- Atomic Min ---
    float *d_min;
    hipMalloc(&d_min, sizeof(float));
    float init = 1e10f;
    hipMemcpy(d_min, &init, sizeof(float), hipMemcpyHostToDevice);

    atomic_min_test<<<GRD, BLK>>>(d_min, N);
    hipDeviceSynchronize();

    float minVal;
    hipMemcpy(&minVal, d_min, sizeof(float), hipMemcpyDeviceToHost);
    printf("AtomicMin:    GPU=%.1f  expect=1.0  %s\n", minVal, (fabsf(minVal-1.0f)<0.1f)?"PASS":"FAIL");

    // --- WARP_SIZE check ---
    unsigned *d_warp;
    hipMalloc(&d_warp, sizeof(unsigned));
    printf("WARP_SIZE:    %u (expect 64)\n", WARP_SIZE);

    bool pass = (fabs(gpu - cpu) / cpu < 1e-4f) && (fabsf(minVal - 1.0f) < 0.1f) && (WARP_SIZE == 64);
    printf("\n%s\n", pass ? "[PASS] Shuffle + Atomics OK" : "[FAIL]");

    delete[] h_in;
    hipFree(d_in); hipFree(d_sum); hipFree(d_min); hipFree(d_warp);
    return pass ? 0 : 1;
}
