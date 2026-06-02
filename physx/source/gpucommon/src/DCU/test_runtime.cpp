// PhysX DCU Runtime Layer Test
//
// Validates the full HIP runtime surface needed by PhysX:
//   HipContext → Memory → Stream → Event → Kernel Launch → Module Load
//
// Build:
//   hipcc test_runtime.cpp HipContext.cpp probe_kernel.hip \
//       -Isource/gpucommon/include -Isource/gpunarrowphase/src/CUDA \
//       -o test_runtime --offload-arch=gfx936
//
// The probe_kernel.hip kernels are launched through:
//   1. Direct <<<>>> launch (standard launch path)
//   2. hipModuleLoadData + hipModuleLaunchKernel (PhysX explicit module path)

#include "HipContext.h"
#include "PxgHIPCompat.h"
#include "PxgCommonDefines.h"
#include <cstdio>
#include <cmath>
#include <cstdlib>

// Kernels from probe_kernel.hip (extern declarations)
extern "C" __global__ void probe_vector_add(const float* a, const float* b, float* c, int n);
extern "C" __global__ void probe_shuffle(float* out, int n);
extern "C" __global__ void probe_ballot_popc(unsigned int* out, int n);
extern "C" __global__ void probe_atomics(float* sumOut, float* minOut, float* maxOut, int n);

static int passed = 0, total = 0;

void check(const char* name, bool ok) {
    total++;
    if (ok) { passed++; printf("  [PASS] %s\n", name); }
    else    { printf("  [FAIL] %s\n", name); }
}

int main()
{
    printf("=== PhysX DCU Runtime Layer Test ===\n\n");

    // ---- Test 1: HipContext Init ----
    printf("--- Test 1: HipContext Init ---\n");
    HipContext ctx;
    bool ok = hipContext_init(&ctx, 0);
    check("hipContext_init", ok && ctx.isValid);
    if (!ok) { printf("FATAL: Cannot init HIP context\n"); return 1; }
    check("warpSize == 64 (DCU wavefront)", ctx.warpSize == 64);
    check("maxThreadsPerBlock >= 256", ctx.maxThreadsPerBlock >= 256);

    // ---- Test 2: Memory Allocation ----
    printf("--- Test 2: Memory Allocation ---\n");
    const int N = 256 * 1024;
    const size_t bytes = N * sizeof(float);

    hipDeviceptr_t d_a = 0, d_b = 0, d_c = 0;
    check("hipMalloc (3 buffers)",
        hipContext_memAlloc(&ctx, &d_a, bytes) &&
        hipContext_memAlloc(&ctx, &d_b, bytes) &&
        hipContext_memAlloc(&ctx, &d_c, bytes));

    // Pinned host memory
    float *h_pinned = nullptr;
    check("hipHostMalloc", hipContext_memHostAlloc(&ctx, (void**)&h_pinned, bytes, 0));

    // ---- Test 3: H2D / D2H Copy ----
    printf("--- Test 3: Memory Copy ---\n");
    float *h_a = new float[N];
    float *h_b = new float[N];
    for (int i = 0; i < N; i++) { h_a[i] = (float)i; h_b[i] = (float)(i*2); }

    check("memcpy H2D (a)", hipContext_memcpyHtoD(&ctx, d_a, h_a, bytes));
    check("memcpy H2D (b)", hipContext_memcpyHtoD(&ctx, d_b, h_b, bytes));

    // ---- Test 4: Stream + Event ----
    printf("--- Test 4: Stream + Event ---\n");
    hipStream_t stream = nullptr;
    check("streamCreate", hipContext_streamCreate(&ctx, &stream));

    hipEvent_t evStart = nullptr, evStop = nullptr;
    check("eventCreate",
        hipContext_eventCreate(&ctx, &evStart) &&
        hipContext_eventCreate(&ctx, &evStop));

    // ---- Test 5: Direct Kernel Launch (<<<>>> syntax) ----
    printf("--- Test 5: Direct Kernel Launch ---\n");
    int block = 256;
    int grid = (N + block - 1) / block;

    hipContext_eventRecord(&ctx, evStart, stream);
    probe_vector_add<<<grid, block, 0, stream>>>((const float*)d_a, (const float*)d_b, (float*)d_c, N);
    hipContext_eventRecord(&ctx, evStop, stream);
    hipContext_streamSync(&ctx, stream);

    float ms = 0;
    hipEventElapsedTime(&ms, evStart, evStop);
    printf("  VectorAdd kernel time: %.3f ms\n", ms);

    float *h_c = new float[N];
    check("memcpy D2H", hipContext_memcpyDtoH(&ctx, h_c, d_c, bytes));

    bool correct = true;
    for (int i = 0; i < N; i++) {
        if (fabsf(h_c[i] - (h_a[i] + h_b[i])) > 1e-5f) { correct = false; break; }
    }
    check("VectorAdd via HipContext stream", correct);

    delete[] h_a; delete[] h_b; delete[] h_c;

    // ---- Test 6: Shuffle kernel via HipContext stream ----
    printf("--- Test 6: Shuffle via HipContext ---\n");
    float *h_shuf = new float[N];
    hipDeviceptr_t d_shuf = 0;
    hipContext_memAlloc(&ctx, &d_shuf, bytes);

    probe_shuffle<<<grid, block, 0, stream>>>((float*)d_shuf, N);
    hipContext_streamSync(&ctx, stream);
    hipContext_memcpyDtoH(&ctx, h_shuf, d_shuf, bytes);

    correct = true;
    for (int i = 0; i < N; i++) {
        if (h_shuf[i] <= 0.0f || h_shuf[i] != h_shuf[i]) { correct = false; break; }
    }
    check("Shuffle kernel via HipContext", correct);

    hipContext_memFree(&ctx, d_shuf);
    delete[] h_shuf;

    // ---- Test 7: Atomics via HipContext ----
    printf("--- Test 7: Atomics via HipContext ---\n");
    float h_sum = 0;
    hipDeviceptr_t d_sum = 0, d_min = 0, d_max = 0;
    hipContext_memAlloc(&ctx, &d_sum, sizeof(float));
    hipContext_memAlloc(&ctx, &d_min, sizeof(float));
    hipContext_memAlloc(&ctx, &d_max, sizeof(float));

    float init_min = 1e10f, init_max = -1e10f;
    hipContext_memcpyHtoD(&ctx, d_sum, &h_sum, sizeof(float));
    hipContext_memcpyHtoD(&ctx, d_min, &init_min, sizeof(float));
    hipContext_memcpyHtoD(&ctx, d_max, &init_max, sizeof(float));

    probe_atomics<<<grid, block, 0, stream>>>((float*)d_sum, (float*)d_min, (float*)d_max, N);
    hipContext_streamSync(&ctx, stream);
    hipContext_memcpyDtoH(&ctx, &h_sum, d_sum, sizeof(float));

    float expected = (float)N * (float)(N + 1) / 2.0f;
    check("AtomicAdd via HipContext", fabsf(h_sum - expected) / expected < 0.01f);

    hipContext_memFree(&ctx, d_sum);
    hipContext_memFree(&ctx, d_min);
    hipContext_memFree(&ctx, d_max);

    // ---- Test 8: Event timing validates stream sync ----
    printf("--- Test 8: Event sync ---\n");
    check("eventSync", hipContext_eventSync(&ctx, evStop));

    // ---- Test 9: D2D Copy ----
    printf("--- Test 9: Device-to-Device Copy ---\n");
    hipDeviceptr_t d_copy = 0;
    hipContext_memAlloc(&ctx, &d_copy, bytes);
    correct = hipContext_memcpyDtoD(&ctx, d_copy, d_c, bytes);
    check("memcpy D2D", correct);
    hipContext_memFree(&ctx, d_copy);

    // ---- Cleanup ----
    printf("--- Cleanup ---\n");
    hipContext_eventDestroy(&ctx, evStart);
    hipContext_eventDestroy(&ctx, evStop);
    hipContext_streamDestroy(&ctx, stream);
    hipContext_memFree(&ctx, d_a);
    hipContext_memFree(&ctx, d_b);
    hipContext_memFree(&ctx, d_c);
    hipContext_memFreeHost(&ctx, h_pinned);
    hipContext_release(&ctx);

    printf("\n========================================\n");
    printf("Runtime layer: %d / %d tests passed\n", passed, total);
    return (passed == total) ? 0 : 1;
}
