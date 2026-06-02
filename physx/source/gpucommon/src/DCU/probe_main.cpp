// DCU Toolchain Probe — Host-side launcher
//
// Build (standalone):
//   hipcc probe_kernel.hip probe_main.cpp -o probe_dcu --offload-arch=gfx936
//
// Build (via CMake):
//   mkdir build && cd build
//   cmake .. -DCMAKE_HIP_COMPILER=hipcc
//   make
//
// Run:
//   ./probe_dcu
//
// Verifies: hipcc → device code gen → hiprt → kernel launch → memory transfer

#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstring>

// Match the kernels declared in probe_kernel.hip
extern "C" __global__ void probe_vector_add(const float* a, const float* b, float* c, int n);
extern "C" __global__ void probe_shuffle(float* out, int n);
extern "C" __global__ void probe_ballot_popc(unsigned int* out, int n);
extern "C" __global__ void probe_atomics(float* sumOut, float* minOut, float* maxOut, int n);
extern "C" __global__ void probe_shared_sync(float* out, int n);
extern "C" __global__ void probe_launch_bounds(float* out, int n);

#define CHECK_HIP(call) do { \
    hipError_t err = call; \
    if (err != hipSuccess) { \
        fprintf(stderr, "HIP ERROR at %s:%d — %s\n", __FILE__, __LINE__, \
                hipGetErrorString(err)); \
        exit(1); \
    } \
} while(0)

static int g_totalTests = 0;
static int g_passedTests = 0;

void test_result(const char* name, bool passed)
{
    g_totalTests++;
    if (passed) {
        g_passedTests++;
        printf("  [PASS] %s\n", name);
    } else {
        printf("  [FAIL] %s\n", name);
    }
}

int main()
{
    printf("=== DCU Toolchain Probe ===\n\n");

    // Query device
    int deviceCount = 0;
    CHECK_HIP(hipGetDeviceCount(&deviceCount));
    printf("Device count: %d\n", deviceCount);

    hipDeviceProp_t props;
    CHECK_HIP(hipGetDeviceProperties(&props, 0));
    printf("Device 0: %s\n", props.name);
    printf("  Compute: gfx%d\n", props.gcnArch);
    printf("  TotalMem: %.1f GB\n", props.totalGlobalMem / (1024.0*1024.0*1024.0));
    printf("  Max threads/block: %d\n", props.maxThreadsPerBlock);
    printf("  Shared mem/block: %zu KB\n", props.sharedMemPerBlock / 1024);
    printf("  Multi-processors: %d\n", props.multiProcessorCount);
    printf("  Warp (wavefront) size: %d\n", props.warpSize);
    printf("\n");

    // Verify wavefront size = 64
    test_result("Wavefront size == 64", props.warpSize == 64);
    printf("\n");

    const int N = 1024 * 1024;  // 1M elements
    const int blockSize = 256;
    const int gridSize = (N + blockSize - 1) / blockSize;

    // ---- Test 1: Vector Add ----
    {
        printf("--- Test 1: Vector Add ---\n");
        float *h_a = new float[N];
        float *h_b = new float[N];
        float *h_c = new float[N];
        for (int i = 0; i < N; i++) { h_a[i] = (float)i; h_b[i] = (float)(i * 2); }

        float *d_a, *d_b, *d_c;
        CHECK_HIP(hipMalloc(&d_a, N * sizeof(float)));
        CHECK_HIP(hipMalloc(&d_b, N * sizeof(float)));
        CHECK_HIP(hipMalloc(&d_c, N * sizeof(float)));
        CHECK_HIP(hipMemcpy(d_a, h_a, N * sizeof(float), hipMemcpyHostToDevice));
        CHECK_HIP(hipMemcpy(d_b, h_b, N * sizeof(float), hipMemcpyHostToDevice));

        probe_vector_add<<<gridSize, blockSize>>>(d_a, d_b, d_c, N);
        CHECK_HIP(hipGetLastError());
        CHECK_HIP(hipDeviceSynchronize());

        CHECK_HIP(hipMemcpy(h_c, d_c, N * sizeof(float), hipMemcpyDeviceToHost));

        bool ok = true;
        for (int i = 0; i < N; i++) {
            if (fabsf(h_c[i] - (h_a[i] + h_b[i])) > 1e-5f) { ok = false; break; }
        }
        test_result("Vector add", ok);

        CHECK_HIP(hipFree(d_a)); CHECK_HIP(hipFree(d_b)); CHECK_HIP(hipFree(d_c));
        delete[] h_a; delete[] h_b; delete[] h_c;
    }

    // ---- Test 2: Warp Shuffle ----
    {
        printf("--- Test 2: Warp Shuffle ---\n");
        float *h_out = new float[N];
        float *d_out;
        CHECK_HIP(hipMalloc(&d_out, N * sizeof(float)));

        probe_shuffle<<<gridSize, blockSize>>>(d_out, N);
        CHECK_HIP(hipGetLastError());
        CHECK_HIP(hipDeviceSynchronize());
        CHECK_HIP(hipMemcpy(h_out, d_out, N * sizeof(float), hipMemcpyDeviceToHost));

        bool ok = true;
        for (int i = 0; i < N; i++) {
            if (h_out[i] <= 0.0f || (h_out[i] != h_out[i])) { ok = false; break; }
        }
        test_result("Warp shuffle (shfl_sync + shfl_xor_sync)", ok);

        CHECK_HIP(hipFree(d_out));
        delete[] h_out;
    }

    // ---- Test 3: Ballot + Popc ----
    {
        printf("--- Test 3: Ballot + Popc ---\n");
        unsigned int *h_out = new unsigned int[N], *d_out;
        CHECK_HIP(hipMalloc(&d_out, N * sizeof(unsigned int)));

        probe_ballot_popc<<<gridSize, blockSize>>>(d_out, N);
        CHECK_HIP(hipGetLastError());
        CHECK_HIP(hipDeviceSynchronize());
        CHECK_HIP(hipMemcpy(h_out, d_out, N * sizeof(unsigned int), hipMemcpyDeviceToHost));

        bool ok = true;
        for (int i = 0; i < N; i++) {
            unsigned int count = h_out[i] & 0xFFFF;
            // With wavefront=64, even lanes = 32 threads -> popc should be 32
            if (count != 32) { ok = false; break; }
        }
        test_result("Ballot + popcll (64-bit mask, count=32)", ok);

        CHECK_HIP(hipFree(d_out));
        delete[] h_out;
    }

    // ---- Test 4: Atomics ----
    {
        printf("--- Test 4: Atomics ---\n");
        float h_sum = 0.0f, h_min = 1e10f, h_max = -1e10f;
        float *d_sum, *d_min, *d_max;
        CHECK_HIP(hipMalloc(&d_sum, sizeof(float)));
        CHECK_HIP(hipMalloc(&d_min, sizeof(float)));
        CHECK_HIP(hipMalloc(&d_max, sizeof(float)));
        CHECK_HIP(hipMemcpy(d_sum, &h_sum, sizeof(float), hipMemcpyHostToDevice));
        CHECK_HIP(hipMemcpy(d_min, &h_min, sizeof(float), hipMemcpyHostToDevice));
        CHECK_HIP(hipMemcpy(d_max, &h_max, sizeof(float), hipMemcpyHostToDevice));

        probe_atomics<<<gridSize, blockSize>>>(d_sum, d_min, d_max, N);
        CHECK_HIP(hipGetLastError());
        CHECK_HIP(hipDeviceSynchronize());
        CHECK_HIP(hipMemcpy(&h_sum, d_sum, sizeof(float), hipMemcpyDeviceToHost));
        CHECK_HIP(hipMemcpy(&h_min, d_min, sizeof(float), hipMemcpyDeviceToHost));
        CHECK_HIP(hipMemcpy(&h_max, d_max, sizeof(float), hipMemcpyDeviceToHost));

        // sum = 1 + 2 + ... + N = N*(N+1)/2
        float expected = (float)N * (float)(N + 1) / 2.0f;
        test_result("Atomic add float", fabsf(h_sum - expected) / expected < 0.01f);

        CHECK_HIP(hipFree(d_sum)); CHECK_HIP(hipFree(d_min)); CHECK_HIP(hipFree(d_max));
    }

    // ---- Test 5: Shared memory + sync ----
    {
        printf("--- Test 5: Shared memory + sync ---\n");
        float *h_out = new float[N], *d_out;
        CHECK_HIP(hipMalloc(&d_out, N * sizeof(float)));

        size_t smemSize = blockSize * sizeof(float);
        probe_shared_sync<<<gridSize, blockSize, smemSize>>>(d_out, N);
        CHECK_HIP(hipGetLastError());
        CHECK_HIP(hipDeviceSynchronize());
        CHECK_HIP(hipMemcpy(h_out, d_out, N * sizeof(float), hipMemcpyDeviceToHost));

        bool ok = true;
        for (int i = 0; i < N; i++) {
            int lane = i & (blockSize - 1);
            float expected = (float)((lane + 1) % blockSize);
            if (fabsf(h_out[i] - expected) > 1e-5f) { ok = false; break; }
        }
        test_result("Shared memory + syncthreads + syncwarp", ok);

        CHECK_HIP(hipFree(d_out));
        delete[] h_out;
    }

    // ---- Test 6: __launch_bounds__ ----
    {
        printf("--- Test 6: launch_bounds ---\n");
        float *h_out = new float[N], *d_out;
        CHECK_HIP(hipMalloc(&d_out, N * sizeof(float)));

        probe_launch_bounds<<<gridSize, blockSize>>>(d_out, N);
        CHECK_HIP(hipGetLastError());
        CHECK_HIP(hipDeviceSynchronize());
        CHECK_HIP(hipMemcpy(h_out, d_out, N * sizeof(float), hipMemcpyDeviceToHost));

        bool ok = true;
        for (int i = 0; i < N; i++) {
            if ((int)h_out[i] != i) { ok = false; break; }
        }
        test_result("launch_bounds kernel", ok);

        CHECK_HIP(hipFree(d_out));
        delete[] h_out;
    }

    // ---- Summary ----
    printf("\n========================================\n");
    printf("Results: %d / %d tests passed\n", g_passedTests, g_totalTests);

    if (g_passedTests == g_totalTests) {
        printf("DCU toolchain probe: ALL TESTS PASSED\n");
        return 0;
    } else {
        printf("DCU toolchain probe: %d TEST(S) FAILED\n", g_totalTests - g_passedTests);
        return 1;
    }
}
