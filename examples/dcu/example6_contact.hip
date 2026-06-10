// Example 6: Contact Reduction / Sparse Remove
// Tests: ballot+popc stream compaction (used by gpunarrowphase contact processing)
// Modules: gpucommon (contactReduction.cuh) + gpunarrowphase (contact output)

#include "PxgCommonDefines.h"
#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdlib>

using namespace physx;

// Contact pair: two body indices + distance
struct GpuContact { PxU32 bodyA; PxU32 bodyB; float dist; };

// CPU reference: filter contacts with dist < threshold
int filter_cpu(const GpuContact* in, int n, int* out_indices, float threshold)
{
    int count = 0;
    for (int i = 0; i < n; i++) {
        if (in[i].dist < threshold) out_indices[count++] = i;
    }
    return count;
}

// GPU kernel: ballot+popc to compact active contacts
extern "C" __global__ void gpu_filter_contacts(
    GpuContact* contacts, int n,
    PxU32* active_mask, float threshold)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int lane = tid & (WARP_SIZE - 1);

    // Each warp processes WARP_SIZE contacts
    int warp_start = (tid / WARP_SIZE) * WARP_SIZE;
    int idx = warp_start + lane;

    bool valid = (idx < n) && (contacts[idx].dist < threshold);

    // Ballot: which lanes in this warp have valid contacts
    unsigned long long mask = __ballot_sync(FULL_MASK, valid);

    // Popc: count active contacts in this warp
    int warp_count = __popcll(mask);

    // Prefix sum within warp for compaction offset
    int prefix = __popcll(mask & ((1ULL << lane) - 1));

    if (lane == 0 && warp_count > 0)
        atomicAdd(active_mask, warp_count);

    // Store prefix for compaction (later kernel uses this)
    // For validation, we just output the mask
    if (lane == 0)
        contacts[warp_start / WARP_SIZE].bodyA = (PxU32)mask;
}

int main()
{
    printf("========================================\n");
    printf(" Example 6: Contact Filter (narrowphase)\n");
    printf("========================================\n\n");

    const int N = 8192;
    GpuContact *h_in = new GpuContact[N];

    srand(77);
    int expected = 0;
    for (int i = 0; i < N; i++) {
        h_in[i].bodyA = rand() % 1000;
        h_in[i].bodyB = rand() % 1000;
        h_in[i].dist = (float)(rand() % 200) / 100.0f;  // 0.0 to 2.0
        if (h_in[i].dist < 0.5f) expected++;
    }

    // CPU reference
    int *cpu_indices = new int[N];
    int cpu_count = filter_cpu(h_in, N, cpu_indices, 0.5f);

    // GPU
    GpuContact *d_in;
    PxU32 *d_mask;
    hipMalloc(&d_in, N * sizeof(GpuContact));
    hipMalloc(&d_mask, sizeof(PxU32));
    hipMemcpy(d_in, h_in, N * sizeof(GpuContact), hipMemcpyHostToDevice);

    PxU32 zero = 0;
    hipMemcpy(d_mask, &zero, sizeof(PxU32), hipMemcpyHostToDevice);

    int blk = 256, grd = (N + blk - 1) / blk;
    gpu_filter_contacts<<<grd, blk>>>(d_in, N, d_mask, 0.5f);
    hipDeviceSynchronize();

    PxU32 gpu_count;
    hipMemcpy(&gpu_count, d_mask, sizeof(PxU32), hipMemcpyDeviceToHost);

    printf("Contact count (threshold<0.5):\n");
    printf("  CPU reference: %d\n", cpu_count);
    printf("  GPU result:    %u\n", gpu_count);
    printf("  Expected:      ~%d\n", expected);

    bool pass = (gpu_count == (PxU32)cpu_count) && (gpu_count > 0);
    printf("\n%s\n", pass ? "[PASS] GPU ballot+popc matches CPU" : "[FAIL]");

    delete[] h_in; delete[] cpu_indices;
    hipFree(d_in); hipFree(d_mask);
    return pass ? 0 : 1;
}
