// gpucommon module HIP integration test
// Uses real PhysX types from the full build system (no stubs)

#include "PxgCommonDefines.h"
#include "shuffle.cuh"
#include <cstdio>

extern "C" __global__ void test_warp_size(unsigned int* out)
{
    int tid = threadIdx.x;
    if (tid == 0) {
        out[0] = WARP_SIZE;
        out[1] = (unsigned int)(FULL_MASK & 0xFFFFFFFFULL);
        out[2] = (unsigned int)((FULL_MASK >> 32) & 0xFFFFFFFFULL);
    }
}

int main()
{
    unsigned int *d_info, h_info[3];
    hipMalloc(&d_info, 3 * sizeof(unsigned int));

    test_warp_size<<<1, 1>>>(d_info);
    hipDeviceSynchronize();
    hipMemcpy(h_info, d_info, 3 * sizeof(unsigned int), hipMemcpyDeviceToHost);
    hipFree(d_info);

    printf("=== PhysX gpucommon Integration Test ===\n");
    printf("WARP_SIZE: %u (expect 64)\n", h_info[0]);
    printf("FULL_MASK lo32: 0x%08x (expect 0xffffffff)\n", h_info[1]);
    printf("FULL_MASK hi32: 0x%08x (expect 0xffffffff)\n", h_info[2]);

    bool pass = (h_info[0] == 64) && (h_info[1] == 0xffffffff) && (h_info[2] == 0xffffffff);
    printf("%s\n", pass ? "gpucommon CMake build: ALL PASS" : "FAIL");
    return pass ? 0 : 1;
}
