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
