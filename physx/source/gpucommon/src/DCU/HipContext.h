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

// Minimal HIP Runtime Context for PhysX HCU Port
//
// Provides device management, memory allocation, stream/event control,
// module loading, and kernel launching — the full GPU runtime surface
// needed by PhysX.

#ifndef PXG_HIP_CONTEXT_H
#define PXG_HIP_CONTEXT_H

#include <hip/hip_runtime.h>

struct HipContext {
    hipDevice_t     device;
    hipCtx_t        context;
    int             deviceOrdinal;
    char            deviceName[256];
    int             computeUnits;
    int             maxThreadsPerBlock;
    size_t          sharedMemPerBlock;
    size_t          totalGlobalMem;
    int             warpSize;

    bool            isValid;
};

// ---- Lifecycle ----
bool hipContext_init(HipContext* ctx, int deviceOrdinal);
void hipContext_release(HipContext* ctx);

// ---- Memory ----
bool hipContext_memAlloc(HipContext* ctx, hipDeviceptr_t* dptr, size_t bytes);
bool hipContext_memFree(HipContext* ctx, hipDeviceptr_t dptr);
bool hipContext_memHostAlloc(HipContext* ctx, void** ptr, size_t bytes, unsigned int flags);
bool hipContext_memFreeHost(HipContext* ctx, void* ptr);

// ---- Copy ----
bool hipContext_memcpyHtoD(HipContext* ctx, hipDeviceptr_t dst, const void* src, size_t bytes);
bool hipContext_memcpyDtoH(HipContext* ctx, void* dst, hipDeviceptr_t src, size_t bytes);
bool hipContext_memcpyDtoD(HipContext* ctx, hipDeviceptr_t dst, hipDeviceptr_t src, size_t bytes);

// ---- Stream ----
bool hipContext_streamCreate(HipContext* ctx, hipStream_t* stream);
bool hipContext_streamDestroy(HipContext* ctx, hipStream_t stream);
bool hipContext_streamSync(HipContext* ctx, hipStream_t stream);

// ---- Event ----
bool hipContext_eventCreate(HipContext* ctx, hipEvent_t* event);
bool hipContext_eventRecord(HipContext* ctx, hipEvent_t event, hipStream_t stream);
bool hipContext_eventSync(HipContext* ctx, hipEvent_t event);
bool hipContext_eventDestroy(HipContext* ctx, hipEvent_t event);

// ---- Module ----
bool hipContext_moduleLoadData(HipContext* ctx, hipModule_t* module, const void* image);
bool hipContext_moduleGetFunc(HipContext* ctx, hipFunction_t* func, hipModule_t module, const char* name);
bool hipContext_moduleUnload(HipContext* ctx, hipModule_t module);

// ---- Kernel Launch ----
bool hipContext_launchKernel(HipContext* ctx,
    hipFunction_t func,
    unsigned int gridX, unsigned int gridY, unsigned int gridZ,
    unsigned int blockX, unsigned int blockY, unsigned int blockZ,
    unsigned int sharedMemBytes,
    hipStream_t stream,
    void** kernelParams,
    void** extra);

#endif
