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

// HIP GPU Dispatcher — replaces PxGpu.cpp + PxPhysXGpuModuleLoader.cpp for HCU
// Provides the GPU work dispatch layer that PxScene::simulate() needs.
// All functions are compiled directly into the executable (no dlopen).
//
// On HCU, kernel function resolution is handled by the KernelWrangler
// (HipKernelWrangler.cpp) which directly calls hipModuleGetFunction for each
// kernel name — bypassing the static PxKernelIndex function table entirely.
// The stubs below exist only to satisfy link-time dependencies from code that
// references PxGpuGetCudaFunctionTable / PxGpuCudaRegisterFunction.

#include "foundation/PxPreprocessor.h"
#if PX_SUPPORT_GPU_PHYSX

#include "gpu/PxGpu.h"
#include "cudamanager/PxCudaContextManager.h"
#include "PxPhysXConfig.h"
#include <cstdio>

namespace physx {

// ---- GPU module loading (replaces PxPhysXGpuModuleLoader.cpp) ----
// On HCU, all GPU functions are compiled in — no .so to dlopen.

void PxLoadPhysxGPUModule(const char*) {
	// HCU: GPU functions are compiled-in, no separate module to load
}

void PxUnloadPhysxGPUModule() {}

// ---- Function pointer table ----
// These function pointers are used by the PhysX GPU dispatch code.
// On HCU, they point to our locally-compiled implementations.

typedef PxCudaContextManager* (*PxCreateCudaContextManager_FUNC)(PxFoundation&, const PxCudaContextManagerDesc&, PxProfilerCallback*, bool);
extern "C" PxCudaContextManager* PxCreateCudaContextManager(PxFoundation&, const PxCudaContextManagerDesc&, PxProfilerCallback*, bool);

PxCreateCudaContextManager_FUNC g_PxCudaContextManagerFunc = PxCreateCudaContextManager;

// ---- Kernel function table stubs ----
// The HCU KernelWrangler (HipKernelWrangler.cpp) resolves functions directly
// via hipModuleGetFunction and does NOT use this table.  These stubs exist to
// satisfy link-time references from code compiled from the shared PhysX GPU
// source files.

PX_PHYSX_CORE_API PxU32 PxGpuGetCudaFunctionTableSize() { return 0; }
PX_PHYSX_CORE_API PxKernelIndex* PxGpuGetCudaFunctionTable() { return nullptr; }

// PxGpuCudaRegisterFunction is called by __hipRegisterFunction (in
// gpu_module_table.cpp inside libPhysXGpuHCU.so) during static init.
// On HCU we track registrations but the KernelWrangler bypasses this table.
static const PxU32 MAX_FUNCTIONS = 1024;
static PxKernelIndex s_FunctionTable[MAX_FUNCTIONS];
static PxU32 s_numFunctions = 0;

void PxGpuCudaRegisterFunction(int moduleIndex, const char* functionName) {
	if (s_numFunctions < MAX_FUNCTIONS) {
		s_FunctionTable[s_numFunctions].moduleIndex = (PxU32)moduleIndex;
		s_FunctionTable[s_numFunctions].functionName = functionName;
		s_numFunctions++;
	}
}

} // namespace physx

#endif // PX_SUPPORT_GPU_PHYSX
