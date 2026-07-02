// HIP GPU Dispatcher — replaces PxGpu.cpp + PxPhysXGpuModuleLoader.cpp for DCU
// Provides the GPU work dispatch layer that PxScene::simulate() needs.
// All functions are compiled directly into the executable (no dlopen).
//
// On DCU, kernel function resolution is handled by the KernelWrangler
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
// On DCU, all GPU functions are compiled in — no .so to dlopen.

void PxLoadPhysxGPUModule(const char*) {
	// DCU: GPU functions are compiled-in, no separate module to load
}

void PxUnloadPhysxGPUModule() {}

// ---- Function pointer table ----
// These function pointers are used by the PhysX GPU dispatch code.
// On DCU, they point to our locally-compiled implementations.

typedef PxCudaContextManager* (*PxCreateCudaContextManager_FUNC)(PxFoundation&, const PxCudaContextManagerDesc&, PxProfilerCallback*, bool);
extern "C" PxCudaContextManager* PxCreateCudaContextManager(PxFoundation&, const PxCudaContextManagerDesc&, PxProfilerCallback*, bool);

PxCreateCudaContextManager_FUNC g_PxCudaContextManagerFunc = PxCreateCudaContextManager;

// ---- Kernel function table stubs ----
// The DCU KernelWrangler (HipKernelWrangler.cpp) resolves functions directly
// via hipModuleGetFunction and does NOT use this table.  These stubs exist to
// satisfy link-time references from code compiled from the shared PhysX GPU
// source files.

PX_PHYSX_CORE_API PxU32 PxGpuGetCudaFunctionTableSize() { return 0; }
PX_PHYSX_CORE_API PxKernelIndex* PxGpuGetCudaFunctionTable() { return nullptr; }

// PxGpuCudaRegisterFunction is called by __hipRegisterFunction (in
// gpu_module_table.cpp inside libPhysXGpuDCU.so) during static init.
// On DCU we track registrations but the KernelWrangler bypasses this table.
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
