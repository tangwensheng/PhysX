// HipKernelWrangler — DCU KernelWrangler: resolves GPU kernel functions
// from multiple HIP modules (one per .cu file).
#include "foundation/PxErrorCallback.h"
#include "foundation/PxString.h"
#include "CudaKernelWrangler.h"
#include "cudamanager/PxCudaContextManager.h"
#include "cudamanager/PxCudaContext.h"
#include <cstdio>

extern unsigned int g_DCU_ModuleCount;

using namespace physx;

const char* KernelWrangler::getCuFunctionName(uint16_t funcIndex) const
{
	return mKernelNames[funcIndex];
}

KernelWrangler::KernelWrangler(
	PxCudaContextManager& cudaContextManager,
	PxErrorCallback&       errorCallback,
	const char**           funcNames,
	uint16_t               numFuncs)
	: mError(false)
	, mKernelNames(NULL)
	, mCuFunctions("CuFunctions")
	, mCudaContextManager(cudaContextManager)
	, mCudaContext(cudaContextManager.getCudaContext())
	, mErrorCallback(errorCallback)
{
	PX_ASSERT(funcNames);
	mKernelNames = funcNames;

	CUmodule* cuModules = cudaContextManager.getCuModules();
	PxU32 nbModules = g_DCU_ModuleCount;
	if(nbModules == 0) nbModules = 1;
	mCuFunctions.resize(numFuncs, NULL);

	if(mCudaContextManager.tryAcquireContext())
	{
		for(uint32_t i = 0; i < numFuncs; ++i)
		{
			PxCUresult ret = CUDA_ERROR_NOT_READY;
			int foundModule = -1;
			for(PxU32 mod = 0; mod < nbModules; mod++)
			{
				ret = mCudaContext->moduleGetFunction(
					&mCuFunctions[i], cuModules[mod], funcNames[i]);
				if(ret == CUDA_SUCCESS) {
					foundModule = (int)mod;
					break;
				}
			}

			if(ret != CUDA_SUCCESS)
			{
				char buffer[256];
				Pxsnprintf(buffer, 256,
					"Could not find GPU function '%s' in %u modules.",
					funcNames[i], nbModules);
				mErrorCallback.reportError(
					PxErrorCode::eINTERNAL_ERROR, buffer, PX_FL);
				mError = true;
			}
		}
		mCudaContextManager.releaseContext();
	}
	else
	{
		mErrorCallback.reportError(PxErrorCode::eINTERNAL_ERROR,
			"Failed to acquire the GPU context.", PX_FL);
		mError = true;
	}
}
