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

// HipKernelWrangler — HCU KernelWrangler: resolves GPU kernel functions
// from multiple HIP modules (one per .cu file).
#include "foundation/PxErrorCallback.h"
#include "foundation/PxString.h"
#include "CudaKernelWrangler.h"
#include "cudamanager/PxCudaContextManager.h"
#include "cudamanager/PxCudaContext.h"
#include <cstdio>

extern unsigned int g_HCU_ModuleCount;

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
	PxU32 nbModules = g_HCU_ModuleCount;
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
