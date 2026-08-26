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
//
// Copyright (c) 2008-2025 NVIDIA Corporation. All rights reserved.
// Copyright (c) 2004-2008 AGEIA Technologies, Inc. All rights reserved.
// Copyright (c) 2001-2004 NovodeX AG. All rights reserved.  

#include <assert.h>
#include "SparseRemove.cuh"

#include "cudaNpCommon.h"
#include "PxgPersistentContactManifold.h"
#include "PxsContactManagerState.h"
#include "PxgContactManager.h"
#include "PxgNpKernelIndices.h"

namespace physx
{
	class PxsContactManager;

	extern "C" __host__ void initNarrowphaseKernels7() {}
}

#if defined(PX_DCU_PORT) && PX_DCU_PORT
extern "C" __device__ __constant__ char gPxgDcuPairRemoveStageVersion[] = "PX_DCU_PAIR_REMOVE_STAGE_V55_SERIAL_COMPACT";

template<typename Manifold>
static __device__ void copyContactManagerSlotDcu(const PxgPairManagementData* PX_RESTRICT pairData,
	const PxU32 dstIndex, const PxU32 srcIndex, const bool copyManifold)
{
	pairData->mContactManagerInputData[dstIndex] = pairData->mContactManagerInputData[srcIndex];
	pairData->mContactManagerOutputData[dstIndex] = pairData->mContactManagerOutputData[srcIndex];
	pairData->mCpuContactManagerMapping[dstIndex] = pairData->mCpuContactManagerMapping[srcIndex];
	pairData->mShapeInteractions[dstIndex] = pairData->mShapeInteractions[srcIndex];
	pairData->mRestDistances[dstIndex] = pairData->mRestDistances[srcIndex];
	pairData->mTorsionalData[dstIndex] = pairData->mTorsionalData[srcIndex];

	if(copyManifold)
	{
		Manifold* manifolds = reinterpret_cast<Manifold*>(pairData->mPersistentContactManagers);
		float4* dst = reinterpret_cast<float4*>(&manifolds[dstIndex]);
		const float4* src = reinterpret_cast<const float4*>(&manifolds[srcIndex]);
		for(PxU32 i = 0; i < sizeof(Manifold) / sizeof(float4); ++i)
			dst[i] = src[i];
	}
}

template<typename Manifold>
static __device__ void removeContactManagersSerialDcu(const PxgPairManagementData* PX_RESTRICT pairData,
	const bool copyManifold)
{
	if(blockIdx.x != 0 || blockIdx.y != 0 || blockIdx.z != 0 ||
		threadIdx.x != 0 || threadIdx.y != 0 || threadIdx.z != 0)
		return;

	const PxU32 totalSize = pairData->mNbPairs;
	const PxU32 nbToRemove = pairData->mNbToRemove;
	if(nbToRemove == 0 || nbToRemove > totalSize)
		return;

	const PxU32* PX_RESTRICT removeIndices = pairData->mRemoveIndices;
	for(PxU32 i = 0; i < nbToRemove; ++i)
	{
		const PxU32 removedIndex = removeIndices[i];
		if(removedIndex >= totalSize || (i > 0 && removedIndex <= removeIndices[i - 1]))
			return;
	}

	const PxU32 newSize = totalSize - nbToRemove;
	PxU32 lowRemoveCount = 0;
	while(lowRemoveCount < nbToRemove && removeIndices[lowRemoveCount] < newSize)
		++lowRemoveCount;

	PxU32 tailRemoveIndex = lowRemoveCount;
	PxU32 srcIndex = newSize;
	for(PxU32 i = 0; i < lowRemoveCount; ++i)
	{
		while(tailRemoveIndex < nbToRemove && removeIndices[tailRemoveIndex] == srcIndex)
		{
			++tailRemoveIndex;
			++srcIndex;
		}

		if(srcIndex >= totalSize)
			return;

		copyContactManagerSlotDcu<Manifold>(pairData, removeIndices[i], srcIndex, copyManifold);
		++srcIndex;
	}
}
#endif

extern "C" __global__ void removeContactManagers_Stage1(const PxgPairManagementData* PX_RESTRICT pairData)
{
#if defined(PX_DCU_PORT) && PX_DCU_PORT
	PX_UNUSED(pairData);
	return;
#else
	initializeKeepDropBuffer(pairData->mTempAccumulator, pairData->mNbPairs, pairData->mNbToRemove);
#endif
}

extern "C" __global__ void removeContactManagers_Stage2(const PxgPairManagementData* PX_RESTRICT pairData)
{
#if defined(PX_DCU_PORT) && PX_DCU_PORT
	PX_UNUSED(pairData);
	return;
#else
	markKeepDropBuff(pairData->mRemoveIndices, pairData->mNbToRemove, pairData->mTempAccumulator, pairData->mNbPairs);
#endif
}

extern "C" __global__ void removeContactManagers_Stage3(const PxgPairManagementData* PX_RESTRICT pairData)
{
#if defined(PX_DCU_PORT) && PX_DCU_PORT
	PX_UNUSED(pairData);
	return;
#else
	processKeepDropBuff<PxgNarrowPhaseBlockDims::REMOVE_CONTACT_MANAGERS, PxgNarrowPhaseGridDims::REMOVE_CONTACT_MANAGERS>
		(pairData->mTempAccumulator, pairData->mNbPairs, pairData->mBlockSharedAccumulator);
#endif
}

extern "C" __global__ void removeContactManagers_Stage4(const PxgPairManagementData* PX_RESTRICT pairData)
{
#if defined(PX_DCU_PORT) && PX_DCU_PORT
	PX_UNUSED(pairData);
	return;
#else
	accumulateKeepDrop<PxgNarrowPhaseGridDims::REMOVE_CONTACT_MANAGERS>(pairData->mTempAccumulator, pairData->mNbPairs, pairData->mBlockSharedAccumulator);
#endif
}

extern "C" __global__ void removeContactManagers_Stage5(const PxgPairManagementData* PX_RESTRICT pairData, const bool copyManifold)
{
#if defined(PX_DCU_PORT) && PX_DCU_PORT
	removeContactManagersSerialDcu<PxgPersistentContactManifold>(pairData, copyManifold);
#else
	const PxU32* PX_RESTRICT globalRunSumBuffer = pairData->mTempAccumulator;
	const PxU32 totalSize = pairData->mNbPairs;
	PxU32 nbToSwap = getNbSwapsRequired(globalRunSumBuffer, totalSize, pairData->mNbToRemove);
	PxgContactManagerInput* inputData = pairData->mContactManagerInputData;
	PxsContactManagerOutput* outputData = pairData->mContactManagerOutputData;
	PxgPersistentContactManifold* manifolds = reinterpret_cast<PxgPersistentContactManifold*>(pairData->mPersistentContactManagers);
	PxsContactManager** cms = pairData->mCpuContactManagerMapping;
	Sc::ShapeInteraction** sis = pairData->mShapeInteractions;
	PxReal* rest = pairData->mRestDistances;
	PxsTorsionalFrictionData* torsional = pairData->mTorsionalData;

	for(PxU32 i = (threadIdx.x + WARP_SIZE * threadIdx.y + blockIdx.x * blockDim.x * blockDim.y)/16;
		i < nbToSwap; i+= ((blockDim.x * blockDim.y * gridDim.x)/16))
	{
		PxU32 srcIndex, dstIndex;
		getSwapIndices(globalRunSumBuffer, totalSize, i, nbToSwap, dstIndex, srcIndex);

		if((threadIdx.x&15) == 0)
		{
			inputData[dstIndex] = inputData[srcIndex];
			outputData[dstIndex] = outputData[srcIndex];
			cms[dstIndex] = cms[srcIndex];
			sis[dstIndex] = sis[srcIndex];
			rest[dstIndex] = rest[srcIndex];
			torsional[dstIndex] = torsional[srcIndex];
		}

		if (copyManifold)
		{
			float4* dst = (float4*)&manifolds[dstIndex];
			float4* src = (float4*)&manifolds[srcIndex];

			const PxU32 nbFloat4 = sizeof(PxgPersistentContactManifold)/sizeof(float4);

			for(PxU32 idx = threadIdx.x&15; idx < nbFloat4; idx += 16)			
				dst[idx] = src[idx];
		}
	}
#endif
}

extern "C" __global__ void removeContactManagers_Stage5_CvxTri(const PxgPairManagementData* PX_RESTRICT pairData, const bool copyManifold)
{
#if defined(PX_DCU_PORT) && PX_DCU_PORT
	removeContactManagersSerialDcu<PxgPersistentContactMultiManifold>(pairData, copyManifold);
#else
	const PxU32* PX_RESTRICT globalRunSumBuffer = pairData->mTempAccumulator;
	const PxU32 totalSize = pairData->mNbPairs;
	PxU32 nbToSwap = getNbSwapsRequired(globalRunSumBuffer, totalSize, pairData->mNbToRemove);
	PxgContactManagerInput* inputData = pairData->mContactManagerInputData;
	PxsContactManagerOutput* outputData = pairData->mContactManagerOutputData;
	PxgPersistentContactMultiManifold* manifolds = reinterpret_cast<PxgPersistentContactMultiManifold*>( pairData->mPersistentContactManagers);
	PxsContactManager** cms = pairData->mCpuContactManagerMapping;
	Sc::ShapeInteraction** sis = pairData->mShapeInteractions;
	PxReal* rest = pairData->mRestDistances;
	PxsTorsionalFrictionData* torsional = pairData->mTorsionalData;

	for (PxU32 i = (threadIdx.x + WARP_SIZE * threadIdx.y + blockIdx.x * blockDim.x * blockDim.y)/WARP_SIZE;
		i < nbToSwap; i += ((blockDim.x * blockDim.y * gridDim.x))/WARP_SIZE)
	{
		PxU32 srcIndex, dstIndex;
		getSwapIndices(globalRunSumBuffer, totalSize, i, nbToSwap, dstIndex, srcIndex);

		if(threadIdx.x == 0)
		{
			inputData[dstIndex] = inputData[srcIndex];
			outputData[dstIndex] = outputData[srcIndex];
			cms[dstIndex] = cms[srcIndex];
			sis[dstIndex] = sis[srcIndex];
			rest[dstIndex] = rest[srcIndex];
			torsional[dstIndex] = torsional[srcIndex];
		}

		if (copyManifold)
		{
			float4* dst = (float4*)&manifolds[dstIndex];
			float4* src = (float4*)&manifolds[srcIndex];

			for (PxU32 i = threadIdx.x; i < sizeof(PxgPersistentContactMultiManifold) / sizeof(float4); i += WARP_SIZE)
			{
				dst[i] = src[i];
			}
		}
	}
#endif
}

extern "C" __global__ void initializeManifolds(float4* destination, const float4* source, PxU32 dataSize, PxU32 nbTimesToReplicate)
{
	const PxU32 MaxStructureSize = 4096;

	__shared__ float sourceData[MaxStructureSize / sizeof(float)];

	//Number of threads required per element (rounded up to a warp size)
	const PxU32 nbThreadsPerElement = dataSize / sizeof(float4);
	//Number to process per block
	const PxU32 nbPerBlock = blockDim.x / nbThreadsPerElement;
	const PxU32 MaxThreadIdx = nbPerBlock*nbThreadsPerElement;

	const PxU32 nbBlocksRequired = (nbTimesToReplicate + nbPerBlock - 1) / nbPerBlock;

	if (threadIdx.x < nbThreadsPerElement)
	{
		const float4 value = source[threadIdx.x];
		sourceData[threadIdx.x] = value.x;
		sourceData[threadIdx.x + nbThreadsPerElement] = value.y;
		sourceData[threadIdx.x + 2*nbThreadsPerElement] = value.z;
		sourceData[threadIdx.x + 3*nbThreadsPerElement] = value.w;
	}

	const PxU32 threadReadIdx = threadIdx.x%nbThreadsPerElement;

	__syncthreads();

	for (PxU32 a = blockIdx.x; a < nbBlocksRequired; a += gridDim.x)
	{
		PxU32 startBlockIdx = nbPerBlock*a;
		float4* startPtr = destination + startBlockIdx*nbThreadsPerElement;

		PxU32 idx = startBlockIdx + threadIdx.x / nbThreadsPerElement;

		if (threadIdx.x < MaxThreadIdx && idx < nbTimesToReplicate)
		{
			startPtr[threadIdx.x] = make_float4(sourceData[threadReadIdx], sourceData[threadReadIdx + nbThreadsPerElement],
				sourceData[threadReadIdx + 2 * nbThreadsPerElement], sourceData[threadReadIdx + 3 * nbThreadsPerElement]);
		}
	}
}
