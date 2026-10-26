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
// Copyright (c) 2026 Hygon Information Technology Co., Ltd.

#include "RadixSort.cuh"
#include "PxgRadixSortDesc.h"
#include "PxgRadixSortKernelIndices.h"
#include "stdio.h"
#include "PxNodeIndex.h"

#if defined(PX_DCU_PORT) && PX_DCU_PORT && defined(PX_DCU_VOLUME_RADIX_O3_P38) && PX_DCU_VOLUME_RADIX_O3_P38
extern "C" __device__ __constant__ char gPxgDcuVolumeRadixO3P38Version[] = "PX_DCU_RADIX_O3_P38_GFX936_DEVICE_V1";
#endif

extern "C" __host__ void initCommonKernels1() {}

#if defined(PX_DCU_PORT) && PX_DCU_PORT
extern "C" __device__ __constant__ char gPxgDcuBroadphaseRadixStageVersion[] = "PX_DCU_GPU_BROADPHASE_RADIX_V3_SERIAL_COUNTING_PASS";
extern "C" __device__ __constant__ char gPxgDcuFemClothRadixStageVersion[] = "PX_DCU_FEM_CLOTH_RADIX_V2_SERIAL_COUNTING_1024";
extern "C" __device__ __constant__ char gPxgDcuPbdRadixStageVersion[] = "PX_DCU_PBD_RADIX_NO_COUNT_V1_SERIAL_STABLE_4BIT";
extern "C" __device__ __constant__ char gPxgDcuPbdPrimitiveRadixStageVersion[] = "PX_DCU_PBD_PRIMITIVE_RADIX_V4_SERIAL_COUNTING_4096";
extern "C" __device__ __constant__ char gPxgDcuPbdPrimitiveRadixChunkedStageVersion[] = "PX_DCU_PBD_PRIMITIVE_RADIX_V5_CHUNKED_COUNTING_32";
extern "C" __device__ __constant__ char gPxgDcuPbdPrimitiveRadixInlineStageVersion[] = "PX_DCU_PBD_PRIMITIVE_RADIX_V7_INLINE_RADIX_32";
extern "C" __device__ __constant__ char gPxgDcuPbdPrimitiveRadixFused64StageVersion[] = "PX_DCU_PBD_PRIMITIVE_RADIX_V63_FUSED_64_INLINE";
extern "C" __device__ __constant__ char gPxgDcuPbdSpatialRadixChunkedStageVersion[] = "PX_DCU_PBD_SPATIAL_RADIX_V6_CHUNKED_COUNTING_32";
#endif


//gNumOfKeys is device ptr
extern "C" __global__ 
__launch_bounds__(PxgRadixSortKernelBlockDim::RADIX_SORT, 1)
void radixSortMultiBlockLaunch(PxgRadixSortBlockDesc* desc, const PxU32 gStartBit)
{
	const uint4* PX_RESTRICT gInputKeys = reinterpret_cast<uint4*>(desc[blockIdx.y].inputKeys);
	const uint4* PX_RESTRICT gInputRanks = reinterpret_cast<uint4*>(desc[blockIdx.y].inputRanks);
	PxU32* gRadixCount = desc[blockIdx.y].radixBlockCounts;

	const PxU32 numKeys = *desc[blockIdx.y].numKeys;

	radixSortSingleBlock<PxgRadixSortKernelBlockDim::RADIX_SORT / WARP_SIZE>(gInputKeys, gInputRanks, numKeys, gStartBit, gRadixCount);
}

//gNumOfKeys is device ptr
extern "C" __global__ 
__launch_bounds__(PxgRadixSortKernelBlockDim::RADIX_SORT, 1)
void radixSortMultiCalculateRanksLaunch(PxgRadixSortBlockDesc* desc, const PxU32 gStartBit)
{
	const uint4* PX_RESTRICT gInputKeys = reinterpret_cast<const uint4*>(desc[blockIdx.y].inputKeys);
	const uint4* PX_RESTRICT gInputRanks = reinterpret_cast<const uint4*>(desc[blockIdx.y].inputRanks);
	PxU32* gOutputKeys = desc[blockIdx.y].outputKeys;
	PxU32* gOutputRanks = desc[blockIdx.y].outputRanks;

	PxU32* gRadixCount = desc[blockIdx.y].radixBlockCounts;

	const PxU32 numKeys = *desc[blockIdx.y].numKeys;

#if defined(PX_DCU_PORT) && PX_DCU_PORT
	// The gfx936 multiblock rank gather is unstable once FEM cloth plane
	// contacts exceed the old 128-key fallback. Keep the validated matrix range
	// on a stable per-pass counting sort while preserving descriptor ping-pong.
	if(numKeys <= 1024)
	{
		if(blockIdx.x == 0 && threadIdx.x == 0)
		{
			const PxU32* PX_RESTRICT inputKeys = reinterpret_cast<const PxU32*>(gInputKeys);
			const PxU32* PX_RESTRICT inputRanks = reinterpret_cast<const PxU32*>(gInputRanks);
			PxU32 offsets[RADIX_SIZE];

			for(PxU32 radix = 0; radix < RADIX_SIZE; ++radix)
				offsets[radix] = 0;

			for(PxU32 i = 0; i < numKeys; ++i)
				++offsets[getRadix(inputKeys[i], gStartBit)];

			PxU32 runningOffset = 0;
			for(PxU32 radix = 0; radix < RADIX_SIZE; ++radix)
			{
				const PxU32 radixCount = offsets[radix];
				offsets[radix] = runningOffset;
				runningOffset += radixCount;
			}

			for(PxU32 i = 0; i < numKeys; ++i)
			{
				const PxU32 radix = getRadix(inputKeys[i], gStartBit);
				const PxU32 outputIndex = offsets[radix]++;
				gOutputKeys[outputIndex] = inputKeys[i];
				gOutputRanks[outputIndex] = inputRanks[i];
			}
		}
		return;
	}
#endif

	radixSortCalculateRanks<PxgRadixSortKernelBlockDim::RADIX_SORT / WARP_SIZE>(gInputKeys, gInputRanks, numKeys, gStartBit, gRadixCount, gOutputKeys, gOutputRanks);
}

#if defined(PX_DCU_PORT) && PX_DCU_PORT
extern "C" __global__
__launch_bounds__(1, 1)
void radixSortPbdPrimitiveSerialCountingPassDcuV4(PxgRadixSortBlockDesc* desc, const PxU32 gStartBit)
{
	const PxgRadixSortBlockDesc& radixDesc = desc[blockIdx.y];
	const PxU32 numKeys = *radixDesc.numKeys;
	const PxU32* PX_RESTRICT inputKeys = radixDesc.inputKeys;
	const PxU32* PX_RESTRICT inputRanks = radixDesc.inputRanks;
	PxU32* PX_RESTRICT outputKeys = radixDesc.outputKeys;
	PxU32* PX_RESTRICT outputRanks = radixDesc.outputRanks;
	PxU32 offsets[RADIX_SIZE];

	for(PxU32 radix = 0; radix < RADIX_SIZE; ++radix)
		offsets[radix] = 0;

	for(PxU32 i = 0; i < numKeys; ++i)
		++offsets[getRadix(inputKeys[i], gStartBit)];

	PxU32 runningOffset = 0;
	for(PxU32 radix = 0; radix < RADIX_SIZE; ++radix)
	{
		const PxU32 radixCount = offsets[radix];
		offsets[radix] = runningOffset;
		runningOffset += radixCount;
	}

	for(PxU32 i = 0; i < numKeys; ++i)
	{
		const PxU32 radix = getRadix(inputKeys[i], gStartBit);
		const PxU32 outputIndex = offsets[radix]++;
		outputKeys[outputIndex] = inputKeys[i];
		outputRanks[outputIndex] = inputRanks[i];
	}
}

extern "C" __global__
__launch_bounds__(32, 1)
void radixSortPbdPrimitiveChunkedCounting32DcuV5(PxgRadixSortBlockDesc* desc0, PxgRadixSortBlockDesc* desc1)
{
	const PxU32 lane = threadIdx.x;
	const PxU32 descriptorIndex = blockIdx.y;
	const PxU32 numKeys = *desc0[descriptorIndex].numKeys;
	const PxU32 chunkSize = (numKeys + 31u) / 32u;
	const PxU32 chunkBegin = lane * chunkSize;
	const PxU32 chunkEnd = PxMin(chunkBegin + chunkSize, numKeys);

	__shared__ PxU32 chunkOffsets[RADIX_SIZE * 32];
	__shared__ PxU32 radixBases[RADIX_SIZE];

	for(PxU32 pass = 0; pass < 8; ++pass)
	{
		PxgRadixSortBlockDesc& radixDesc = (pass & 1u) ? desc1[descriptorIndex] : desc0[descriptorIndex];
		const PxU32* PX_RESTRICT inputKeys = radixDesc.inputKeys;
		const PxU32* PX_RESTRICT inputRanks = radixDesc.inputRanks;
		PxU32* PX_RESTRICT outputKeys = radixDesc.outputKeys;
		PxU32* PX_RESTRICT outputRanks = radixDesc.outputRanks;
		const PxU32 startBit = pass * 4u;

		for(PxU32 radix = 0; radix < RADIX_SIZE; ++radix)
			chunkOffsets[radix * 32u + lane] = 0;
		__syncthreads();

		for(PxU32 i = chunkBegin; i < chunkEnd; ++i)
			++chunkOffsets[getRadix(inputKeys[i], startBit) * 32u + lane];
		__syncthreads();

		if(lane < RADIX_SIZE)
		{
			PxU32 radixCount = 0;
			for(PxU32 chunk = 0; chunk < 32; ++chunk)
				radixCount += chunkOffsets[lane * 32u + chunk];
			radixBases[lane] = radixCount;
		}
		__syncthreads();

		if(lane == 0)
		{
			PxU32 radixBase = 0;
			for(PxU32 radix = 0; radix < RADIX_SIZE; ++radix)
			{
				const PxU32 radixCount = radixBases[radix];
				radixBases[radix] = radixBase;
				radixBase += radixCount;
			}
		}
		__syncthreads();

		if(lane < RADIX_SIZE)
		{
			PxU32 outputOffset = radixBases[lane];
			for(PxU32 chunk = 0; chunk < 32; ++chunk)
			{
				const PxU32 chunkCount = chunkOffsets[lane * 32u + chunk];
				chunkOffsets[lane * 32u + chunk] = outputOffset;
				outputOffset += chunkCount;
			}
		}
		__syncthreads();

		for(PxU32 i = chunkBegin; i < chunkEnd; ++i)
		{
			const PxU32 radix = getRadix(inputKeys[i], startBit);
			const PxU32 outputIndex = chunkOffsets[radix * 32u + lane]++;
			outputKeys[outputIndex] = inputKeys[i];
			outputRanks[outputIndex] = inputRanks[i];
		}
		__syncthreads();
	}
}

PX_FORCE_INLINE static __device__ PxU32 getPbdPrimitiveRadixInlineDcuV7(PxU32 input, PxU32 startBit)
{
	return (input >> startBit) & 0xFu;
}

extern "C" __global__
__launch_bounds__(32, 1)
void radixSortPbdPrimitiveChunkedCounting32InlineDcuV7(PxgRadixSortBlockDesc* desc0, PxgRadixSortBlockDesc* desc1)
{
	const PxU32 lane = threadIdx.x;
	const PxU32 descriptorIndex = blockIdx.y;
	const PxU32 numKeys = *desc0[descriptorIndex].numKeys;
	const PxU32 chunkSize = (numKeys + 31u) / 32u;
	const PxU32 chunkBegin = lane * chunkSize;
	const PxU32 chunkEnd = PxMin(chunkBegin + chunkSize, numKeys);

	__shared__ PxU32 chunkOffsets[RADIX_SIZE * 32];
	__shared__ PxU32 radixBases[RADIX_SIZE];

	for(PxU32 pass = 0; pass < 8; ++pass)
	{
		PxgRadixSortBlockDesc& radixDesc = (pass & 1u) ? desc1[descriptorIndex] : desc0[descriptorIndex];
		const PxU32* PX_RESTRICT inputKeys = radixDesc.inputKeys;
		const PxU32* PX_RESTRICT inputRanks = radixDesc.inputRanks;
		PxU32* PX_RESTRICT outputKeys = radixDesc.outputKeys;
		PxU32* PX_RESTRICT outputRanks = radixDesc.outputRanks;
		const PxU32 startBit = pass * 4u;

		for(PxU32 radix = 0; radix < RADIX_SIZE; ++radix)
			chunkOffsets[radix * 32u + lane] = 0;
		__syncthreads();

		for(PxU32 i = chunkBegin; i < chunkEnd; ++i)
			++chunkOffsets[getPbdPrimitiveRadixInlineDcuV7(inputKeys[i], startBit) * 32u + lane];
		__syncthreads();

		if(lane < RADIX_SIZE)
		{
			PxU32 radixCount = 0;
			for(PxU32 chunk = 0; chunk < 32; ++chunk)
				radixCount += chunkOffsets[lane * 32u + chunk];
			radixBases[lane] = radixCount;
		}
		__syncthreads();

		if(lane == 0)
		{
			PxU32 radixBase = 0;
			for(PxU32 radix = 0; radix < RADIX_SIZE; ++radix)
			{
				const PxU32 radixCount = radixBases[radix];
				radixBases[radix] = radixBase;
				radixBase += radixCount;
			}
		}
		__syncthreads();

		if(lane < RADIX_SIZE)
		{
			PxU32 outputOffset = radixBases[lane];
			for(PxU32 chunk = 0; chunk < 32; ++chunk)
			{
				const PxU32 chunkCount = chunkOffsets[lane * 32u + chunk];
				chunkOffsets[lane * 32u + chunk] = outputOffset;
				outputOffset += chunkCount;
			}
		}
		__syncthreads();

		for(PxU32 i = chunkBegin; i < chunkEnd; ++i)
		{
			const PxU32 radix = getPbdPrimitiveRadixInlineDcuV7(inputKeys[i], startBit);
			const PxU32 outputIndex = chunkOffsets[radix * 32u + lane]++;
			outputKeys[outputIndex] = inputKeys[i];
			outputRanks[outputIndex] = inputRanks[i];
		}
		__syncthreads();
	}
}

extern "C" __global__
__launch_bounds__(32, 1)
void radixSortPbdPrimitiveChunkedCounting64InlineDcuV63(PxgRadixSortBlockDesc* desc0,
	PxgRadixSortBlockDesc* desc1, const PxU64* originalKeys0, const PxU64* originalKeys1)
{
	const PxU32 lane = threadIdx.x;
	const PxU32 descriptorIndex = blockIdx.y;
	const PxU32 numKeys = *desc0[descriptorIndex].numKeys;
	const PxU32 chunkSize = (numKeys + 31u) / 32u;
	const PxU32 chunkBegin = lane * chunkSize;
	const PxU32 chunkEnd = PxMin(chunkBegin + chunkSize, numKeys);
	const PxU64* PX_RESTRICT originalKeys = descriptorIndex == 0 ? originalKeys0 : originalKeys1;

	__shared__ PxU32 chunkOffsets[RADIX_SIZE * 32];
	__shared__ PxU32 radixBases[RADIX_SIZE];

	for(PxU32 pass = 0; pass < 16; ++pass)
	{
		if(pass == 8)
		{
			PxU32* PX_RESTRICT highKeys = desc0[descriptorIndex].inputKeys;
			const PxU32* PX_RESTRICT lowSortedRanks = desc0[descriptorIndex].inputRanks;
			for(PxU32 i = chunkBegin; i < chunkEnd; ++i)
				highKeys[i] = PxU32(originalKeys[lowSortedRanks[i]] >> 32);
			__syncthreads();
		}

		PxgRadixSortBlockDesc& radixDesc = (pass & 1u) ? desc1[descriptorIndex] : desc0[descriptorIndex];
		const PxU32* PX_RESTRICT inputKeys = radixDesc.inputKeys;
		const PxU32* PX_RESTRICT inputRanks = radixDesc.inputRanks;
		PxU32* PX_RESTRICT outputKeys = radixDesc.outputKeys;
		PxU32* PX_RESTRICT outputRanks = radixDesc.outputRanks;
		const PxU32 startBit = (pass & 7u) * 4u;

		for(PxU32 radix = 0; radix < RADIX_SIZE; ++radix)
			chunkOffsets[radix * 32u + lane] = 0;
		__syncthreads();

		for(PxU32 i = chunkBegin; i < chunkEnd; ++i)
			++chunkOffsets[getPbdPrimitiveRadixInlineDcuV7(inputKeys[i], startBit) * 32u + lane];
		__syncthreads();

		if(lane < RADIX_SIZE)
		{
			PxU32 radixCount = 0;
			for(PxU32 chunk = 0; chunk < 32; ++chunk)
				radixCount += chunkOffsets[lane * 32u + chunk];
			radixBases[lane] = radixCount;
		}
		__syncthreads();

		if(lane == 0)
		{
			PxU32 radixBase = 0;
			for(PxU32 radix = 0; radix < RADIX_SIZE; ++radix)
			{
				const PxU32 radixCount = radixBases[radix];
				radixBases[radix] = radixBase;
				radixBase += radixCount;
			}
		}
		__syncthreads();

		if(lane < RADIX_SIZE)
		{
			PxU32 outputOffset = radixBases[lane];
			for(PxU32 chunk = 0; chunk < 32; ++chunk)
			{
				const PxU32 chunkCount = chunkOffsets[lane * 32u + chunk];
				chunkOffsets[lane * 32u + chunk] = outputOffset;
				outputOffset += chunkCount;
			}
		}
		__syncthreads();

		for(PxU32 i = chunkBegin; i < chunkEnd; ++i)
		{
			const PxU32 radix = getPbdPrimitiveRadixInlineDcuV7(inputKeys[i], startBit);
			const PxU32 outputIndex = chunkOffsets[radix * 32u + lane]++;
			outputKeys[outputIndex] = inputKeys[i];
			outputRanks[outputIndex] = inputRanks[i];
		}
		__syncthreads();
	}
}

extern "C" __global__
__launch_bounds__(32, 1)
void radixSortPbdSpatialChunkedCounting32DcuV6(PxgRadixSortDesc* desc0, PxgRadixSortDesc* desc1)
{
	const PxU32 lane = threadIdx.x;
	const PxU32 numKeys = desc0[0].count;
	const PxU32 chunkSize = (numKeys + 31u) / 32u;
	const PxU32 chunkBegin = lane * chunkSize;
	const PxU32 chunkEnd = PxMin(chunkBegin + chunkSize, numKeys);

	__shared__ PxU32 chunkOffsets[RADIX_SIZE * 32];
	__shared__ PxU32 radixBases[RADIX_SIZE];

	for(PxU32 pass = 0; pass < 8; ++pass)
	{
		PxgRadixSortDesc& radixDesc = (pass & 1u) ? desc1[0] : desc0[0];
		const PxU32* PX_RESTRICT inputKeys = radixDesc.inputKeys;
		const PxU32* PX_RESTRICT inputRanks = radixDesc.inputRanks;
		PxU32* PX_RESTRICT outputKeys = radixDesc.outputKeys;
		PxU32* PX_RESTRICT outputRanks = radixDesc.outputRanks;
		const PxU32 startBit = pass * 4u;

		for(PxU32 radix = 0; radix < RADIX_SIZE; ++radix)
			chunkOffsets[radix * 32u + lane] = 0;
		__syncthreads();

		for(PxU32 i = chunkBegin; i < chunkEnd; ++i)
			++chunkOffsets[getRadix(inputKeys[i], startBit) * 32u + lane];
		__syncthreads();

		if(lane < RADIX_SIZE)
		{
			PxU32 radixCount = 0;
			for(PxU32 chunk = 0; chunk < 32; ++chunk)
				radixCount += chunkOffsets[lane * 32u + chunk];
			radixBases[lane] = radixCount;
		}
		__syncthreads();

		if(lane == 0)
		{
			PxU32 radixBase = 0;
			for(PxU32 radix = 0; radix < RADIX_SIZE; ++radix)
			{
				const PxU32 radixCount = radixBases[radix];
				radixBases[radix] = radixBase;
				radixBase += radixCount;
			}
		}
		__syncthreads();

		if(lane < RADIX_SIZE)
		{
			PxU32 outputOffset = radixBases[lane];
			for(PxU32 chunk = 0; chunk < 32; ++chunk)
			{
				const PxU32 chunkCount = chunkOffsets[lane * 32u + chunk];
				chunkOffsets[lane * 32u + chunk] = outputOffset;
				outputOffset += chunkCount;
			}
		}
		__syncthreads();

		for(PxU32 i = chunkBegin; i < chunkEnd; ++i)
		{
			const PxU32 radix = getRadix(inputKeys[i], startBit);
			const PxU32 outputIndex = chunkOffsets[radix * 32u + lane]++;
			outputKeys[outputIndex] = inputKeys[i];
			outputRanks[outputIndex] = inputRanks[i];
		}
		__syncthreads();
	}
}
#endif

extern "C" __global__
__launch_bounds__(PxgRadixSortKernelBlockDim::RADIX_SORT, 1)
void radixSortMultiBlockLaunchWithoutCount(PxgRadixSortDesc* desc, const PxU32 gStartBit)
{
#if defined(PX_DCU_PORT) && PX_DCU_PORT
	// PBD's no-count path does not consume the per-block histogram directly.
	// Keep the 1024-thread histogram kernel out of the gfx936 execution path;
	// the following rank kernel performs the complete stable digit pass.
	return;
#endif

	const uint4* PX_RESTRICT gInputKeys = reinterpret_cast<uint4*>(desc[blockIdx.y].inputKeys);
	const uint4* PX_RESTRICT gInputRanks = reinterpret_cast<uint4*>(desc[blockIdx.y].inputRanks);
	PxU32* gRadixCount = desc[blockIdx.y].radixBlockCounts;

	radixSortSingleBlock<PxgRadixSortKernelBlockDim::RADIX_SORT / WARP_SIZE>(gInputKeys, gInputRanks, desc[blockIdx.y].count, gStartBit, gRadixCount);
}

//gNumOfKeys is device ptr
extern "C" __global__
__launch_bounds__(PxgRadixSortKernelBlockDim::RADIX_SORT, 1)
void radixSortMultiCalculateRanksLaunchWithoutCount(PxgRadixSortDesc* desc, const PxU32 gStartBit)
{
	const uint4* PX_RESTRICT gInputKeys = reinterpret_cast<uint4*>(desc[blockIdx.y].inputKeys);
	const uint4* PX_RESTRICT gInputRanks = reinterpret_cast<uint4*>(desc[blockIdx.y].inputRanks);
	PxU32* gOutputKeys = desc[blockIdx.y].outputKeys;
	PxU32* gOutputRanks = desc[blockIdx.y].outputRanks;
	PxU32* gRadixCount = desc[blockIdx.y].radixBlockCounts;
	const PxU32 numKeys = desc[blockIdx.y].count;

#if defined(PX_DCU_PORT) && PX_DCU_PORT
	// Avoid the gfx936 1024-thread shared-LDS rank path for the PBD no-count
	// descriptor. This is a stable least-significant 4-bit counting pass, so
	// descriptor ping-pong remains unchanged across all eight passes.
	if(blockIdx.x == 0 && threadIdx.x == 0)
	{
		const PxU32* PX_RESTRICT keys = reinterpret_cast<const PxU32*>(gInputKeys);
		const PxU32* PX_RESTRICT ranks = reinterpret_cast<const PxU32*>(gInputRanks);
		PxU32 offsets[RADIX_SIZE];

		for(PxU32 radix = 0; radix < RADIX_SIZE; ++radix)
			offsets[radix] = 0;

		for(PxU32 i = 0; i < numKeys; ++i)
			++offsets[getRadix(keys[i], gStartBit)];

		PxU32 runningOffset = 0;
		for(PxU32 radix = 0; radix < RADIX_SIZE; ++radix)
		{
			const PxU32 radixCount = offsets[radix];
			offsets[radix] = runningOffset;
			runningOffset += radixCount;
		}

		for(PxU32 i = 0; i < numKeys; ++i)
		{
			const PxU32 radix = getRadix(keys[i], gStartBit);
			const PxU32 outputIndex = offsets[radix]++;
			gOutputKeys[outputIndex] = keys[i];
			gOutputRanks[outputIndex] = ranks[i];
		}
	}
	return;
#endif

	radixSortCalculateRanks<PxgRadixSortKernelBlockDim::RADIX_SORT / WARP_SIZE>(gInputKeys, gInputRanks, numKeys, gStartBit, gRadixCount, gOutputKeys, gOutputRanks);
}


//gNumOfKeys is device ptr
extern "C" __global__ 
__launch_bounds__(PxgRadixSortKernelBlockDim::RADIX_SORT, 1)
void radixSortMultiBlockLaunchWithCount(PxgRadixSortDesc* desc, const PxU32 numKeys, const PxU32 gStartBit)
{
	const uint4* PX_RESTRICT gInputKeys = reinterpret_cast<uint4*>(desc[blockIdx.y].inputKeys);
	const uint4* PX_RESTRICT gInputRanks = reinterpret_cast<uint4*>(desc[blockIdx.y].inputRanks);
	PxU32* gRadixCount = desc[blockIdx.y].radixBlockCounts;

	radixSortSingleBlock<PxgRadixSortKernelBlockDim::RADIX_SORT / WARP_SIZE>(gInputKeys, gInputRanks, numKeys, gStartBit, gRadixCount);
}

//gNumOfKeys is device ptr
extern "C" __global__ 
__launch_bounds__(PxgRadixSortKernelBlockDim::RADIX_SORT, 1)
void radixSortMultiCalculateRanksLaunchWithCount(PxgRadixSortDesc* desc, const PxU32 numKeys, const PxU32 gStartBit)
{
	const uint4* PX_RESTRICT gInputKeys = reinterpret_cast<uint4*>(desc[blockIdx.y].inputKeys);
	const uint4* PX_RESTRICT gInputRanks = reinterpret_cast<uint4*>(desc[blockIdx.y].inputRanks);
	PxU32* gOutputKeys = desc[blockIdx.y].outputKeys;
	PxU32* gOutputRanks = desc[blockIdx.y].outputRanks;
	PxU32* gRadixCount = desc[blockIdx.y].radixBlockCounts;

#if defined(PX_DCU_PORT) && PX_DCU_PORT
	// The 64-body TGS eGPU smoke test produces 132 projection keys. Keep this
	// small set off the unstable gfx936 1024-thread wave/LDS rank path.
	if(numKeys <= WARP_SIZE * 8)
	{
		if(numKeys > 0 && blockIdx.x == 0 && threadIdx.x == 0)
		{
			const PxU32* keys = reinterpret_cast<const PxU32*>(gInputKeys);
			const PxU32* ranks = reinterpret_cast<const PxU32*>(gInputRanks);

			for(PxU32 i = 0; i < numKeys; ++i)
			{
				gOutputKeys[i] = keys[i];
				gOutputRanks[i] = ranks[i];
			}

			for(PxU32 i = 1; i < numKeys; ++i)
			{
				const PxU32 key = gOutputKeys[i];
				const PxU32 rank = gOutputRanks[i];
				PxU32 j = i;

				while(j > 0 && gOutputKeys[j - 1] > key)
				{
					gOutputKeys[j] = gOutputKeys[j - 1];
					gOutputRanks[j] = gOutputRanks[j - 1];
					--j;
				}

				gOutputKeys[j] = key;
				gOutputRanks[j] = rank;
			}
		}
		return;
	}
#endif

	radixSortCalculateRanks<PxgRadixSortKernelBlockDim::RADIX_SORT / WARP_SIZE>(gInputKeys, gInputRanks, numKeys, gStartBit, gRadixCount, gOutputKeys, gOutputRanks);
}

#if defined(PX_DCU_PORT) && PX_DCU_PORT
extern "C" __global__
__launch_bounds__(1, 1)
void radixSortBroadPhaseSerialCountingPassDcuV3(PxgRadixSortDesc* desc, const PxU32 numKeys, const PxU32 gStartBit)
{
	const PxgRadixSortDesc& radixDesc = desc[blockIdx.y];
	const PxU32* PX_RESTRICT inputKeys = radixDesc.inputKeys;
	const PxU32* PX_RESTRICT inputRanks = radixDesc.inputRanks;
	PxU32* PX_RESTRICT outputKeys = radixDesc.outputKeys;
	PxU32* PX_RESTRICT outputRanks = radixDesc.outputRanks;
	PxU32 offsets[RADIX_SIZE];

	for(PxU32 radix = 0; radix < RADIX_SIZE; ++radix)
		offsets[radix] = 0;

	for(PxU32 i = 0; i < numKeys; ++i)
		++offsets[getRadix(inputKeys[i], gStartBit)];

	PxU32 runningOffset = 0;
	for(PxU32 radix = 0; radix < RADIX_SIZE; ++radix)
	{
		const PxU32 radixCount = offsets[radix];
		offsets[radix] = runningOffset;
		runningOffset += radixCount;
	}

	// Incrementing each bucket offset while traversing the input preserves the
	// stable ordering required by the eight least-significant-digit passes.
	for(PxU32 i = 0; i < numKeys; ++i)
	{
		const PxU32 radix = getRadix(inputKeys[i], gStartBit);
		const PxU32 outputIndex = offsets[radix]++;
		outputKeys[outputIndex] = inputKeys[i];
		outputRanks[outputIndex] = inputRanks[i];
	}
}
#endif

extern "C" __global__ void radixSortCopyHigh32Bits(const PxU64* inValue, PxU32* outValue, PxU32* rank, const PxU32* numKeys)
{
	const PxU32 numElems = *numKeys;
	const PxU32 numIternations = (numElems + blockDim.x * gridDim.x - 1) / blockDim.x * gridDim.x;

	for (PxU32 i = 0; i < numIternations; ++i)
	{
		const PxU32 workIndex = threadIdx.x + blockIdx.x * blockDim.x + i * blockDim.x * gridDim.x;

		if (workIndex >= numElems)
			return;

		const PxU32 index = rank[workIndex];
		outValue[workIndex] = PxU32(inValue[index] >> 32);

		//printf("Copy 32 workIndex %i index %i blockDim.x %i gridDim.x %i\n", workIndex, index, blockDim.x, gridDim.x);

	}
}

extern "C" __global__ void radixSortDoubleCopyHigh32Bits(const PxU64 * inValue0, PxU32 * outValue0, PxU32 * rank0, const PxU64 * inValue1, PxU32 * outValue1, PxU32 * rank1, const PxU32 * numKeys)
{
	const PxU32 numElems = *numKeys;
	const PxU32 numIternations = (numElems + blockDim.x * gridDim.x - 1) / blockDim.x * gridDim.x;

	for (PxU32 i = 0; i < numIternations; ++i)
	{
		const PxU32 workIndex = threadIdx.x + blockIdx.x * blockDim.x + i * blockDim.x * gridDim.x;

		if (workIndex >= numElems)
			return;

		const PxU32 index0 = rank0[workIndex];
		outValue0[workIndex] = PxU32(inValue0[index0] >> 32);

		const PxU32 index1 = rank1[workIndex];
		outValue1[workIndex] = PxU32(inValue1[index1] >> 32);

		//printf("Copy 32 workIndex %i index %i blockDim.x %i gridDim.x %i\n", workIndex, index, blockDim.x, gridDim.x);

	}
}


extern "C" __global__ void radixSortCopy(const PxU64* inValue, PxU64* outValue, PxU32* rank, const PxU32* numKeys)
{

	const PxU32 numElems = *numKeys;
	const PxU32 numIternations = (numElems + blockDim.x * gridDim.x - 1) / blockDim.x * gridDim.x;


	for (PxU32 i = 0; i < numIternations; ++i)
	{
		const PxU32 workIndex = threadIdx.x + blockIdx.x * blockDim.x + i * blockDim.x * gridDim.x;

		if (workIndex >= numElems)
			return;

		const PxU32 index = rank[workIndex];
		outValue[workIndex] = inValue[index];

		//const PxNodeIndex nodeIndex = reinterpret_cast<const PxNodeIndex&>(inValue[index]);

		//printf("Copy 64 workIndex %i index %i value %i blockDim.x %i gridDim.x %i\n", workIndex, index, nodeIndex.index(), blockDim.x, gridDim.x);
	}

}

extern "C" __global__ void radixSortDoubleCopy(
	const PxU64 * inValue0, PxU64 * outValue0, PxU32 * rank0, 
	const PxU64 * inValue1, PxU64 * outValue1, PxU32 * rank1, 
	const PxU32 * numKeys)
{

	const PxU32 numElems = *numKeys;
	const PxU32 numIternations = (numElems + blockDim.x * gridDim.x - 1) / blockDim.x * gridDim.x;


	for (PxU32 i = 0; i < numIternations; ++i)
	{
		const PxU32 workIndex = threadIdx.x + blockIdx.x * blockDim.x + i * blockDim.x * gridDim.x;

		if (workIndex >= numElems)
			return;

		const PxU32 index0 = rank0[workIndex];
		outValue0[workIndex] = inValue0[index0];

		const PxU32 index1 = rank1[workIndex];
		outValue1[workIndex] = inValue1[index1];

		//const PxNodeIndex nodeIndex = reinterpret_cast<const PxNodeIndex&>(inValue[index]);

		//printf("Copy 64 workIndex %i index %i value %i blockDim.x %i gridDim.x %i\n", workIndex, index, nodeIndex.index(), blockDim.x, gridDim.x);
	}

}


extern "C" __global__ void radixSortCopyBits2(const PxU64* inValue, PxU32* outValue, PxU32* rank, const PxU32 numKeys,
	const bool lowBit)
{
	const PxU64 lowerMask = 0x00000000ffffffffull;
	const PxU32 globalThreadIndex = threadIdx.x + blockDim.x * blockIdx.x;

	if (globalThreadIndex < numKeys)
	{
		const PxU32 index = rank[globalThreadIndex];

		//this is aggregate, nodeIndex should be 0xffffffff
		if (index == 0xffffffff)
			outValue[globalThreadIndex] = 0xffffffff;
		else
			outValue[globalThreadIndex] = lowBit ? PxU32(inValue[index] & lowerMask) : PxU32(inValue[index] >> 32ll);
	}
}


extern "C" __global__ void radixSortCopy2(const PxU64* inValue, PxU64* outValue, PxU32* rank, const PxU32 numKeys)
{

	const PxU32 globalThreadIndex = threadIdx.x + blockDim.x * blockIdx.x;

	if(globalThreadIndex < numKeys)
	{
		const PxU32 index = rank[globalThreadIndex];
		const bool aggregate = (index == 0xffffffff);
		outValue[globalThreadIndex] = aggregate ? 0xffffffffffffffff : inValue[index];
	}

}
