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


#ifndef __SCANWARP_CUH__
#define __SCANWARP_CUH__

#if defined(__HIPCC__)
#include "PxgHIPCompat.h"
#endif

/* Scan&Reduce operators */

template <typename T>
struct AddOP
{
	static __device__ T apply(T a, T b) { return a + b; }
	static __device__ T identity() { return T(); }
};

template <typename T>
struct AndOP
{
	static __device__ T apply(T a, T b) { return a & b; }
	static __device__ T identity() { return T(); }
};

typedef AddOP<unsigned int> AddOPu;
typedef AddOP<float> AddOPf;

template <typename T>
struct MaxOPu
{
	static __device__ T apply(T a, T b) { return a > b ? a : b; }
	static __device__ T identity() { return T(); }
};

struct MinOPf
{
	static __device__ float apply(float a, float b) { return fminf(a, b); }
	static __device__ float applyIdx(float a, float b, unsigned int &oindex, unsigned int aindex, unsigned int bindex)
	{
		if(b < a)
		{
			oindex = bindex;
			return b;
		}
		else
		{
			oindex = aindex;
			return a;
		}
	}
	static __device__ float identity() { return +FLT_MAX; }
};

struct MaxOPf
{
	static __device__ float apply(float a, float b) { return fmaxf(a, b); }
	static __device__ float applyIdx(float a, float b, unsigned int &oindex, unsigned int aindex, unsigned int bindex)
	{
		if(b > a)
		{
			oindex = bindex;
			return b;
		}
		else
		{
			oindex = aindex;
			return a;
		}
	}

	static __device__ float identity() { return -FLT_MAX; }
};

/* Reduce */

template <typename T, class OP>
__device__ static inline void reduceWarp(volatile T* sdata)
{
	unsigned int idx = threadIdx.x;
	// Generalised for WARP_SIZE = 32 or 64
	if ((idx & (WARP_SIZE - 1)) < (WARP_SIZE / 2))
	{
		for (unsigned int step = WARP_SIZE / 2; step > 0; step >>= 1)
		{
			sdata[idx] = OP::apply(sdata[idx], sdata[idx + step]);
		}
	}
}

template <typename T, class OP>
__device__ static inline void reduceWarp(volatile T* sdata, unsigned int idx)
{
	for (unsigned int step = WARP_SIZE / 2; step > 0; step >>= 1)
	{
		sdata[idx] = OP::apply(sdata[idx], sdata[idx + step]);
	}
}

template <typename T, class OP>
__device__ static inline void reduceHalfWarp(volatile T* sdata)
{
	unsigned int idx = threadIdx.x;
	// Half a warp/wavefront reduction
	for (unsigned int step = WARP_SIZE / 4; step > 0; step >>= 1)
	{
		sdata[idx] = OP::apply(sdata[idx], sdata[idx + step]);
	}
}

template <typename T, typename TIndex, class OP>
__device__ static inline void reduceWarpKeepIndex(volatile T* sdata, volatile TIndex* sindices, unsigned int idx)
{
	TIndex ia, ib, oindex;

	for (unsigned int step = WARP_SIZE / 2; step > 0; step >>= 1)
	{
		if ((idx & (WARP_SIZE - 1)) < step)
		{
			ia = sindices[idx];
			ib = sindices[idx + step];
			sdata[idx] = OP::applyIdx(sdata[idx], sdata[idx + step], oindex, ia, ib);
			sindices[idx] = oindex;
		}
	}
}

/* Scan */

template <typename T, typename OP>
__device__ static inline void scanWarp(unsigned int scanIdx, volatile T* sdata)
{
	sdata[scanIdx] = OP::apply(sdata[scanIdx], sdata[scanIdx -  1]); 
	sdata[scanIdx] = OP::apply(sdata[scanIdx], sdata[scanIdx -  2]); 
	sdata[scanIdx] = OP::apply(sdata[scanIdx], sdata[scanIdx -  4]); 
	sdata[scanIdx] = OP::apply(sdata[scanIdx], sdata[scanIdx -  8]); 
	sdata[scanIdx] = OP::apply(sdata[scanIdx], sdata[scanIdx - 16]); 
}

/// The number of warp scan steps
#define STEPS LOG2_WARP_SIZE

// The SHFL mask for logically splitting warps into sub-segments
// Generalized for WARP_SIZE = 32 or 64
#define SHFL_MASK (((unsigned int)(-1) << STEPS) & (WARP_SIZE - 1)) << 8

template <>
__device__ inline void scanWarp<PxU32, AddOP<PxU32> >(unsigned int scanIdx, volatile PxU32 * sdata)
{
	PxU32 input = sdata[scanIdx];

#if defined(__HIPCC__)
	// HIP/DCU: use __shfl_up intrinsic (portable, works on all GPUs)
	#pragma unroll
	for (int STEP = 0; STEP < STEPS; STEP++)
	{
		PxU32 up = __shfl_up(input, 1 << STEP, WARP_SIZE);
		if ((int)(scanIdx & (WARP_SIZE - 1)) >= (1 << STEP))
			input = AddOP<PxU32>::apply(input, up);
	}
#else
	// CUDA: use PTX shfl.up.b32 with predicate for optimal codegen
	#pragma unroll
	for (int STEP = 0; STEP < STEPS; STEP++)
	{
		asm(
			"{"
			"  .reg .u32 r0;"
			"  .reg .pred p;"
			"  shfl.up.b32 r0|p, %1, %2, %3;"
			"  @p add.u32 r0, r0, %4;"
			"  mov.u32 %0, r0;"
			"}"
			: "=r"(input) : "r"(input), "r"(1 << STEP), "r"(SHFL_MASK), "r"(input));
	}
#endif
	sdata[scanIdx] = input;
}

template <>
__device__ inline void scanWarp<int, AddOP<int> >(unsigned int scanIdx, volatile int* sdata)
{
	PxU32 input = sdata[scanIdx];

#if defined(__HIPCC__)
	#pragma unroll
	for (int STEP = 0; STEP < STEPS; STEP++)
	{
		PxU32 up = __shfl_up(input, 1 << STEP, WARP_SIZE);
		if ((int)(scanIdx & (WARP_SIZE - 1)) >= (1 << STEP))
			input = AddOP<int>::apply(input, up);
	}
#else
	#pragma unroll
	for (int STEP = 0; STEP < STEPS; STEP++)
	{
		asm(
			"{"
			"  .reg .u32 r0;"
			"  .reg .pred p;"
			"  shfl.up.b32 r0|p, %1, %2, %3;"
			"  @p add.u32 r0, r0, %4;"
			"  mov.u32 %0, r0;"
			"}"
			: "=r"(input) : "r"(input), "r"(1 << STEP), "r"(SHFL_MASK), "r"(input));
	}
#endif
	sdata[scanIdx] = input;
}


#endif
