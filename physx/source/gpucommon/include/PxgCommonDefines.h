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

#ifndef PXG_COMMON_DEFINES_H
#define PXG_COMMON_DEFINES_H

// A place for shared defines for the GPU libs

// HIP compatibility: when compiling with hipcc, this header maps
// CUDA intrinsics (__shfl_sync, __ballot_sync, etc.) to HIP equivalents.
// Included here so all kernel files that include PxgCommonDefines.h
// automatically get HIP compatibility.
#if defined(__HIPCC__)
#include "PxgHIPCompat.h"
#endif

#define PXG_MAX_NUM_POINTS_PER_CONTACT_PATCH 6 // corresponding CPU define is CONTACT_REDUCTION_MAX_CONTACTS

// DCU/HIP platform detection: __HIPCC__ is defined by the hipcc compiler
#if defined(__HIPCC__) || defined(PX_DCU_PORT)
	// Hygon DCU: wavefront size = 64
	#define LOG2_WARP_SIZE 6
	#define WARP_SIZE (1U << LOG2_WARP_SIZE)  // 64
	#define FULL_MASK 0xffffffffffffffffULL    // full mask for 64 threads in a wavefront
#else
	// NVIDIA CUDA: warp size = 32
	#define LOG2_WARP_SIZE 5
	#define WARP_SIZE (1U << LOG2_WARP_SIZE)  // 32
	#define FULL_MASK 0xffffffff              // full mask for 32 threads in a warp
#endif


#endif