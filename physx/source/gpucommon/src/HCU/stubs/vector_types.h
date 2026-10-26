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

// HCU stub for <vector_types.h> — provides CUDA-compatible float4/int4/etc.
//
// When compiling with hipcc (kernel .cu files → HIP): HIP runtime already
// provides these types — just delegate to <hip/hip_runtime.h>.
//
// When compiling with g++ (host-side GPU objects): HIP runtime must NOT be
// included (would define float4 as "using" alias, conflicting with PhysX's
// "struct float4" forward declarations).  Define plain structs instead.
#pragma once

#if defined(__HIPCC__)
// Kernel code compiled by hipcc — HIP types are available
#include <hip/hip_runtime.h>
#else
// Host-side code compiled by g++ — define CUDA-compatible plain structs

struct float2  { float  x, y; };
struct float3  { float  x, y, z; };
struct float4  { float  x, y, z, w; };

struct int2    { int    x, y; };
struct int3    { int    x, y, z; };
struct int4    { int    x, y, z, w; };

struct uint2   { unsigned int x, y; };
struct uint3   { unsigned int x, y, z; };
struct uint4   { unsigned int x, y, z, w; };

struct double2 { double x, y; };
struct uchar4  { unsigned char x, y, z, w; };
struct char4   { char x, y, z, w; };

// dim3 — used in kernel launch configuration
struct dim3 {
	unsigned int x, y, z;
	dim3(unsigned int _x = 1, unsigned int _y = 1, unsigned int _z = 1) : x(_x), y(_y), z(_z) {}
};

struct uint3_host {
	unsigned int x, y, z;
};
#endif // __HIPCC__
