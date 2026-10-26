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

// DCU stub for <vector_functions.h>
#pragma once

#if defined(__HIPCC__)
// Kernel code compiled by hipcc — HIP provides make_float4 etc.
#include <hip/hip_runtime.h>
#else
// Host-side g++: bare-minimum multi-arg constructors.
// cutil_math.h provides scalar overloads and operator overloads.
#include "vector_types.h"

#define __host__
#define __device__

inline float2 make_float2(float x, float y) { float2 v = {x, y}; return v; }
inline float3 make_float3(float x, float y, float z) { float3 v = {x, y, z}; return v; }
inline float3 make_float3(const float4& a)        { float3 v = {a.x, a.y, a.z}; return v; }
inline float4 make_float4(float x, float y, float z, float w) { float4 v = {x, y, z, w}; return v; }

inline int2 make_int2(int x, int y) { int2 v = {x, y}; return v; }
inline int3 make_int3(int x, int y, int z) { int3 v = {x, y, z}; return v; }
inline int4 make_int4(int x, int y, int z, int w) { int4 v = {x, y, z, w}; return v; }

inline uint2 make_uint2(unsigned int x, unsigned int y) { uint2 v = {x, y}; return v; }
inline uint3 make_uint3(unsigned int x, unsigned int y, unsigned int z) { uint3 v = {x, y, z}; return v; }
inline uint4 make_uint4(unsigned int x, unsigned int y, unsigned int z, unsigned int w) { uint4 v = {x, y, z, w}; return v; }

inline double2 make_double2(double x, double y) { double2 v = {x, y}; return v; }
#endif
