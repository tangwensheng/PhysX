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
