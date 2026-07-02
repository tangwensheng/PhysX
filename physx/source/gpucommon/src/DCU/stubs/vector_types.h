// DCU stub for <vector_types.h> — provides CUDA-compatible float4/int4/etc.
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
