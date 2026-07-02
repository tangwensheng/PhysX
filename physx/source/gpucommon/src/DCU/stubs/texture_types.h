// DCU stub for <texture_types.h>
#pragma once
#if defined(__HIPCC__)
#include <hip/hip_runtime.h>
#endif
// Host-side g++: minimal types
#ifndef __HIPCC__
struct textureReference;
typedef unsigned int cudaTextureObject_t;
struct cudaChannelFormatDesc {
	int x, y, z, w;
	enum cudaChannelFormatKind {
		cudaChannelFormatKindSigned = 0,
		cudaChannelFormatKindUnsigned = 1,
		cudaChannelFormatKindFloat = 2
	} f;
};
#endif
