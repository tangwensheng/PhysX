// DCU stub for <driver_types.h>
#pragma once
#if defined(__HIPCC__)
#include <hip/hip_runtime.h>
#endif
// Host-side g++: minimal types only (cudaError_t etc.)
#ifndef __HIPCC__
typedef enum cudaError_enum { cudaSuccess = 0 } cudaError_t;
typedef struct CUstream_st* cudaStream_t;
#endif
