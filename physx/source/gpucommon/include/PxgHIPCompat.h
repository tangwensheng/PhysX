// HIP / DCU Compatibility Layer for PhysX
//
// Maps CUDA intrinsic names to HIP equivalents.
// Included before shuffle.cuh / atomic.cuh / warpHelpers.cuh when compiling with hipcc.
//
// DCU uses older HIP API (no _sync suffix on shuffle/ballot intrinsics).
// This header provides the mapping so the rest of the PhysX kernel code
// can continue to use CUDA-style intrinsic names without changes.

#ifndef PXG_HIP_COMPAT_H
#define PXG_HIP_COMPAT_H

#if defined(__HIPCC__)

// ---- Includes ----
// HIP runtime provides device intrinsics (__shfl, __ballot, atomicAdd, etc.)
#include <hip/hip_runtime.h>

// PhysX foundation types needed by kernel code (PxVec3, PxU32, PxReal, etc.)
// In the CUDA build these come transitively through nvcc include chains.
// For HIP, we ensure they're available at the top level.
#include "foundation/PxPreprocessor.h"
#include "foundation/PxSimpleTypes.h"
#include "foundation/PxVec3.h"
#include "foundation/PxVec4.h"
#include "foundation/PxMat33.h"
#include "foundation/PxMat44.h"
#include "foundation/PxTransform.h"

// ---- Architecture constants ----
// WARP_SIZE and FULL_MASK are already defined in PxgCommonDefines.h
// which detects __HIPCC__ and sets WARP_SIZE=64, FULL_MASK=0xffffffffffffffffULL

// ---- Warp shuffle intrinsics ----
// DCU wavefront=64, HIP __shfl with width=32 correctly splits into 32-lane
// virtual warps. width=32 is the CUDA warpSize default. No manual lane math needed.
#define __shfl_sync_4(mask, var, lane, width)  __shfl((var), (int)(lane), (int)(width))
#define __shfl_sync_3(mask, var, lane)         __shfl((var), (int)(lane), 32)
#define __shfl_sync_DISP(_1,_2,_3,_4,NAME,...) NAME
#define __shfl_sync(...) \
    __shfl_sync_DISP(__VA_ARGS__, __shfl_sync_4, __shfl_sync_3, _DUMMY)(__VA_ARGS__)

#define __shfl_xor_sync_4(mask, var, offset, width)  __shfl_xor((var), (offset), (width))
#define __shfl_xor_sync_3(mask, var, offset)         __shfl_xor((var), (offset), 32)
#define __shfl_xor_sync(...)  __shfl_sync_DISP(__VA_ARGS__, __shfl_xor_sync_4, __shfl_xor_sync_3, _DUMMY)(__VA_ARGS__)

#define __shfl_up_sync_4(mask, var, delta, width)    __shfl_up((var), (delta), (width))
#define __shfl_up_sync_3(mask, var, delta)           __shfl_up((var), (delta), 32)
#define __shfl_up_sync(...)  __shfl_sync_DISP(__VA_ARGS__, __shfl_up_sync_4, __shfl_up_sync_3, _DUMMY)(__VA_ARGS__)

#define __shfl_down_sync_4(mask, var, delta, width)  __shfl_down((var), (delta), (width))
#define __shfl_down_sync_3(mask, var, delta)         __shfl_down((var), (delta), 32)
#define __shfl_down_sync(...)  __shfl_sync_DISP(__VA_ARGS__, __shfl_down_sync_4, __shfl_down_sync_3, _DUMMY)(__VA_ARGS__)

// ---- Hardware lane ----
#define PXG_HW_LANE ((int)((threadIdx.z * blockDim.y + threadIdx.y) * blockDim.x + threadIdx.x) & 63)

// ---- Warp vote intrinsics ----
// Split 64-bit wavefront ballot into 32-bit virtual warps
#define __ballot_sync(mask, pred) \
    (unsigned int)(((PXG_HW_LANE & 32) ? (__ballot(pred) >> 32) : __ballot(pred)))
// all/any: temporary revert to simple compare for boxBoxNphase_Kernel stability
#define __all_sync(mask, pred)  __all(pred)
#define __any_sync(mask, pred)  __any(pred)

// ---- Warp synchronization ----
// __syncwarp is not available in old HIP; use __syncthreads as fallback
// (PhysX typically uses this within a single warp, so __syncthreads is safe)
#define __syncwarp(mask)             __syncthreads()

// ---- Bit operation intrinsics ----
// CUDA __popc is 32-bit, HIP provides both __popc(32b) and __popcll(64b)
// Keep __popc for 32-bit compatibility
// On HIP, __popc(unsigned int) works on 32-bit values just like CUDA
// Note: when ballot returns 64-bit, use __popcll explicitly in callers

// ---- Device-side memory fence ----
// These have the same names in HIP
// __threadfence(), __threadfence_block(), __threadfence_system()

// ---- Other device intrinsics (same names in HIP) ----
// atomicAdd, atomicCAS, atomicOr, atomicAnd, atomicMin, atomicMax, atomicExch
// __int_as_float, __float_as_int
// __syncthreads, __syncthreads_and, __syncthreads_or
// __ldg (read-only cache load)
// __launch_bounds__

// ---- Architecture macro remap ----
// CUDA code uses __CUDA_ARCH__ for conditional compilation
// HIP provides __HIP_DEVICE_COMPILE__ and __HIP_ARCH__
#define __CUDA_ARCH__  __HIP_DEVICE_COMPILE__

// ---- Inline assembly compatibility ----
// red.global.add.f32 (L2 cache-level atomic) -> fall back to atomicAdd
// The original code used inline asm for this optimization; on DCU we degrade
// to the regular atomicAdd which is functionally correct but ~20% slower
#define PX_RED_GLOBAL_ADD_F32(addr, val) atomicAdd((addr), (val))

// ---- Miscellaneous ----
// CUDA alignment attribute → HIP/Clang
#define __builtin_align__(N)  __attribute__((aligned(N)))

// __ffsll: CUDA 64-bit find-first-set. In HIP, __ffsll exists but may have
// ambiguous overloads for unsigned types. Cast to long long for safety.
#define __ffsll(x)  __ffsll((long long)(x))

// ---- Solver/Dynamics constants (for files that lack the full CPU header chain) ----
// These are normally provided by DyConstraint.h / PxcNpWorkUnit.h / PxvConfig.h
// but are not available when compiling kernels standalone without the full build system.
namespace physx { namespace Dy {
    const int MAX_CONSTRAINT_ROWS = 32;
} }
#define DY_SC_FLAG_SPRING               0x0001
#define DY_SC_FLAG_ACCELERATION_SPRING  0x0002
#define DY_SC_FLAG_OUTPUT_FORCE         0x0004
#define DY_SC_FLAG_KEEP_BIAS            0x0008
#define DY_SC_FLAG_INEQUALITY           0x0010
#define DY_SC_FLAG_ORTHO_TARGET         0x0020
#define DY_SC_FLAG_ROT_EQ               0x0040

// Note: PxConstraintFlag comes from PxConstraint.h, included via the full build system

// CUDA's WARP_SIZE definition is in PxgCommonDefines.h, already handled

#endif // defined(__HIPCC__)
#endif // PXG_HIP_COMPAT_H
