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
// which detects __HIPCC__ and keeps CUDA's logical 32-lane warp size.

// ---- Hardware lane ----
#define PXG_HW_LANE ((int)((threadIdx.z * blockDim.y + threadIdx.y) * blockDim.x + threadIdx.x) & 63)

// ---- Warp shuffle intrinsics ----
// DCU wavefront=64 while PhysX keeps CUDA's logical WARP_SIZE=32. Route
// explicitly inside the current 32-lane virtual warp instead of relying on
// HIP's width argument, whose semantics differ across ROCm/DCU toolchains.
template <typename T>
__device__ __forceinline__ T pxgDcuLaneRead32(T var, int hwLane)
{
    union Bits
    {
        T value;
        unsigned int words[(sizeof(T) + sizeof(unsigned int) - 1) / sizeof(unsigned int)];
    } in, out;

    in.value = var;
#pragma unroll
    for (int i = 0; i < int((sizeof(T) + sizeof(unsigned int) - 1) / sizeof(unsigned int)); ++i)
        out.words[i] = __builtin_amdgcn_ds_bpermute(hwLane * int(sizeof(unsigned int)), in.words[i]);
    return out.value;
}

template <typename T>
__device__ __forceinline__ T pxgDcuShfl(T var, int lane, int width = 32)
{
    const int localLane = PXG_HW_LANE & 31;
    const int baseLane = PXG_HW_LANE & ~31;
    const int groupBase = (localLane / width) * width;
    return pxgDcuLaneRead32(var, baseLane + groupBase + (lane & (width - 1)));
}

template <typename T>
__device__ __forceinline__ T pxgDcuShflXor(T var, int offset, int width = 32)
{
    const int localLane = PXG_HW_LANE & 31;
    const int baseLane = PXG_HW_LANE & ~31;
    const int targetLane = localLane ^ offset;
    return ((targetLane / width) == (localLane / width)) ? pxgDcuLaneRead32(var, baseLane + targetLane) : var;
}

template <typename T>
__device__ __forceinline__ T pxgDcuShflUp(T var, int delta, int width = 32)
{
    const int localLane = PXG_HW_LANE & 31;
    const int baseLane = PXG_HW_LANE & ~31;
    const int groupBase = (localLane / width) * width;
    const int laneInGroup = localLane - groupBase;
    return (laneInGroup >= delta) ? pxgDcuLaneRead32(var, baseLane + groupBase + laneInGroup - delta) : var;
}

template <typename T>
__device__ __forceinline__ T pxgDcuShflDown(T var, int delta, int width = 32)
{
    const int localLane = PXG_HW_LANE & 31;
    const int baseLane = PXG_HW_LANE & ~31;
    const int groupBase = (localLane / width) * width;
    const int laneInGroup = localLane - groupBase;
    return (laneInGroup + delta < width) ? pxgDcuLaneRead32(var, baseLane + groupBase + laneInGroup + delta) : var;
}

#define __shfl_sync_4(mask, var, lane, width)  pxgDcuShfl((var), (int)(lane), (int)(width))
#define __shfl_sync_3(mask, var, lane)         pxgDcuShfl((var), (int)(lane), 32)
#define __shfl_sync_DISP(_1,_2,_3,_4,NAME,...) NAME
#define __shfl_sync(...)     __shfl_sync_DISP(__VA_ARGS__, __shfl_sync_4, __shfl_sync_3, _DUMMY)(__VA_ARGS__)

#define __shfl_xor_sync_4(mask, var, offset, width)  pxgDcuShflXor((var), (int)(offset), (int)(width))
#define __shfl_xor_sync_3(mask, var, offset)         pxgDcuShflXor((var), (int)(offset), 32)
#define __shfl_xor_sync(...)     __shfl_sync_DISP(__VA_ARGS__, __shfl_xor_sync_4, __shfl_xor_sync_3, _DUMMY)(__VA_ARGS__)

#define __shfl_up_sync_4(mask, var, delta, width)    pxgDcuShflUp((var), (int)(delta), (int)(width))
#define __shfl_up_sync_3(mask, var, delta)           pxgDcuShflUp((var), (int)(delta), 32)
#define __shfl_up_sync(...)     __shfl_sync_DISP(__VA_ARGS__, __shfl_up_sync_4, __shfl_up_sync_3, _DUMMY)(__VA_ARGS__)

#define __shfl_down_sync_4(mask, var, delta, width)  pxgDcuShflDown((var), (int)(delta), (int)(width))
#define __shfl_down_sync_3(mask, var, delta)         pxgDcuShflDown((var), (int)(delta), 32)
#define __shfl_down_sync(...)     __shfl_sync_DISP(__VA_ARGS__, __shfl_down_sync_4, __shfl_down_sync_3, _DUMMY)(__VA_ARGS__)

// ---- Warp vote intrinsics ----
// Split 64-bit wavefront ballot into 32-bit virtual warps.
#define __ballot_sync(mask, pred) \
    (unsigned int)(((PXG_HW_LANE & 32) ? (__ballot(pred) >> 32) : __ballot(pred)))
// Match CUDA vote semantics by applying the active mask after splitting.
#define __any_sync(mask, pred) \
    ((__ballot_sync((mask), (pred)) & (unsigned int)(mask)) != 0u)
#define __all_sync(mask, pred) \
    (((__ballot_sync((mask), (pred)) & (unsigned int)(mask)) == (unsigned int)(mask)))

// ---- Warp synchronization ----
// CUDA 语义：__syncwarp() 只同步 32-lane 逻辑 warp，并充当 shared memory 编译器栅栏。
// 它经常被放在 if(threadIdx.x < 32) 这类 divergent 分支内（见 convexMeshMidphase.cu
// 的 triangleTriangleCollision）。若映射成 __syncthreads()，workgroup 级 s_barrier 会
// 出现在只有部分线程进入的分支里，与后续 block-uniform 的 __syncthreads() 配对错乱：
// 其它 wavefront 提前越过屏障，读到尚未写入的 shared（如 mesh 描述符）-> 垃圾指针 ->
// KERNEL VMFault / memory aperture violation。
// GCN/CDNA 的 wavefront 天然 lock-step，warp 内执行已同步，这里只需要 block 级 memory
// fence 保证 shared 写入可见即可，绝不能用整块同步。
// 变参形式同时兼容 __syncwarp() 与 __syncwarp(mask) 两种调用写法。
#define __syncwarp(...)              __threadfence_block()

// ---- Bit operation intrinsics ----
// CUDA __popc is 32-bit, HIP provides both __popc(32b) and __popcll(64b).
// Keep __popc for 32-bit compatibility.

// ---- Device-side memory fence ----
// These have the same names in HIP: __threadfence(), __threadfence_block(), etc.

// ---- Other device intrinsics (same names in HIP) ----
// atomicAdd, atomicCAS, atomicOr, atomicAnd, atomicMin, atomicMax, atomicExch
// __int_as_float, __float_as_int, __syncthreads, __ldg, __launch_bounds__

// ---- Architecture macro remap ----
// CUDA code uses __CUDA_ARCH__ for conditional compilation.
#define __CUDA_ARCH__  __HIP_DEVICE_COMPILE__

// ---- Inline assembly compatibility ----
// red.global.add.f32 (L2 cache-level atomic) -> fall back to atomicAdd.
#define PX_RED_GLOBAL_ADD_F32(addr, val) atomicAdd((addr), (val))

// ---- Miscellaneous ----
// CUDA alignment attribute -> HIP/Clang.
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

// Note: PxConstraintFlag comes from PxConstraint.h, included via the full build system.

// CUDA's WARP_SIZE definition is in PxgCommonDefines.h, already handled.

#endif // defined(__HIPCC__)
#endif // PXG_HIP_COMPAT_H
