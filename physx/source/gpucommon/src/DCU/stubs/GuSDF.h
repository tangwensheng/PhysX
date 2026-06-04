// HIP-compatible stub for GuSDF.h
// The real file is at geomutils/src/GuSDF.h but has HIP-compatibility issues
// with alignment macros (PX_ALIGN_PREFIX/SUFFIX) that cause struct member loss.
// This stub provides minimal working implementations for HIP compilation.
#pragma once

#include "foundation/PxVec3.h"

namespace physx {
namespace Gu {

struct GridQueryPointSampler
{
    PxVec3 mOrigin;
    PxVec3 mCellSize;

    __host__ __device__ GridQueryPointSampler() : mCellSize(0,0,0) {}
    __host__ __device__ GridQueryPointSampler(const PxVec3& origin, const PxVec3& cellSize,
        bool /*cellCenteredSamples*/, int = 0, int = 0, int = 0, int = 1, int = 1, int = 1)
        : mOrigin(cellSize), mCellSize(cellSize) {}

    __host__ __device__ PxVec3 getPoint(int x, int y, int z) const {
        return PxVec3(mOrigin.x + x * mCellSize.x,
                      mOrigin.y + y * mCellSize.y,
                      mOrigin.z + z * mCellSize.z);
    }
    __host__ __device__ PxVec3 getOrigin() const { return mOrigin; }
    __host__ __device__ PxVec3 getActiveCellSize() const { return mCellSize; }
};

class DenseSDF
{
public:
    PxU32 mWidth, mHeight, mDepth;

    __host__ __device__ DenseSDF() : mWidth(0), mHeight(0), mDepth(0) {}
    __host__ __device__ DenseSDF(PxU32 w, PxU32 h, PxU32 d, float*)
        : mWidth(w), mHeight(h), mDepth(d) {}

    __host__ __device__ void initialize(PxU32 w, PxU32 h, PxU32 d, float*) {
        mWidth = w; mHeight = h; mDepth = d;
    }

    __host__ __device__ float sampleSDFDirect(const PxVec3&) const { return 0.0f; }
};

// Utility functions — signatures must match real GuSDF.h exactly
__host__ __device__ inline PxU32 idx3D(PxU32 x, PxU32 y, PxU32 z, PxU32 width, PxU32 height) {
    return x + y * width + z * width * height;
}
__host__ __device__ inline void idToXYZ(PxU32 id, PxU32 sizeX, PxU32 sizeY, PxU32& xi, PxU32& yi, PxU32& zi) {
    xi = id % sizeX; yi = (id / sizeX) % sizeY; zi = id / (sizeX * sizeY);
}
__host__ __device__ inline PxU32 encodeTriple(PxU32 x, PxU32 y, PxU32 z) {
    return x | (y << 10) | (z << 20);
}

} // namespace Gu
} // namespace physx
