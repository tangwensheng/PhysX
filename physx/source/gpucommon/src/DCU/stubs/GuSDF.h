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
