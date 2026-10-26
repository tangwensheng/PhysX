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

// PhysX CPU Library Integration Test
//
// Validates that the PhysX foundation library compiles and works on HCU.
// Uses PxVec3, PxMat33, PxTransform — the core math types.

#include "foundation/PxVec3.h"
#include "foundation/PxVec4.h"
#include "foundation/PxMat33.h"
#include "foundation/PxMat44.h"
#include "foundation/PxTransform.h"
#include "foundation/PxSimpleTypes.h"
#include <cstdio>
#include <cmath>

using namespace physx;

int main()
{
    printf("=== PhysX CPU Foundation Test ===\n\n");

    int passed = 0, total = 0;

    // Test 1: PxVec3 basic operations
    printf("--- Test 1: PxVec3 ---\n");
    PxVec3 a(1.0f, 2.0f, 3.0f);
    PxVec3 b(4.0f, 5.0f, 6.0f);
    PxVec3 c = a + b;

    total++;
    if (c.x == 5.0f && c.y == 7.0f && c.z == 9.0f) {
        passed++;
        printf("  [PASS] PxVec3 addition\n");
    } else {
        printf("  [FAIL] PxVec3 addition\n");
    }

    // dot product
    float d = a.dot(b);
    total++;
    if (fabsf(d - 32.0f) < 0.001f) {
        passed++;
        printf("  [PASS] PxVec3 dot product\n");
    } else {
        printf("  [FAIL] PxVec3 dot product\n");
    }

    // cross product
    PxVec3 e = a.cross(b);
    total++;
    if (fabsf(e.x + 3.0f) < 0.001f && fabsf(e.y - 6.0f) < 0.001f && fabsf(e.z + 3.0f) < 0.001f) {
        passed++;
        printf("  [PASS] PxVec3 cross product\n");
    } else {
        printf("  [FAIL] PxVec3 cross product\n");
    }

    // magnitude
    float mag = a.magnitude();
    total++;
    if (fabsf(mag - sqrtf(14.0f)) < 0.001f) {
        passed++;
        printf("  [PASS] PxVec3 magnitude\n");
    } else {
        printf("  [FAIL] PxVec3 magnitude\n");
    }

    // normalize
    PxVec3 n = a.getNormalized();
    total++;
    if (fabsf(n.magnitude() - 1.0f) < 0.001f) {
        passed++;
        printf("  [PASS] PxVec3 normalize\n");
    } else {
        printf("  [FAIL] PxVec3 normalize\n");
    }

    // Test 2: PxMat33
    printf("\n--- Test 2: PxMat33 ---\n");
    PxMat33 m = PxMat33(PxIdentity);
    PxVec3 r = m * a;
    total++;
    if (fabsf(r.x - a.x) < 0.001f && fabsf(r.y - a.y) < 0.001f && fabsf(r.z - a.z) < 0.001f) {
        passed++;
        printf("  [PASS] PxMat33 identity transform\n");
    } else {
        printf("  [FAIL] PxMat33 identity transform\n");
    }

    // Test 3: PxTransform
    printf("\n--- Test 3: PxTransform ---\n");
    PxTransform t1(PxIdentity);
    PxTransform t2(PxVec3(1.0f, 2.0f, 3.0f), PxQuat(PxIdentity));
    PxTransform t3 = t1 * t2;
    total++;
    if (fabsf(t3.p.x - 1.0f) < 0.001f) {
        passed++;
        printf("  [PASS] PxTransform multiply\n");
    } else {
        printf("  [FAIL] PxTransform multiply\n");
    }

    // summary
    printf("\n========================================\n");
    printf(" Results: %d / %d passed\n", passed, total);
    printf("========================================\n");

    return (passed == total) ? 0 : 1;
}
