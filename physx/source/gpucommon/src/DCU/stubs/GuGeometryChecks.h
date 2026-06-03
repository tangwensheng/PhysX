// HIP stub for GuGeometryChecks.h
// The real GuGeometryChecks.h is at geomutils/src/ but our stub overrides it
// (stubs dir comes first in include path). So we redirect to the real one.
#pragma once

// Include all geometry types (same as real GuGeometryChecks.h)
#include "geometry/PxBoxGeometry.h"
#include "geometry/PxSphereGeometry.h"
#include "geometry/PxCapsuleGeometry.h"
#include "geometry/PxPlaneGeometry.h"
#include "geometry/PxConvexMeshGeometry.h"
#include "geometry/PxTriangleMeshGeometry.h"
#include "geometry/PxHeightFieldGeometry.h"
#include "geometry/PxParticleSystemGeometry.h"
#include "geometry/PxTetrahedronMeshGeometry.h"
#include "geometry/PxCustomGeometry.h"
#include "geometry/PxConvexCoreGeometry.h"

// Declare the PxcGeometryTraits template expected by PxvGeometry.h
namespace physx {
    template <typename T> struct PxcGeometryTraits {};
}
