// HIP stub for PxcNpWorkUnit.h — CPU-side narrowphase types
#pragma once
namespace physx {
    typedef unsigned int PxcNpWorkUnitFlag;
    struct PxgMaterialContactData { float dummy; };
    struct PxsShapeCore { char _pad[16]; };
}
#define DY_SC_FLAG_SPRING 0x01
