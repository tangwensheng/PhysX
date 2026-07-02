// DcuNonRigidStubs — method stubs for non-rigid-body modules
// (soft body, FEM cloth, particles, articulation) not yet ported to DCU.
// Only provides the method implementations; class declarations come from
// the existing PhysX GPU headers which ARE in the include path.
#include "PxgArticulationCore.h"
#include "PxgSoftBodyCore.h"
#include "PxgFEMClothCore.h"
#include "PxgParticleSystemCore.h"
#include "PxgPBDParticleSystemCore.h"
#include "PxgFEMCore.h"
#include "PxgSimulationController.h"
#include "PxgContext.h"
#include <cstdio>

using namespace physx;

#define DCU_STUB(CLASS, METHOD, ...) \
	void CLASS::METHOD(__VA_ARGS__) {}

#define DCU_STUB_RET(RET, CLASS, METHOD, ...) \
	RET CLASS::METHOD(__VA_ARGS__) { return RET(); }

// ---- PxgArticulationCore ----
PxgArticulationCore::PxgArticulationCore(PxgCudaKernelWranglerManager*, PxCudaContextManager*, PxgHeapMemoryAllocatorManager*) { fprintf(stderr,"[DCU] PxgArticulationCore stub\n"); }
PxgArticulationCore::~PxgArticulationCore() {}
void PxgArticulationCore::syncStream() {}
void PxgArticulationCore::saveVelocities() {}
void PxgArticulationCore::syncUnconstrainedVelocities() {}
void PxgArticulationCore::updateBodies(float, bool, bool) {}
void PxgArticulationCore::precomputeDependencies(unsigned int) {}
void PxgArticulationCore::computeUnconstrainedVelocities(unsigned int, unsigned int, float, const PxVec3&, float, bool, bool) {}
void PxgArticulationCore::setupInternalConstraints(unsigned int, float, float, float, bool) {}
void PxgArticulationCore::createStaticContactAndConstraintsBatch(unsigned int) {}
void PxgArticulationCore::synchronizedStreams(CUstream*, CUstream*) {}
void PxgArticulationCore::updateArticulationsKinematic(bool, const unsigned int*, unsigned int) {}
void PxgArticulationCore::outputVelocity(PxU64, CUstream*, bool) {}
void PxgArticulationCore::allocDeltaVBuffer(unsigned int, unsigned int, CUstream*) {}
void PxgArticulationCore::layoutDeltaVBuffer(unsigned int, unsigned int, CUstream*) {}
void PxgArticulationCore::averageDeltaV(unsigned int, CUstream*, float4*, unsigned int, bool, PxU64) {}
void PxgArticulationCore::propagateRigidBodyImpulsesAndSolveInternalConstraints(float, float, bool, float, float, const Dy::ArticulationConstraintProcessingConfigGPU&, unsigned int*, unsigned int*, PxU64, bool, bool, bool, bool) {}
void PxgArticulationCore::gpuMemDMAbackArticulation(PxArray<PxU8, PxVirtualAllocator>&, PxArray<PxgSolverBodySleepData, PxVirtualAllocator>&, PxArray<Dy::ErrorAccumulator, PxVirtualAllocator>&, PxArray<Dy::ErrorAccumulator, PxVirtualAllocator>&) {}
void PxgArticulationCore::getArticulationData(void*, const unsigned int*, PxArticulationGPUAPIReadType::Enum, unsigned int, CUevent*, CUevent*, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int) const {}
void PxgArticulationCore::setArticulationData(const void*, const unsigned int*, PxArticulationGPUAPIWriteType::Enum, unsigned int, CUevent*, CUevent*, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int) {}
void PxgArticulationCore::computeArticulationData(void*, const unsigned int*, PxArticulationGPUAPIComputeType::Enum, unsigned int, unsigned int, unsigned int, CUevent*, CUevent*) {}

// ---- PxgSoftBodyCore ----
void PxgSoftBodyCore::constraintPrep(PxgDevicePointer<PxgPrePrepDesc>, PxgDevicePointer<PxgConstraintPrepareDesc>, float, PxgDevicePointer<PxgSolverSharedDescBase>, CUstream*, bool, unsigned int, unsigned int) {}
void PxgSoftBodyCore::solve(PxgDevicePointer<PxgPrePrepDesc>, PxgDevicePointer<PxgConstraintPrepareDesc>, PxgDevicePointer<PxgSolverCoreDesc>, PxgDevicePointer<PxgSolverSharedDescBase>, PxgDevicePointer<PxgArticulationCoreDesc>, float, CUstream*, bool) {}
void PxgSoftBodyCore::resetContactCounts() {}
void PxgSoftBodyCore::sortContacts(unsigned int) {}
void PxgSoftBodyCore::selfCollision() {}
void PxgSoftBodyCore::syncSoftBodies() {}
void PxgSoftBodyCore::checkBufferOverflows() {}
void PxgSoftBodyCore::createActivatedDeactivatedLists() {}
void PxgSoftBodyCore::updateTetraRotations() {}
void PxgSoftBodyCore::finalizeVelocities(float, float, bool) {}
void PxgSoftBodyCore::preIntegrateSystems(PxgSoftBody*, unsigned int*, unsigned int, PxVec3, float) {}
void PxgSoftBodyCore::refitBound(PxgSoftBody*, unsigned int) {}
void PxgSoftBodyCore::updateUserData(PxArray<PxgSoftBody, PxVirtualAllocator>&, PxArray<unsigned int, PxReflectionAllocator<unsigned int> >&, const unsigned int*, unsigned int, void**) {}
void PxgSoftBodyCore::copySoftBodyDataDEPRECATED(void**, void*, void*, PxSoftBodyGpuDataFlag::Enum, unsigned int, unsigned int, CUevent*) {}
void PxgSoftBodyCore::applySoftBodyDataDEPRECATED(void**, void*, void*, PxSoftBodyGpuDataFlag::Enum, unsigned int, unsigned int, CUevent*, CUevent*) {}

// ---- PxgFEMClothCore ----
void PxgFEMClothCore::constraintPrep(PxgDevicePointer<PxgPrePrepDesc>, PxgDevicePointer<PxgConstraintPrepareDesc>, float, PxgDevicePointer<PxgSolverSharedDescBase>, CUstream*, unsigned int, unsigned int) {}
void PxgFEMClothCore::solve(PxgDevicePointer<PxgPrePrepDesc>, PxgDevicePointer<PxgSolverCoreDesc>, PxgDevicePointer<PxgSolverSharedDescBase>, PxgDevicePointer<PxgArticulationCoreDesc>, float, CUstream*, unsigned int, unsigned int, bool, const PxVec3&) {}
void PxgFEMClothCore::resetClothVsNonclothContactCounts() {}
void PxgFEMClothCore::sortContacts(unsigned int) {}
void PxgFEMClothCore::syncCloths() {}
void PxgFEMClothCore::checkBufferOverflows() {}
void PxgFEMClothCore::createActivatedDeactivatedLists() {}
void PxgFEMClothCore::preIteration() {}
void PxgFEMClothCore::finalizeVelocities(float) {}
void PxgFEMClothCore::preIntegrateSystems(unsigned int, const PxVec3&, float) {}
void PxgFEMClothCore::refitBound(unsigned int, CUstream*) {}
void PxgFEMClothCore::updateUserData(PxArray<PxgFEMCloth, PxVirtualAllocator>&, PxArray<unsigned int, PxReflectionAllocator<unsigned int> >&, const unsigned int*, unsigned int, void**) {}
void PxgFEMClothCore::partitionTriangleSimData(PxgFEMCloth&, PxgFEMClothData&, PxArray<unsigned int, PxReflectionAllocator<unsigned int> >&, const PxArray<unsigned int, PxReflectionAllocator<unsigned int> >&, PxsHeapMemoryAllocator*) {}
void PxgFEMClothCore::partitionTrianglePairSimData(PxgFEMCloth&, PxgFEMClothData&, unsigned int, PxArray<unsigned int, PxReflectionAllocator<unsigned int> >&, const PxArray<unsigned int, PxReflectionAllocator<unsigned int> >&, const PxArray<uint4, PxReflectionAllocator<uint4> >&, bool, PxsHeapMemoryAllocator*) {}

// ---- PxgParticleSystemCore ----
void PxgParticleSystemCore::resetContactCounts() {}
void PxgParticleSystemCore::sortContacts(unsigned int) {}
void PxgParticleSystemCore::updateParticleSystemData(PxgParticleSystem&, Dy::ParticleSystemCore&) {}

// ---- PxgPBDParticleSystemCore ----
void PxgPBDParticleSystemCore::applyParticleBufferDataDEPRECATED(const unsigned int*, const PxGpuParticleBufferIndexPair*, const PxFlags<PxParticleBufferFlag::Enum, unsigned int>*, unsigned int, CUevent*, CUevent*) {}

// ---- PxgFEMCore ----
void PxgFEMCore::copyContactCountsToHost() {}
void PxgFEMCore::reserveRigidDeltaVelBuf(unsigned int) {}

// ---- PxgSoftBody ----
void PxgSoftBody::deallocate(PxsHeapMemoryAllocator*) {}

// ---- PxgFEMCloth ----
void PxgFEMCloth::deallocate(PxsHeapMemoryAllocator*) {}

// ---- PxgSoftBodyUtil ----
PxU64 PxgSoftBodyUtil::computeTetMeshByteSize(const Gu::BVTetrahedronMesh*) { return 0; }
void PxgSoftBodyUtil::initialTetData(PxgSoftBody&, const Gu::BVTetrahedronMesh*, const Gu::TetrahedronMesh*, const Gu::DeformableVolumeAuxData*, const PxU16*, PxsHeapMemoryAllocator*) {}
void PxgSoftBodyUtil::loadOutTetMesh(void*, const Gu::BVTetrahedronMesh*) {}

// ---- PxgFEMClothUtil ----
PxU64 PxgFEMClothUtil::computeTriangleMeshByteSize(const Gu::TriangleMesh*) { return 0; }
void PxgFEMClothUtil::loadOutTriangleMesh(void*, const Gu::TriangleMesh*) {}
void PxgFEMClothUtil::initialTriangleData(PxgFEMCloth&, PxArray<uint2, PxReflectionAllocator<uint2> >&, PxArray<uint4, PxReflectionAllocator<uint4> >&, const Gu::TriangleMesh*, const PxU16*, PxsDeformableSurfaceMaterialData*, unsigned int, PxsHeapMemoryAllocator*) {}
void PxgFEMClothUtil::categorizeClothConstraints(PxArray<unsigned int, PxReflectionAllocator<unsigned int> >&, PxArray<unsigned int, PxReflectionAllocator<unsigned int> >&, PxArray<unsigned int, PxReflectionAllocator<unsigned int> >&, PxgFEMCloth&, const PxArray<uint2, PxReflectionAllocator<uint2> >&) {}
void PxgFEMClothUtil::computeNonSharedTriangleConfiguration(PxgFEMCloth&, const PxArray<unsigned int, PxReflectionAllocator<unsigned int> >&, const PxArray<unsigned int, PxReflectionAllocator<unsigned int> >&, const Gu::TriangleMesh*) {}
void PxgFEMClothUtil::computeTrianglePairConfiguration(PxgFEMCloth&, PxArray<uint2, PxReflectionAllocator<uint2> >&, const PxArray<uint4, PxReflectionAllocator<uint4> >&, const PxArray<unsigned int, PxReflectionAllocator<unsigned int> >&, const PxArray<unsigned int, PxReflectionAllocator<unsigned int> >&, const Gu::TriangleMesh*, const PxsDeformableSurfaceMaterialData*, bool, bool) {}
