// HipPhysXGpu — DCU PxPhysXGpu: creates real GPU objects via the existing
// Pxg* host-side implementations (PxgCudaBroadPhaseSap, PxgDynamicsContext,
// PxgNphaseImplementationContext, PxgSimulationController, etc.)
//
// This replaces the NVIDIA PxgPhysXGpu.cpp factory for DCU builds.

#include "PxPhysXGpu.h"
#include "PxsKernelWrangler.h"
#include "PxsHeapMemoryAllocator.h"
#include "PxsMemoryManager.h"
#include "cudamanager/PxCudaContextManager.h"

// Real GPU object headers (NVIDIA source, compiled for DCU)
#include "PxgMemoryManager.h"
#include "PxgHeapMemAllocator.h"
#include "PxgKernelWrangler.h"
#include "PxgCudaBroadPhaseSap.h"
#include "PxgAABBManager.h"
#include "PxgNphaseImplementationContext.h"
#include "PxgSimulationController.h"
#include "PxgDynamicsContext.h"
#include "PxgTGSDynamicsContext.h"
#include "PxgBroadPhase.h"
#include "PxgCommon.h"
#include "PxgNarrowphase.h"
#include "PxgSolver.h"

#include "foundation/PxHashMap.h"
#include "foundation/PxFoundation.h"
#include <new>
#include <cstdio>

#define DCU_TRACE(fmt, ...) fprintf(stderr, "[DCU GPU] " fmt "\n", ##__VA_ARGS__)

#if PX_SUPPORT_GPU_PHYSX
namespace physx {

// Forward declarations
PxsMemoryManager* createPxgMemoryManager(PxCudaContextManager* cudaContextManager);

// Per-context KernelWrangler map (same pattern as PxgPhysXGpu)
struct HipKernelWranglerMap
{
	typedef PxHashMap<PxCudaContextManager*, PxgCudaKernelWranglerManager*> Map;
	Map                       mMap;
	const Map::Entry*         find(PxCudaContextManager* k) const  { return mMap.find(k); }
	void                      insert(PxCudaContextManager* k, PxgCudaKernelWranglerManager* v)
	                                                               { mMap.insert(k, v); }
	Map::Iterator             getIterator()                        { return mMap.getIterator(); }
	void                      clear()                              { mMap.clear(); }
};

// NOTE: PxPhysXGpu does NOT inherit from PxUserAllocated, so we cannot use
// PX_NEW / PX_DELETE_THIS.  Plain new/delete.
struct HipPhysXGpu final : public PxPhysXGpu
{
	// ---- Lifecycle ----
	void release() override {
		for(HipKernelWranglerMap::Map::Iterator iter = mKernelWranglerInstances.getIterator();
		    !iter.done(); ++iter)
		{
			if(iter->second) PX_DELETE(iter->second);
		}
		mKernelWranglerInstances.clear();
		delete this;
	}

	static HipPhysXGpu& getInstance() {
		static HipPhysXGpu* inst = nullptr;
		if(!inst) inst = new (std::nothrow) HipPhysXGpu;
		return *inst;
	}

	// ---- Memory Manager ----
	PxsMemoryManager* createGpuMemoryManager(PxCudaContextManager* ctx) override {
		DCU_TRACE("createGpuMemoryManager");
		return createPxgMemoryManager(ctx);
	}

	// ---- Kernel Wrangler Manager ----
	PxsKernelWranglerManager* getGpuKernelWranglerManager(PxCudaContextManager* ctx) override {
		if(!ctx) return nullptr;
		DCU_TRACE("getGpuKernelWranglerManager");

		const HipKernelWranglerMap::Map::Entry* entry = mKernelWranglerInstances.find(ctx);
		if(entry) return entry->second;

		DCU_TRACE("  creating PxgCudaKernelWranglerManager...");
		PxgCudaKernelWranglerManager* wrangler =
			PX_NEW(PxgCudaKernelWranglerManager)(*ctx, *PxGetErrorCallback());
		DCU_TRACE("  done");
		mKernelWranglerInstances.insert(ctx, wrangler);
		return wrangler;
	}

	// ---- Heap Memory Allocator ----
	PxsHeapMemoryAllocatorManager* createGpuHeapMemoryAllocatorManager(
		PxU32 heapCapacity, PxsMemoryManager* memoryManager, PxU32) override
	{
		DCU_TRACE("createGpuHeapMemoryAllocatorManager cap=%u", heapCapacity);
		return PX_NEW(PxgHeapMemoryAllocatorManager)(heapCapacity, memoryManager);
	}

	// ---- GPU BroadPhase ----
	Bp::BroadPhase* createGpuBroadPhase(const PxGpuBroadPhaseDesc& desc,
		PxsKernelWranglerManager* gpuKernelWrangler, PxCudaContextManager* cudaContextManager,
		PxU32 gpuComputeVersion, const PxGpuDynamicsMemoryConfig& config,
		PxsHeapMemoryAllocatorManager* heapMemoryManager, PxU64 contextID) override
	{
		DCU_TRACE("createGpuBroadPhase");
		if(gpuComputeVersion != 0) return nullptr;
		return PX_PLACEMENT_NEW(
			PX_ALLOC(sizeof(PxgCudaBroadPhaseSap), "PxgCudaBroadPhaseSap"),
			PxgCudaBroadPhaseSap)(desc,
			static_cast<PxgCudaKernelWranglerManager*>(gpuKernelWrangler),
			cudaContextManager, config,
			static_cast<PxgHeapMemoryAllocatorManager*>(heapMemoryManager), contextID);
	}

	// ---- GPU AABB Manager ----
	Bp::AABBManagerBase* createGpuAABBManager(
		PxsKernelWranglerManager* gpuKernelWrangler,
		PxCudaContextManager* cudaContextManager, PxU32 gpuComputeVersion,
		const PxGpuDynamicsMemoryConfig& config, PxsHeapMemoryAllocatorManager* heapMemoryManager,
		Bp::BroadPhase& bp, Bp::BoundsArray& boundsArray, PxFloatArrayPinnedSafe& contactDistance,
		PxU32 maxNbAggregates, PxU32 maxNbShapes, PxVirtualAllocator& allocator, PxU64 contextID,
		PxPairFilteringMode::Enum kineKineFilteringMode,
		PxPairFilteringMode::Enum staticKineFilteringMode) override
	{
		DCU_TRACE("createGpuAABBManager");
		if(gpuComputeVersion != 0) return nullptr;
		return PX_PLACEMENT_NEW(
			PX_ALLOC(sizeof(PxgAABBManager), "PxgAABBManager"), PxgAABBManager)(
			static_cast<PxgCudaKernelWranglerManager*>(gpuKernelWrangler),
			cudaContextManager, static_cast<PxgHeapMemoryAllocatorManager*>(heapMemoryManager),
			config, bp, boundsArray, contactDistance, maxNbAggregates, maxNbShapes, allocator,
			contextID, kineKineFilteringMode, staticKineFilteringMode);
	}

	// ---- GPU Bounds Array ----
	Bp::BoundsArray* createGpuBounds(PxVirtualAllocator&) override { return nullptr; }

	// ---- GPU NarrowPhase Implementation Context ----
	PxvNphaseImplementationContext* createGpuNphaseImplementationContext(
		PxsContext& context, PxsKernelWranglerManager* gpuKernelWrangler,
		PxvNphaseImplementationFallback* fallbackForUnsupportedCMs,
		const PxGpuDynamicsMemoryConfig& gpuDynamicsConfig,
		void* contactStreamBase, void* patchStreamBase, void* forceAndIndiceStreamBase,
		PxBoundsArrayPinned& bounds, IG::IslandSim* islandSim, Dy::Context* dynamicsContext,
		PxU32, PxsHeapMemoryAllocatorManager* heapMemoryManager, bool useGPUBP) override
	{
		DCU_TRACE("createGpuNphaseImplementationContext");
		return PX_PLACEMENT_NEW(
			PX_ALLOC(sizeof(PxgNphaseImplementationContext), "PxgNphaseImplementationContext"),
			PxgNphaseImplementationContext)(context, gpuKernelWrangler,
			fallbackForUnsupportedCMs, gpuDynamicsConfig,
			contactStreamBase, patchStreamBase, forceAndIndiceStreamBase,
			bounds, islandSim, dynamicsContext,
			static_cast<PxgHeapMemoryAllocatorManager*>(heapMemoryManager), useGPUBP);
	}

	// ---- GPU Simulation Controller ----
	PxsSimulationController* createGpuSimulationController(
		PxsKernelWranglerManager* gpuWranglerManagers, PxCudaContextManager* cudaContextManager,
		Dy::Context* dynamicContext, PxvNphaseImplementationContext* npContext,
		Bp::BroadPhase* bp, bool useGpuBroadphase, PxsSimulationControllerCallback* callback,
		PxU32, PxsHeapMemoryAllocatorManager* heapMemoryManager,
		PxU32 maxSoftBodyContacts, PxU32 maxFemClothContacts,
		PxU32 maxParticleContacts, PxU32 collisionStackSizeBytes,
		bool enableBodyAccelerations) override
	{
		DCU_TRACE("createGpuSimulationController");
		return PX_PLACEMENT_NEW(
			PX_ALLOC(sizeof(PxgSimulationController), "PxgSimulationController"),
			PxgSimulationController)(gpuWranglerManagers, cudaContextManager,
			static_cast<PxgDynamicsContext*>(dynamicContext),
			static_cast<PxgNphaseImplementationContext*>(npContext),
			bp, useGpuBroadphase, callback,
			static_cast<PxgHeapMemoryAllocatorManager*>(heapMemoryManager),
			maxSoftBodyContacts, maxFemClothContacts, maxParticleContacts,
			collisionStackSizeBytes, enableBodyAccelerations);
	}

	// ---- GPU Dynamics Context ----
	Dy::Context* createGpuDynamicsContext(
		Cm::FlushPool& taskPool, PxsKernelWranglerManager* gpuKernelWrangler,
		PxCudaContextManager* cudaContextManager, const PxGpuDynamicsMemoryConfig& config,
		IG::SimpleIslandManager& islandManager,
		PxU32 maxNumPartitions, PxU32 maxNumStaticPartitions,
		bool enableStabilization, bool useEnhancedDeterminism,
		bool solveArticulationContactLast, PxReal maxBiasCoefficient,
		PxU32, PxvSimStats& simStats, PxsHeapMemoryAllocatorManager* heapMemoryManager,
		bool frictionEveryIteration, bool externalForcesEveryTgsIterationEnabled,
		PxSolverType::Enum solverType, PxReal lengthScale, bool enableDirectGPUAPI,
		PxU64 contextID, bool isResidualReportingEnabled) override
	{
		DCU_TRACE("createGpuDynamicsContext solverType=%d", (int)solverType);
		if(solverType == PxSolverType::eTGS)
		{
			return PX_PLACEMENT_NEW(
				PX_ALLOC(sizeof(PxgTGSDynamicsContext), "PxgTGSDynamicsContext"),
				PxgTGSDynamicsContext)(taskPool, gpuKernelWrangler, cudaContextManager,
				config, islandManager, maxNumPartitions, maxNumStaticPartitions,
				enableStabilization, useEnhancedDeterminism,
				solveArticulationContactLast, maxBiasCoefficient,
				simStats, static_cast<PxgHeapMemoryAllocatorManager*>(heapMemoryManager),
				externalForcesEveryTgsIterationEnabled, lengthScale, enableDirectGPUAPI,
				contextID, isResidualReportingEnabled);
		}

		return PX_PLACEMENT_NEW(
			PX_ALLOC(sizeof(PxgDynamicsContext), "PxgDynamicsContext"),
			PxgDynamicsContext)(taskPool, gpuKernelWrangler, cudaContextManager,
			config, islandManager, maxNumPartitions, maxNumStaticPartitions,
			enableStabilization, useEnhancedDeterminism,
			solveArticulationContactLast, maxBiasCoefficient,
			simStats, static_cast<PxgHeapMemoryAllocatorManager*>(heapMemoryManager),
			frictionEveryIteration, lengthScale, enableDirectGPUAPI,
			contextID, isResidualReportingEnabled);
	}

	// ---- Particle buffers (not yet implemented for DCU) ----
	PxsParticleBuffer*              createParticleBuffer(PxU32, PxU32, PxCudaContextManager&) override { return nullptr; }
	PxsParticleAndDiffuseBuffer*    createParticleAndDiffuseBuffer(PxU32, PxU32, PxU32, PxCudaContextManager&) override { return nullptr; }
	PxsParticleClothBuffer*         createParticleClothBuffer(PxU32, PxU32, PxU32, PxU32, PxU32, PxCudaContextManager&) override { return nullptr; }
	PxsParticleRigidBuffer*         createParticleRigidBuffer(PxU32, PxU32, PxU32, PxCudaContextManager&) override { return nullptr; }

private:
	HipKernelWranglerMap mKernelWranglerInstances;
};

} // namespace physx

// PxCreatePhysXGpu is declared with PX_C_EXPORT (extern "C") in PxPhysXGpu.h
// and must have C linkage for PxvGetPhysXGpu to resolve it.
extern "C" physx::PxPhysXGpu* PxCreatePhysXGpu() {
	fprintf(stderr, "[DCU GPU] PxCreatePhysXGpu called\n");
	return &physx::HipPhysXGpu::getInstance();
}
#endif // PX_SUPPORT_GPU_PHYSX
