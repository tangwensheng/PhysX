// PhysX DCU smoke test: minimal PBD particle cloth path.

#include "PxPhysicsAPI.h"
#include "cudamanager/PxCudaContextManager.h"
#include "extensions/PxParticleExt.h"
#include "extensions/PxCudaHelpersExt.h"
#include "gpu/PxGpu.h"
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <thread>
#include <vector>

using namespace physx;

static bool finiteVec(const PxVec3& v)
{
    return std::isfinite(v.x) && std::isfinite(v.y) && std::isfinite(v.z);
}

static PxParticleSpring makeSpring(PxU32 a, PxU32 b, PxReal length, PxReal stiffness, PxReal damping)
{
    PxParticleSpring s;
    s.ind0 = a;
    s.ind1 = b;
    s.length = length;
    s.stiffness = stiffness;
    s.damping = damping;
    s.pad = 0.0f;
    return s;
}

int main(int argc, char** argv)
{
    int steps = 60;
    PxU32 dim = 8;
    bool cleanExit = false;
    bool forceGpuBroadphase = false;
    bool printAllSteps = false;
    int cleanExitWaitSeconds = 5;
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--dim") == 0 && i + 1 < argc) {
            dim = PxU32(std::atoi(argv[++i]));
        } else if (std::strcmp(argv[i], "--force-gpu-bp") == 0) {
            forceGpuBroadphase = true;
        } else if (std::strcmp(argv[i], "--print-all-steps") == 0) {
            printAllSteps = true;
        } else if (std::strcmp(argv[i], "--cleanexit-wait") == 0 && i + 1 < argc) {
            cleanExit = true;
            cleanExitWaitSeconds = std::atoi(argv[++i]);
            if (cleanExitWaitSeconds < 0)
                cleanExitWaitSeconds = 0;
        } else if (std::strcmp(argv[i], "--cleanexit") == 0 || std::strcmp(argv[i], "cleanexit") == 0) {
            cleanExit = true;
        } else {
            steps = std::atoi(argv[i]);
        }
    }
    if (steps < 0)
        steps = 60;
    if (dim < 2)
        dim = 2;

    printf("========================================\n");
    printf(" PhysX DCU Smoke - PBD Cloth\n");
    printf("========================================\n\n");

    static PxDefaultErrorCallback gErr;
    static PxDefaultAllocator gAlloc;
    PxFoundation* fnd = PxCreateFoundation(PX_PHYSICS_VERSION, gAlloc, gErr);
    PxPhysics* phy = PxCreatePhysics(PX_PHYSICS_VERSION, *fnd, PxTolerancesScale());

    PxCudaContextManagerDesc gpuDesc;
    gpuDesc.deviceOrdinal = 0;
    PxCudaContextManager* gpuMgr = PxCreateCudaContextManager(*fnd, gpuDesc, nullptr, false);
    const bool gpuOk = gpuMgr && gpuMgr->contextIsValid();
    printf("GPU: %s - %s\n", gpuOk ? gpuMgr->getDeviceName() : "NONE", gpuOk ? "enabled" : "disabled");
    if (!gpuOk)
        return 2;

    PxSceneDesc sd(phy->getTolerancesScale());
    sd.gravity = PxVec3(0.0f, -9.81f, 0.0f);
    PxDefaultCpuDispatcher* dsp = PxDefaultCpuDispatcherCreate(0);
    sd.cpuDispatcher = dsp;
    sd.filterShader = PxDefaultSimulationFilterShader;
    sd.cudaContextManager = gpuMgr;
    sd.flags |= PxSceneFlag::eENABLE_GPU_DYNAMICS;
    sd.flags |= PxSceneFlag::eENABLE_PCM;
    if (forceGpuBroadphase)
        sd.broadPhaseType = PxBroadPhaseType::eGPU;
    sd.gpuMaxNumPartitions = 8;
    sd.gpuDynamicsConfig.heapCapacity = 512u * 1024u * 1024u;
    sd.gpuDynamicsConfig.tempBufferCapacity = 128u * 1024u * 1024u;
    sd.gpuDynamicsConfig.collisionStackSize = 256u * 1024u * 1024u;

    printf("Creating GPU dynamics scene: broadphase=%s...\n", forceGpuBroadphase ? "eGPU" : "default");
    fflush(stdout);
    PxScene* scene = phy->createScene(sd);
    if (!scene) {
        printf("FAIL: createScene returned null\n");
        return 3;
    }

    const PxU32 numParticles = dim * dim;
    const PxU32 numTriangles = 2 * (dim - 1) * (dim - 1);
    const PxU32 numSprings = dim * (dim - 1) * 2;
    const PxReal spacing = 0.1f;
    const PxReal restOffset = 0.05f;
    const PxReal totalMass = 10.0f;
    const PxReal invParticleMass = PxReal(numParticles) / totalMass;

    printf("Creating PBD material and particle system: dim=%u particles=%u triangles=%u springs=%u...\n",
           dim, numParticles, numTriangles, numSprings);
    fflush(stdout);
    PxPBDMaterial* mat = phy->createPBDMaterial(0.8f, 0.05f, 1e+6f, 0.001f, 0.5f, 0.005f, 0.05f, 0.0f, 0.0f);
    PxPBDParticleSystem* ps = phy->createPBDParticleSystem(*gpuMgr);
    if (!mat || !ps) {
        printf("FAIL: create PBD material/system failed mat=%p ps=%p\n", static_cast<void*>(mat), static_cast<void*>(ps));
        return 4;
    }
    ps->setRestOffset(restOffset);
    ps->setContactOffset(restOffset + 0.02f);
    ps->setParticleContactOffset(restOffset + 0.02f);
    ps->setSolidRestOffset(restOffset);
    ps->setFluidRestOffset(0.0f);
    scene->addActor(*ps);
    const PxU32 phaseValue = ps->createPhase(mat, PxParticlePhaseFlags(PxParticlePhaseFlag::eParticlePhaseSelfCollideFilter | PxParticlePhaseFlag::eParticlePhaseSelfCollide));

    printf("Allocating cloth helper and pinned buffers...\n");
    fflush(stdout);
    ExtGpu::PxParticleClothBufferHelper* helper = ExtGpu::PxCreateParticleClothBufferHelper(1, numTriangles, numSprings, numParticles, gpuMgr);
    PxU32* phases = PX_EXT_PINNED_MEMORY_ALLOC(PxU32, *gpuMgr, numParticles);
    PxVec4* positions = PX_EXT_PINNED_MEMORY_ALLOC(PxVec4, *gpuMgr, numParticles);
    PxVec4* velocities = PX_EXT_PINNED_MEMORY_ALLOC(PxVec4, *gpuMgr, numParticles);
    std::vector<PxParticleSpring> springs;
    std::vector<PxU32> triangles;
    springs.reserve(numSprings);
    triangles.reserve(numTriangles * 3u);

    for (PxU32 y = 0; y < dim; ++y) {
        for (PxU32 x = 0; x < dim; ++x) {
            const PxU32 id = y * dim + x;
            positions[id] = PxVec4(PxReal(x) * spacing, 2.0f, PxReal(y) * spacing, invParticleMass);
            velocities[id] = PxVec4(0.0f);
            phases[id] = phaseValue;
            if (x > 0)
                springs.push_back(makeSpring(id, id - 1, spacing, 1.0f, 0.0f));
            if (y > 0)
                springs.push_back(makeSpring(id, id - dim, spacing, 1.0f, 0.0f));
        }
    }
    for (PxU32 y = 0; y + 1 < dim; ++y) {
        for (PxU32 x = 0; x + 1 < dim; ++x) {
            const PxU32 i0 = y * dim + x;
            const PxU32 i1 = i0 + 1;
            const PxU32 i2 = i0 + dim;
            const PxU32 i3 = i2 + 1;
            triangles.push_back(i0); triangles.push_back(i2); triangles.push_back(i1);
            triangles.push_back(i1); triangles.push_back(i2); triangles.push_back(i3);
        }
    }

    printf("Adding cloth desc: springs=%zu triangles=%zu...\n", springs.size(), triangles.size() / 3);
    fflush(stdout);
    helper->addCloth(0.0f, 0.0f, 0.0f, triangles.data(), numTriangles, springs.data(), numSprings, positions, numParticles);

    ExtGpu::PxParticleBufferDesc desc;
    desc.maxParticles = numParticles;
    desc.numActiveParticles = numParticles;
    desc.positions = positions;
    desc.velocities = velocities;
    desc.phases = phases;

    printf("Partitioning cloth springs...\n");
    fflush(stdout);
    const PxParticleClothDesc& clothDesc = helper->getParticleClothDesc();
    PxParticleClothPreProcessor* pre = PxCreateParticleClothPreProcessor(gpuMgr);
    PxPartitionedParticleCloth output;
    pre->partitionSprings(clothDesc, output);
    pre->release();

    printf("Creating and populating particle cloth buffer...\n");
    fflush(stdout);
    PxParticleClothBuffer* clothBuffer = ExtGpu::PxCreateAndPopulateParticleClothBuffer(desc, clothDesc, output, gpuMgr);
    if (!clothBuffer) {
        printf("FAIL: PxCreateAndPopulateParticleClothBuffer returned null\n");
        return 5;
    }
    ps->addParticleBuffer(clothBuffer);
    helper->release();
    PX_EXT_PINNED_MEMORY_FREE(*gpuMgr, positions);
    PX_EXT_PINNED_MEMORY_FREE(*gpuMgr, velocities);
    PX_EXT_PINNED_MEMORY_FREE(*gpuMgr, phases);

    printf("Starting simulation: steps=%d dim=%u\n", steps, dim);
    fflush(stdout);
    for (int i = 0; i < steps; ++i) {
        if (printAllSteps || i < 10 || i == steps - 1) {
            printf("Simulate step %d/%d...\n", i + 1, steps);
            fflush(stdout);
        }
        scene->simulate(1.0f / 60.0f);
        if (printAllSteps || i < 10 || i == steps - 1) {
            printf("Fetch step %d/%d...\n", i + 1, steps);
            fflush(stdout);
        }
        scene->fetchResults(true);
    }

    PxBounds3 bounds = ps->getWorldBounds(1.0f);
    const bool pass = bounds.isValid() && finiteVec(bounds.minimum) && finiteVec(bounds.maximum);
    printf("PBD cloth system bounds min=(%.6f %.6f %.6f) max=(%.6f %.6f %.6f)\n",
           bounds.minimum.x, bounds.minimum.y, bounds.minimum.z, bounds.maximum.x, bounds.maximum.y, bounds.maximum.z);
    printf("VERDICT: %s\n", pass ? "PASS" : "FAIL");
    fflush(stdout);

    printf("Releasing scene...\n"); fflush(stdout); scene->release();
    printf("Releasing dispatcher...\n"); fflush(stdout); dsp->release();
    printf("Releasing GPU manager...\n"); fflush(stdout); gpuMgr->release();
    printf("Releasing physics...\n"); fflush(stdout); phy->release();
    printf("Releasing foundation...\n"); fflush(stdout); fnd->release();
    printf("Release finished.\n"); fflush(stdout);
    if (cleanExit) {
        printf("Clean exit after explicit PhysX release (--cleanexit), waiting %d seconds for DCU/HSA cleanup...\n", cleanExitWaitSeconds);
        fflush(stdout);
        if (cleanExitWaitSeconds > 0)
            std::this_thread::sleep_for(std::chrono::seconds(cleanExitWaitSeconds));
        printf("Clean exit wait finished.\n");
        fflush(stdout);
        std::_Exit(pass ? 0 : 1);
    }
    return pass ? 0 : 1;
}
