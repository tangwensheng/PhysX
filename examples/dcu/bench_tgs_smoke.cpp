// PhysX DCU smoke test: GPU dynamics with TGS solver.

#include "PxPhysicsAPI.h"
#include "cudamanager/PxCudaContextManager.h"
#include "gpu/PxGpu.h"
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <thread>
#include <vector>

using namespace physx;

static bool isFinitePose(const PxRigidDynamic* body)
{
    const PxTransform pose = body->getGlobalPose();
    const PxVec3 p = pose.p;
    const PxVec3 v = body->getLinearVelocity();
    return std::isfinite(p.x) && std::isfinite(p.y) && std::isfinite(p.z) &&
           std::isfinite(v.x) && std::isfinite(v.y) && std::isfinite(v.z);
}

int main(int argc, char** argv)
{
    int steps = 180;
    bool useTgs = true;
    bool printAllSteps = false;
    bool cleanExit = false;
    int cleanExitWaitSeconds = 5;
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "pgs") == 0 || std::strcmp(argv[i], "PGS") == 0) {
            useTgs = false;
        } else if (std::strcmp(argv[i], "tgs") == 0 || std::strcmp(argv[i], "TGS") == 0) {
            useTgs = true;
        } else if (std::strcmp(argv[i], "--print-all-steps") == 0) {
            printAllSteps = true;
        } else if (std::strcmp(argv[i], "cleanexit") == 0 || std::strcmp(argv[i], "--cleanexit") == 0) {
            cleanExit = true;
        } else if (std::strcmp(argv[i], "--cleanexit-wait") == 0 && i + 1 < argc) {
            cleanExit = true;
            cleanExitWaitSeconds = std::atoi(argv[++i]);
            if (cleanExitWaitSeconds < 0)
                cleanExitWaitSeconds = 0;
        } else {
            const int parsedSteps = std::atoi(argv[i]);
            if (parsedSteps > 0)
                steps = parsedSteps;
        }
    }
    if (steps <= 0)
        steps = 180;

    printf("========================================\n");
    printf(" PhysX DCU Smoke - %s Solver\n", useTgs ? "TGS" : "PGS");
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
    sd.cudaContextManager = gpuMgr;
    PxDefaultCpuDispatcher* dsp = PxDefaultCpuDispatcherCreate(0);
    sd.cpuDispatcher = dsp;
    sd.filterShader = PxDefaultSimulationFilterShader;
    sd.flags |= PxSceneFlag::eENABLE_GPU_DYNAMICS;
    // Match the stable tower benchmark first: GPU dynamics on, broadphase left
    // at the scene default. Forcing eGPU currently hits a DCU launch-bound
    // issue in computeStartAndActiveRegionHistogram.
    sd.solverType = useTgs ? PxSolverType::eTGS : PxSolverType::ePGS;
    sd.gpuMaxNumPartitions = 8;
    sd.gpuDynamicsConfig.maxRigidContactCount = 1024u * 1024u * 8u;
    sd.gpuDynamicsConfig.maxRigidPatchCount = 1024u * 1024u;
    sd.gpuDynamicsConfig.foundLostPairsCapacity = 1024u * 1024u * 2u;
    sd.gpuDynamicsConfig.heapCapacity = 512u * 1024u * 1024u;
    sd.gpuDynamicsConfig.collisionStackSize = 256u * 1024u * 1024u;
    sd.gpuDynamicsConfig.tempBufferCapacity = 128u * 1024u * 1024u;

    printf("Creating scene with solver=%s...\n", useTgs ? "TGS" : "PGS");
    fflush(stdout);
    PxScene* scene = phy->createScene(sd);
    if (!scene) {
        printf("FAIL: createScene returned null\n");
        dsp->release();
        gpuMgr->release();
        phy->release();
        fnd->release();
        return 3;
    }
    printf("Scene solver type: %s\n", scene->getSolverType() == PxSolverType::eTGS ? "TGS" : "PGS");

    PxMaterial* mat = phy->createMaterial(0.5f, 0.5f, 0.2f);
    scene->addActor(*PxCreatePlane(*phy, PxPlane(0, 1, 0, 0), *mat));

    PxShape* boxShape = phy->createShape(PxBoxGeometry(0.5f, 0.5f, 0.5f), *mat);
    std::vector<PxRigidDynamic*> bodies;
    for (int y = 0; y < 8; ++y) {
        for (int x = 0; x < 8; ++x) {
            PxVec3 pos((x - 3.5f) * 1.05f, 0.55f + y * 1.02f, 0.0f);
            PxRigidDynamic* b = phy->createRigidDynamic(PxTransform(pos));
            b->attachShape(*boxShape);
            PxRigidBodyExt::updateMassAndInertia(*b, 1.0f);
            scene->addActor(*b);
            bodies.push_back(b);
        }
    }

    float minY = 1e30f;
    float maxY = -1e30f;
    int bad = 0;
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

    for (PxRigidDynamic* b : bodies) {
        const PxVec3 p = b->getGlobalPose().p;
        if (!isFinitePose(b))
            bad++;
        if (p.y < minY)
            minY = p.y;
        if (p.y > maxY)
            maxY = p.y;
    }

    printf("Steps: %d\n", steps);
    printf("Bodies: %zu\n", bodies.size());
    printf("Height range: [%.6f, %.6f]\n", minY, maxY);
    printf("Bad bodies: %d\n", bad);

    const bool pass = bad == 0 && minY > -0.25f && maxY < 20.0f;
    printf("VERDICT: %s\n", pass ? "PASS" : "FAIL");

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
