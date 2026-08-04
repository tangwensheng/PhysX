// PhysX DCU smoke test: minimal GPU deformable volume path.

#include "PxPhysicsAPI.h"
#include "cudamanager/PxCudaContextManager.h"
#include "gpu/PxGpu.h"
#include "extensions/PxDeformableVolumeExt.h"
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <thread>

using namespace physx;

static bool finiteVec(const PxVec3& v)
{
    return std::isfinite(v.x) && std::isfinite(v.y) && std::isfinite(v.z);
}

int main(int argc, char** argv)
{
    int steps = 60;
    PxU32 voxels = 4;
    bool addPlane = true;
    bool cleanExit = false;
    bool printAllSteps = false;
    PxReal height = 2.0f;
    PxReal planeY = 0.0f;
    int cleanExitWaitSeconds = 5;

    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--voxels") == 0 && i + 1 < argc) {
            voxels = PxU32(std::atoi(argv[++i]));
        } else if (std::strcmp(argv[i], "--height") == 0 && i + 1 < argc) {
            height = PxReal(std::atof(argv[++i]));
        } else if (std::strcmp(argv[i], "--plane-y") == 0 && i + 1 < argc) {
            planeY = PxReal(std::atof(argv[++i]));
        } else if (std::strcmp(argv[i], "--print-all-steps") == 0) {
            printAllSteps = true;
        } else if (std::strcmp(argv[i], "--no-plane") == 0) {
            addPlane = false;
        } else if (std::strcmp(argv[i], "cleanexit") == 0 || std::strcmp(argv[i], "--cleanexit") == 0) {
            cleanExit = true;
        } else if (std::strcmp(argv[i], "--cleanexit-wait") == 0 && i + 1 < argc) {
            cleanExit = true;
            cleanExitWaitSeconds = std::atoi(argv[++i]);
            if (cleanExitWaitSeconds < 0)
                cleanExitWaitSeconds = 0;
        } else {
            steps = std::atoi(argv[i]);
        }
    }
    if (steps < 0)
        steps = 60;
    if (voxels < 2)
        voxels = 2;

    printf("========================================\n");
    printf(" PhysX DCU Smoke - Deformable Volume\n");
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
    sd.flags |= PxSceneFlag::eENABLE_PCM;
    sd.gpuMaxNumPartitions = 8;
    sd.gpuDynamicsConfig.heapCapacity = 512u * 1024u * 1024u;
    sd.gpuDynamicsConfig.tempBufferCapacity = 128u * 1024u * 1024u;
    sd.gpuDynamicsConfig.collisionStackSize = 256u * 1024u * 1024u;

    printf("Creating GPU dynamics scene...\n");
    fflush(stdout);
    PxScene* scene = phy->createScene(sd);
    if (!scene) {
        printf("FAIL: createScene returned null\n");
        return 3;
    }

    PxMaterial* rigidMat = phy->createMaterial(0.5f, 0.5f, 0.0f);
    if (addPlane) {
        printf("Adding rigid plane at y=%.3f...\n", planeY);
        fflush(stdout);
        scene->addActor(*PxCreatePlane(*phy, PxPlane(0, 1, 0, -planeY), *rigidMat));
    } else {
        printf("Skipping rigid plane (--no-plane).\n");
        fflush(stdout);
    }

    printf("Creating deformable volume material...\n");
    fflush(stdout);
    PxDeformableVolumeMaterial* volMat = phy->createDeformableVolumeMaterial(2.0e5f, 0.3f, 0.1f);
    if (!volMat) {
        printf("FAIL: createDeformableVolumeMaterial returned null\n");
        return 4;
    }

    printf("Creating deformable volume box: voxels=%u height=%.3f...\n", voxels, height);
    fflush(stdout);
    PxDeformableVolume* volume = PxDeformableVolumeExt::createDeformableVolumeBox(
        PxTransform(PxVec3(0.0f, height, 0.0f)),
        PxVec3(1.0f, 1.0f, 1.0f),
        *volMat,
        *gpuMgr,
        0.0f,
        100.0f,
        voxels,
        1.0f);
    printf("createDeformableVolumeBox returned %p\n", static_cast<void*>(volume));
    fflush(stdout);
    if (!volume) {
        printf("FAIL: createDeformableVolumeBox returned null\n");
        return 5;
    }

    volume->setDeformableBodyFlag(PxDeformableBodyFlag::eDISABLE_SELF_COLLISION, true);
    volume->setSolverIterationCounts(30);

    printf("Adding deformable volume actor...\n");
    fflush(stdout);
    scene->addActor(*volume);

    printf("Starting simulation: steps=%d voxels=%u plane=%s height=%.3f planeY=%.3f\n",
           steps, voxels, addPlane ? "yes" : "no", height, planeY);
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
        if (printAllSteps) {
            PxBounds3 stepBounds = volume->getWorldBounds(1.0f);
            printf("Step %d bounds min=(%.6f %.6f %.6f) max=(%.6f %.6f %.6f)\n",
                   i + 1,
                   stepBounds.minimum.x, stepBounds.minimum.y, stepBounds.minimum.z,
                   stepBounds.maximum.x, stepBounds.maximum.y, stepBounds.maximum.z);
            fflush(stdout);
        }
    }

    PxBounds3 bounds = volume->getWorldBounds(1.0f);
    const bool pass = bounds.isValid() && finiteVec(bounds.minimum) && finiteVec(bounds.maximum) && bounds.minimum.y > -5.0f && bounds.maximum.y < 6.0f;
    printf("Volume bounds min=(%.6f %.6f %.6f) max=(%.6f %.6f %.6f)\n",
           bounds.minimum.x, bounds.minimum.y, bounds.minimum.z, bounds.maximum.x, bounds.maximum.y, bounds.maximum.z);
    printf("VERDICT: %s\n", pass ? "PASS" : "FAIL");
    fflush(stdout);

    printf("Releasing scene...\n");
    fflush(stdout);
    scene->release();
    printf("Releasing dispatcher...\n");
    fflush(stdout);
    dsp->release();
    printf("Releasing GPU manager...\n");
    fflush(stdout);
    gpuMgr->release();
    printf("Releasing physics...\n");
    fflush(stdout);
    phy->release();
    printf("Releasing foundation...\n");
    fflush(stdout);
    fnd->release();
    printf("Release finished.\n");
    fflush(stdout);

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
