// PhysX DCU smoke test: reduced-coordinate articulation in a GPU scene.

#include "PxPhysicsAPI.h"
#include "cudamanager/PxCudaContextManager.h"
#include "gpu/PxGpu.h"
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>

using namespace physx;

static bool finiteVec(const PxVec3& v)
{
    return std::isfinite(v.x) && std::isfinite(v.y) && std::isfinite(v.z);
}

int main(int argc, char** argv)
{
    int steps = 180;
    bool useTgs = false;
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "tgs") == 0 || std::strcmp(argv[i], "TGS") == 0) {
            useTgs = true;
        } else if (std::strcmp(argv[i], "pgs") == 0 || std::strcmp(argv[i], "PGS") == 0) {
            useTgs = false;
        } else {
            steps = std::atoi(argv[i]);
        }
    }
    if (steps <= 0)
        steps = 180;

    printf("========================================\n");
    printf(" PhysX DCU Smoke - Articulation (%s)\n", useTgs ? "TGS" : "PGS");
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

    printf("Scene created. Adding ground...\n");
    fflush(stdout);
    PxMaterial* mat = phy->createMaterial(0.5f, 0.5f, 0.2f);
    scene->addActor(*PxCreatePlane(*phy, PxPlane(0, 1, 0, 0), *mat));
    printf("Ground added. Creating articulation...\n");
    fflush(stdout);

    PxArticulationReducedCoordinate* art = phy->createArticulationReducedCoordinate();
    if (!art) {
        printf("FAIL: createArticulationReducedCoordinate returned null\n");
        scene->release();
        dsp->release();
        gpuMgr->release();
        phy->release();
        fnd->release();
        return 4;
    }

    printf("Articulation created. Setting flags/shapes...\n");
    fflush(stdout);
    art->setArticulationFlag(PxArticulationFlag::eFIX_BASE, true);

    PxShape* baseShape = phy->createShape(PxBoxGeometry(0.25f, 0.25f, 0.25f), *mat);
    PxShape* linkShape = phy->createShape(PxCapsuleGeometry(0.12f, 0.45f), *mat);

    printf("Creating base link...\n");
    fflush(stdout);
    PxArticulationLink* base = art->createLink(nullptr, PxTransform(PxVec3(0.0f, 4.0f, 0.0f)));
    base->attachShape(*baseShape);
    PxRigidBodyExt::updateMassAndInertia(*base, 1.0f);

    printf("Base link created. Creating child links...\n");
    fflush(stdout);
    PxArticulationLink* prev = base;
    PxArticulationLink* last = base;
    for (int i = 0; i < 4; ++i) {
        PxArticulationLink* link = art->createLink(prev, PxTransform(PxVec3(0.0f, 3.5f - 0.55f * i, 0.0f)));
        link->attachShape(*linkShape);
        PxRigidBodyExt::updateMassAndInertia(*link, 1.0f);

        PxArticulationJointReducedCoordinate* joint = link->getInboundJoint();
        joint->setJointType(PxArticulationJointType::eREVOLUTE);
        joint->setParentPose(PxTransform(PxVec3(0.0f, -0.3f, 0.0f)));
        joint->setChildPose(PxTransform(PxVec3(0.0f, 0.3f, 0.0f)));
        joint->setMotion(PxArticulationAxis::eTWIST, PxArticulationMotion::eFREE);
        joint->setLimitParams(PxArticulationAxis::eTWIST, PxArticulationLimit(-PxPi / 4.0f, PxPi / 4.0f));
        joint->setDriveParams(PxArticulationAxis::eTWIST, PxArticulationDrive(10.0f, 1.0f, 100.0f, PxArticulationDriveType::eFORCE));

        prev = link;
        last = link;
    }

    printf("Child links created. Adding articulation to scene...\n");
    fflush(stdout);
    if (!scene->addArticulation(*art)) {
        printf("FAIL: addArticulation returned false\n");
        art->release();
        scene->release();
        dsp->release();
        gpuMgr->release();
        phy->release();
        fnd->release();
        return 5;
    }

    printf("Articulation added. Starting simulation...\n");
    fflush(stdout);
    for (int i = 0; i < steps; ++i) {
        if (i < 5 || i == steps - 1) {
            printf("Simulate step %d/%d...\n", i + 1, steps);
            fflush(stdout);
        }
        scene->simulate(1.0f / 60.0f);
        if (i < 5 || i == steps - 1) {
            printf("Fetch step %d/%d...\n", i + 1, steps);
            fflush(stdout);
        }
        scene->fetchResults(true);
    }
    printf("Simulation finished. Reading poses...\n");
    fflush(stdout);

    PxVec3 basePos = base->getGlobalPose().p;
    PxVec3 lastPos = last->getGlobalPose().p;
    bool pass = finiteVec(basePos) && finiteVec(lastPos) && lastPos.y > 0.0f && lastPos.y < 4.0f;

    printf("Steps: %d\n", steps);
    printf("Base position: [%.6f, %.6f, %.6f]\n", basePos.x, basePos.y, basePos.z);
    printf("Last position: [%.6f, %.6f, %.6f]\n", lastPos.x, lastPos.y, lastPos.z);
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
    return pass ? 0 : 1;
}
