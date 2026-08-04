// PhysX DCU smoke test: dynamic boxes colliding with a static triangle mesh.

#include "PxPhysicsAPI.h"
#include "cudamanager/PxCudaContextManager.h"
#include "gpu/PxGpu.h"
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <chrono>
#include <thread>
#include <vector>

using namespace physx;

static bool finiteVec(const PxVec3& v)
{
    return std::isfinite(v.x) && std::isfinite(v.y) && std::isfinite(v.z);
}

static PxTriangleMesh* createPlatformMesh(PxPhysics* phy)
{
    // A small non-flat platform made of 8 triangles.
    static PxVec3 verts[] = {
        PxVec3(-6.0f, 0.0f, -6.0f), PxVec3(0.0f, 0.0f, -6.0f), PxVec3(6.0f, 0.0f, -6.0f),
        PxVec3(-6.0f, 0.0f,  0.0f), PxVec3(0.0f, 0.6f,  0.0f), PxVec3(6.0f, 0.0f,  0.0f),
        PxVec3(-6.0f, 0.0f,  6.0f), PxVec3(0.0f, 0.0f,  6.0f), PxVec3(6.0f, 0.0f,  6.0f)
    };
    static PxU32 indices[] = {
        0, 3, 1,  1, 3, 4,
        1, 4, 2,  2, 4, 5,
        3, 6, 4,  4, 6, 7,
        4, 7, 5,  5, 7, 8
    };

    PxTriangleMeshDesc desc;
    desc.points.count = 9;
    desc.points.stride = sizeof(PxVec3);
    desc.points.data = verts;
    desc.triangles.count = 8;
    desc.triangles.stride = 3 * sizeof(PxU32);
    desc.triangles.data = indices;

    PxTolerancesScale scale;
    PxCookingParams params(scale);
    params.meshPreprocessParams |= PxMeshPreprocessingFlag::eWELD_VERTICES;
    params.meshWeldTolerance = 0.001f;

    return PxCreateTriangleMesh(params, desc, phy->getPhysicsInsertionCallback());
}

int main(int argc, char** argv)
{
    int steps = 240;
    bool cleanExit = false;
    int cleanExitWaitSeconds = 5;
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "cleanexit") == 0 || std::strcmp(argv[i], "--cleanexit") == 0) {
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
    if (steps <= 0)
        steps = 240;

    printf("========================================\n");
    printf(" PhysX DCU Smoke - Triangle Mesh Contact\n");
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
    sd.gpuMaxNumPartitions = 8;
    sd.gpuDynamicsConfig.maxRigidContactCount = 1024u * 1024u * 8u;
    sd.gpuDynamicsConfig.maxRigidPatchCount = 1024u * 1024u;
    sd.gpuDynamicsConfig.foundLostPairsCapacity = 1024u * 1024u * 2u;
    sd.gpuDynamicsConfig.heapCapacity = 512u * 1024u * 1024u;
    sd.gpuDynamicsConfig.collisionStackSize = 256u * 1024u * 1024u;
    sd.gpuDynamicsConfig.tempBufferCapacity = 128u * 1024u * 1024u;

    PxScene* scene = phy->createScene(sd);
    if (!scene) {
        printf("FAIL: createScene returned null\n");
        dsp->release();
        gpuMgr->release();
        phy->release();
        fnd->release();
        return 3;
    }

    PxMaterial* mat = phy->createMaterial(0.5f, 0.5f, 0.2f);
    PxTriangleMesh* triMesh = createPlatformMesh(phy);
    if (!triMesh) {
        printf("FAIL: createPlatformMesh returned null\n");
        scene->release();
        dsp->release();
        gpuMgr->release();
        phy->release();
        fnd->release();
        return 4;
    }

    PxRigidStatic* platform = phy->createRigidStatic(PxTransform(PxIdentity));
    PxShape* meshShape = phy->createShape(PxTriangleMeshGeometry(triMesh), *mat);
    platform->attachShape(*meshShape);
    scene->addActor(*platform);

    PxShape* boxShape = phy->createShape(PxBoxGeometry(0.35f, 0.35f, 0.35f), *mat);
    std::vector<PxRigidDynamic*> boxes;
    for (int i = 0; i < 16; ++i) {
        float x = -3.0f + (i % 4) * 2.0f;
        float z = -3.0f + (i / 4) * 2.0f;
        float y = 4.0f + 0.2f * i;
        PxRigidDynamic* box = phy->createRigidDynamic(PxTransform(PxVec3(x, y, z)));
        box->attachShape(*boxShape);
        PxRigidBodyExt::updateMassAndInertia(*box, 1.0f);
        scene->addActor(*box);
        boxes.push_back(box);
    }

    for (int i = 0; i < steps; ++i) {
        scene->simulate(1.0f / 60.0f);
        scene->fetchResults(true);
    }

    int bad = 0;
    float minY = 1e30f;
    float maxY = -1e30f;
    float maxSpeed = 0.0f;
    for (PxRigidDynamic* b : boxes) {
        const PxVec3 p = b->getGlobalPose().p;
        const PxVec3 v = b->getLinearVelocity();
        if (!finiteVec(p) || !finiteVec(v))
            bad++;
        if (p.y < minY)
            minY = p.y;
        if (p.y > maxY)
            maxY = p.y;
        float speed = v.magnitude();
        if (speed > maxSpeed)
            maxSpeed = speed;
    }

    printf("Steps: %d\n", steps);
    printf("Boxes: %zu\n", boxes.size());
    printf("Height range: [%.6f, %.6f]\n", minY, maxY);
    printf("Max speed: %.6f\n", maxSpeed);
    printf("Bad boxes: %d\n", bad);

    const bool pass = bad == 0 && minY > -0.5f && maxY < 8.0f && maxSpeed < 20.0f;
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
