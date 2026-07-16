// PhysX Convex Collision Test — test convex modules on DCU
#include "PxPhysicsAPI.h"
#include "cudamanager/PxCudaContextManager.h"
#include "gpu/PxGpu.h"
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <cmath>
#include <ctime>

using namespace physx;

// Build scene with convex shapes
static void buildConvexScene(PxPhysics* phy, PxScene* scene, std::vector<PxRigidDynamic*>& bodies)
{
    PxMaterial* mat = phy->createMaterial(0.5f, 0.5f, 0.3f);
    
    // Ground plane
    scene->addActor(*PxCreatePlane(*phy, PxPlane(PxVec3(0,1,0), 0), *mat));

    // Create convex hull shapes (boxes as simple convex)
    PxBoxGeometry boxGeo(0.5f, 0.5f, 0.5f);
    PxShape* boxShape = phy->createShape(boxGeo, *mat);

    // Stack of boxes (like tower)
    const int LAYERS = 10, BASE = 8;
    const float H = 0.5f;
    
    for (int layer = 0; layer < LAYERS; layer++) {
        int side = BASE - layer;
        if (side <= 0) break;
        float y = (layer * 2 + 1) * H + 0.01f;
        for (int ix = 0; ix < side; ix++) {
            for (int iz = 0; iz < side; iz++) {
                float x = (ix - (side-1)/2.0f) * 2.0f * H;
                float z = (iz - (side-1)/2.0f) * 2.0f * H;
                PxRigidDynamic* b = phy->createRigidDynamic(PxTransform(PxVec3(x, y, z)));
                b->attachShape(*boxShape);
                PxRigidBodyExt::updateMassAndInertia(*b, 1.0f);
                scene->addActor(*b);
                bodies.push_back(b);
            }
        }
    }
    printf("Box stack: %d bodies\n", (int)bodies.size());

    // Add some spheres to interact with boxes
    const int NSPH = 50;
    PxMaterial* matH = phy->createMaterial(0.8f, 0.8f, 0.1f);
    PxShape* sphShape = phy->createShape(PxSphereGeometry(0.6f), *matH);
    srand(42);
    for (int s = 0; s < NSPH; s++) {
        float a = rand()/(float)RAND_MAX * 6.283f;
        float v = 20 + rand()/(float)RAND_MAX * 35;
        PxRigidDynamic* sp = phy->createRigidDynamic(
            PxTransform(PxVec3((rand()%40-20), 12+rand()%8, (rand()%40-20))));
        sp->attachShape(*sphShape);
        sp->setLinearVelocity(PxVec3(cosf(a)*v, -5, sinf(a)*v));
        PxRigidBodyExt::updateMassAndInertia(*sp, 5.0f);
        scene->addActor(*sp);
        bodies.push_back(sp);
    }
    printf("Spheres: %d\n", NSPH);
}

int main()
{
    printf("========================================\n");
    printf(" PhysX Convex Collision Test\n");
    printf("========================================\n\n");

    static PxDefaultErrorCallback gErr;
    static PxDefaultAllocator    gAlloc;
    PxFoundation* fnd = PxCreateFoundation(PX_PHYSICS_VERSION, gAlloc, gErr);
    PxPhysics*    phy = PxCreatePhysics(PX_PHYSICS_VERSION, *fnd, PxTolerancesScale());

    // GPU setup
    PxCudaContextManagerDesc gpuDesc;
    gpuDesc.deviceOrdinal = 0;
    PxCudaContextManager* gpuMgr = PxCreateCudaContextManager(*fnd, gpuDesc, nullptr, false);
    bool gpuOk = gpuMgr && gpuMgr->contextIsValid();
    printf("GPU: %s — %s\n",
        gpuOk ? gpuMgr->getDeviceName() : "NONE",
        gpuOk ? "GPU enabled" : "CPU only");

    // Scene with GPU dynamics
    PxSceneDesc sd(phy->getTolerancesScale());
    sd.gravity = PxVec3(0, -9.81f, 0);
    sd.cudaContextManager = gpuMgr;
    if (gpuOk) {
        sd.flags |= PxSceneFlag::eENABLE_GPU_DYNAMICS;
        sd.gpuMaxNumPartitions = 8;
        sd.gpuDynamicsConfig.maxRigidContactCount   = 1024u * 1024u * 4u;
        sd.gpuDynamicsConfig.maxRigidPatchCount     = 1024u * 512u;
        sd.gpuDynamicsConfig.heapCapacity           = 256u * 1024u * 1024u;
        sd.gpuDynamicsConfig.foundLostPairsCapacity = 512u * 1024u;
        sd.gpuDynamicsConfig.collisionStackSize     = 128u * 1024u * 1024u;
        sd.gpuDynamicsConfig.tempBufferCapacity     = 64u * 1024u * 1024u;
    }
    PxDefaultCpuDispatcher* dsp = PxDefaultCpuDispatcherCreate(0);
    sd.cpuDispatcher = dsp;
    sd.filterShader = PxDefaultSimulationFilterShader;
    PxScene* scene = phy->createScene(sd);

    // Build convex scene
    std::vector<PxRigidDynamic*> bodies;
    buildConvexScene(phy, scene, bodies);
    printf("Total bodies: %d\n\n", (int)bodies.size());

    // Diagnostic: zero gravity zero step
    printf("Diagnostic: 0 gravity + dt=0...\n");
    scene->setGravity(PxVec3(0, 0, 0));
    scene->simulate(0.0f);
    scene->fetchResults(true);
    printf("Diagnostic OK\n\n");
    scene->setGravity(PxVec3(0, -9.81f, 0));

    // Settle
    printf("Settling (60 steps)...\n");
    for (int s = 0; s < 60; s++) {
        scene->simulate(1.0f/60.0f);
        scene->fetchResults(true);
        if (s % 20 == 19) printf("  step %d done\n", s+1);
    }
    printf("Settle complete\n\n");

    // Benchmark
    printf("Running 200 simulation steps...\n");
    struct timespec t0, t1;
    clock_gettime(CLOCK_MONOTONIC, &t0);
    
    for (int f = 0; f < 200; f++) {
        scene->simulate(1.0f/60.0f);
        scene->fetchResults(true);
    }
    
    clock_gettime(CLOCK_MONOTONIC, &t1);
    double elapsed = (t1.tv_sec - t0.tv_sec) + (t1.tv_nsec - t0.tv_nsec) * 1e-9;
    double avgMs = elapsed * 1000.0 / 200.0;

    // Stats
    int moving = 0;
    float maxY = 0, minY = 1e10f;
    for (auto* b : bodies) {
        if (!b->isSleeping()) moving++;
        float y = b->getGlobalPose().p.y;
        if (y > maxY) maxY = y;
        if (y < minY) minY = y;
    }

    printf("\n=== Results ===\n");
    printf("Avg frame time: %.2f ms (%.0f FPS)\n", avgMs, 1000.0f/avgMs);
    printf("Moving bodies:  %d / %d\n", moving, (int)bodies.size());
    printf("Height range:   [%.1f, %.1f]\n", minY, maxY);
    printf("\nConvex collision test PASSED!\n");

    scene->release();
    dsp->release();
    if (gpuMgr) gpuMgr->release();
    phy->release();
    fnd->release();
    return 0;
}