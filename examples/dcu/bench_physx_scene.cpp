// PhysX Scene Benchmark — PxScene::simulate() on GPU
// Same code for DCU (g++) and A800 (g++).

#include "PxPhysicsAPI.h"
#include "cudamanager/PxCudaContextManager.h"
#include "gpu/PxGpu.h"
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <cmath>
#include <ctime>
#if defined(__HIPCC__)
#include <hip/hip_runtime.h>
#endif

using namespace physx;

int main()
{
    printf("========================================\n");
    printf(" PhysX Scene Benchmark — Tower Collapse\n");
    printf("========================================\n\n");

    static PxDefaultErrorCallback gErr;
    static PxDefaultAllocator       gAlloc;
    PxFoundation* fnd = PxCreateFoundation(PX_PHYSICS_VERSION, gAlloc, gErr);
    PxPhysics*    phy = PxCreatePhysics(PX_PHYSICS_VERSION, *fnd, PxTolerancesScale());

    // GPU setup
    PxCudaContextManagerDesc gpuDesc;
    gpuDesc.deviceOrdinal = 0;
    PxCudaContextManager* gpuMgr = PxCreateCudaContextManager(*fnd, gpuDesc, nullptr, false);
    bool gpuOk = gpuMgr && gpuMgr->contextIsValid();
    printf("GPU: %s (%d CUs, %.1f GB) — %s\n",
        gpuOk ? gpuMgr->getDeviceName() : "NONE",
        gpuOk ? gpuMgr->getMultiprocessorCount() : 0,
        gpuOk ? gpuMgr->getDeviceTotalMemBytes()/(1024.0*1024.0*1024.0) : 0.0,
        gpuOk ? "GPU enabled" : "CPU only");

    // ---- CPU warm-up: initialize ROCm/PhysX internals ----
    printf("CPU warmup...\n");
    {
        PxDefaultCpuDispatcher* dspWarm = PxDefaultCpuDispatcherCreate(0);
        PxSceneDesc sdCpu(phy->getTolerancesScale());
        sdCpu.gravity = PxVec3(0, -9.81f, 0);
        sdCpu.cpuDispatcher = dspWarm;
        sdCpu.filterShader = PxDefaultSimulationFilterShader;
        PxScene* cpuScene = phy->createScene(sdCpu);
        cpuScene->simulate(1.0f/60.0f);
        cpuScene->fetchResults(true);
        cpuScene->release();
        dspWarm->release();
        printf("CPU warmup OK\n");
    }

    PxSceneDesc sd(phy->getTolerancesScale());
    sd.gravity = PxVec3(0, -9.81f, 0);
    sd.cudaContextManager = gpuMgr;
    if (gpuOk) {
        sd.flags |= PxSceneFlag::eENABLE_GPU_DYNAMICS;
        sd.gpuMaxNumPartitions = 8;
        sd.gpuDynamicsConfig.maxRigidContactCount   = 1024u * 1024u * 8u;
        sd.gpuDynamicsConfig.maxRigidPatchCount     = 1024u * 1024u;
        sd.gpuDynamicsConfig.foundLostPairsCapacity = 1024u * 1024u * 2u;
        sd.gpuDynamicsConfig.heapCapacity           = 512u * 1024u * 1024u;
        sd.gpuDynamicsConfig.collisionStackSize     = 256u * 1024u * 1024u;
        sd.gpuDynamicsConfig.tempBufferCapacity     = 128u * 1024u * 1024u;
    }
    PxDefaultCpuDispatcher* dsp = PxDefaultCpuDispatcherCreate(0);
    sd.cpuDispatcher = dsp;
    sd.filterShader  = PxDefaultSimulationFilterShader;
    PxScene* scene = phy->createScene(sd);

    // ---- Build Tower ----
    const int LAYERS = 20, BASE = 44;
    const float H = 0.5f;
    PxMaterial* mat = phy->createMaterial(0.5f, 0.5f, 0.3f);
    PxShape* boxShape = phy->createShape(PxBoxGeometry(H, H, H), *mat);
    scene->addActor(*PxCreatePlane(*phy, PxPlane(PxVec3(0,1,0), 0), *mat));

    std::vector<PxRigidDynamic*> bodies;
    int nBox = 0;

    for (int layer = 0; layer < LAYERS; layer++) {
        int side = BASE - layer * 2; if (side <= 0) break;
        float y = (layer * 2 + 1) * H + 0.01f;
        for (int ix = 0; ix < side; ix++) for (int iz = 0; iz < side; iz++) {
            float x = (ix - (side-1)/2.0f) * 2.0f * H;
            float z = (iz - (side-1)/2.0f) * 2.0f * H;
            PxRigidDynamic* b = phy->createRigidDynamic(PxTransform(PxVec3(x, y, z)));
            b->attachShape(*boxShape);
            PxRigidBodyExt::updateMassAndInertia(*b, 1.0f);
            scene->addActor(*b);
            bodies.push_back(b);
            nBox++;
        }
    }
    int nTotal = (int)bodies.size();
    printf("Bodies: %d (%d layers)\n", nBox, LAYERS);

    // ---- Sphere Cannon ----
    const int NSPH = 300;
    PxMaterial* matH = phy->createMaterial(0.8f, 0.8f, 0.1f);
    PxShape* sphShape = phy->createShape(PxSphereGeometry(0.6f), *matH);
    srand(42);
    for (int s = 0; s < NSPH; s++) {
        float a = rand()/(float)RAND_MAX * 6.283f;
        float v = 20 + rand()/(float)RAND_MAX * 35;
        PxRigidDynamic* sp = phy->createRigidDynamic(
            PxTransform(PxVec3((rand()%60-30), 15+rand()%10, (rand()%60-30))));
        sp->attachShape(*sphShape);
        sp->setLinearVelocity(PxVec3(cosf(a)*v, -5, sinf(a)*v));
        PxRigidBodyExt::updateMassAndInertia(*sp, 5.0f);
        scene->addActor(*sp);
        bodies.push_back(sp);
    }
    nTotal = (int)bodies.size();
    printf("Projectiles: %d | Total: %d\n\n", NSPH, nTotal);

    // ---- Diagnostic: zero-gravity zero-time step ----
    printf("Diagnostic: 0 gravity + dt=0...\n");
    scene->setGravity(PxVec3(0, 0, 0));
    scene->simulate(0.0f);
#if defined(__HIPCC__)
    hipError_t e = hipDeviceSynchronize();
    if (e != hipSuccess) {
        printf("FATAL: crash during zero-step (init phase): %s\n", hipGetErrorString(e));
        exit(1);
    }
#endif
    scene->fetchResults(true);
    printf("Diagnostic OK\n");
    scene->setGravity(PxVec3(0, -9.81f, 0));

    // ---- Settle ----
    printf("Settling...\n");
#if defined(__HIPCC__)
    for (int s = 0; s < 120; s++) {
        if (s % 10 == 0) printf("  step %d...\n", s);
        scene->simulate(1.0f/60.0f);
        e = hipDeviceSynchronize();
        if (e != hipSuccess) {
            printf("FATAL step %d: %s\n", s, hipGetErrorString(e));
            exit(1);
        }
        scene->fetchResults(true);
    }
#else
    for (int s = 0; s < 120; s++) { scene->simulate(1.0f/60.0f); scene->fetchResults(true); }
#endif

    // ---- Benchmark ----
    printf("Running 300 simulation steps...\n");
    float step = 1.0f/60.0f;
    const int BENCH_STEPS = 300;
    struct timespec t0, t1;
    clock_gettime(CLOCK_MONOTONIC, &t0);
    for (int f = 0; f < BENCH_STEPS; f++) {
        scene->simulate(step);
        scene->fetchResults(true);
    }
    clock_gettime(CLOCK_MONOTONIC, &t1);
    double elapsed = (t1.tv_sec - t0.tv_sec) + (t1.tv_nsec - t0.tv_nsec) * 1e-9;
    double avgMs = elapsed * 1000.0 / BENCH_STEPS;

    // ---- Stats ----
    int aboveGround = 0;
    float maxY = 0, minY = 1e10f;
    for (auto* b : bodies) {
        if (b->isSleeping()) continue;
        float y = b->getGlobalPose().p.y;
        if (y > 0.5f) aboveGround++;
        if (y > maxY) maxY = y;
        if (y < minY) minY = y;
    }

    printf("\n=== Results ===\n");
    printf("Avg frame time: %.2f ms (%.0f FPS)\n", avgMs, 1000.0f/avgMs);
    printf("Above ground:   %d / %d\n", aboveGround, nTotal);
    printf("Height range:   [%.1f, %.1f]\n", minY, maxY);

    scene->release();
    dsp->release();
    if (gpuMgr) gpuMgr->release();
    phy->release();
    fnd->release();
    return 0;
}
