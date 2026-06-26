// PhysX Scene Benchmark — PxScene::simulate() on CPU
// Same code for DCU (g++) and A800 (g++).
// Measures real PhysX simulation frame time with Tower Collapse scene.

#include "PxPhysicsAPI.h"
#include "cudamanager/PxCudaContextManager.h"
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <cmath>

using namespace physx;

int main()
{
    printf("========================================\n");
    printf(" PhysX Scene Benchmark — Tower Collapse\n");
    printf("========================================\n\n");

    // ---- PhysX Setup ----
    static PxDefaultErrorCallback gErr;
    static PxDefaultAllocator       gAlloc;
    PxFoundation* fnd = PxCreateFoundation(PX_PHYSICS_VERSION, gAlloc, gErr);
    PxPhysics*    phy = PxCreatePhysics(PX_PHYSICS_VERSION, *fnd, PxTolerancesScale());

    // GPU setup via HIP context manager
    PxCudaContextManagerDesc gpuDesc;
    gpuDesc.deviceOrdinal = 0;
    PxCudaContextManager* gpuMgr = PxCreateCudaContextManager(*fnd, gpuDesc, nullptr, false);
    bool gpuOk = gpuMgr && gpuMgr->contextIsValid();
    printf("GPU: %s (%d CUs, %.1f GB) — %s\n",
        gpuOk ? gpuMgr->getDeviceName() : "NONE",
        gpuOk ? gpuMgr->getMultiprocessorCount() : 0,
        gpuOk ? gpuMgr->getDeviceTotalMemBytes()/(1024.0*1024.0*1024.0) : 0.0,
        gpuOk ? "GPU enabled" : "CPU only");

    PxSceneDesc sd(phy->getTolerancesScale());
    sd.gravity = PxVec3(0, -9.81f, 0);
    sd.cudaContextManager = gpuMgr;  // GPU auto-enabled when context is valid
    PxDefaultCpuDispatcher* dsp = PxDefaultCpuDispatcherCreate(0);
    sd.cpuDispatcher = dsp;
    sd.filterShader  = PxDefaultSimulationFilterShader;
    PxScene* scene = phy->createScene(sd);

    // ---- Build Tower ----
    const int LAYERS = 14, BASE = 30;
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
    const int NSPH = 200;
    PxMaterial* matH = phy->createMaterial(0.8f, 0.8f, 0.1f);
    PxShape* sphShape = phy->createShape(PxSphereGeometry(0.6f), *matH);
    srand(42);

    for (int s = 0; s < NSPH; s++) {
        float a = rand() / (float)RAND_MAX * 6.283f;
        float v = 20 + rand() / (float)RAND_MAX * 35;
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

    // ---- Settle ----
    printf("Settling...\n");
    for (int s = 0; s < 120; s++) { scene->simulate(1.0f/60.0f); scene->fetchResults(true); }

    // ---- Benchmark ----
    printf("Running 300 simulation steps...\n");
    float step = 1.0f / 60.0f;

    // Precision timing: use clock_gettime for wall-clock
    struct timespec t0, t1;
    clock_gettime(CLOCK_MONOTONIC, &t0);

    for (int f = 0; f < 300; f++) {
        scene->simulate(step);
        scene->fetchResults(true);
    }

    clock_gettime(CLOCK_MONOTONIC, &t1);
    double elapsed = (t1.tv_sec - t0.tv_sec) + (t1.tv_nsec - t0.tv_nsec) * 1e-9;
    double avgMs = elapsed * 1000.0 / 300.0;

    // ---- Stats ----
    int aboveGround = 0;
    float maxY = 0, minY = 1e10f;
    for (auto* b : bodies) {
        float y = b->getGlobalPose().p.y;
        if (y > 0.5f) aboveGround++;
        if (y > maxY) maxY = y;
        if (y < minY) minY = y;
    }

    printf("\n=== Results ===\n");
    printf("Avg frame time: %.2f ms (%.0f FPS)\n", avgMs, 1000.0f/avgMs);
    printf("Above ground:   %d / %d\n", aboveGround, nTotal);
    printf("Height range:   [%.1f, %.1f]\n", minY, maxY);

    // ---- Cleanup ----
    scene->release(); phy->release(); fnd->release(); dsp->release();
    return 0;
}
