// PhysX ConvexMesh & HeightField Test — test convex modules on DCU
#include "PxPhysicsAPI.h"
#include "cudamanager/PxCudaContextManager.h"
#include "gpu/PxGpu.h"
#include "cooking/PxCooking.h"
#include "extensions/PxDefaultStreams.h"
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <cmath>
#include <ctime>

using namespace physx;

// Create a convex mesh (pyramid)
static PxConvexMesh* createConvexMesh(PxPhysics* phy)
{
    // Pyramid vertices (5 vertices: 4 base + 1 top)
    PxVec3 vertices[] = {
        PxVec3(-0.5f, 0, -0.5f),  // base corner
        PxVec3( 0.5f, 0, -0.5f),  // base corner
        PxVec3( 0.5f, 0,  0.5f),  // base corner
        PxVec3(-0.5f, 0,  0.5f),  // base corner
        PxVec3( 0.0f, 1.0f, 0.0f)  // top
    };
    
    PxConvexMeshDesc desc;
    desc.points.count = 5;
    desc.points.data = vertices;
    desc.points.stride = sizeof(PxVec3);
    desc.flags = PxConvexFlag::eCOMPUTE_CONVEX;
    
    PxDefaultMemoryOutputStream buf;
    PxConvexMeshCookingResult::Enum result;
    
    // Use PxCookingParams
    PxCookingParams params(phy->getTolerancesScale());
    if (!PxCookConvexMesh(params, desc, buf, &result)) {
        printf("ERROR: Failed to cook convex mesh, result=%d\n", result);
        return nullptr;
    }
    
    // Use PxDefaultMemoryInputData instead of PxDefaultMemoryInputStream
    PxDefaultMemoryInputData input(buf.getData(), buf.getSize());
    return phy->createConvexMesh(input);
}

// Create heightfield
static PxHeightField* createHeightField(PxPhysics* phy)
{
    const int ROWS = 32, COLS = 32;
    PxHeightFieldSample samples[ROWS * COLS];
    
    for (int i = 0; i < ROWS; i++) {
        for (int j = 0; j < COLS; j++) {
            // Create wavy terrain
            float h = sinf(i * 0.3f) * 0.3f + cosf(j * 0.3f) * 0.3f;
            samples[i * COLS + j].height = h;
        }
    }
    
    PxHeightFieldDesc desc;
    desc.nbRows = ROWS;
    desc.nbColumns = COLS;
    desc.samples.data = samples;
    desc.samples.stride = sizeof(PxHeightFieldSample);
    desc.convexEdgeThreshold = 0.0f;
    
    PxDefaultMemoryOutputStream buf;
    if (!PxCookHeightField(desc, buf)) {
        printf("ERROR: Failed to cook heightfield\n");
        return nullptr;
    }
    
    PxDefaultMemoryInputData input(buf.getData(), buf.getSize());
    return phy->createHeightField(input);
}

// Simple random float in [-1, 1]
static float frand()
{
    return (rand() / (float)RAND_MAX) * 2.0f - 1.0f;
}

static void buildConvexMeshScene(PxPhysics* phy, PxScene* scene, 
                                  PxConvexMesh* convex, PxHeightField* hf,
                                  std::vector<PxRigidDynamic*>& bodies)
{
    PxMaterial* mat = phy->createMaterial(0.5f, 0.5f, 0.3f);
    
    // Ground plane
    scene->addActor(*PxCreatePlane(*phy, PxPlane(PxVec3(0,1,0), 0), *mat));

    // === Test 1: Convex Mesh shapes ===
    printf("Creating convex mesh objects...\n");
    PxConvexMeshGeometry convexGeo(convex);
    PxShape* convexShape = phy->createShape(convexGeo, *mat);
    
    const int NCONVEX = 100;
    srand(123);
    for (int i = 0; i < NCONVEX; i++) {
        float x = (rand() % 40 - 20) * 1.0f;
        float z = (rand() % 40 - 20) * 1.0f;
        float y = 5.0f + rand() % 20;
        
        PxRigidDynamic* b = phy->createRigidDynamic(PxTransform(PxVec3(x, y, z)));
        b->attachShape(*convexShape);
        PxRigidBodyExt::updateMassAndInertia(*b, 1.0f);
        
        // Random rotation
        PxQuat q(frand(), frand(), frand(), frand());
        b->setGlobalPose(PxTransform(b->getGlobalPose().p, q.getNormalized()));
        
        scene->addActor(*b);
        bodies.push_back(b);
    }
    printf("Convex mesh objects: %d\n", NCONVEX);

    // === Test 2: HeightField ===
    printf("Creating heightfield...\n");
    // Use old-style constructor: hf, flags, heightScale, rowScale, columnScale
    PxHeightFieldGeometry hfGeo(hf, PxMeshGeometryFlag::Enum(0), 1.0f, 1.0f, 1.0f);
    PxShape* hfShape = phy->createShape(hfGeo, *mat);
    
    PxTransform hfTrans(PxVec3(0, -1.0f, 0), PxQuat(PxPi/2, PxVec3(1,0,0)));
    PxRigidStatic* hfActor = phy->createRigidStatic(hfTrans);
    hfActor->attachShape(*hfShape);
    scene->addActor(*hfActor);
    printf("Heightfield created\n");

    // === Test 3: Spheres to interact ===
    const int NSPH = 30;
    PxMaterial* matH = phy->createMaterial(0.8f, 0.8f, 0.1f);
    PxShape* sphShape = phy->createShape(PxSphereGeometry(0.5f), *matH);
    
    for (int s = 0; s < NSPH; s++) {
        float a = rand() / (float)RAND_MAX * 6.283f;
        float v = 25 + rand() / (float)RAND_MAX * 30;
        PxRigidDynamic* sp = phy->createRigidDynamic(
            PxTransform(PxVec3((rand()%30-15), 20+rand()%10, (rand()%30-15))));
        sp->attachShape(*sphShape);
        sp->setLinearVelocity(PxVec3(cosf(a)*v, -5, sinf(a)*v));
        PxRigidBodyExt::updateMassAndInertia(*sp, 3.0f);
        scene->addActor(*sp);
        bodies.push_back(sp);
    }
    printf("Spheres: %d\n", NSPH);
    printf("Total bodies: %d\n", (int)bodies.size());
}

int main()
{
    printf("========================================\n");
    printf(" PhysX ConvexMesh & HeightField Test\n");
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

    // Create convex mesh and heightfield
    printf("\nCreating convex mesh...\n");
    PxConvexMesh* convex = createConvexMesh(phy);
    if (!convex) { printf("FAILED to create convex mesh\n"); return 1; }
    printf("Convex mesh created OK\n");

    printf("Creating heightfield...\n");
    PxHeightField* hf = createHeightField(phy);
    if (!hf) { printf("FAILED to create heightfield\n"); return 1; }
    printf("Heightfield created OK\n");

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

    // Build scene
    std::vector<PxRigidDynamic*> bodies;
    buildConvexMeshScene(phy, scene, convex, hf, bodies);
    printf("\n");

    // Diagnostic
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
    float maxY = -1e10f, minY = 1e10f;
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
    printf("\nConvexMesh & HeightField test PASSED!\n");

    scene->release();
    dsp->release();
    if (gpuMgr) gpuMgr->release();
    phy->release();
    fnd->release();
    return 0;
}