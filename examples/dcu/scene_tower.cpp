// Example 8: PhysX Tower Collapse Scene
//
// 2000-box pyramid tower under sphere cannon fire.
// GPU kernel validates contact positions in parallel.
// Links against full PhysX SDK + GPU kernel module.
//
// Build: add to cmakecpu CMakeLists.txt

#include "PxPhysicsAPI.h"
#include "foundation/PxVec3.h"
#include <cstdio>
#include <cmath>
#include <vector>
#include <cstdlib>
#include <hip/hip_runtime.h>

using namespace physx;

// GPU kernel helper (defined in scene_tower_kernel.hip)
extern "C" void launch_contact_filter(const PxVec3* d_pos, int n, float groundY, int* d_count);

int main()
{
    printf("========================================\n");
    printf(" PhysX DCU — Tower Collapse Scene\n");
    printf("========================================\n\n");

    // ---- PhysX SDK Setup ----
    static PxDefaultErrorCallback gErr;
    static PxDefaultAllocator       gAlloc;

    PxFoundation* foundation = PxCreateFoundation(PX_PHYSICS_VERSION, gAlloc, gErr);
    if (!foundation) { printf("ERROR: PxCreateFoundation\n"); return 1; }

    PxPhysics* physics = PxCreatePhysics(PX_PHYSICS_VERSION, *foundation, PxTolerancesScale());
    if (!physics) { printf("ERROR: PxCreatePhysics\n"); return 1; }

    PxSceneDesc sceneDesc(physics->getTolerancesScale());
    sceneDesc.gravity = PxVec3(0, -9.81f, 0);
    PxDefaultCpuDispatcher* dispatcher = PxDefaultCpuDispatcherCreate(4);
    sceneDesc.cpuDispatcher = dispatcher;
    sceneDesc.filterShader  = PxDefaultSimulationFilterShader;

    PxScene* scene = physics->createScene(sceneDesc);
    if (!scene) { printf("ERROR: createScene\n"); return 1; }

    // ---- Material ----
    PxMaterial* mat = physics->createMaterial(0.5f, 0.5f, 0.3f);

    // ---- Ground ----
    PxRigidStatic* ground = PxCreatePlane(*physics, PxPlane(PxVec3(0,1,0), 0), *mat);
    scene->addActor(*ground);

    // ---- Build Pyramid Tower ----
    const int LAYERS = 18;
    const int BASE   = 36;
    const float H    = 0.5f;
    PxShape* boxShape = physics->createShape(PxBoxGeometry(H, H, H), *mat);

    std::vector<PxRigidDynamic*> bodies;
    int boxCount = 0;

    for (int layer = 0; layer < LAYERS; layer++) {
        int side = BASE - layer * 2;
        if (side <= 0) break;
        float y = (layer * 2 + 1) * H + 0.01f;
        for (int ix = 0; ix < side; ix++) {
            for (int iz = 0; iz < side; iz++) {
                float x = (ix - (side-1)/2.0f) * 2.0f * H;
                float z = (iz - (side-1)/2.0f) * 2.0f * H;
                PxRigidDynamic* box = physics->createRigidDynamic(PxTransform(PxVec3(x, y, z)));
                box->attachShape(*boxShape);
                PxRigidBodyExt::updateMassAndInertia(*box, 1.0f);
                scene->addActor(*box);
                bodies.push_back(box);
                boxCount++;
            }
        }
    }
    printf("Tower:  %d boxes, %d layers\n", boxCount, LAYERS);

    // ---- Sphere Cannon ----
    const int SPHERES = 200;
    PxShape* sphereShape = physics->createShape(PxSphereGeometry(0.6f), *mat);
    srand(42);

    for (int s = 0; s < SPHERES; s++) {
        float angle = rand() / (float)RAND_MAX * 6.28318f;
        float speed = 25.0f + rand() / (float)RAND_MAX * 35.0f;
        PxVec3 pos((rand()%60-30), 15.0f + rand()%8, (rand()%60-30));
        PxVec3 vel(cosf(angle)*speed, -5.0f, sinf(angle)*speed);
        PxRigidDynamic* sphere = physics->createRigidDynamic(PxTransform(pos));
        sphere->attachShape(*sphereShape);
        sphere->setLinearVelocity(vel);
        PxRigidBodyExt::updateMassAndInertia(*sphere, 5.0f);
        scene->addActor(*sphere);
        bodies.push_back(sphere);
    }
    printf("Spheres: %d\n", SPHERES);
    printf("Total bodies: %d\n\n", (int)bodies.size());

    // ---- GPU Memory ----
    int totalBodies = (int)bodies.size();
    PxVec3* h_pos = new PxVec3[totalBodies];
    PxVec3* d_pos;
    int *d_count;
    hipMalloc(&d_pos, totalBodies * sizeof(PxVec3));
    hipMalloc(&d_count, sizeof(int));

    // ---- Simulation ----
    printf("Simulating 300 steps...\n");
    float stepSize = 1.0f / 60.0f;
    float simTime = 0, maxSimTime = 5.0f;
    int frames = 0;

    hipEvent_t gpuStart, gpuStop;
    hipEventCreate(&gpuStart); hipEventCreate(&gpuStop);
    hipEventRecord(gpuStart);

    while (simTime < maxSimTime) {
        scene->simulate(stepSize);
        scene->fetchResults(true);
        simTime += stepSize;
        frames++;
    }

    hipEventRecord(gpuStop);
    hipDeviceSynchronize();
    float wallMs;
    hipEventElapsedTime(&wallMs, gpuStart, gpuStop);

    // ---- GPU Contact Validation ----
    for (int i = 0; i < totalBodies; i++)
        h_pos[i] = bodies[i]->getGlobalPose().p;
    hipMemcpy(d_pos, h_pos, totalBodies * sizeof(PxVec3), hipMemcpyHostToDevice);

    hipMemset(d_count, 0, sizeof(int));
    hipEventRecord(gpuStart);
    launch_contact_filter(d_pos, totalBodies, -5.0f, d_count);
    hipEventRecord(gpuStop);
    hipDeviceSynchronize();

    float kernelMs;
    hipEventElapsedTime(&kernelMs, gpuStart, gpuStop);

    int belowCount;
    hipMemcpy(&belowCount, d_count, sizeof(int), hipMemcpyDeviceToHost);

    // ---- Stats ----
    int above = 0;
    for (int i = 0; i < totalBodies; i++)
        if (h_pos[i].y > 0.5f) above++;

    printf("\n=== Results ===\n");
    printf("Sim time:    %.2f s (%.0f FPS)\n", wallMs/1000.0f, frames / (wallMs/1000.0f) * 1000.0f);
    printf("GPU kernel:  %.3f ms\n", kernelMs);
    printf("Below ground: %d / %d\n", belowCount, totalBodies);
    printf("Above 0.5m:   %d\n", above);

    bool pass = (belowCount < 50);
    printf("\n%s\n", pass ? "[PASS] Tower collapse physics OK" : "[FAIL]");

    // Cleanup
    scene->release(); physics->release(); foundation->release();
    dispatcher->release();
    delete[] h_pos;
    hipFree(d_pos); hipFree(d_count);
    hipEventDestroy(gpuStart); hipEventDestroy(gpuStop);
    return pass ? 0 : 1;
}
