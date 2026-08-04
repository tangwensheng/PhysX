// PhysX DCU smoke test: many rigid bodies colliding with a complex static triangle mesh.

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

static float terrainHeight(float x, float z)
{
    const float wave = 0.35f * std::sin(0.45f * x) * std::cos(0.35f * z);
    const float ridge = 0.18f * std::sin(0.9f * (x + 0.4f * z));
    const float ramp = 0.015f * x;
    return wave + ridge + ramp;
}

static PxTriangleMesh* createTerrainMesh(PxPhysics* phy, int grid, float spacing)
{
    std::vector<PxVec3> verts;
    std::vector<PxU32> indices;
    verts.reserve(size_t(grid + 1) * size_t(grid + 1));
    indices.reserve(size_t(grid) * size_t(grid) * 6u);

    const float half = 0.5f * grid * spacing;
    for (int z = 0; z <= grid; ++z) {
        for (int x = 0; x <= grid; ++x) {
            const float wx = x * spacing - half;
            const float wz = z * spacing - half;
            verts.push_back(PxVec3(wx, terrainHeight(wx, wz), wz));
        }
    }

    for (int z = 0; z < grid; ++z) {
        for (int x = 0; x < grid; ++x) {
            const PxU32 i0 = PxU32(z * (grid + 1) + x);
            const PxU32 i1 = i0 + 1;
            const PxU32 i2 = i0 + PxU32(grid + 1);
            const PxU32 i3 = i2 + 1;
            indices.push_back(i0); indices.push_back(i2); indices.push_back(i1);
            indices.push_back(i1); indices.push_back(i2); indices.push_back(i3);
        }
    }

    PxTriangleMeshDesc desc;
    desc.points.count = PxU32(verts.size());
    desc.points.stride = sizeof(PxVec3);
    desc.points.data = verts.data();
    desc.triangles.count = PxU32(indices.size() / 3);
    desc.triangles.stride = 3 * sizeof(PxU32);
    desc.triangles.data = indices.data();

    PxCookingParams params(phy->getTolerancesScale());
    params.meshPreprocessParams = PxMeshPreprocessingFlags(PxMeshPreprocessingFlag::eWELD_VERTICES);
    params.meshWeldTolerance = 0.001f;
    params.buildTriangleAdjacencies = false;
    params.buildGPUData = true;

    PxTriangleMeshCookingResult::Enum result;
    PxTriangleMesh* mesh = PxCreateTriangleMesh(params, desc, phy->getPhysicsInsertionCallback(), &result);
    printf("Terrain cooking: grid=%d verts=%zu tris=%zu result=%d mesh=%p\n",
           grid, verts.size(), indices.size() / 3, int(result), static_cast<void*>(mesh));
    fflush(stdout);
    return mesh;
}

int main(int argc, char** argv)
{
    int steps = 720;
    int bodyCount = 256;
    int grid = 48;
    bool cleanExit = false;
    bool printAllSteps = false;
    int cleanExitWaitSeconds = 5;

    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "cleanexit") == 0 || std::strcmp(argv[i], "--cleanexit") == 0) {
            cleanExit = true;
        } else if (std::strcmp(argv[i], "--cleanexit-wait") == 0 && i + 1 < argc) {
            cleanExit = true;
            cleanExitWaitSeconds = std::atoi(argv[++i]);
            if (cleanExitWaitSeconds < 0)
                cleanExitWaitSeconds = 0;
        } else if (std::strcmp(argv[i], "--print-all-steps") == 0) {
            printAllSteps = true;
        } else if (std::strcmp(argv[i], "--bodies") == 0 && i + 1 < argc) {
            bodyCount = std::atoi(argv[++i]);
        } else if (std::strcmp(argv[i], "--grid") == 0 && i + 1 < argc) {
            grid = std::atoi(argv[++i]);
        } else {
            steps = std::atoi(argv[i]);
        }
    }
    if (steps <= 0)
        steps = 720;
    if (bodyCount < 0)
        bodyCount = 0;
    if (grid < 4)
        grid = 4;

    printf("========================================\n");
    printf(" PhysX DCU Smoke - Complex Triangle Mesh\n");
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

    printf("Cooking terrain mesh...\n");
    fflush(stdout);
    PxTriangleMesh* terrainMesh = createTerrainMesh(phy, grid, 0.5f);
    if (!terrainMesh) {
        printf("FAIL: terrain mesh cooking returned null\n");
        gpuMgr->release();
        phy->release();
        fnd->release();
        return 3;
    }

    PxSceneDesc sd(phy->getTolerancesScale());
    sd.gravity = PxVec3(0.0f, -9.81f, 0.0f);
    sd.cudaContextManager = gpuMgr;
    PxDefaultCpuDispatcher* dsp = PxDefaultCpuDispatcherCreate(0);
    sd.cpuDispatcher = dsp;
    sd.filterShader = PxDefaultSimulationFilterShader;
    sd.flags |= PxSceneFlag::eENABLE_GPU_DYNAMICS;
    sd.gpuMaxNumPartitions = 8;
    sd.gpuDynamicsConfig.maxRigidContactCount = 1024u * 1024u * 8u;
    sd.gpuDynamicsConfig.maxRigidPatchCount = 1024u * 1024u;
    sd.gpuDynamicsConfig.foundLostPairsCapacity = 1024u * 1024u * 2u;
    sd.gpuDynamicsConfig.heapCapacity = 512u * 1024u * 1024u;
    sd.gpuDynamicsConfig.collisionStackSize = 256u * 1024u * 1024u;
    sd.gpuDynamicsConfig.tempBufferCapacity = 128u * 1024u * 1024u;

    printf("Creating GPU dynamics scene...\n");
    fflush(stdout);
    PxScene* scene = phy->createScene(sd);
    if (!scene) {
        printf("FAIL: createScene returned null\n");
        dsp->release();
        gpuMgr->release();
        phy->release();
        fnd->release();
        return 4;
    }

    PxMaterial* mat = phy->createMaterial(0.55f, 0.55f, 0.05f);
    PxRigidStatic* terrain = phy->createRigidStatic(PxTransform(PxIdentity));
    PxShape* terrainShape = phy->createShape(PxTriangleMeshGeometry(terrainMesh), *mat);
    terrain->attachShape(*terrainShape);
    scene->addActor(*terrain);

    PxShape* boxShape = phy->createShape(PxBoxGeometry(0.22f, 0.22f, 0.22f), *mat);
    std::vector<PxRigidDynamic*> boxes;
    boxes.reserve(size_t(bodyCount));

    const int side = int(std::ceil(std::sqrt(float(bodyCount))));
    const float spacing = 0.72f;
    const float start = -0.5f * spacing * (side - 1);
    for (int i = 0; i < bodyCount; ++i) {
        const int ix = i % side;
        const int iz = i / side;
        const float x = start + ix * spacing;
        const float z = start + iz * spacing;
        const float y = 3.0f + 0.035f * float(i);
        PxRigidDynamic* box = phy->createRigidDynamic(PxTransform(PxVec3(x, y, z)));
        box->attachShape(*boxShape);
        box->setLinearDamping(0.05f);
        box->setAngularDamping(0.05f);
        PxRigidBodyExt::updateMassAndInertia(*box, 1.0f);
        box->setSolverIterationCounts(8, 1);
        scene->addActor(*box);
        boxes.push_back(box);
    }

    printf("Starting simulation: steps=%d bodies=%zu grid=%d\n", steps, boxes.size(), grid);
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

    int bad = 0;
    int belowTerrain = 0;
    float minY = 1e30f;
    float maxY = -1e30f;
    float maxSpeed = 0.0f;
    for (PxRigidDynamic* b : boxes) {
        const PxVec3 p = b->getGlobalPose().p;
        const PxVec3 v = b->getLinearVelocity();
        if (!finiteVec(p) || !finiteVec(v))
            bad++;
        const float h = terrainHeight(p.x, p.z);
        if (p.y < h - 0.4f)
            belowTerrain++;
        if (p.y < minY)
            minY = p.y;
        if (p.y > maxY)
            maxY = p.y;
        const float speed = v.magnitude();
        if (speed > maxSpeed)
            maxSpeed = speed;
    }

    if (boxes.empty()) {
        minY = 0.0f;
        maxY = 0.0f;
        maxSpeed = 0.0f;
    }

    printf("Steps: %d\n", steps);
    printf("Boxes: %zu\n", boxes.size());
    printf("Height range: [%.6f, %.6f]\n", minY, maxY);
    printf("Max speed: %.6f\n", maxSpeed);
    printf("Bad boxes: %d\n", bad);
    printf("Below terrain boxes: %d\n", belowTerrain);

    const bool pass = bad == 0 && belowTerrain == 0 && (boxes.empty() || (minY > -2.0f && maxY < 40.0f && maxSpeed < 50.0f));
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
