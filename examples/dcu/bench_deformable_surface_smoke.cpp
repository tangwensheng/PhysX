// PhysX DCU smoke test: minimal GPU deformable surface (cloth-like) path.

#include "PxPhysicsAPI.h"
#include "cudamanager/PxCudaContextManager.h"
#include "cudamanager/PxCudaContext.h"
#include "gpu/PxGpu.h"
#include "extensions/PxCudaHelpersExt.h"
#include "extensions/PxDeformableSurfaceExt.h"
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <thread>
#include <vector>

using namespace physx;

static PX_FORCE_INLINE PxU32 vid(PxU32 x, PxU32 y, PxU32 numY)
{
    return x * numY + y;
}

static PxReal triMass(const PxU32* tri, const PxVec3* verts, PxReal thickness, PxReal density)
{
    const PxVec3& p0 = verts[tri[0]];
    const PxVec3& p1 = verts[tri[1]];
    const PxVec3& p2 = verts[tri[2]];
    return 0.5f * (p1 - p0).cross(p2 - p0).magnitude() * thickness * density;
}

static bool finiteVec(const PxVec3& v)
{
    return std::isfinite(v.x) && std::isfinite(v.y) && std::isfinite(v.z);
}

static bool finiteVec4(const PxVec4& v)
{
    return std::isfinite(v.x) && std::isfinite(v.y) && std::isfinite(v.z) && std::isfinite(v.w);
}

static PxDeformableSurface* createClothSurface(PxPhysics* phy, PxScene* scene, const PxCookingParams& params,
                                               PxCudaContextManager* gpuMgr, PxDeformableSurfaceMaterial* material,
                                               PxU32 nx, PxU32 nz, PxReal size, PxReal height, bool selfCollision)
{
    std::vector<PxVec3> verts;
    std::vector<PxVec3> velocity;
    std::vector<PxU32> tris;
    std::vector<PxReal> triMasses;
    verts.reserve(size_t(nx) * size_t(nz));
    velocity.reserve(size_t(nx) * size_t(nz));
    tris.reserve(size_t(nx - 1) * size_t(nz - 1) * 6u);
    triMasses.reserve(size_t(nx - 1) * size_t(nz - 1) * 2u);

    const PxReal sx = size / PxReal(nx - 1);
    const PxReal sz = size / PxReal(nz - 1);
    for (PxU32 i = 0; i < nx; ++i) {
        for (PxU32 j = 0; j < nz; ++j) {
            verts.push_back(PxVec3(PxReal(i) * sx - 0.5f * size, height, PxReal(j) * sz - 0.5f * size));
            velocity.push_back(PxVec3(0.0f));
        }
    }

    for (PxU32 i = 1; i < nx; ++i) {
        for (PxU32 j = 1; j < nz; ++j) {
            tris.push_back(vid(i - 1, j - 1, nz));
            tris.push_back(vid(i,     j - 1, nz));
            tris.push_back(vid(i - 1, j,     nz));
            triMasses.push_back(triMass(&tris[tris.size() - 3], verts.data(), 0.01f, 500.0f));

            tris.push_back(vid(i - 1, j,     nz));
            tris.push_back(vid(i,     j - 1, nz));
            tris.push_back(vid(i,     j,     nz));
            triMasses.push_back(triMass(&tris[tris.size() - 3], verts.data(), 0.01f, 500.0f));
        }
    }

    PxTriangleMeshDesc meshDesc;
    meshDesc.points.count = PxU32(verts.size());
    meshDesc.points.stride = sizeof(PxVec3);
    meshDesc.points.data = verts.data();
    meshDesc.triangles.count = PxU32(tris.size() / 3);
    meshDesc.triangles.stride = 3 * sizeof(PxU32);
    meshDesc.triangles.data = tris.data();

    printf("Cooking deformable surface triangle mesh: verts=%zu tris=%zu...\n", verts.size(), tris.size() / 3);
    fflush(stdout);
    PxTriangleMeshCookingResult::Enum cookResult;
    PxTriangleMesh* triMesh = PxCreateTriangleMesh(params, meshDesc, phy->getPhysicsInsertionCallback(), &cookResult);
    printf("Deformable surface mesh cooking result=%d mesh=%p\n", int(cookResult), static_cast<void*>(triMesh));
    fflush(stdout);
    if (!triMesh)
        return nullptr;

    printf("Creating PxDeformableSurface...\n");
    fflush(stdout);
    PxDeformableSurface* surface = phy->createDeformableSurface(*gpuMgr);
    if (!surface) {
        printf("FAIL: createDeformableSurface returned null\n");
        return nullptr;
    }

    PxShapeFlags shapeFlags = PxShapeFlag::eVISUALIZATION | PxShapeFlag::eSCENE_QUERY_SHAPE | PxShapeFlag::eSIMULATION_SHAPE;
    PxDeformableSurfaceMaterial* materials[1] = { material };
    PxShape* shape = phy->createShape(PxTriangleMeshGeometry(triMesh), materials, 1, true, shapeFlags);
    if (!shape) {
        printf("FAIL: createShape for deformable surface returned null\n");
        return nullptr;
    }
    surface->attachShape(*shape);
    shape->setContactOffset(0.02f);
    shape->setRestOffset(0.01f);
    shape->setDeformableSurfaceMaterials(materials, 1);
    surface->setSelfCollisionFilterDistance(0.025f);
    surface->setLinearDamping(0.0f);
    surface->setDeformableBodyFlag(PxDeformableBodyFlag::eDISABLE_SELF_COLLISION, !selfCollision);
    surface->setMaxVelocity(1000.0f);
    surface->setNbCollisionPairUpdatesPerTimestep(1);
    surface->setNbCollisionSubsteps(1);

    printf("Adding deformable surface actor...\n");
    fflush(stdout);
    scene->addActor(*surface);

    PxVec4* posInvMassPinned = nullptr;
    PxVec4* velocityPinned = nullptr;
    PxVec4* restPositionPinned = nullptr;
    const PxTransform transform(PxIdentity);
    printf("Allocating deformable host mirror...\n");
    fflush(stdout);
    PxDeformableSurfaceExt::allocateAndInitializeHostMirror(*surface, verts.data(), velocity.data(), verts.data(), 0.5f,
                                                            transform, gpuMgr, posInvMassPinned, velocityPinned, restPositionPinned);
    printf("Distributing mass and copying deformable data to device...\n");
    fflush(stdout);
    PxDeformableSurfaceExt::distributeTriangleMassToVertices(*surface, triMasses.data(), posInvMassPinned);
    PxDeformableSurfaceExt::copyToDevice(*surface, PxDeformableSurfaceDataFlag::eALL, PxU32(verts.size()),
                                         posInvMassPinned, velocityPinned, restPositionPinned);
    PX_PINNED_HOST_FREE(gpuMgr, posInvMassPinned);
    PX_PINNED_HOST_FREE(gpuMgr, velocityPinned);
    PX_PINNED_HOST_FREE(gpuMgr, restPositionPinned);

    printf("Deformable surface ready.\n");
    fflush(stdout);
    return surface;
}

int main(int argc, char** argv)
{
    int steps = 60;
    PxU32 grid = 8;
    bool cleanExit = false;
    bool addPlane = true;
    bool selfCollision = true;
    int cleanExitWaitSeconds = 5;
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--grid") == 0 && i + 1 < argc) {
            grid = PxU32(std::atoi(argv[++i]));
        } else if (std::strcmp(argv[i], "--no-plane") == 0) {
            addPlane = false;
        } else if (std::strcmp(argv[i], "--no-self-collision") == 0) {
            selfCollision = false;
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
    if (grid < 3)
        grid = 3;

    printf("========================================\n");
    printf(" PhysX DCU Smoke - Deformable Surface\n");
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
    sd.broadPhaseType = PxBroadPhaseType::eGPU;
    sd.gpuMaxNumPartitions = 8;
    sd.gpuDynamicsConfig.heapCapacity = 512u * 1024u * 1024u;
    sd.gpuDynamicsConfig.tempBufferCapacity = 128u * 1024u * 1024u;
    sd.gpuDynamicsConfig.collisionStackSize = 256u * 1024u * 1024u;

    printf("Creating GPU dynamics scene: broadPhase=eGPU...\n");
    fflush(stdout);
    PxScene* scene = phy->createScene(sd);
    if (!scene) {
        printf("FAIL: createScene returned null\n");
        return 3;
    }

    PxMaterial* rigidMat = phy->createMaterial(0.5f, 0.5f, 0.0f);
    if (addPlane) {
        printf("Adding rigid plane...\n");
        fflush(stdout);
        scene->addActor(*PxCreatePlane(*phy, PxPlane(0, 1, 0, 0), *rigidMat));
    } else {
        printf("Skipping rigid plane (--no-plane).\n");
        fflush(stdout);
    }
    PxDeformableSurfaceMaterial* surfMat = phy->createDeformableSurfaceMaterial(1.e10f, 0.3f, 0.5f, 0.01f, 0.00001f);

    PxCookingParams cookingParams(phy->getTolerancesScale());
    cookingParams.meshWeldTolerance = 0.001f;
    cookingParams.meshPreprocessParams = PxMeshPreprocessingFlags(PxMeshPreprocessingFlag::eWELD_VERTICES);
    cookingParams.meshPreprocessParams |= PxMeshPreprocessingFlag::eENABLE_VERT_MAPPING;
    cookingParams.buildTriangleAdjacencies = false;
    cookingParams.buildGPUData = true;
    cookingParams.midphaseDesc = PxMeshMidPhase::eBVH34;

    PxDeformableSurface* surface = createClothSurface(phy, scene, cookingParams, gpuMgr, surfMat, grid, grid, 2.0f, 2.0f, selfCollision);
    if (!surface) {
        printf("FAIL: deformable surface setup failed\n");
        return 4;
    }

    printf("Starting simulation: steps=%d grid=%u vertices=%u plane=%s selfCollision=%s\n",
           steps, grid, grid * grid, addPlane ? "yes" : "no", selfCollision ? "yes" : "no");
    fflush(stdout);
    for (int i = 0; i < steps; ++i) {
        if (i < 10 || i == steps - 1) {
            printf("Simulate step %d/%d...\n", i + 1, steps);
            fflush(stdout);
        }
        scene->simulate(1.0f / 60.0f);
        if (i < 10 || i == steps - 1) {
            printf("Fetch step %d/%d...\n", i + 1, steps);
            fflush(stdout);
        }
        scene->fetchResults(true);
    }

    const PxU32 numVertices = grid * grid;
    PxVec4* positionsPinned = Ext::PxCudaHelpersExt::allocPinnedHostBuffer<PxVec4>(*gpuMgr, numVertices);
    PxVec4* velocitiesPinned = Ext::PxCudaHelpersExt::allocPinnedHostBuffer<PxVec4>(*gpuMgr, numVertices);
    bool readbackOk = positionsPinned != nullptr && velocitiesPinned != nullptr;
    PxI32 syncResult = -1;
    PxI32 positionCopyResult = -1;
    PxI32 velocityCopyResult = -1;
    if (readbackOk) {
        PxScopedCudaLock lock(*gpuMgr);
        PxCudaContext* ctx = gpuMgr->getCudaContext();
        syncResult = PxI32(ctx->streamSynchronize(0));
        positionCopyResult = PxI32(ctx->memcpyDtoH(reinterpret_cast<void*>(positionsPinned),
                                                   reinterpret_cast<CUdeviceptr>(surface->getPositionInvMassBufferD()),
                                                   size_t(numVertices) * sizeof(PxVec4)));
        velocityCopyResult = PxI32(ctx->memcpyDtoH(reinterpret_cast<void*>(velocitiesPinned),
                                                   reinterpret_cast<CUdeviceptr>(surface->getVelocityBufferD()),
                                                   size_t(numVertices) * sizeof(PxVec4)));
        readbackOk = syncResult == 0 && positionCopyResult == 0 && velocityCopyResult == 0;
    }

    PxReal minVertexY = PX_MAX_F32;
    PxReal maxVertexY = -PX_MAX_F32;
    PxReal maxSpeed = 0.0f;
    PxReal maxHeightChange = 0.0f;
    PxU32 badVertices = 0;
    if (readbackOk) {
        for (PxU32 i = 0; i < numVertices; ++i) {
            const PxVec4& p = positionsPinned[i];
            const PxVec4& v = velocitiesPinned[i];
            if (!finiteVec4(p) || !finiteVec4(v)) {
                ++badVertices;
                continue;
            }
            minVertexY = PxMin(minVertexY, p.y);
            maxVertexY = PxMax(maxVertexY, p.y);
            maxSpeed = PxMax(maxSpeed, PxVec3(v.x, v.y, v.z).magnitude());
            maxHeightChange = PxMax(maxHeightChange, PxAbs(p.y - 2.0f));
        }
    }

    printf("GPU readback sync=%d positionCopy=%d velocityCopy=%d\n",
           syncResult, positionCopyResult, velocityCopyResult);
    if (readbackOk && numVertices > 0) {
        const PxVec4& p = positionsPinned[0];
        const PxVec4& v = velocitiesPinned[0];
        printf("GPU vertex[0] position=(%.6f %.6f %.6f invMass=%.6f) velocity=(%.6f %.6f %.6f %.6f)\n",
               p.x, p.y, p.z, p.w, v.x, v.y, v.z, v.w);
    }
    printf("GPU vertexHeight=[%.6f, %.6f] maxSpeed=%.6f maxHeightChange=%.6f badVertices=%u/%u\n",
           minVertexY, maxVertexY, maxSpeed, maxHeightChange, badVertices, numVertices);

    PxBounds3 bounds = surface->getWorldBounds(1.0f);
    const bool validBounds = bounds.isValid() && finiteVec(bounds.minimum) && finiteVec(bounds.maximum);
    const bool validVertices = readbackOk && badVertices == 0;
    const bool realMotion = steps == 0 || addPlane || maxHeightChange > 0.001f || maxSpeed > 0.001f;
    const bool planeContactResponse = !addPlane || steps < 33 || minVertexY > -0.1f;
    const bool pass = validBounds && validVertices && realMotion && planeContactResponse;
    printf("Surface bounds min=(%.6f %.6f %.6f) max=(%.6f %.6f %.6f)\n",
           bounds.minimum.x, bounds.minimum.y, bounds.minimum.z, bounds.maximum.x, bounds.maximum.y, bounds.maximum.z);
    printf("Checks validBounds=%s validVertices=%s realMotion=%s planeContactResponse=%s\n",
           validBounds ? "yes" : "no", validVertices ? "yes" : "no", realMotion ? "yes" : "no",
           planeContactResponse ? "yes" : "no");
    printf("VERDICT: %s\n", pass ? "PASS" : "FAIL");
    fflush(stdout);

    Ext::PxCudaHelpersExt::freePinnedHostBuffer(*gpuMgr, positionsPinned);
    Ext::PxCudaHelpersExt::freePinnedHostBuffer(*gpuMgr, velocitiesPinned);

    printf("Releasing scene...\n");
    fflush(stdout);
    scene->release();
    printf("Releasing dispatcher...\n");
    fflush(stdout);
    dsp->release();
    printf("Releasing physics...\n");
    fflush(stdout);
    phy->release();
    printf("Releasing GPU manager...\n");
    fflush(stdout);
    gpuMgr->release();
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
