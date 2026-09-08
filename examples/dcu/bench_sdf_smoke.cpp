// PhysX DCU smoke test: triangle mesh SDF cooking and GPU dynamics contact.

#include "PxPhysicsAPI.h"
#include "cudamanager/PxCudaContextManager.h"
#include "gpu/PxGpu.h"
#include <cmath>
#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <chrono>
#include <thread>
#include <vector>

using namespace physx;

class SmokeErrorCallback : public PxErrorCallback
{
public:
    void reportError(PxErrorCode::Enum code, const char* message, const char* file, int line) override
    {
        mErrorCount.fetch_add(1, std::memory_order_relaxed);
        if (code == PxErrorCode::eINVALID_PARAMETER ||
            code == PxErrorCode::eINVALID_OPERATION ||
            code == PxErrorCode::eOUT_OF_MEMORY ||
            code == PxErrorCode::eINTERNAL_ERROR ||
            code == PxErrorCode::eABORT)
            mFatalErrorCount.fetch_add(1, std::memory_order_relaxed);
        mDefaultCallback.reportError(code, message, file, line);
    }

    PxU32 getErrorCount() const
    {
        return PxU32(mErrorCount.load(std::memory_order_relaxed));
    }

    PxU32 getFatalErrorCount() const
    {
        return PxU32(mFatalErrorCount.load(std::memory_order_relaxed));
    }

private:
    PxDefaultErrorCallback mDefaultCallback;
    std::atomic<PxU32> mErrorCount{0};
    std::atomic<PxU32> mFatalErrorCount{0};
};

static bool finiteVec(const PxVec3& v)
{
    return std::isfinite(v.x) && std::isfinite(v.y) && std::isfinite(v.z);
}

static PxFilterFlags planeOnlyFilterShader(
    PxFilterObjectAttributes attributes0, PxFilterData filterData0,
    PxFilterObjectAttributes attributes1, PxFilterData filterData1,
    PxPairFlags& pairFlags, const void* constantBlock, PxU32 constantBlockSize)
{
    if (PxGetFilterObjectType(attributes0) == PxFilterObjectType::eRIGID_DYNAMIC &&
        PxGetFilterObjectType(attributes1) == PxFilterObjectType::eRIGID_DYNAMIC)
        return PxFilterFlag::eSUPPRESS;

    return PxDefaultSimulationFilterShader(attributes0, filterData0, attributes1, filterData1,
        pairFlags, constantBlock, constantBlockSize);
}

static PxTriangleMesh* createSdfCubeMesh(PxPhysics* phy, PxCookingParams& params)
{
    static PxVec3 verts[] = {
        PxVec3(-0.5f, -0.5f, -0.5f), PxVec3( 0.5f, -0.5f, -0.5f),
        PxVec3( 0.5f,  0.5f, -0.5f), PxVec3(-0.5f,  0.5f, -0.5f),
        PxVec3(-0.5f, -0.5f,  0.5f), PxVec3( 0.5f, -0.5f,  0.5f),
        PxVec3( 0.5f,  0.5f,  0.5f), PxVec3(-0.5f,  0.5f,  0.5f)
    };
    static PxU32 indices[] = {
        0, 2, 1, 0, 3, 2,
        4, 5, 6, 4, 6, 7,
        0, 1, 5, 0, 5, 4,
        3, 6, 2, 3, 7, 6,
        1, 2, 6, 1, 6, 5,
        0, 4, 7, 0, 7, 3
    };

    PxTriangleMeshDesc desc;
    desc.points.count = 8;
    desc.points.stride = sizeof(PxVec3);
    desc.points.data = verts;
    desc.triangles.count = 12;
    desc.triangles.stride = 3 * sizeof(PxU32);
    desc.triangles.data = indices;

    PxSDFDesc sdfDesc;
    sdfDesc.spacing = 0.05f;
    sdfDesc.subgridSize = 6;
    sdfDesc.bitsPerSubgridPixel = PxSdfBitsPerSubgridPixel::e16_BIT_PER_PIXEL;
    sdfDesc.numThreadsForSdfConstruction = 4;
    desc.sdfDesc = &sdfDesc;

    params.meshPreprocessParams = PxMeshPreprocessingFlags(PxMeshPreprocessingFlag::eWELD_VERTICES);
    params.meshPreprocessParams |= PxMeshPreprocessingFlag::eENABLE_INERTIA;
    params.meshWeldTolerance = 0.001f;
    params.buildTriangleAdjacencies = false;
    params.buildGPUData = true;

    PxTriangleMeshCookingResult::Enum result;
    PxTriangleMesh* mesh = PxCreateTriangleMesh(params, desc, phy->getPhysicsInsertionCallback(), &result);
    printf("SDF cube cooking result: %d, mesh=%p\n", int(result), static_cast<void*>(mesh));
    fflush(stdout);
    return mesh;
}

int main(int argc, char** argv)
{
    int steps = 240;
    int bodyCount = 8;
    float bodySpacing = 1.25f;
    bool useTgs = false;
    bool useGpuBroadphase = false;
    bool cookOnly = false;
    bool printAllSteps = false;
    bool planeOnly = false;
    bool cleanExit = true;
    int cleanExitWaitSeconds = 0;
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "tgs") == 0 || std::strcmp(argv[i], "TGS") == 0) {
            useTgs = true;
        } else if (std::strcmp(argv[i], "pgs") == 0 || std::strcmp(argv[i], "PGS") == 0) {
            useTgs = false;
        } else if (std::strcmp(argv[i], "gpubp") == 0 || std::strcmp(argv[i], "--gpubp") == 0) {
            useGpuBroadphase = true;
        } else if (std::strcmp(argv[i], "cookonly") == 0 || std::strcmp(argv[i], "--cookonly") == 0) {
            cookOnly = true;
        } else if (std::strcmp(argv[i], "--print-all-steps") == 0) {
            printAllSteps = true;
        } else if (std::strcmp(argv[i], "--plane-only") == 0) {
            planeOnly = true;
        } else if (std::strcmp(argv[i], "--cleanexit") == 0) {
            cleanExit = true;
        } else if (std::strcmp(argv[i], "--no-cleanexit") == 0) {
            cleanExit = false;
        } else if (std::strcmp(argv[i], "--cleanexit-wait") == 0 && i + 1 < argc) {
            cleanExit = true;
            cleanExitWaitSeconds = std::atoi(argv[++i]);
            if (cleanExitWaitSeconds < 0)
                cleanExitWaitSeconds = 0;
        } else if (std::strcmp(argv[i], "--bodies") == 0 && i + 1 < argc) {
            bodyCount = std::atoi(argv[++i]);
        } else if (std::strcmp(argv[i], "--spacing") == 0 && i + 1 < argc) {
            bodySpacing = static_cast<float>(std::atof(argv[++i]));
        } else {
            steps = std::atoi(argv[i]);
        }
    }
    if (steps <= 0)
        steps = 240;
    if (bodyCount <= 0)
        bodyCount = 1;
    if (bodySpacing < 1.05f)
        bodySpacing = 1.05f;

    printf("========================================\n");
    printf(" PhysX DCU Smoke - SDF Triangle Mesh (%s)\n", useTgs ? "TGS" : "PGS");
    printf("========================================\n\n");
    printf("SDF_CORRECTNESS_CRITERIA_SCALE_AWARE_V2\n");

    static SmokeErrorCallback gErr;
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

    PxCookingParams cookingParams(phy->getTolerancesScale());
    printf("Cooking SDF cube mesh...\n");
    fflush(stdout);
    PxTriangleMesh* sdfMesh = createSdfCubeMesh(phy, cookingParams);
    if (!sdfMesh) {
        printf("FAIL: SDF mesh cooking returned null\n");
        gpuMgr->release();
        phy->release();
        fnd->release();
        return 3;
    }
    if (cookOnly) {
        printf("VERDICT: PASS (SDF cooking only)\n");
        fflush(stdout);
        std::_Exit(0);
    }

    PxSceneDesc sd(phy->getTolerancesScale());
    sd.gravity = PxVec3(0.0f, -9.81f, 0.0f);
    sd.cudaContextManager = gpuMgr;
    PxDefaultCpuDispatcher* dsp = PxDefaultCpuDispatcherCreate(0);
    sd.cpuDispatcher = dsp;
    sd.filterShader = planeOnly ? planeOnlyFilterShader : PxDefaultSimulationFilterShader;
    sd.flags |= PxSceneFlag::eENABLE_GPU_DYNAMICS;
    sd.flags |= PxSceneFlag::eENABLE_PCM;
    sd.solverType = useTgs ? PxSolverType::eTGS : PxSolverType::ePGS;
    if (useGpuBroadphase)
        sd.broadPhaseType = PxBroadPhaseType::eGPU;
    sd.gpuMaxNumPartitions = 8;
    sd.gpuDynamicsConfig.maxRigidContactCount = 1024u * 1024u * 8u;
    sd.gpuDynamicsConfig.maxRigidPatchCount = 1024u * 1024u;
    sd.gpuDynamicsConfig.foundLostPairsCapacity = 1024u * 1024u * 2u;
    sd.gpuDynamicsConfig.heapCapacity = 512u * 1024u * 1024u;
    sd.gpuDynamicsConfig.collisionStackSize = 256u * 1024u * 1024u;
    sd.gpuDynamicsConfig.tempBufferCapacity = 128u * 1024u * 1024u;

    printf("Creating scene solver=%s broadphase=%s...\n", useTgs ? "TGS" : "PGS", useGpuBroadphase ? "GPU" : "default");
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

    PxMaterial* mat = phy->createMaterial(0.5f, 0.5f, 0.05f);
    scene->addActor(*PxCreatePlane(*phy, PxPlane(0, 1, 0, 0), *mat));

    PxTriangleMeshGeometry sdfGeom(sdfMesh);
    std::vector<PxRigidDynamic*> bodies;
    const float startX = -0.5f * bodySpacing * float(bodyCount - 1);
    const float initialMaxY = 3.0f + 0.35f * float(bodyCount - 1);
    const float linearDamping = 0.2f;
    for (int i = 0; i < bodyCount; ++i) {
        PxVec3 pos(startX + bodySpacing * i, 3.0f + 0.35f * i, 0.0f);
        PxRigidDynamic* body = phy->createRigidDynamic(PxTransform(pos));
        body->setLinearDamping(linearDamping);
        body->setAngularDamping(0.1f);
        body->setRigidBodyFlag(PxRigidBodyFlag::eENABLE_GYROSCOPIC_FORCES, true);
        body->setRigidBodyFlag(PxRigidBodyFlag::eENABLE_SPECULATIVE_CCD, true);
        PxShape* shape = PxRigidActorExt::createExclusiveShape(*body, sdfGeom, *mat);
        shape->setContactOffset(0.05f);
        shape->setRestOffset(0.0f);
        PxRigidBodyExt::updateMassAndInertia(*body, 100.0f);
        body->setWakeCounter(100000000.0f);
        body->setSolverIterationCounts(50, 1);
        body->setMaxDepenetrationVelocity(5.0f);
        scene->addActor(*body);
        bodies.push_back(body);
    }

    printf("Starting simulation with %zu SDF mesh bodies...\n", bodies.size());
    fflush(stdout);
    int completedSteps = 0;
    bool nonFiniteState = false;
    float trajectoryMinY = 1e30f;
    float trajectoryMaxY = -1e30f;
    float trajectoryMaxSpeed = 0.0f;
    const int timingWarmupSteps = steps > 10 ? 10 : 0;
    double simulationMs = 0.0;
    for (int i = 0; i < steps; ++i) {
        if (i < 3 || i == steps - 1) {
            printf("Simulate step %d/%d...\n", i + 1, steps);
            fflush(stdout);
        }
        const std::chrono::steady_clock::time_point stepStart = std::chrono::steady_clock::now();
        scene->simulate(1.0f / 60.0f);
        scene->fetchResults(true);
		if (i >= timingWarmupSteps)
			simulationMs += std::chrono::duration<double, std::milli>(
				std::chrono::steady_clock::now() - stepStart).count();
		completedSteps = i + 1;
		if (gErr.getFatalErrorCount() != 0) {
			printf("Stopping after step %d due to fatal PhysX errors.\n", completedSteps);
			fflush(stdout);
			break;
		}

        float stepMinY = 1e30f;
        float stepMaxY = -1e30f;
        float stepMaxSpeed = 0.0f;
        for (PxRigidDynamic* body : bodies) {
            const PxVec3 position = body->getGlobalPose().p;
            const PxVec3 velocity = body->getLinearVelocity();
            if (!finiteVec(position) || !finiteVec(velocity))
                nonFiniteState = true;
            stepMinY = PxMin(stepMinY, position.y);
            stepMaxY = PxMax(stepMaxY, position.y);
            stepMaxSpeed = PxMax(stepMaxSpeed, velocity.magnitude());
        }
        trajectoryMinY = PxMin(trajectoryMinY, stepMinY);
        trajectoryMaxY = PxMax(trajectoryMaxY, stepMaxY);
        trajectoryMaxSpeed = PxMax(trajectoryMaxSpeed, stepMaxSpeed);

		if (printAllSteps) {
			printf("Step state %d/%d: height=[%.6f, %.6f] maxSpeed=%.6f\n",
				i + 1, steps, stepMinY, stepMaxY, stepMaxSpeed);
			fflush(stdout);
		}
    }
    const int completedWarmupSteps = completedSteps < timingWarmupSteps ? completedSteps : timingWarmupSteps;
    const int measuredSteps = completedSteps - completedWarmupSteps;
    printf("PERF marker=PX_DCU_BENCH_SIMULATION_TIMING_V1 warmup_steps=%d measured_steps=%d simulation_ms=%.3f ms_per_step=%.6f steps_per_second=%.3f\n",
           completedWarmupSteps, measuredSteps, simulationMs,
           measuredSteps > 0 ? simulationMs / double(measuredSteps) : 0.0,
           simulationMs > 0.0 ? double(measuredSteps) * 1000.0 / simulationMs : 0.0);

    int bad = 0;
    float minY = 1e30f;
    float maxY = -1e30f;
    float maxSpeed = 0.0f;
    for (PxRigidDynamic* b : bodies) {
        const PxVec3 p = b->getGlobalPose().p;
        const PxVec3 v = b->getLinearVelocity();
        if (!finiteVec(p) || !finiteVec(v))
            bad++;
        if (p.y < minY)
            minY = p.y;
        if (p.y > maxY)
            maxY = p.y;
        const float speed = v.magnitude();
        if (speed > maxSpeed)
            maxSpeed = speed;
    }

    printf("Steps: %d/%d\n", completedSteps, steps);
    printf("SDF bodies: %zu\n", bodies.size());
    printf("Height range: [%.6f, %.6f]\n", minY, maxY);
    printf("Max speed: %.6f\n", maxSpeed);
    printf("Trajectory height range: [%.6f, %.6f]\n", trajectoryMinY, trajectoryMaxY);
    printf("Trajectory max speed: %.6f\n", trajectoryMaxSpeed);
    printf("Bad bodies: %d\n", bad);
    printf("PhysX errors: %u (fatal: %u)\n", gErr.getErrorCount(), gErr.getFatalErrorCount());

    const float simulationSeconds = float(steps) / 60.0f;
    const float freeFallSpeedLimit = linearDamping > 0.0f
        ? 9.81f / linearDamping * (1.0f - std::exp(-linearDamping * simulationSeconds))
        : 9.81f * simulationSeconds;
    const float maxSpeedLimit = PxMax(20.0f, freeFallSpeedLimit + 8.0f);
    const float maxHeightLimit = initialMaxY + 8.0f;
    printf("Correctness limits: height=[0.1, %.6f] speed<=%.6f\n",
        maxHeightLimit, maxSpeedLimit);

    const bool pass = completedSteps == steps && bad == 0 && !nonFiniteState &&
        gErr.getFatalErrorCount() == 0 && trajectoryMinY > 0.1f &&
        trajectoryMaxY < maxHeightLimit && trajectoryMaxSpeed <= maxSpeedLimit;
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
