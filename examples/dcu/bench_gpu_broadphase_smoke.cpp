// gfx936 forced-GPU-broadphase correctness and pair-churn smoke test.

#include "PxPhysicsAPI.h"
#include "cudamanager/PxCudaContextManager.h"
#include "gpu/PxGpu.h"

#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

using namespace physx;

static const char* const kBuildMarker = "PX_DCU_GPU_BROADPHASE_SMOKE_V1_PAIR_CHURN";

class CountingErrorCallback : public PxErrorCallback
{
public:
    CountingErrorCallback() : mErrorCount(0) {}

    void reportError(PxErrorCode::Enum code, const char* message, const char* file, int line) PX_OVERRIDE
    {
        std::printf("[PHYSX] code=%d file=%s line=%d message=%s\n",
                    int(code), file ? file : "<unknown>", line, message ? message : "<empty>");
        std::fflush(stdout);
        if (code != PxErrorCode::eNO_ERROR && code != PxErrorCode::eDEBUG_INFO)
            mErrorCount.fetch_add(1, std::memory_order_relaxed);
    }

    int getErrorCount() const
    {
        return mErrorCount.load(std::memory_order_relaxed);
    }

private:
    std::atomic<int> mErrorCount;
};

struct EventSnapshot
{
    PxU64 found;
    PxU64 lost;
    PxU64 wake;
    PxU64 sleep;
};

class SimulationEventCounter : public PxSimulationEventCallback
{
public:
    SimulationEventCounter() : mFound(0), mLost(0), mWake(0), mSleep(0) {}

    EventSnapshot snapshot() const
    {
        EventSnapshot result = {mFound, mLost, mWake, mSleep};
        return result;
    }

    void onConstraintBreak(PxConstraintInfo*, PxU32) PX_OVERRIDE {}

    void onWake(PxActor**, PxU32 count) PX_OVERRIDE
    {
        mWake += count;
    }

    void onSleep(PxActor**, PxU32 count) PX_OVERRIDE
    {
        mSleep += count;
    }

    void onContact(const PxContactPairHeader&, const PxContactPair* pairs, PxU32 count) PX_OVERRIDE
    {
        for (PxU32 i = 0; i < count; ++i) {
            if (pairs[i].events & PxPairFlag::eNOTIFY_TOUCH_FOUND)
                ++mFound;
            if (pairs[i].events & PxPairFlag::eNOTIFY_TOUCH_LOST)
                ++mLost;
        }
    }

    void onTrigger(PxTriggerPair*, PxU32) PX_OVERRIDE {}
    void onAdvance(const PxRigidBody* const*, const PxTransform*, PxU32) PX_OVERRIDE {}

private:
    PxU64 mFound;
    PxU64 mLost;
    PxU64 mWake;
    PxU64 mSleep;
};

static PxFilterFlags pairChurnFilterShader(
    PxFilterObjectAttributes,
    PxFilterData,
    PxFilterObjectAttributes,
    PxFilterData,
    PxPairFlags& pairFlags,
    const void*,
    PxU32)
{
    pairFlags = PxPairFlag::eCONTACT_DEFAULT |
                PxPairFlag::eNOTIFY_TOUCH_FOUND |
                PxPairFlag::eNOTIFY_TOUCH_LOST;
    return PxFilterFlag::eDEFAULT;
}

struct RunMetrics
{
    RunMetrics()
        : simulationSteps(0), fetchFailures(0), badStates(0), firstBadStep(0),
          maxPoseError(0.0f), maxLinearSpeed(0.0f), maxAngularSpeed(0.0f),
          measuredSteps(0), simulationMs(0.0)
    {
    }

    int simulationSteps;
    int fetchFailures;
    int badStates;
    int firstBadStep;
    float maxPoseError;
    float maxLinearSpeed;
    float maxAngularSpeed;
    int measuredSteps;
    double simulationMs;
};

static bool isFinite(const PxVec3& value)
{
    return std::isfinite(value.x) && std::isfinite(value.y) && std::isfinite(value.z);
}

static bool isFinite(const PxQuat& value)
{
    return std::isfinite(value.x) && std::isfinite(value.y) &&
           std::isfinite(value.z) && std::isfinite(value.w);
}

static bool simulateSteps(
    PxScene& scene,
    const std::vector<PxRigidDynamic*>& bodies,
    const std::vector<PxVec3>& nearPositions,
    const PxVec3& offset,
    int steps,
    RunMetrics& metrics)
{
    const float dt = 1.0f / 60.0f;
    for (int localStep = 0; localStep < steps; ++localStep) {
        const std::chrono::steady_clock::time_point stepStart = std::chrono::steady_clock::now();
        scene.simulate(dt);
        const bool fetchSucceeded = scene.fetchResults(true);
        if (!fetchSucceeded) {
            ++metrics.fetchFailures;
            if (metrics.firstBadStep == 0)
                metrics.firstBadStep = metrics.simulationSteps + 1;
            return false;
        }
        if (metrics.simulationSteps >= 10) {
            metrics.simulationMs += std::chrono::duration<double, std::milli>(
                std::chrono::steady_clock::now() - stepStart).count();
            ++metrics.measuredSteps;
        }

        ++metrics.simulationSteps;
        for (size_t i = 0; i < bodies.size(); ++i) {
            const PxTransform pose = bodies[i]->getGlobalPose();
            const PxVec3 linearVelocity = bodies[i]->getLinearVelocity();
            const PxVec3 angularVelocity = bodies[i]->getAngularVelocity();
            const bool finite = isFinite(pose.p) && isFinite(pose.q) &&
                                isFinite(linearVelocity) && isFinite(angularVelocity);
            if (!finite) {
                ++metrics.badStates;
                if (metrics.firstBadStep == 0)
                    metrics.firstBadStep = metrics.simulationSteps;
                continue;
            }

            const float poseError = (pose.p - (nearPositions[i] + offset)).magnitude();
            metrics.maxPoseError = PxMax(metrics.maxPoseError, poseError);
            metrics.maxLinearSpeed = PxMax(metrics.maxLinearSpeed, linearVelocity.magnitude());
            metrics.maxAngularSpeed = PxMax(metrics.maxAngularSpeed, angularVelocity.magnitude());
        }
    }
    return true;
}

static int countSleeping(const std::vector<PxRigidDynamic*>& bodies)
{
    int sleeping = 0;
    for (PxRigidDynamic* body : bodies) {
        if (body->isSleeping())
            ++sleeping;
    }
    return sleeping;
}

static PxU64 counterDelta(PxU64 after, PxU64 before)
{
    return after >= before ? after - before : 0;
}

static bool checkPhaseEvents(
    const char* phase,
    int cycle,
    const EventSnapshot& before,
    const EventSnapshot& after,
    PxU64 expectedFound,
    PxU64 expectedLost)
{
    const PxU64 found = counterDelta(after.found, before.found);
    const PxU64 lost = counterDelta(after.lost, before.lost);
    const PxU64 wake = counterDelta(after.wake, before.wake);
    const PxU64 sleep = counterDelta(after.sleep, before.sleep);
    const bool pass = found == expectedFound && lost == expectedLost;

    std::printf(
        "Cycle %d %s: found=%llu/%llu lost=%llu/%llu wake=%llu sleep=%llu %s\n",
        cycle,
        phase,
        static_cast<unsigned long long>(found),
        static_cast<unsigned long long>(expectedFound),
        static_cast<unsigned long long>(lost),
        static_cast<unsigned long long>(expectedLost),
        static_cast<unsigned long long>(wake),
        static_cast<unsigned long long>(sleep),
        pass ? "PASS" : "FAIL");
    return pass;
}

static bool parsePositiveInt(const char* text, int minimum, int maximum, int& value)
{
    char* end = NULL;
    const long parsed = std::strtol(text, &end, 10);
    if (!text[0] || !end || *end != '\0' || parsed < minimum || parsed > maximum)
        return false;
    value = static_cast<int>(parsed);
    return true;
}

static void printUsage(const char* executable)
{
    std::printf(
        "Usage: %s [--bodies N] [--cycles N] [--transition-steps N] [--settle-steps N]\n",
        executable);
}

int main(int argc, char** argv)
{
    int bodyCount = 512;
    int cycles = 8;
    int transitionSteps = 2;
    int settleSteps = 60;

    for (int i = 1; i < argc; ++i) {
        int* destination = NULL;
        int minimum = 1;
        int maximum = 0;
        if (std::strcmp(argv[i], "--bodies") == 0) {
            destination = &bodyCount;
            minimum = 64;
            maximum = 4096;
        } else if (std::strcmp(argv[i], "--cycles") == 0) {
            destination = &cycles;
            maximum = 100;
        } else if (std::strcmp(argv[i], "--transition-steps") == 0) {
            destination = &transitionSteps;
            maximum = 30;
        } else if (std::strcmp(argv[i], "--settle-steps") == 0) {
            destination = &settleSteps;
            minimum = 30;
            maximum = 600;
        } else if (std::strcmp(argv[i], "--help") == 0) {
            printUsage(argv[0]);
            return 0;
        } else {
            std::printf("FAIL: unknown argument: %s\n", argv[i]);
            printUsage(argv[0]);
            return 2;
        }

        if (++i >= argc || !parsePositiveInt(argv[i], minimum, maximum, *destination)) {
            std::printf("FAIL: invalid value for %s\n", argv[i - 1]);
            printUsage(argv[0]);
            return 2;
        }
    }

    std::printf("========================================\n");
    std::printf(" PhysX DCU Forced GPU Broadphase Smoke\n");
    std::printf("========================================\n");
    std::printf("Marker: %s\n", kBuildMarker);
    std::printf("Configuration: bodies=%d cycles=%d transition_steps=%d settle_steps=%d\n",
                bodyCount, cycles, transitionSteps, settleSteps);

    CountingErrorCallback errorCallback;
    PxDefaultAllocator allocator;
    PxFoundation* foundation = PxCreateFoundation(PX_PHYSICS_VERSION, allocator, errorCallback);
    if (!foundation) {
        std::printf("VERDICT: FAIL\n");
        return 3;
    }

    PxPhysics* physics = PxCreatePhysics(PX_PHYSICS_VERSION, *foundation, PxTolerancesScale());
    if (!physics) {
        foundation->release();
        std::printf("VERDICT: FAIL\n");
        return 3;
    }

    PxCudaContextManagerDesc gpuDesc;
    gpuDesc.deviceOrdinal = 0;
    PxCudaContextManager* gpuManager = PxCreateCudaContextManager(*foundation, gpuDesc, NULL, false);
    const bool gpuValid = gpuManager && gpuManager->contextIsValid();
    std::printf("GPU: %s - %s\n",
                gpuValid ? gpuManager->getDeviceName() : "NONE",
                gpuValid ? "enabled" : "disabled");
    if (!gpuValid) {
        if (gpuManager)
            gpuManager->release();
        physics->release();
        foundation->release();
        std::printf("VERDICT: FAIL\n");
        return 4;
    }

    SimulationEventCounter eventCounter;
    PxDefaultCpuDispatcher* dispatcher = PxDefaultCpuDispatcherCreate(0);
    PxSceneDesc sceneDesc(physics->getTolerancesScale());
    sceneDesc.gravity = PxVec3(0.0f);
    sceneDesc.cpuDispatcher = dispatcher;
    sceneDesc.cudaContextManager = gpuManager;
    sceneDesc.filterShader = pairChurnFilterShader;
    sceneDesc.simulationEventCallback = &eventCounter;
    sceneDesc.flags |= PxSceneFlag::eENABLE_GPU_DYNAMICS;
    sceneDesc.broadPhaseType = PxBroadPhaseType::eGPU;
    sceneDesc.solverType = PxSolverType::ePGS;
    sceneDesc.gpuMaxNumPartitions = 8;
    sceneDesc.gpuDynamicsConfig.maxRigidContactCount = 1024u * 1024u * 8u;
    sceneDesc.gpuDynamicsConfig.maxRigidPatchCount = 1024u * 1024u;
    sceneDesc.gpuDynamicsConfig.foundLostPairsCapacity = 1024u * 1024u * 2u;
    sceneDesc.gpuDynamicsConfig.heapCapacity = 512u * 1024u * 1024u;
    sceneDesc.gpuDynamicsConfig.collisionStackSize = 256u * 1024u * 1024u;
    sceneDesc.gpuDynamicsConfig.tempBufferCapacity = 128u * 1024u * 1024u;

    PxScene* scene = dispatcher ? physics->createScene(sceneDesc) : NULL;
    if (!scene) {
        std::printf("FAIL: createScene returned null\n");
        if (dispatcher)
            dispatcher->release();
        gpuManager->release();
        physics->release();
        foundation->release();
        std::printf("VERDICT: FAIL\n");
        return 5;
    }

    const PxBroadPhaseType::Enum actualBroadphase = scene->getBroadPhaseType();
    std::printf("Scene broadphase type: %s\n",
                actualBroadphase == PxBroadPhaseType::eGPU ? "eGPU" : "non-GPU");

    PxMaterial* material = physics->createMaterial(0.5f, 0.5f, 0.0f);
    PxShape* staticShape = material ? physics->createShape(PxBoxGeometry(0.45f, 0.45f, 0.45f), *material) : NULL;
    PxShape* dynamicShape = material ? physics->createShape(PxBoxGeometry(0.45f, 0.45f, 0.45f), *material) : NULL;
    std::vector<PxRigidStatic*> staticActors;
    std::vector<PxRigidDynamic*> bodies;
    std::vector<PxVec3> nearPositions;
    staticActors.reserve(bodyCount);
    bodies.reserve(bodyCount);
    nearPositions.reserve(bodyCount);

    bool creationOk = material && staticShape && dynamicShape;
    int columns = 1;
    while (columns * columns < bodyCount)
        ++columns;

    const float spacing = 2.5f;
    const PxVec3 farOffset(0.0f, 50.0f, 0.0f);
    const PxRigidDynamicLockFlags lockFlags =
        PxRigidDynamicLockFlag::eLOCK_LINEAR_X |
        PxRigidDynamicLockFlag::eLOCK_LINEAR_Y |
        PxRigidDynamicLockFlag::eLOCK_LINEAR_Z |
        PxRigidDynamicLockFlag::eLOCK_ANGULAR_X |
        PxRigidDynamicLockFlag::eLOCK_ANGULAR_Y |
        PxRigidDynamicLockFlag::eLOCK_ANGULAR_Z;

    for (int i = 0; creationOk && i < bodyCount; ++i) {
        const int xIndex = i % columns;
        const int zIndex = i / columns;
        const float x = (xIndex - 0.5f * float(columns - 1)) * spacing;
        const float z = (zIndex - 0.5f * float(columns - 1)) * spacing;
        const PxVec3 staticPosition(x, 0.0f, z);
        const PxVec3 dynamicPosition(x, 0.86f, z);

        PxRigidStatic* anchor = physics->createRigidStatic(PxTransform(staticPosition));
        PxRigidDynamic* body = physics->createRigidDynamic(PxTransform(dynamicPosition));
        if (!anchor || !body) {
            if (anchor)
                anchor->release();
            if (body)
                body->release();
            creationOk = false;
            break;
        }

        anchor->attachShape(*staticShape);
        body->attachShape(*dynamicShape);
        body->setRigidDynamicLockFlags(lockFlags);
        body->setActorFlag(PxActorFlag::eSEND_SLEEP_NOTIFIES, true);
        PxRigidBodyExt::updateMassAndInertia(*body, 1.0f);
        scene->addActor(*anchor);
        scene->addActor(*body);
        staticActors.push_back(anchor);
        bodies.push_back(body);
        nearPositions.push_back(dynamicPosition);
    }

    RunMetrics metrics;
    int phaseFailures = 0;
    bool simulationOk = creationOk && static_cast<int>(bodies.size()) == bodyCount;
    if (!simulationOk)
        std::printf("FAIL: actor creation stopped at %zu/%d bodies\n", bodies.size(), bodyCount);

    if (simulationOk) {
        const EventSnapshot before = eventCounter.snapshot();
        simulationOk = simulateSteps(*scene, bodies, nearPositions, PxVec3(0.0f),
                                     transitionSteps, metrics);
        const EventSnapshot after = eventCounter.snapshot();
        if (!checkPhaseEvents("initial-found", 0, before, after,
                              PxU64(bodyCount), 0))
            ++phaseFailures;
    }

    for (int cycle = 1; simulationOk && cycle <= cycles; ++cycle) {
        EventSnapshot before = eventCounter.snapshot();
        for (size_t i = 0; i < bodies.size(); ++i) {
            bodies[i]->setGlobalPose(PxTransform(nearPositions[i] + farOffset), false);
            bodies[i]->putToSleep();
        }
        simulationOk = simulateSteps(*scene, bodies, nearPositions, farOffset,
                                     transitionSteps, metrics);
        EventSnapshot after = eventCounter.snapshot();
        if (!checkPhaseEvents("move-out", cycle, before, after,
                              0, PxU64(bodyCount)))
            ++phaseFailures;
        const int sleepingAfterMoveOut = countSleeping(bodies);
        std::printf("Cycle %d move-out sleeping=%d/%d %s\n",
                    cycle, sleepingAfterMoveOut, bodyCount,
                    sleepingAfterMoveOut == bodyCount ? "PASS" : "FAIL");
        if (sleepingAfterMoveOut != bodyCount)
            ++phaseFailures;

        before = eventCounter.snapshot();
        for (size_t i = 0; i < bodies.size(); ++i) {
            bodies[i]->setGlobalPose(PxTransform(nearPositions[i]), false);
            bodies[i]->wakeUp();
        }
        simulationOk = simulateSteps(*scene, bodies, nearPositions, PxVec3(0.0f),
                                     transitionSteps, metrics);
        after = eventCounter.snapshot();
        if (!checkPhaseEvents("move-in", cycle, before, after,
                              PxU64(bodyCount), 0))
            ++phaseFailures;
        const int sleepingAfterMoveIn = countSleeping(bodies);
        std::printf("Cycle %d move-in awake=%d/%d %s\n",
                    cycle, bodyCount - sleepingAfterMoveIn, bodyCount,
                    sleepingAfterMoveIn == 0 ? "PASS" : "FAIL");
        if (sleepingAfterMoveIn != 0)
            ++phaseFailures;
    }

    if (simulationOk) {
        const EventSnapshot before = eventCounter.snapshot();
        simulationOk = simulateSteps(*scene, bodies, nearPositions, PxVec3(0.0f),
                                     settleSteps, metrics);
        const EventSnapshot after = eventCounter.snapshot();
        const PxU64 settleSleep = counterDelta(after.sleep, before.sleep);
        const PxU64 settleFound = counterDelta(after.found, before.found);
        const PxU64 settleLost = counterDelta(after.lost, before.lost);
        const int finalSleeping = countSleeping(bodies);
        const bool settlePass = finalSleeping == bodyCount && settleFound == 0 && settleLost == 0;
        std::printf("Final settle: sleeping=%d/%d sleep_events=%llu found=%llu lost=%llu %s\n",
                    finalSleeping,
                    bodyCount,
                    static_cast<unsigned long long>(settleSleep),
                    static_cast<unsigned long long>(settleFound),
                    static_cast<unsigned long long>(settleLost),
                    settlePass ? "PASS" : "FAIL");
        if (!settlePass)
            ++phaseFailures;
    }

    const EventSnapshot totals = eventCounter.snapshot();
    const PxU64 expectedFound = PxU64(bodyCount) * PxU64(cycles + 1);
    const PxU64 expectedLost = PxU64(bodyCount) * PxU64(cycles);
    const PxU64 minimumWake = PxU64(bodyCount) * PxU64(cycles + 1);
    const PxU64 minimumSleep = PxU64(bodyCount) * PxU64(cycles + 1);
    const bool totalsMatch = totals.found == expectedFound && totals.lost == expectedLost;
    const bool sleepWakeCovered = totals.wake >= minimumWake && totals.sleep >= minimumSleep;
    const bool broadphaseMatches = actualBroadphase == PxBroadPhaseType::eGPU;
    const bool statePass = metrics.fetchFailures == 0 && metrics.badStates == 0 &&
                           metrics.firstBadStep == 0 && metrics.maxPoseError < 1.0e-4f &&
                           metrics.maxLinearSpeed < 1.0e-4f && metrics.maxAngularSpeed < 1.0e-4f;
    const int preReleaseErrors = errorCallback.getErrorCount();
    bool pass = simulationOk && broadphaseMatches && totalsMatch && sleepWakeCovered &&
                phaseFailures == 0 && statePass && preReleaseErrors == 0;

    std::printf("\n=== Results ===\n");
    std::printf("Simulation steps: %d\n", metrics.simulationSteps);
    std::printf("PERF marker=PX_DCU_BENCH_SIMULATION_TIMING_V1 warmup_steps=%d measured_steps=%d simulation_ms=%.3f ms_per_step=%.6f steps_per_second=%.3f\n",
                metrics.simulationSteps - metrics.measuredSteps, metrics.measuredSteps, metrics.simulationMs,
                metrics.measuredSteps > 0 ? metrics.simulationMs / double(metrics.measuredSteps) : 0.0,
                metrics.simulationMs > 0.0 ? double(metrics.measuredSteps) * 1000.0 / metrics.simulationMs : 0.0);
    std::printf("Bodies: %d\n", bodyCount);
    std::printf("Cycles: %d\n", cycles);
    std::printf("Touch found: %llu / %llu\n",
                static_cast<unsigned long long>(totals.found),
                static_cast<unsigned long long>(expectedFound));
    std::printf("Touch lost: %llu / %llu\n",
                static_cast<unsigned long long>(totals.lost),
                static_cast<unsigned long long>(expectedLost));
    std::printf("Wake events: %llu / >= %llu\n",
                static_cast<unsigned long long>(totals.wake),
                static_cast<unsigned long long>(minimumWake));
    std::printf("Sleep events: %llu / >= %llu\n",
                static_cast<unsigned long long>(totals.sleep),
                static_cast<unsigned long long>(minimumSleep));
    std::printf("Phase failures: %d\n", phaseFailures);
    std::printf("Fetch failures: %d\n", metrics.fetchFailures);
    std::printf("Bad states: %d\n", metrics.badStates);
    std::printf("First bad step: %d\n", metrics.firstBadStep);
    std::printf("Max pose error: %.9f\n", metrics.maxPoseError);
    std::printf("Max linear speed: %.9f\n", metrics.maxLinearSpeed);
    std::printf("Max angular speed: %.9f\n", metrics.maxAngularSpeed);
    std::printf("PhysX errors/warnings before release: %d\n", preReleaseErrors);
    std::printf("Broadphase selection: %s\n", broadphaseMatches ? "PASS" : "FAIL");
    std::printf("Pair totals: %s\n", totalsMatch ? "PASS" : "FAIL");
    std::printf("Sleep/wake coverage: %s\n", sleepWakeCovered ? "PASS" : "FAIL");
    std::fflush(stdout);

    for (PxRigidDynamic* body : bodies)
        body->release();
    for (PxRigidStatic* actor : staticActors)
        actor->release();
    if (dynamicShape)
        dynamicShape->release();
    if (staticShape)
        staticShape->release();
    if (material)
        material->release();
    std::printf("Releasing scene...\n");
    std::fflush(stdout);
    scene->release();
    dispatcher->release();
    gpuManager->release();
    physics->release();
    foundation->release();
    const int finalErrors = errorCallback.getErrorCount();
    pass = pass && finalErrors == 0;
    std::printf("Release finished. PhysX errors/warnings after release: %d\n", finalErrors);
    std::printf("VERDICT: %s\n", pass ? "PASS" : "FAIL");
    std::fflush(stdout);
    return pass ? 0 : 1;
}
