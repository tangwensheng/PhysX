// PhysX DCU smoke test: diffuse buffer generation, update, and primitive collision paths.

#include "PxPhysicsAPI.h"
#include "cudamanager/PxCudaContext.h"
#include "cudamanager/PxCudaContextManager.h"
#include "gpu/PxGpu.h"
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <thread>
#include <vector>

using namespace physx;

class CountingErrorCallback : public PxErrorCallback
{
public:
    CountingErrorCallback() : errorCount(0) {}

    void reportError(PxErrorCode::Enum code, const char* message, const char* file, int line) override
    {
        if (code != PxErrorCode::eDEBUG_INFO && code != PxErrorCode::eDEBUG_WARNING)
            ++errorCount;
        std::fprintf(stderr, "%s (%d) : PhysX error %d: %s\n", file, line, int(code), message);
    }

    PxU32 errorCount;
};

static bool finiteVec4(const PxVec4& value)
{
    return std::isfinite(value.x) && std::isfinite(value.y) && std::isfinite(value.z) && std::isfinite(value.w);
}

static bool uploadParticles(PxParticleAndDiffuseBuffer* buffer, const std::vector<PxVec4>& positions,
                            const std::vector<PxVec4>& velocities, const std::vector<PxU32>& phases,
                            PxCudaContextManager* gpuManager)
{
#if PX_SUPPORT_GPU_PHYSX
    PxScopedCudaLock lock(*gpuManager);
    PxCudaContext* context = gpuManager->getCudaContext();
    if (!context)
        return false;

    PxVec4* positionsDevice = buffer->getPositionInvMasses();
    PxVec4* velocitiesDevice = buffer->getVelocities();
    PxU32* phasesDevice = buffer->getPhases();
    PxVec4* diffusePositionsDevice = buffer->getDiffusePositionLifeTime();
    PxVec4* diffuseVelocitiesDevice = buffer->getDiffuseVelocities();

    printf("Device ptrs: pos=%p vel=%p phases=%p diffusePos=%p diffuseVel=%p\n",
           static_cast<void*>(positionsDevice), static_cast<void*>(velocitiesDevice),
           static_cast<void*>(phasesDevice), static_cast<void*>(diffusePositionsDevice),
           static_cast<void*>(diffuseVelocitiesDevice));
    fflush(stdout);

    if (!positionsDevice || !velocitiesDevice || !phasesDevice ||
        !diffusePositionsDevice || !diffuseVelocitiesDevice)
        return false;

    const size_t particleCount = positions.size();
    bool copyOk = true;
    copyOk &= PxU32(context->memcpyHtoD(CUdeviceptr(positionsDevice), positions.data(), particleCount * sizeof(PxVec4))) == 0;
    copyOk &= PxU32(context->memcpyHtoD(CUdeviceptr(velocitiesDevice), velocities.data(), particleCount * sizeof(PxVec4))) == 0;
    copyOk &= PxU32(context->memcpyHtoD(CUdeviceptr(phasesDevice), phases.data(), particleCount * sizeof(PxU32))) == 0;
    copyOk &= PxU32(context->streamSynchronize(0)) == 0;
    return copyOk;
#else
    PX_UNUSED(buffer);
    PX_UNUSED(positions);
    PX_UNUSED(velocities);
    PX_UNUSED(phases);
    PX_UNUSED(gpuManager);
    return false;
#endif
}

int main(int argc, char** argv)
{
    int steps = 10;
    bool cleanExit = false;
    bool generateDiffuse = false;
    bool generateFirstStepOnly = false;
    bool addPlane = false;
    bool printAllSteps = false;
    int cleanExitWaitSeconds = 5;

    for (int index = 1; index < argc; ++index) {
        if (std::strcmp(argv[index], "--print-all-steps") == 0) {
            printAllSteps = true;
        } else if (std::strcmp(argv[index], "--generate") == 0) {
            generateDiffuse = true;
        } else if (std::strcmp(argv[index], "--generate-first-step-only") == 0) {
            generateDiffuse = true;
            generateFirstStepOnly = true;
        } else if (std::strcmp(argv[index], "--plane") == 0) {
            addPlane = true;
        } else if (std::strcmp(argv[index], "--cleanexit") == 0) {
            cleanExit = true;
        } else if (std::strcmp(argv[index], "--cleanexit-wait") == 0 && index + 1 < argc) {
            cleanExit = true;
            cleanExitWaitSeconds = std::atoi(argv[++index]);
            if (cleanExitWaitSeconds < 0)
                cleanExitWaitSeconds = 0;
        } else if (argv[index][0] != '-') {
            steps = std::atoi(argv[index]);
        }
    }
    if (steps < 0)
        steps = 0;

    const PxU32 particleCount = 8;
    const PxU32 maxDiffuseParticles = 32;
    printf("========================================\n");
    printf(" PhysX DCU Smoke - PBD Diffuse\n");
    printf("========================================\n");
    printf("steps=%d particles=%u maxDiffuse=%u gravity=%s plane=%s generation=%s firstStepOnly=%s\n",
           steps, particleCount, maxDiffuseParticles, addPlane ? "-9.81y" : "0",
           addPlane ? "yes" : "no", generateDiffuse ? "enabled" : "disabled",
           generateFirstStepOnly ? "yes" : "no");
    fflush(stdout);

    static CountingErrorCallback errorCallback;
    static PxDefaultAllocator allocator;
    PxFoundation* foundation = PxCreateFoundation(PX_PHYSICS_VERSION, allocator, errorCallback);
    PxPhysics* physics = foundation ? PxCreatePhysics(PX_PHYSICS_VERSION, *foundation, PxTolerancesScale()) : nullptr;
    if (!foundation || !physics) {
        printf("FAIL: create foundation/physics failed foundation=%p physics=%p\n",
               static_cast<void*>(foundation), static_cast<void*>(physics));
        return 2;
    }

    PxCudaContextManagerDesc gpuDesc;
    gpuDesc.deviceOrdinal = 0;
    PxCudaContextManager* gpuManager = PxCreateCudaContextManager(*foundation, gpuDesc, nullptr, false);
    const bool gpuOk = gpuManager && gpuManager->contextIsValid();
    printf("GPU: %s - %s\n", gpuOk ? gpuManager->getDeviceName() : "NONE", gpuOk ? "enabled" : "disabled");
    if (!gpuOk)
        return 3;

    PxSceneDesc sceneDesc(physics->getTolerancesScale());
    sceneDesc.gravity = addPlane ? PxVec3(0.0f, -9.81f, 0.0f) : PxVec3(0.0f);
    PxDefaultCpuDispatcher* dispatcher = PxDefaultCpuDispatcherCreate(0);
    sceneDesc.cpuDispatcher = dispatcher;
    sceneDesc.filterShader = PxDefaultSimulationFilterShader;
    sceneDesc.cudaContextManager = gpuManager;
    sceneDesc.flags |= PxSceneFlag::eENABLE_GPU_DYNAMICS;
    sceneDesc.flags |= PxSceneFlag::eENABLE_PCM;
    sceneDesc.broadPhaseType = PxBroadPhaseType::eGPU;
    sceneDesc.gpuMaxNumPartitions = 8;
    sceneDesc.gpuDynamicsConfig.heapCapacity = 512u * 1024u * 1024u;
    sceneDesc.gpuDynamicsConfig.tempBufferCapacity = 128u * 1024u * 1024u;
    sceneDesc.gpuDynamicsConfig.collisionStackSize = 256u * 1024u * 1024u;

    printf("Creating GPU dynamics scene: broadPhase=eGPU...\n");
    fflush(stdout);
    PxScene* scene = physics->createScene(sceneDesc);
    if (!scene) {
        printf("FAIL: createScene returned null\n");
        return 4;
    }

    PxMaterial* planeMaterial = nullptr;
    PxRigidStatic* plane = nullptr;
    if (addPlane) {
        planeMaterial = physics->createMaterial(0.5f, 0.5f, 0.0f);
        plane = planeMaterial ? PxCreatePlane(*physics, PxPlane(0, 1, 0, 0), *planeMaterial) : nullptr;
        if (!planeMaterial || !plane) {
            printf("FAIL: create plane failed material=%p plane=%p\n",
                   static_cast<void*>(planeMaterial), static_cast<void*>(plane));
            return 5;
        }
        scene->addActor(*plane);
    }

    const PxReal spacing = 0.1f;
    const PxReal restOffset = 0.5f * spacing / 0.6f;
    const PxReal fluidRestOffset = restOffset * 0.6f;
    const PxReal particleMass = 1000.0f * 1.333f * 3.14159f * spacing * spacing * spacing;

    PxPBDMaterial* particleMaterial = physics->createPBDMaterial(0.05f, 0.05f, 0.0f, 0.001f, 0.5f, 0.005f, 0.01f, 0.0f, 0.0f);
    PxPBDParticleSystem* particleSystem = physics->createPBDParticleSystem(*gpuManager, 96);
    if (!particleMaterial || !particleSystem) {
        printf("FAIL: create PBD material/system failed material=%p system=%p\n",
               static_cast<void*>(particleMaterial), static_cast<void*>(particleSystem));
        return 5;
    }
    particleSystem->setRestOffset(restOffset);
    particleSystem->setContactOffset(restOffset + 0.01f);
    particleSystem->setParticleContactOffset(fluidRestOffset / 0.6f);
    particleSystem->setSolidRestOffset(restOffset);
    particleSystem->setFluidRestOffset(fluidRestOffset);
    particleSystem->setParticleFlag(PxParticleFlag::eENABLE_SPECULATIVE_CCD, false);
    particleSystem->setMaxVelocity(restOffset * 100.0f);
    scene->addActor(*particleSystem);

    const PxU32 phase = particleSystem->createPhase(
        particleMaterial,
        PxParticlePhaseFlags(PxParticlePhaseFlag::eParticlePhaseFluid | PxParticlePhaseFlag::eParticlePhaseSelfCollide));

    std::vector<PxVec4> positions(particleCount);
    std::vector<PxVec4> velocities(
        particleCount, generateDiffuse ? PxVec4(3.0f, 0.0f, 0.0f, 0.0f) : PxVec4(0.0f));
    std::vector<PxU32> phases(particleCount, phase);
    PxU32 particleIndex = 0;
    for (PxU32 x = 0; x < 2; ++x) {
        for (PxU32 y = 0; y < 2; ++y) {
            for (PxU32 z = 0; z < 2; ++z) {
                positions[particleIndex++] = PxVec4(
                    (PxReal(x) - 0.5f) * spacing,
                    1.0f + PxReal(y) * spacing,
                    (PxReal(z) - 0.5f) * spacing,
                    1.0f / particleMass);
            }
        }
    }

    printf("Creating PxParticleAndDiffuseBuffer...\n");
    fflush(stdout);
    PxParticleAndDiffuseBuffer* diffuseBuffer =
        physics->createParticleAndDiffuseBuffer(particleCount, 0, maxDiffuseParticles, gpuManager);
    printf("createParticleAndDiffuseBuffer returned %p\n", static_cast<void*>(diffuseBuffer));
    fflush(stdout);
    if (!diffuseBuffer) {
        printf("VERDICT=UNSUPPORTED\n");
        fflush(stdout);
        return 77;
    }

    if (!uploadParticles(diffuseBuffer, positions, velocities, phases, gpuManager)) {
        printf("FAIL: diffuse buffer device upload failed\n");
        return 6;
    }

    PxDiffuseParticleParams diffuseParams;
    diffuseParams.threshold = generateDiffuse ? 0.01f : 1.0f;
    diffuseParams.lifetime = 2.0f;
    diffuseParams.kineticEnergyWeight = generateDiffuse ? 1.0f : 0.0f;
    diffuseParams.pressureWeight = 0.0f;
    diffuseParams.divergenceWeight = 0.0f;
    diffuseBuffer->setDiffuseParticleParams(diffuseParams);
    diffuseBuffer->setMaxActiveDiffuseParticles(maxDiffuseParticles);
    diffuseBuffer->setNbActiveParticles(particleCount);
    diffuseBuffer->setNbParticleVolumes(0);
    particleSystem->addParticleBuffer(diffuseBuffer);

    PxU32 maxObservedDiffuseParticles = diffuseBuffer->getNbActiveDiffuseParticles();
    PxU32 firstStepDiffuseParticles = 0;
    PxReal minObservedDiffuseHeight = PX_MAX_F32;
    PxU32 diffuseObservationReadbackFailures = 0;
    std::vector<PxVec4> diffuseObservationReadback(maxDiffuseParticles);
    printf("Initial active diffuse particles=%u\n", maxObservedDiffuseParticles);
    printf("Starting simulation...\n");
    fflush(stdout);
    for (int step = 0; step < steps; ++step) {
        if (printAllSteps || step < 10 || step == steps - 1) {
            printf("Simulate/fetch step %d/%d...\n", step + 1, steps);
            fflush(stdout);
        }
        scene->simulate(1.0f / 60.0f);
        scene->fetchResults(true);
        const PxU32 activeDiffuseParticles = diffuseBuffer->getNbActiveDiffuseParticles();
        maxObservedDiffuseParticles = PxMax(maxObservedDiffuseParticles, activeDiffuseParticles);
        if (addPlane && activeDiffuseParticles > 0) {
            PxI32 observationSyncResult = -1;
            PxI32 observationCopyResult = -1;
            {
                PxScopedCudaLock lock(*gpuManager);
                PxCudaContext* context = gpuManager->getCudaContext();
                observationSyncResult = PxI32(context->streamSynchronize(0));
                observationCopyResult = PxI32(context->memcpyDtoH(
                    diffuseObservationReadback.data(), CUdeviceptr(diffuseBuffer->getDiffusePositionLifeTime()),
                    activeDiffuseParticles * sizeof(PxVec4)));
            }
            if (observationSyncResult == 0 && observationCopyResult == 0) {
                for (PxU32 index = 0; index < activeDiffuseParticles; ++index) {
                    if (finiteVec4(diffuseObservationReadback[index]))
                        minObservedDiffuseHeight = PxMin(minObservedDiffuseHeight, diffuseObservationReadback[index].y);
                }
            } else {
                ++diffuseObservationReadbackFailures;
            }
        }
        if (step == 0) {
            firstStepDiffuseParticles = activeDiffuseParticles;
            if (generateFirstStepOnly) {
                PxDiffuseParticleParams disabledParams = diffuseParams;
                disabledParams.kineticEnergyWeight = 0.0f;
                disabledParams.pressureWeight = 0.0f;
                disabledParams.divergenceWeight = 0.0f;
                diffuseBuffer->setDiffuseParticleParams(disabledParams);
                printf("Generation disabled after step 1; active diffuse particles=%u\n",
                       firstStepDiffuseParticles);
                fflush(stdout);
            }
        }
    }

    std::vector<PxVec4> positionsReadback(particleCount);
    std::vector<PxVec4> velocitiesReadback(particleCount);
    const PxU32 finalDiffuseParticles = diffuseBuffer->getNbActiveDiffuseParticles();
    std::vector<PxVec4> diffusePositionsReadback(finalDiffuseParticles);
    std::vector<PxVec4> diffuseVelocitiesReadback(finalDiffuseParticles);
    PxI32 syncResult = -1;
    PxI32 positionCopyResult = -1;
    PxI32 velocityCopyResult = -1;
    PxI32 diffusePositionCopyResult = 0;
    PxI32 diffuseVelocityCopyResult = 0;
    {
        PxScopedCudaLock lock(*gpuManager);
        PxCudaContext* context = gpuManager->getCudaContext();
        syncResult = PxI32(context->streamSynchronize(0));
        positionCopyResult = PxI32(context->memcpyDtoH(positionsReadback.data(),
                                                       CUdeviceptr(diffuseBuffer->getPositionInvMasses()),
                                                       particleCount * sizeof(PxVec4)));
        velocityCopyResult = PxI32(context->memcpyDtoH(velocitiesReadback.data(),
                                                       CUdeviceptr(diffuseBuffer->getVelocities()),
                                                       particleCount * sizeof(PxVec4)));
        if (finalDiffuseParticles > 0) {
            diffusePositionCopyResult = PxI32(context->memcpyDtoH(
                diffusePositionsReadback.data(), CUdeviceptr(diffuseBuffer->getDiffusePositionLifeTime()),
                finalDiffuseParticles * sizeof(PxVec4)));
            diffuseVelocityCopyResult = PxI32(context->memcpyDtoH(
                diffuseVelocitiesReadback.data(), CUdeviceptr(diffuseBuffer->getDiffuseVelocities()),
                finalDiffuseParticles * sizeof(PxVec4)));
        }
    }

    PxU32 badParticles = 0;
    PxReal maxDisplacement = 0.0f;
    PxReal maxSpeed = 0.0f;
    PxReal minParticleHeight = PX_MAX_F32;
    PxReal maxParticleHeight = -PX_MAX_F32;
    PxReal minParticleVerticalVelocity = PX_MAX_F32;
    PxReal maxParticleVerticalVelocity = -PX_MAX_F32;
    for (PxU32 index = 0; index < particleCount; ++index) {
        if (!finiteVec4(positionsReadback[index]) || !finiteVec4(velocitiesReadback[index])) {
            ++badParticles;
            continue;
        }
        const PxVec3 initialPosition(positions[index].x, positions[index].y, positions[index].z);
        const PxVec3 finalPosition(positionsReadback[index].x, positionsReadback[index].y, positionsReadback[index].z);
        const PxVec3 velocity(velocitiesReadback[index].x, velocitiesReadback[index].y, velocitiesReadback[index].z);
        maxDisplacement = PxMax(maxDisplacement, (finalPosition - initialPosition).magnitude());
        maxSpeed = PxMax(maxSpeed, velocity.magnitude());
        minParticleHeight = PxMin(minParticleHeight, finalPosition.y);
        maxParticleHeight = PxMax(maxParticleHeight, finalPosition.y);
        minParticleVerticalVelocity = PxMin(minParticleVerticalVelocity, velocity.y);
        maxParticleVerticalVelocity = PxMax(maxParticleVerticalVelocity, velocity.y);
    }

    PxU32 badDiffuseParticles = 0;
    PxReal minDiffuseLifetime = PX_MAX_F32;
    PxReal maxDiffuseLifetime = -PX_MAX_F32;
    PxReal minDiffuseHeight = PX_MAX_F32;
    PxReal maxDiffuseHeight = -PX_MAX_F32;
    PxReal minDiffuseVerticalVelocity = PX_MAX_F32;
    PxReal maxDiffuseVerticalVelocity = -PX_MAX_F32;
    for (PxU32 index = 0; index < finalDiffuseParticles; ++index) {
        if (!finiteVec4(diffusePositionsReadback[index]) || !finiteVec4(diffuseVelocitiesReadback[index])) {
            ++badDiffuseParticles;
            continue;
        }
        minDiffuseLifetime = PxMin(minDiffuseLifetime, diffusePositionsReadback[index].w);
        maxDiffuseLifetime = PxMax(maxDiffuseLifetime, diffusePositionsReadback[index].w);
        minDiffuseHeight = PxMin(minDiffuseHeight, diffusePositionsReadback[index].y);
        maxDiffuseHeight = PxMax(maxDiffuseHeight, diffusePositionsReadback[index].y);
        minDiffuseVerticalVelocity = PxMin(minDiffuseVerticalVelocity, diffuseVelocitiesReadback[index].y);
        maxDiffuseVerticalVelocity = PxMax(maxDiffuseVerticalVelocity, diffuseVelocitiesReadback[index].y);
    }

    const bool readbackOk = syncResult == 0 && positionCopyResult == 0 && velocityCopyResult == 0 &&
                            diffusePositionCopyResult == 0 && diffuseVelocityCopyResult == 0;
    const bool validParticles = readbackOk && badParticles == 0;
    const bool diffuseCapacityValid = diffuseBuffer->getMaxDiffuseParticles() == maxDiffuseParticles;
    const bool diffuseCountValid = generateFirstStepOnly
        ? firstStepDiffuseParticles > 0 && finalDiffuseParticles == firstStepDiffuseParticles
        : (generateDiffuse
            ? maxObservedDiffuseParticles > 0 && finalDiffuseParticles > 0 && finalDiffuseParticles <= maxDiffuseParticles
            : maxObservedDiffuseParticles == 0 && finalDiffuseParticles == 0);
    const bool validDiffuseParticles = !generateDiffuse ||
        (badDiffuseParticles == 0 && minDiffuseLifetime > 0.0f && maxDiffuseLifetime <= diffuseParams.lifetime);
    const bool particlesReachedPlane = !addPlane || minParticleHeight < 0.2f;
    const bool noParticlePlanePenetration = !addPlane || minParticleHeight > -0.02f;
    const bool diffuseObservationValid = !addPlane || !generateDiffuse ||
        (diffuseObservationReadbackFailures == 0 && minObservedDiffuseHeight < PX_MAX_F32);
    const bool diffuseReachedPlane = !addPlane || !generateDiffuse ||
        (diffuseObservationValid && minObservedDiffuseHeight < 0.2f);
    const bool noDiffusePlanePenetration = !addPlane || !generateDiffuse ||
        (diffuseObservationValid && minObservedDiffuseHeight > -0.02f);
    const bool noPhysxErrors = errorCallback.errorCount == 0;
    const bool pass = validParticles && diffuseCapacityValid && diffuseCountValid &&
                      validDiffuseParticles && particlesReachedPlane && noParticlePlanePenetration &&
                      diffuseReachedPlane && noDiffusePlanePenetration &&
                      noPhysxErrors;

    printf("GPU readback sync=%d position=%d velocity=%d diffusePosition=%d diffuseVelocity=%d\n",
           syncResult, positionCopyResult, velocityCopyResult,
           diffusePositionCopyResult, diffuseVelocityCopyResult);
    printf("particles=%u badParticles=%u/%u maxDisplacement=%.6f maxSpeed=%.6f\n",
           particleCount, badParticles, particleCount, maxDisplacement, maxSpeed);
    printf("particle height=[%.6f,%.6f] verticalVelocity=[%.6f,%.6f]\n",
           minParticleHeight, maxParticleHeight,
           minParticleVerticalVelocity, maxParticleVerticalVelocity);
    printf("diffuse max=%u firstStepActive=%u maxObservedActive=%u finalActive=%u PhysXErrors=%u\n",
           diffuseBuffer->getMaxDiffuseParticles(), firstStepDiffuseParticles,
           maxObservedDiffuseParticles, finalDiffuseParticles, errorCallback.errorCount);
    printf("diffuse badParticles=%u/%u lifetime=[%.6f,%.6f]\n",
           badDiffuseParticles, finalDiffuseParticles,
           finalDiffuseParticles ? minDiffuseLifetime : 0.0f,
           finalDiffuseParticles ? maxDiffuseLifetime : 0.0f);
    printf("diffuse height=[%.6f,%.6f] verticalVelocity=[%.6f,%.6f]\n",
           finalDiffuseParticles ? minDiffuseHeight : 0.0f,
           finalDiffuseParticles ? maxDiffuseHeight : 0.0f,
           finalDiffuseParticles ? minDiffuseVerticalVelocity : 0.0f,
           finalDiffuseParticles ? maxDiffuseVerticalVelocity : 0.0f);
    printf("diffuse observedMinHeight=%.6f observationReadbackFailures=%u\n",
           minObservedDiffuseHeight < PX_MAX_F32 ? minObservedDiffuseHeight : 0.0f,
           diffuseObservationReadbackFailures);
    printf("Checks validParticles=%s diffuseCapacityValid=%s diffuseCountValid=%s validDiffuseParticles=%s particlesReachedPlane=%s noParticlePlanePenetration=%s diffuseObservationValid=%s diffuseReachedPlane=%s noDiffusePlanePenetration=%s noPhysxErrors=%s\n",
           validParticles ? "yes" : "no", diffuseCapacityValid ? "yes" : "no",
           diffuseCountValid ? "yes" : "no", validDiffuseParticles ? "yes" : "no",
           particlesReachedPlane ? "yes" : "no", noParticlePlanePenetration ? "yes" : "no",
           diffuseObservationValid ? "yes" : "no",
           diffuseReachedPlane ? "yes" : "no", noDiffusePlanePenetration ? "yes" : "no",
           noPhysxErrors ? "yes" : "no");
    printf("VERDICT=%s\n", pass ? "PASS" : "FAIL");
    fflush(stdout);

    diffuseBuffer->release();
    particleSystem->release();
    particleMaterial->release();
    if (plane)
        plane->release();
    if (planeMaterial)
        planeMaterial->release();
    scene->release();
    dispatcher->release();
    gpuManager->release();
    physics->release();
    foundation->release();

    if (cleanExit) {
        printf("Clean exit after explicit PhysX release, waiting %d seconds for DCU/HSA cleanup...\n", cleanExitWaitSeconds);
        fflush(stdout);
        if (cleanExitWaitSeconds > 0)
            std::this_thread::sleep_for(std::chrono::seconds(cleanExitWaitSeconds));
        printf("Clean exit wait finished.\n");
        fflush(stdout);
        std::_Exit(pass ? 0 : 1);
    }
    return pass ? 0 : 1;
}
