// PhysX DCU smoke test: single shape-matched PBD rigid particle cluster.

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

static bool finiteVec3(const PxVec3& value)
{
    return std::isfinite(value.x) && std::isfinite(value.y) && std::isfinite(value.z);
}

static bool finiteVec4(const PxVec4& value)
{
    return std::isfinite(value.x) && std::isfinite(value.y) && std::isfinite(value.z) && std::isfinite(value.w);
}

static bool copyRigidToDevice(PxParticleRigidBuffer* buffer, const std::vector<PxVec4>& positions,
                              const std::vector<PxVec4>& velocities, const std::vector<PxU32>& phases,
                              const PxU32* rigidOffsets, const PxReal* rigidCoefficients,
                              const PxVec4* rigidTranslations, const PxVec4* rigidRotations,
                              const std::vector<PxVec4>& localPositions,
                              const std::vector<PxVec4>& localNormals,
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
    PxU32* offsetsDevice = buffer->getRigidOffsets();
    PxReal* coefficientsDevice = buffer->getRigidCoefficients();
    PxVec4* translationsDevice = buffer->getRigidTranslations();
    PxVec4* rotationsDevice = buffer->getRigidRotations();
    PxVec4* localPositionsDevice = buffer->getRigidLocalPositions();
    PxVec4* localNormalsDevice = buffer->getRigidLocalNormals();

    printf("Device ptrs: pos=%p vel=%p phases=%p offsets=%p coeff=%p translation=%p rotation=%p localPos=%p localNormal=%p\n",
           static_cast<void*>(positionsDevice), static_cast<void*>(velocitiesDevice),
           static_cast<void*>(phasesDevice), static_cast<void*>(offsetsDevice),
           static_cast<void*>(coefficientsDevice), static_cast<void*>(translationsDevice),
           static_cast<void*>(rotationsDevice), static_cast<void*>(localPositionsDevice),
           static_cast<void*>(localNormalsDevice));
    fflush(stdout);

    if (!positionsDevice || !velocitiesDevice || !phasesDevice || !offsetsDevice ||
        !coefficientsDevice || !translationsDevice || !rotationsDevice ||
        !localPositionsDevice || !localNormalsDevice)
        return false;

    const size_t particleCount = positions.size();
    bool copyOk = true;
    copyOk &= PxU32(context->memcpyHtoD(CUdeviceptr(positionsDevice), positions.data(), particleCount * sizeof(PxVec4))) == 0;
    copyOk &= PxU32(context->memcpyHtoD(CUdeviceptr(velocitiesDevice), velocities.data(), particleCount * sizeof(PxVec4))) == 0;
    copyOk &= PxU32(context->memcpyHtoD(CUdeviceptr(phasesDevice), phases.data(), particleCount * sizeof(PxU32))) == 0;
    copyOk &= PxU32(context->memcpyHtoD(CUdeviceptr(offsetsDevice), rigidOffsets, 2 * sizeof(PxU32))) == 0;
    copyOk &= PxU32(context->memcpyHtoD(CUdeviceptr(coefficientsDevice), rigidCoefficients, sizeof(PxReal))) == 0;
    copyOk &= PxU32(context->memcpyHtoD(CUdeviceptr(translationsDevice), rigidTranslations, sizeof(PxVec4))) == 0;
    copyOk &= PxU32(context->memcpyHtoD(CUdeviceptr(rotationsDevice), rigidRotations, sizeof(PxVec4))) == 0;
    copyOk &= PxU32(context->memcpyHtoD(CUdeviceptr(localPositionsDevice), localPositions.data(), particleCount * sizeof(PxVec4))) == 0;
    copyOk &= PxU32(context->memcpyHtoD(CUdeviceptr(localNormalsDevice), localNormals.data(), particleCount * sizeof(PxVec4))) == 0;
    copyOk &= PxU32(context->streamSynchronize(0)) == 0;
    return copyOk;
#else
    PX_UNUSED(buffer);
    PX_UNUSED(positions);
    PX_UNUSED(velocities);
    PX_UNUSED(phases);
    PX_UNUSED(rigidOffsets);
    PX_UNUSED(rigidCoefficients);
    PX_UNUSED(rigidTranslations);
    PX_UNUSED(rigidRotations);
    PX_UNUSED(localPositions);
    PX_UNUSED(localNormals);
    PX_UNUSED(gpuManager);
    return false;
#endif
}

int main(int argc, char** argv)
{
    int steps = 300;
    bool cleanExit = false;
    bool printAllSteps = false;
    int cleanExitWaitSeconds = 5;

    for (int index = 1; index < argc; ++index) {
        if (std::strcmp(argv[index], "--print-all-steps") == 0) {
            printAllSteps = true;
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

    printf("========================================\n");
    printf(" PhysX DCU Smoke - PBD Particle Rigid\n");
    printf("========================================\n");
    printf("steps=%d particles=8 rigids=1 plane=1\n", steps);
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
    sceneDesc.gravity = PxVec3(0.0f, -9.81f, 0.0f);
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

    PxMaterial* planeMaterial = physics->createMaterial(0.5f, 0.5f, 0.0f);
    PxRigidStatic* plane = planeMaterial ? PxCreatePlane(*physics, PxPlane(0, 1, 0, 0), *planeMaterial) : nullptr;
    if (!planeMaterial || !plane) {
        printf("FAIL: create plane failed material=%p plane=%p\n",
               static_cast<void*>(planeMaterial), static_cast<void*>(plane));
        return 5;
    }
    scene->addActor(*plane);

    const PxU32 numParticles = 8;
    const PxReal halfExtent = 0.075f;
    const PxReal restOffset = 0.05f;
    const PxReal initialCenterY = 1.25f;
    const PxVec3 initialTranslation(0.0f, initialCenterY, 0.0f);
    const PxReal invParticleMass = PxReal(numParticles);

    PxPBDMaterial* particleMaterial = physics->createPBDMaterial(0.8f, 0.05f, 1e+6f, 0.001f, 0.5f, 0.005f, 0.05f, 0.0f, 0.0f);
    PxPBDParticleSystem* particleSystem = physics->createPBDParticleSystem(*gpuManager);
    if (!particleMaterial || !particleSystem) {
        printf("FAIL: create PBD material/system failed material=%p system=%p\n",
               static_cast<void*>(particleMaterial), static_cast<void*>(particleSystem));
        return 6;
    }
    particleSystem->setRestOffset(restOffset);
    particleSystem->setContactOffset(restOffset + 0.02f);
    particleSystem->setParticleContactOffset(restOffset + 0.02f);
    particleSystem->setSolidRestOffset(restOffset);
    particleSystem->setFluidRestOffset(0.0f);
    scene->addActor(*particleSystem);
    const PxU32 phase = particleSystem->createPhase(particleMaterial, PxParticlePhaseFlags());

    std::vector<PxVec4> localPositions(numParticles);
    std::vector<PxVec4> localNormals(numParticles);
    std::vector<PxVec4> positions(numParticles);
    std::vector<PxVec4> velocities(numParticles, PxVec4(0.0f));
    std::vector<PxU32> phases(numParticles, phase);
    PxU32 particleIndex = 0;
    for (PxI32 y = -1; y <= 1; y += 2) {
        for (PxI32 z = -1; z <= 1; z += 2) {
            for (PxI32 x = -1; x <= 1; x += 2) {
                const PxVec3 local(PxReal(x) * halfExtent, PxReal(y) * halfExtent, PxReal(z) * halfExtent);
                const PxVec3 normal = local.getNormalized();
                localPositions[particleIndex] = PxVec4(local, 0.0f);
                localNormals[particleIndex] = PxVec4(normal, -restOffset);
                positions[particleIndex] = PxVec4(initialTranslation + local, invParticleMass);
                ++particleIndex;
            }
        }
    }

    const PxU32 rigidOffsets[2] = {0, numParticles};
    const PxReal rigidCoefficients[1] = {1.0f};
    const PxVec4 rigidTranslations[1] = {PxVec4(initialTranslation, 0.0f)};
    const PxVec4 rigidRotations[1] = {PxVec4(0.0f, 0.0f, 0.0f, 1.0f)};

    printf("Creating PxParticleRigidBuffer...\n");
    fflush(stdout);
    PxParticleRigidBuffer* rigidBuffer = physics->createParticleRigidBuffer(numParticles, 0, 1, gpuManager);
    printf("createParticleRigidBuffer returned %p\n", static_cast<void*>(rigidBuffer));
    fflush(stdout);
    if (!rigidBuffer) {
        printf("VERDICT=UNSUPPORTED\n");
        fflush(stdout);
        return 77;
    }

    if (!copyRigidToDevice(rigidBuffer, positions, velocities, phases, rigidOffsets, rigidCoefficients,
                           rigidTranslations, rigidRotations, localPositions, localNormals, gpuManager)) {
        printf("FAIL: rigid buffer device upload failed\n");
        return 7;
    }
    rigidBuffer->setNbActiveParticles(numParticles);
    rigidBuffer->setNbParticleVolumes(0);
    rigidBuffer->setNbRigids(1);
    particleSystem->addParticleBuffer(rigidBuffer);

    printf("Starting simulation...\n");
    fflush(stdout);
    for (int step = 0; step < steps; ++step) {
        if (printAllSteps || step < 10 || step == steps - 1) {
            printf("Simulate/fetch step %d/%d...\n", step + 1, steps);
            fflush(stdout);
        }
        scene->simulate(1.0f / 60.0f);
        scene->fetchResults(true);
    }

    std::vector<PxVec4> positionsReadback(numParticles);
    std::vector<PxVec4> velocitiesReadback(numParticles);
    PxVec4 translationReadback(0.0f);
    PxVec4 rotationReadback(0.0f);
    PxI32 syncResult = -1;
    PxI32 positionCopyResult = -1;
    PxI32 velocityCopyResult = -1;
    PxI32 translationCopyResult = -1;
    PxI32 rotationCopyResult = -1;
    {
        PxScopedCudaLock lock(*gpuManager);
        PxCudaContext* context = gpuManager->getCudaContext();
        syncResult = PxI32(context->streamSynchronize(0));
        positionCopyResult = PxI32(context->memcpyDtoH(positionsReadback.data(), CUdeviceptr(rigidBuffer->getPositionInvMasses()), numParticles * sizeof(PxVec4)));
        velocityCopyResult = PxI32(context->memcpyDtoH(velocitiesReadback.data(), CUdeviceptr(rigidBuffer->getVelocities()), numParticles * sizeof(PxVec4)));
        translationCopyResult = PxI32(context->memcpyDtoH(&translationReadback, CUdeviceptr(rigidBuffer->getRigidTranslations()), sizeof(PxVec4)));
        rotationCopyResult = PxI32(context->memcpyDtoH(&rotationReadback, CUdeviceptr(rigidBuffer->getRigidRotations()), sizeof(PxVec4)));
    }

    const bool readbackOk = syncResult == 0 && positionCopyResult == 0 && velocityCopyResult == 0 &&
                            translationCopyResult == 0 && rotationCopyResult == 0;
    PxU32 badParticles = 0;
    PxVec3 center(0.0f);
    PxReal minHeight = PX_MAX_F32;
    PxReal maxHeight = -PX_MAX_F32;
    PxReal maxSpeed = 0.0f;
    if (readbackOk) {
        for (PxU32 index = 0; index < numParticles; ++index) {
            if (!finiteVec4(positionsReadback[index]) || !finiteVec4(velocitiesReadback[index])) {
                ++badParticles;
                continue;
            }
            const PxVec3 position(positionsReadback[index].x, positionsReadback[index].y, positionsReadback[index].z);
            const PxVec3 velocity(velocitiesReadback[index].x, velocitiesReadback[index].y, velocitiesReadback[index].z);
            center += position;
            minHeight = PxMin(minHeight, position.y);
            maxHeight = PxMax(maxHeight, position.y);
            maxSpeed = PxMax(maxSpeed, velocity.magnitude());
        }
        if (badParticles == 0)
            center /= PxReal(numParticles);
    }

    PxReal maxPairwiseError = 0.0f;
    if (readbackOk && badParticles == 0) {
        for (PxU32 first = 0; first < numParticles; ++first) {
            for (PxU32 second = first + 1; second < numParticles; ++second) {
                const PxVec3 initialDelta = PxVec3(localPositions[first].x, localPositions[first].y, localPositions[first].z) -
                                            PxVec3(localPositions[second].x, localPositions[second].y, localPositions[second].z);
                const PxVec3 finalDelta = PxVec3(positionsReadback[first].x, positionsReadback[first].y, positionsReadback[first].z) -
                                          PxVec3(positionsReadback[second].x, positionsReadback[second].y, positionsReadback[second].z);
                maxPairwiseError = PxMax(maxPairwiseError, PxAbs(finalDelta.magnitude() - initialDelta.magnitude()));
            }
        }
    }

    const PxVec3 rigidTranslation(translationReadback.x, translationReadback.y, translationReadback.z);
    const PxQuat rigidRotation(rotationReadback.x, rotationReadback.y, rotationReadback.z, rotationReadback.w);
    const bool finiteTransform = finiteVec4(translationReadback) && finiteVec4(rotationReadback);
    const PxReal rotationNormError = finiteTransform ? PxAbs(rigidRotation.magnitude() - 1.0f) : PX_MAX_F32;
    const PxReal centerTransformError = finiteTransform && finiteVec3(center) ? (center - rigidTranslation).magnitude() : PX_MAX_F32;
    const PxReal centerDrop = initialCenterY - center.y;
    const bool validParticles = readbackOk && badParticles == 0;
    const bool shapePreserved = validParticles && maxPairwiseError < 0.03f;
    const bool validTransform = finiteTransform && rotationNormError < 0.02f && centerTransformError < 0.03f;
    const bool realDrop = steps == 0 || centerDrop > 0.25f;
    const bool noPlanePenetration = steps == 0 || minHeight > -0.02f;
    const bool settledOnPlane = steps < 180 || (minHeight > 0.02f && minHeight < 0.10f && maxSpeed < 0.05f);
    const bool noPhysxErrors = errorCallback.errorCount == 0;
    const bool pass = validParticles && shapePreserved && validTransform && realDrop &&
                      noPlanePenetration && settledOnPlane && noPhysxErrors;

    printf("GPU readback sync=%d position=%d velocity=%d translation=%d rotation=%d\n",
           syncResult, positionCopyResult, velocityCopyResult, translationCopyResult, rotationCopyResult);
    printf("particles=%u rigids=1 height=[%.6f,%.6f] center=(%.6f,%.6f,%.6f) centerDrop=%.6f maxSpeed=%.6f badParticles=%u/%u\n",
           numParticles, minHeight, maxHeight, center.x, center.y, center.z, centerDrop, maxSpeed, badParticles, numParticles);
    printf("rigidTranslation=(%.6f,%.6f,%.6f) rigidRotation=(%.6f,%.6f,%.6f,%.6f) rotationNormError=%.6f centerTransformError=%.6f\n",
           translationReadback.x, translationReadback.y, translationReadback.z,
           rotationReadback.x, rotationReadback.y, rotationReadback.z, rotationReadback.w,
           rotationNormError, centerTransformError);
    printf("maxPairwiseError=%.6f PhysXErrors=%u\n", maxPairwiseError, errorCallback.errorCount);
    printf("Checks validParticles=%s shapePreserved=%s validTransform=%s realDrop=%s noPlanePenetration=%s settledOnPlane=%s noPhysxErrors=%s\n",
           validParticles ? "yes" : "no", shapePreserved ? "yes" : "no", validTransform ? "yes" : "no",
           realDrop ? "yes" : "no", noPlanePenetration ? "yes" : "no", settledOnPlane ? "yes" : "no",
           noPhysxErrors ? "yes" : "no");
    printf("VERDICT=%s\n", pass ? "PASS" : "FAIL");
    fflush(stdout);

    rigidBuffer->release();
    particleSystem->release();
    particleMaterial->release();
    plane->release();
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
