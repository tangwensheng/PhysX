// PhysX DCU smoke test: conservative PBD particle buffer path without cloth helper.

#include "PxPhysicsAPI.h"
#include "cudamanager/PxCudaContext.h"
#include "cudamanager/PxCudaContextManager.h"
#include "extensions/PxParticleExt.h"
#include "extensions/PxCudaHelpersExt.h"
#include "gpu/PxGpu.h"
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <thread>
#include <vector>

using namespace physx;

static bool finiteVec(const PxVec3& v)
{
    return std::isfinite(v.x) && std::isfinite(v.y) && std::isfinite(v.z);
}

static bool finiteVec4(const PxVec4& v)
{
    return std::isfinite(v.x) && std::isfinite(v.y) && std::isfinite(v.z) && std::isfinite(v.w);
}

static bool copyParticlesToDevice(PxParticleBuffer* buffer, const ExtGpu::PxParticleBufferDesc& desc, PxCudaContextManager* gpuMgr)
{
#if PX_SUPPORT_GPU_PHYSX
    gpuMgr->acquireContext();
    PxCudaContext* cudaContext = gpuMgr->getCudaContext();
    if (!cudaContext) {
        gpuMgr->releaseContext();
        return false;
    }

    PxVec4* pos = buffer->getPositionInvMasses();
    PxVec4* vel = buffer->getVelocities();
    PxU32* phases = buffer->getPhases();
    PxParticleVolume* volumes = buffer->getParticleVolumes();

    printf("Device ptrs: pos=%p vel=%p phases=%p volumes=%p\n", static_cast<void*>(pos), static_cast<void*>(vel), static_cast<void*>(phases), static_cast<void*>(volumes));
    fflush(stdout);
    if (!pos || !vel || !phases) {
        gpuMgr->releaseContext();
        return false;
    }

    cudaContext->memcpyHtoDAsync(CUdeviceptr(pos), desc.positions, desc.numActiveParticles * sizeof(PxVec4), 0);
    cudaContext->memcpyHtoDAsync(CUdeviceptr(vel), desc.velocities, desc.numActiveParticles * sizeof(PxVec4), 0);
    cudaContext->memcpyHtoDAsync(CUdeviceptr(phases), desc.phases, desc.numActiveParticles * sizeof(PxU32), 0);
    if (desc.numVolumes)
        cudaContext->memcpyHtoDAsync(CUdeviceptr(volumes), desc.volumes, desc.numVolumes * sizeof(PxParticleVolume), 0);
    cudaContext->streamSynchronize(0);
    gpuMgr->releaseContext();
    return true;
#else
    PX_UNUSED(buffer);
    PX_UNUSED(desc);
    PX_UNUSED(gpuMgr);
    return false;
#endif
}

int main(int argc, char** argv)
{
    int steps = 10;
    PxU32 dim = 4;
    bool forceGpuBroadphase = false;
    bool useTgs = false;
    bool skipBuffer = false;
    bool addPlane = true;
    bool printAllSteps = false;
    bool cleanExit = false;
    int cleanExitWaitSeconds = 5;

    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--dim") == 0 && i + 1 < argc) {
            dim = PxU32(std::atoi(argv[++i]));
        } else if (std::strcmp(argv[i], "--force-gpu-bp") == 0) {
            forceGpuBroadphase = true;
        } else if (std::strcmp(argv[i], "--tgs") == 0) {
            useTgs = true;
        } else if (std::strcmp(argv[i], "--skip-buffer") == 0) {
            skipBuffer = true;
        } else if (std::strcmp(argv[i], "--no-plane") == 0) {
            addPlane = false;
        } else if (std::strcmp(argv[i], "--print-all-steps") == 0) {
            printAllSteps = true;
        } else if (std::strcmp(argv[i], "--cleanexit") == 0) {
            cleanExit = true;
        } else if (std::strcmp(argv[i], "--cleanexit-wait") == 0 && i + 1 < argc) {
            cleanExit = true;
            cleanExitWaitSeconds = std::atoi(argv[++i]);
            if (cleanExitWaitSeconds < 0)
                cleanExitWaitSeconds = 0;
        } else if (argv[i][0] != '-') {
            steps = std::atoi(argv[i]);
        }
    }
    if (steps < 0)
        steps = 0;
    if (dim < 2)
        dim = 2;

    printf("PBD particle conservative smoke: steps=%d dim=%u forceGpuBp=%d tgs=%d skipBuffer=%d plane=%d\n",
           steps, dim, forceGpuBroadphase ? 1 : 0, useTgs ? 1 : 0, skipBuffer ? 1 : 0, addPlane ? 1 : 0);
    fflush(stdout);

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
    sd.flags |= PxSceneFlag::eENABLE_GPU_DYNAMICS;
    sd.broadPhaseType = PxBroadPhaseType::eGPU;
    if (useTgs)
        sd.solverType = PxSolverType::eTGS;
    if (forceGpuBroadphase)
        sd.broadPhaseType = PxBroadPhaseType::eGPU;
    printf("Creating scene: gpuDynamics=1 broadPhase=eGPU solver=%s\n", useTgs ? "TGS" : "default");
    fflush(stdout);
    PxDefaultCpuDispatcher* dsp = PxDefaultCpuDispatcherCreate(2);
    sd.cpuDispatcher = dsp;
    sd.filterShader = PxDefaultSimulationFilterShader;
    PxScene* scene = phy->createScene(sd);
    printf("Scene: %p\n", static_cast<void*>(scene));
    if (!scene)
        return 3;

    PxMaterial* rigidMat = phy->createMaterial(0.5f, 0.5f, 0.0f);
    PxRigidStatic* plane = nullptr;
    if (addPlane && rigidMat) {
        printf("Adding rigid plane at y=0.000...\n");
        fflush(stdout);
        plane = PxCreatePlane(*phy, PxPlane(0, 1, 0, 0), *rigidMat);
        if (plane)
            scene->addActor(*plane);
    }
    if (!rigidMat || (addPlane && !plane)) {
        printf("FAIL: create rigid plane failed material=%p plane=%p\n",
               static_cast<void*>(rigidMat), static_cast<void*>(plane));
        return 4;
    }

    const PxU32 numParticles = dim * dim;
    const PxReal spacing = 0.1f;
    const PxReal totalMass = 10.0f;
    const PxReal invParticleMass = PxReal(numParticles) / totalMass;
    const PxReal restOffset = 0.05f;

    PxPBDMaterial* mat = phy->createPBDMaterial(0.8f, 0.05f, 1e+6f, 0.001f, 0.5f, 0.005f, 0.05f, 0.0f, 0.0f);
    PxPBDParticleSystem* ps = phy->createPBDParticleSystem(*gpuMgr);
    if (!mat || !ps) {
        printf("FAIL: create PBD material/system failed mat=%p ps=%p\n", static_cast<void*>(mat), static_cast<void*>(ps));
        return 4;
    }
    ps->setRestOffset(restOffset);
    ps->setContactOffset(restOffset + 0.02f);
    ps->setParticleContactOffset(restOffset + 0.02f);
    ps->setSolidRestOffset(restOffset);
    ps->setFluidRestOffset(0.0f);
    scene->addActor(*ps);
    const PxU32 phaseValue = ps->createPhase(mat, PxParticlePhaseFlags(PxParticlePhaseFlag::eParticlePhaseSelfCollideFilter | PxParticlePhaseFlag::eParticlePhaseSelfCollide));

    printf("Allocating pinned host particle arrays...\n");
    fflush(stdout);
    PxU32* phases = PX_EXT_PINNED_MEMORY_ALLOC(PxU32, *gpuMgr, numParticles);
    PxVec4* positions = PX_EXT_PINNED_MEMORY_ALLOC(PxVec4, *gpuMgr, numParticles);
    PxVec4* velocities = PX_EXT_PINNED_MEMORY_ALLOC(PxVec4, *gpuMgr, numParticles);
    if (!phases || !positions || !velocities) {
        printf("FAIL: pinned allocation failed phases=%p positions=%p velocities=%p\n", static_cast<void*>(phases), static_cast<void*>(positions), static_cast<void*>(velocities));
        return 5;
    }

    for (PxU32 y = 0; y < dim; ++y) {
        for (PxU32 x = 0; x < dim; ++x) {
            const PxU32 id = y * dim + x;
            const PxReal px = (PxReal(x) - PxReal(dim - 1) * 0.5f) * spacing;
            const PxReal pz = (PxReal(y) - PxReal(dim - 1) * 0.5f) * spacing;
            positions[id] = PxVec4(px, 1.5f, pz, invParticleMass);
            velocities[id] = PxVec4(0.0f);
            phases[id] = phaseValue;
        }
    }

    ExtGpu::PxParticleBufferDesc desc;
    desc.maxParticles = numParticles;
    desc.numActiveParticles = numParticles;
    desc.positions = positions;
    desc.velocities = velocities;
    desc.phases = phases;

    printf("Creating plain PxParticleBuffer...\n");
    fflush(stdout);
    PxParticleBuffer* particleBuffer = phy->createParticleBuffer(desc.maxParticles, desc.maxVolumes, gpuMgr);
    printf("createParticleBuffer returned %p\n", static_cast<void*>(particleBuffer));
    fflush(stdout);
    bool unsupportedParticleBuffer = false;
    if (!particleBuffer) {
        unsupportedParticleBuffer = true;
        printf("VERDICT: UNSUPPORTED - createParticleBuffer returned null\n");
        fflush(stdout);
    }
    if (particleBuffer && skipBuffer) {
        printf("Skipping device copy/addParticleBuffer/simulation due to --skip-buffer\n");
        fflush(stdout);
    } else if (particleBuffer) {
        printf("Copying plain particle buffer to device...\n");
        fflush(stdout);
        if (!copyParticlesToDevice(particleBuffer, desc, gpuMgr)) {
            printf("FAIL: copyParticlesToDevice failed\n");
            return 7;
        }
        printf("Setting particle buffer active counts...\n");
        fflush(stdout);
        particleBuffer->setNbActiveParticles(desc.numActiveParticles);
        particleBuffer->setNbParticleVolumes(desc.numVolumes);

        printf("Adding plain particle buffer to PBD system...\n");
        fflush(stdout);
        ps->addParticleBuffer(particleBuffer);
    }

    PX_EXT_PINNED_MEMORY_FREE(*gpuMgr, positions);
    PX_EXT_PINNED_MEMORY_FREE(*gpuMgr, velocities);
    PX_EXT_PINNED_MEMORY_FREE(*gpuMgr, phases);

    bool pass = true;
    bool readbackOk = skipBuffer || !particleBuffer;
    PxReal minParticleY = PX_MAX_F32;
    PxReal maxParticleY = -PX_MAX_F32;
    PxReal maxSpeed = 0.0f;
    PxReal maxHeightChange = 0.0f;
    PxU32 badParticles = 0;
    if (!skipBuffer && particleBuffer) {
        printf("Starting simulation: steps=%d dim=%u\n", steps, dim);
        fflush(stdout);
        const int timingWarmupSteps = steps > 10 ? 10 : 0;
        double simulationMs = 0.0;
        for (int i = 0; i < steps; ++i) {
            if (printAllSteps || i < 10 || i == steps - 1) {
                printf("Simulate step %d/%d...\n", i + 1, steps);
                fflush(stdout);
            }
            const std::chrono::steady_clock::time_point stepStart = std::chrono::steady_clock::now();
            scene->simulate(1.0f / 60.0f);
            const double simulateMs = std::chrono::duration<double, std::milli>(
                std::chrono::steady_clock::now() - stepStart).count();
            if (printAllSteps || i < 10 || i == steps - 1) {
                printf("Fetch step %d/%d...\n", i + 1, steps);
                fflush(stdout);
            }
            const std::chrono::steady_clock::time_point fetchStart = std::chrono::steady_clock::now();
            scene->fetchResults(true);
            if (i >= timingWarmupSteps)
                simulationMs += simulateMs + std::chrono::duration<double, std::milli>(
                    std::chrono::steady_clock::now() - fetchStart).count();
        }
        const int measuredSteps = steps - timingWarmupSteps;
        printf("PERF marker=PX_DCU_BENCH_SIMULATION_TIMING_V1 warmup_steps=%d measured_steps=%d simulation_ms=%.3f ms_per_step=%.6f steps_per_second=%.3f\n",
               timingWarmupSteps, measuredSteps, simulationMs,
               measuredSteps > 0 ? simulationMs / double(measuredSteps) : 0.0,
               simulationMs > 0.0 ? double(measuredSteps) * 1000.0 / simulationMs : 0.0);

        std::vector<PxVec4> positionsReadback(numParticles);
        std::vector<PxVec4> velocitiesReadback(numParticles);
        PxI32 syncResult = -1;
        PxI32 positionCopyResult = -1;
        PxI32 velocityCopyResult = -1;
        {
            PxScopedCudaLock lock(*gpuMgr);
            PxCudaContext* ctx = gpuMgr->getCudaContext();
            syncResult = PxI32(ctx->streamSynchronize(0));
            positionCopyResult = PxI32(ctx->memcpyDtoH(positionsReadback.data(),
                                                       CUdeviceptr(particleBuffer->getPositionInvMasses()),
                                                       size_t(numParticles) * sizeof(PxVec4)));
            velocityCopyResult = PxI32(ctx->memcpyDtoH(velocitiesReadback.data(),
                                                       CUdeviceptr(particleBuffer->getVelocities()),
                                                       size_t(numParticles) * sizeof(PxVec4)));
        }
        readbackOk = syncResult == 0 && positionCopyResult == 0 && velocityCopyResult == 0;
        if (readbackOk) {
            for (PxU32 i = 0; i < numParticles; ++i) {
                const PxVec4& p = positionsReadback[i];
                const PxVec4& v = velocitiesReadback[i];
                if (!finiteVec4(p) || !finiteVec4(v)) {
                    ++badParticles;
                    continue;
                }
                minParticleY = PxMin(minParticleY, p.y);
                maxParticleY = PxMax(maxParticleY, p.y);
                maxSpeed = PxMax(maxSpeed, PxVec3(v.x, v.y, v.z).magnitude());
                maxHeightChange = PxMax(maxHeightChange, PxAbs(p.y - 1.5f));
            }
        }
        printf("GPU readback sync=%d positionCopy=%d velocityCopy=%d\n",
               syncResult, positionCopyResult, velocityCopyResult);
        printf("GPU particleHeight=[%.6f, %.6f] maxSpeed=%.6f maxHeightChange=%.6f badParticles=%u/%u\n",
               minParticleY, maxParticleY, maxSpeed, maxHeightChange, badParticles, numParticles);

        PxBounds3 bounds = ps->getWorldBounds(1.0f);
        const bool validBounds = bounds.isValid() && finiteVec(bounds.minimum) && finiteVec(bounds.maximum);
        const bool validParticles = readbackOk && badParticles == 0;
        const bool realMotion = steps == 0 || maxHeightChange > 0.001f || maxSpeed > 0.001f;
        const bool reachedPlane = !addPlane || steps < 40 || maxParticleY < 0.25f;
        const bool noPlanePenetration = !addPlane || steps == 0 || minParticleY > -0.02f;
        const bool settledOnPlane = !addPlane || steps < 180 ||
            (minParticleY > 0.02f && maxParticleY < 0.10f && maxSpeed < 0.02f);
        pass = validBounds && validParticles && realMotion && reachedPlane && noPlanePenetration && settledOnPlane;
        printf("PBD particle system bounds min=(%.6f %.6f %.6f) max=(%.6f %.6f %.6f)\n",
               bounds.minimum.x, bounds.minimum.y, bounds.minimum.z, bounds.maximum.x, bounds.maximum.y, bounds.maximum.z);
        printf("Checks validBounds=%s validParticles=%s realMotion=%s reachedPlane=%s noPlanePenetration=%s settledOnPlane=%s\n",
               validBounds ? "yes" : "no", validParticles ? "yes" : "no", realMotion ? "yes" : "no",
               reachedPlane ? "yes" : "no", noPlanePenetration ? "yes" : "no", settledOnPlane ? "yes" : "no");
    }
    if (!unsupportedParticleBuffer) {
        printf("VERDICT: %s\n", pass ? "PASS" : "FAIL");
        fflush(stdout);
    }

    if (particleBuffer) { printf("Releasing particle buffer...\n"); fflush(stdout); particleBuffer->release(); }
    if (ps) { printf("Releasing particle system...\n"); fflush(stdout); ps->release(); }
    if (mat) { printf("Releasing PBD material...\n"); fflush(stdout); mat->release(); }
    if (plane) { printf("Releasing rigid plane...\n"); fflush(stdout); plane->release(); }
    if (rigidMat) { printf("Releasing rigid material...\n"); fflush(stdout); rigidMat->release(); }
    printf("Releasing scene...\n"); fflush(stdout); scene->release();
    if (dsp) { printf("Releasing dispatcher...\n"); fflush(stdout); dsp->release(); }
    printf("Releasing GPU manager...\n"); fflush(stdout); gpuMgr->release();
    printf("Releasing physics...\n"); fflush(stdout); phy->release();
    printf("Releasing foundation...\n"); fflush(stdout); fnd->release();
    printf("Release finished.\n"); fflush(stdout);
    if (cleanExit) {
        printf("Clean exit after explicit PhysX release (--cleanexit), waiting %d seconds for DCU/HSA cleanup...\n", cleanExitWaitSeconds);
        fflush(stdout);
        if (cleanExitWaitSeconds > 0)
            std::this_thread::sleep_for(std::chrono::seconds(cleanExitWaitSeconds));
        printf("Clean exit wait finished.\n");
        fflush(stdout);
        std::_Exit(unsupportedParticleBuffer ? 77 : (pass ? 0 : 1));
    }
    return unsupportedParticleBuffer ? 77 : (pass ? 0 : 1);
}
