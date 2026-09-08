// PhysX DCU smoke test: minimal PBD particle cloth path.

#include "PxPhysicsAPI.h"
#include "cudamanager/PxCudaContext.h"
#include "cudamanager/PxCudaContextManager.h"
#include "extensions/PxParticleExt.h"
#include "extensions/PxCudaHelpersExt.h"
#include "gpu/PxGpu.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <thread>
#include <vector>

using namespace physx;

static const char* const gClothDiagnosticMarker = "PX_DCU_PBD_CLOTH_VALIDATION_V61_SCALE_AWARE_TAIL";

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

static bool finiteVec(const PxVec3& v)
{
    return std::isfinite(v.x) && std::isfinite(v.y) && std::isfinite(v.z);
}

static bool finiteVec4(const PxVec4& v)
{
    return std::isfinite(v.x) && std::isfinite(v.y) && std::isfinite(v.z) && std::isfinite(v.w);
}

static PxParticleSpring makeSpring(PxU32 a, PxU32 b, PxReal length, PxReal stiffness, PxReal damping)
{
    PxParticleSpring s;
    s.ind0 = a;
    s.ind1 = b;
    s.length = length;
    s.stiffness = stiffness;
    s.damping = damping;
    s.pad = 0.0f;
    return s;
}

int main(int argc, char** argv)
{
    int steps = 60;
    PxU32 dim = 8;
    bool cleanExit = false;
    bool forceGpuBroadphase = false;
    bool printAllSteps = false;
    bool populateOnly = false;
    bool pinCorners = false;
    bool addPlane = false;
    int cleanExitWaitSeconds = 5;
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--dim") == 0 && i + 1 < argc) {
            dim = PxU32(std::atoi(argv[++i]));
        } else if (std::strcmp(argv[i], "--force-gpu-bp") == 0) {
            forceGpuBroadphase = true;
        } else if (std::strcmp(argv[i], "--print-all-steps") == 0) {
            printAllSteps = true;
        } else if (std::strcmp(argv[i], "--populate-only") == 0) {
            populateOnly = true;
        } else if (std::strcmp(argv[i], "--pin-corners") == 0) {
            pinCorners = true;
        } else if (std::strcmp(argv[i], "--plane") == 0) {
            addPlane = true;
        } else if (std::strcmp(argv[i], "--cleanexit-wait") == 0 && i + 1 < argc) {
            cleanExit = true;
            cleanExitWaitSeconds = std::atoi(argv[++i]);
            if (cleanExitWaitSeconds < 0)
                cleanExitWaitSeconds = 0;
        } else if (std::strcmp(argv[i], "--cleanexit") == 0 || std::strcmp(argv[i], "cleanexit") == 0) {
            cleanExit = true;
        } else {
            steps = std::atoi(argv[i]);
        }
    }
    if (steps < 0)
        steps = 60;
    if (dim < 2)
        dim = 2;

    printf("========================================\n");
    printf(" PhysX DCU Smoke - PBD Cloth\n");
    printf("========================================\n");
    printf("Cloth diagnostic marker: %s\n", gClothDiagnosticMarker);
    printf("steps=%d dim=%u populateOnly=%d pinCorners=%d plane=%d forceGpuBp=%d\n\n",
           steps, dim, populateOnly ? 1 : 0, pinCorners ? 1 : 0, addPlane ? 1 : 0,
           forceGpuBroadphase ? 1 : 0);

    static CountingErrorCallback gErr;
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
    PxDefaultCpuDispatcher* dsp = PxDefaultCpuDispatcherCreate(0);
    sd.cpuDispatcher = dsp;
    sd.filterShader = PxDefaultSimulationFilterShader;
    sd.cudaContextManager = gpuMgr;
    sd.flags |= PxSceneFlag::eENABLE_GPU_DYNAMICS;
    sd.flags |= PxSceneFlag::eENABLE_PCM;
    if (forceGpuBroadphase)
        sd.broadPhaseType = PxBroadPhaseType::eGPU;
    sd.gpuMaxNumPartitions = 8;
    sd.gpuDynamicsConfig.heapCapacity = 512u * 1024u * 1024u;
    sd.gpuDynamicsConfig.tempBufferCapacity = 128u * 1024u * 1024u;
    sd.gpuDynamicsConfig.collisionStackSize = 256u * 1024u * 1024u;

    printf("Creating GPU dynamics scene: broadphase=%s...\n", forceGpuBroadphase ? "eGPU" : "default");
    fflush(stdout);
    PxScene* scene = phy->createScene(sd);
    if (!scene) {
        printf("FAIL: createScene returned null\n");
        return 3;
    }

    PxMaterial* rigidMat = nullptr;
    PxRigidStatic* plane = nullptr;
    if (addPlane) {
        printf("Adding rigid plane at y=0.000...\n");
        fflush(stdout);
        rigidMat = phy->createMaterial(0.5f, 0.5f, 0.0f);
        if (rigidMat)
            plane = PxCreatePlane(*phy, PxPlane(0, 1, 0, 0), *rigidMat);
        if (!plane) {
            printf("FAIL: create rigid plane failed material=%p plane=%p\n",
                   static_cast<void*>(rigidMat), static_cast<void*>(plane));
            return 4;
        }
        scene->addActor(*plane);
    }

    const PxU32 numParticles = dim * dim;
    const PxU32 numTriangles = 2 * (dim - 1) * (dim - 1);
    const PxU32 numSprings = dim * (dim - 1) * 2;
    const PxReal spacing = 0.1f;
    const PxReal restOffset = 0.05f;
    const PxReal totalMass = 10.0f;
    const PxReal invParticleMass = PxReal(numParticles) / totalMass;

    printf("Creating PBD material and particle system: dim=%u particles=%u triangles=%u springs=%u...\n",
           dim, numParticles, numTriangles, numSprings);
    fflush(stdout);
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

    printf("Allocating cloth helper and pinned buffers...\n");
    fflush(stdout);
    ExtGpu::PxParticleClothBufferHelper* helper = ExtGpu::PxCreateParticleClothBufferHelper(1, numTriangles, numSprings, numParticles, gpuMgr);
    PxU32* phases = PX_EXT_PINNED_MEMORY_ALLOC(PxU32, *gpuMgr, numParticles);
    PxVec4* positions = PX_EXT_PINNED_MEMORY_ALLOC(PxVec4, *gpuMgr, numParticles);
    PxVec4* velocities = PX_EXT_PINNED_MEMORY_ALLOC(PxVec4, *gpuMgr, numParticles);
    std::vector<PxParticleSpring> springs;
    std::vector<PxU32> triangles;
    springs.reserve(numSprings);
    triangles.reserve(numTriangles * 3u);

    const PxReal stretchStiffness = 10000.0f;
    const PxReal springDamping = 0.001f;
    for (PxU32 y = 0; y < dim; ++y) {
        for (PxU32 x = 0; x < dim; ++x) {
            const PxU32 id = y * dim + x;
            const bool pinned = pinCorners && y == 0 && (x == 0 || x + 1 == dim);
            positions[id] = PxVec4(PxReal(x) * spacing, 2.0f, PxReal(y) * spacing,
                                   pinned ? 0.0f : invParticleMass);
            velocities[id] = PxVec4(0.0f);
            phases[id] = phaseValue;
            if (x > 0)
                springs.push_back(makeSpring(id, id - 1, spacing, stretchStiffness, springDamping));
            if (y > 0)
                springs.push_back(makeSpring(id, id - dim, spacing, stretchStiffness, springDamping));
        }
    }
    for (PxU32 y = 0; y + 1 < dim; ++y) {
        for (PxU32 x = 0; x + 1 < dim; ++x) {
            const PxU32 i0 = y * dim + x;
            const PxU32 i1 = i0 + 1;
            const PxU32 i2 = i0 + dim;
            const PxU32 i3 = i2 + 1;
            triangles.push_back(i0); triangles.push_back(i2); triangles.push_back(i1);
            triangles.push_back(i1); triangles.push_back(i2); triangles.push_back(i3);
        }
    }

    printf("Adding cloth desc: springs=%zu triangles=%zu...\n", springs.size(), triangles.size() / 3);
    fflush(stdout);
    helper->addCloth(0.0f, 0.0f, 0.0f, triangles.data(), numTriangles, springs.data(), numSprings, positions, numParticles);

    ExtGpu::PxParticleBufferDesc desc;
    desc.maxParticles = numParticles;
    desc.numActiveParticles = numParticles;
    desc.positions = positions;
    desc.velocities = velocities;
    desc.phases = phases;

    printf("Partitioning cloth springs...\n");
    fflush(stdout);
    const PxParticleClothDesc& clothDesc = helper->getParticleClothDesc();
    PxParticleClothPreProcessor* pre = PxCreateParticleClothPreProcessor(gpuMgr);
    PxPartitionedParticleCloth output;
    pre->partitionSprings(clothDesc, output);
    pre->release();

    PxU32 partitionedSpringCount = 0;
    PxU32 partitionRangeErrors = 0;
    PxU32 partitionEndpointErrors = 0;
    PxU32 partitionEndpointConflicts = 0;
    std::vector<PxU64> sourceSpringKeys;
    std::vector<PxU64> orderedSpringKeys;
    sourceSpringKeys.reserve(numSprings);
    orderedSpringKeys.reserve(numSprings);
    for (PxU32 i = 0; i < numSprings; ++i) {
        const PxParticleSpring& spring = springs[i];
        const PxU32 lo = PxMin(spring.ind0, spring.ind1);
        const PxU32 hi = PxMax(spring.ind0, spring.ind1);
        sourceSpringKeys.push_back((PxU64(lo) << 32) | PxU64(hi));
    }
    std::vector<PxI32> endpointPartition(numParticles, -1);
    PxU32 partitionStart = 0;
    for (PxU32 partition = 0; partition < output.nbPartitions; ++partition) {
        const PxU32 partitionEnd = output.accumulatedSpringsPerPartitions[partition];
        if (partitionEnd < partitionStart || partitionEnd > numSprings) {
            ++partitionRangeErrors;
            break;
        }
        for (PxU32 i = partitionStart; i < partitionEnd; ++i) {
            const PxParticleSpring& spring = output.orderedSprings[i];
            if (spring.ind0 >= numParticles || spring.ind1 >= numParticles) {
                ++partitionEndpointErrors;
                continue;
            }
            const PxU32 lo = PxMin(spring.ind0, spring.ind1);
            const PxU32 hi = PxMax(spring.ind0, spring.ind1);
            orderedSpringKeys.push_back((PxU64(lo) << 32) | PxU64(hi));
            if (endpointPartition[spring.ind0] == PxI32(partition))
                ++partitionEndpointConflicts;
            if (endpointPartition[spring.ind1] == PxI32(partition))
                ++partitionEndpointConflicts;
            endpointPartition[spring.ind0] = PxI32(partition);
            endpointPartition[spring.ind1] = PxI32(partition);
        }
        partitionStart = partitionEnd;
    }
    partitionedSpringCount = partitionStart;
    std::sort(sourceSpringKeys.begin(), sourceSpringKeys.end());
    std::sort(orderedSpringKeys.begin(), orderedSpringKeys.end());
    const bool partitionCoverageValid = output.nbSprings == numSprings && partitionedSpringCount == numSprings &&
        partitionRangeErrors == 0 && partitionEndpointErrors == 0 && partitionEndpointConflicts == 0 &&
        sourceSpringKeys == orderedSpringKeys;
    printf("Spring partition diagnostics partitions=%u outputSprings=%u coveredSprings=%u maxPerPartition=%u "
           "remapOutputSize=%u rangeErrors=%u endpointErrors=%u endpointConflicts=%u coverage=%s\n",
           output.nbPartitions, output.nbSprings, partitionedSpringCount, output.maxSpringsPerPartition,
           output.remapOutputSize, partitionRangeErrors, partitionEndpointErrors, partitionEndpointConflicts,
           partitionCoverageValid ? "VALID" : "INVALID");
    if (partitionRangeErrors == 0) {
        for (PxU32 partition = 0; partition < output.nbPartitions; ++partition) {
            const PxU32 start = partition ? output.accumulatedSpringsPerPartitions[partition - 1] : 0;
            const PxU32 end = output.accumulatedSpringsPerPartitions[partition];
            printf("Spring partition %u range=[%u,%u) count=%u\n", partition, start, end, end - start);
        }
    }

    printf("Creating and populating particle cloth buffer...\n");
    fflush(stdout);
    PxParticleClothBuffer* clothBuffer = ExtGpu::PxCreateAndPopulateParticleClothBuffer(desc, clothDesc, output, gpuMgr);
    if (!clothBuffer) {
        printf("FAIL: PxCreateAndPopulateParticleClothBuffer returned null\n");
        return 5;
    }
    helper->release();
    PX_EXT_PINNED_MEMORY_FREE(*gpuMgr, positions);
    PX_EXT_PINNED_MEMORY_FREE(*gpuMgr, velocities);
    PX_EXT_PINNED_MEMORY_FREE(*gpuMgr, phases);

    bool pass = partitionCoverageValid;
    if (populateOnly) {
        printf("Populate-only validation completed; skipping addParticleBuffer/simulation.\n");
        fflush(stdout);
    } else {
        ps->addParticleBuffer(clothBuffer);
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
                                                       CUdeviceptr(clothBuffer->getPositionInvMasses()),
                                                       size_t(numParticles) * sizeof(PxVec4)));
            velocityCopyResult = PxI32(ctx->memcpyDtoH(velocitiesReadback.data(),
                                                       CUdeviceptr(clothBuffer->getVelocities()),
                                                       size_t(numParticles) * sizeof(PxVec4)));
        }

        const bool readbackOk = syncResult == 0 && positionCopyResult == 0 && velocityCopyResult == 0;
        PxU32 badParticles = 0;
        PxReal minParticleY = PX_MAX_F32;
        PxReal maxParticleY = -PX_MAX_F32;
        PxReal maxSpeed = 0.0f;
        PxReal maxPinnedDrift = 0.0f;
        PxReal minDynamicY = PX_MAX_F32;
        PxReal maxSpringError = 0.0f;
        PxReal sumSpringError = 0.0f;
        PxReal sumSquaredSpringError = 0.0f;
        PxReal maxSpringLength = 0.0f;
        PxU32 maxSpringIndex = 0;
        PxU32 validSpringCount = 0;
        PxU32 springsAboveHalfSpacing = 0;
        PxU32 springsAboveSpacing = 0;
        PxU32 springsAboveOneAndHalfSpacing = 0;
        PxU32 springsAboveHalfSpacingByAnchorDistance[5] = { 0, 0, 0, 0, 0 };
        std::vector<PxReal> springErrors;
        springErrors.reserve(numSprings);

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

                const bool pinned = pinCorners && (i == 0 || i + 1 == dim);
                if (pinned) {
                    const PxVec3 initial(PxReal(i) * spacing, 2.0f, 0.0f);
                    maxPinnedDrift = PxMax(maxPinnedDrift, (PxVec3(p.x, p.y, p.z) - initial).magnitude());
                } else {
                    minDynamicY = PxMin(minDynamicY, p.y);
                }
            }

            for (PxU32 i = 0; i < numSprings; ++i) {
                const PxParticleSpring& spring = springs[i];
                const PxVec4& p0 = positionsReadback[spring.ind0];
                const PxVec4& p1 = positionsReadback[spring.ind1];
                if (!finiteVec4(p0) || !finiteVec4(p1))
                    continue;
                const PxReal length = (PxVec3(p1.x, p1.y, p1.z) - PxVec3(p0.x, p0.y, p0.z)).magnitude();
                const PxReal error = PxAbs(length - spring.length);
                if (validSpringCount == 0 || error > maxSpringError) {
                    maxSpringError = error;
                    maxSpringLength = length;
                    maxSpringIndex = i;
                }
                sumSpringError += error;
                sumSquaredSpringError += error * error;
                springErrors.push_back(error);
                ++validSpringCount;
                if (error >= spacing * 0.5f)
                {
                    ++springsAboveHalfSpacing;
                    const PxU32 row0 = spring.ind0 / dim;
                    const PxU32 col0 = spring.ind0 % dim;
                    const PxU32 row1 = spring.ind1 / dim;
                    const PxU32 col1 = spring.ind1 % dim;
                    const PxU32 anchorDistance0 = PxMin(row0 + col0, row0 + (dim - 1 - col0));
                    const PxU32 anchorDistance1 = PxMin(row1 + col1, row1 + (dim - 1 - col1));
                    const PxU32 anchorDistance = PxMin(anchorDistance0, anchorDistance1);
                    ++springsAboveHalfSpacingByAnchorDistance[PxMin(anchorDistance, 4u)];
                }
                if (error >= spacing)
                    ++springsAboveSpacing;
                if (error >= spacing * 1.5f)
                    ++springsAboveOneAndHalfSpacing;
            }
        }

        std::sort(springErrors.begin(), springErrors.end());
        const PxReal averageSpringError = validSpringCount ? sumSpringError / PxReal(validSpringCount) : 0.0f;
        const PxReal rmsSpringError = validSpringCount ? PxSqrt(sumSquaredSpringError / PxReal(validSpringCount)) : 0.0f;
        const PxReal p50SpringError = validSpringCount ? springErrors[(validSpringCount - 1) * 50 / 100] : 0.0f;
        const PxReal p95SpringError = validSpringCount ? springErrors[(validSpringCount - 1) * 95 / 100] : 0.0f;
        const PxReal p99SpringError = validSpringCount ? springErrors[(validSpringCount - 1) * 99 / 100] : 0.0f;
        const PxReal dynamicDrop = minDynamicY < PX_MAX_F32 ? 2.0f - minDynamicY : 0.0f;
        const PxReal heightSpan = maxParticleY > -PX_MAX_F32 ? maxParticleY - minParticleY : 0.0f;
        printf("GPU readback sync=%d positionCopy=%d velocityCopy=%d\n",
               syncResult, positionCopyResult, velocityCopyResult);
        printf("GPU cloth particles height=[%.6f, %.6f] maxSpeed=%.6f badParticles=%u/%u\n",
               minParticleY, maxParticleY, maxSpeed, badParticles, numParticles);
        printf("Spring validation pinnedDrift=%.6f dynamicDrop=%.6f heightSpan=%.6f maxError=%.6f avgError=%.6f\n",
               maxPinnedDrift, dynamicDrop, heightSpan, maxSpringError, averageSpringError);
        printf("Spring error distribution valid=%u/%u p50=%.6f p95=%.6f p99=%.6f rms=%.6f "
               "above0.5spacing=%u above1.0spacing=%u above1.5spacing=%u\n",
               validSpringCount, numSprings, p50SpringError, p95SpringError, p99SpringError, rmsSpringError,
               springsAboveHalfSpacing, springsAboveSpacing, springsAboveOneAndHalfSpacing);
        printf("Springs above 0.5 spacing by nearest-anchor distance d0=%u d1=%u d2=%u d3=%u d4plus=%u\n",
               springsAboveHalfSpacingByAnchorDistance[0], springsAboveHalfSpacingByAnchorDistance[1],
               springsAboveHalfSpacingByAnchorDistance[2], springsAboveHalfSpacingByAnchorDistance[3],
               springsAboveHalfSpacingByAnchorDistance[4]);
        if (validSpringCount) {
            const PxParticleSpring& spring = springs[maxSpringIndex];
            const PxVec4& p0 = positionsReadback[spring.ind0];
            const PxVec4& p1 = positionsReadback[spring.ind1];
            const PxU32 row0 = spring.ind0 / dim;
            const PxU32 col0 = spring.ind0 % dim;
            const PxU32 row1 = spring.ind1 / dim;
            const PxU32 col1 = spring.ind1 % dim;
            const PxU32 anchorDistance0 = PxMin(row0 + col0, row0 + (dim - 1 - col0));
            const PxU32 anchorDistance1 = PxMin(row1 + col1, row1 + (dim - 1 - col1));
            printf("Worst spring index=%u endpoints=(%u[%u,%u],%u[%u,%u]) anchorDistance=(%u,%u) "
                   "rest=%.6f length=%.6f error=%.6f ratio=%.6f\n",
                   maxSpringIndex, spring.ind0, row0, col0, spring.ind1, row1, col1,
                   anchorDistance0, anchorDistance1, spring.length, maxSpringLength, maxSpringError,
                   spring.length > 0.0f ? maxSpringLength / spring.length : 0.0f);
            printf("Worst spring positions p0=(%.6f,%.6f,%.6f,%.6f) p1=(%.6f,%.6f,%.6f,%.6f)\n",
                   p0.x, p0.y, p0.z, p0.w, p1.x, p1.y, p1.z, p1.w);
        }

        PxBounds3 bounds = ps->getWorldBounds(1.0f);
        const bool validBounds = bounds.isValid() && finiteVec(bounds.minimum) && finiteVec(bounds.maximum);
        const bool validParticles = readbackOk && badParticles == 0;
        const bool pinnedStable = !pinCorners || maxPinnedDrift < 0.001f;
        const bool nonRigidMotion = !pinCorners || steps < 30 || (dynamicDrop > 0.05f && heightSpan > 0.05f);
        const bool completeSpringSamples = validSpringCount == numSprings;
        const bool averageSpringErrorBounded = averageSpringError < spacing * 0.1f;
        const bool rmsSpringErrorBounded = rmsSpringError < spacing * 0.25f;
        const bool p95SpringErrorBounded = p95SpringError < spacing * 0.5f;
        const bool p99SpringErrorBounded = p99SpringError < spacing;
        const bool maxSpringErrorBounded = maxSpringError < spacing * 2.0f;
        const bool springDistributionBounded = completeSpringSamples && averageSpringErrorBounded &&
            rmsSpringErrorBounded && p95SpringErrorBounded && p99SpringErrorBounded && maxSpringErrorBounded;
        const bool boundedSpringError = (!pinCorners && !addPlane) || steps == 0 || springDistributionBounded;
        const bool reachedPlane = !addPlane || pinCorners || steps < 60 || maxParticleY < 0.25f;
        const bool noPlanePenetration = !addPlane || steps == 0 || minParticleY > -0.02f;
        const bool settledOnPlane = !addPlane || pinCorners || steps < 180 ||
            (minParticleY > 0.02f && maxParticleY < 0.12f && maxSpeed < 0.05f);
        pass = pass && validBounds && validParticles && pinnedStable && nonRigidMotion && boundedSpringError &&
            reachedPlane && noPlanePenetration && settledOnPlane;
        printf("PBD cloth system bounds min=(%.6f %.6f %.6f) max=(%.6f %.6f %.6f)\n",
               bounds.minimum.x, bounds.minimum.y, bounds.minimum.z,
               bounds.maximum.x, bounds.maximum.y, bounds.maximum.z);
        printf("Checks validBounds=%s validParticles=%s pinnedStable=%s nonRigidMotion=%s boundedSpringError=%s "
               "reachedPlane=%s noPlanePenetration=%s settledOnPlane=%s\n",
               validBounds ? "yes" : "no", validParticles ? "yes" : "no", pinnedStable ? "yes" : "no",
               nonRigidMotion ? "yes" : "no", boundedSpringError ? "yes" : "no",
               reachedPlane ? "yes" : "no", noPlanePenetration ? "yes" : "no", settledOnPlane ? "yes" : "no");
        printf("Spring checks partitionCoverage=%s completeSamples=%s avgLt0.1=%s rmsLt0.25=%s "
               "p95Lt0.5=%s p99Lt1.0=%s maxLt2.0=%s\n",
               partitionCoverageValid ? "yes" : "no", completeSpringSamples ? "yes" : "no",
               averageSpringErrorBounded ? "yes" : "no", rmsSpringErrorBounded ? "yes" : "no",
               p95SpringErrorBounded ? "yes" : "no", p99SpringErrorBounded ? "yes" : "no",
               maxSpringErrorBounded ? "yes" : "no");
    }
    if (gErr.errorCount != 0) {
        printf("FAIL: PhysX reported %u error(s)\n", gErr.errorCount);
        pass = false;
    }
    printf("VERDICT: %s\n", pass ? "PASS" : "FAIL");
    fflush(stdout);

    printf("Releasing particle cloth buffer...\n"); fflush(stdout); clothBuffer->release();
    printf("Releasing particle system...\n"); fflush(stdout); ps->release();
    printf("Releasing PBD material...\n"); fflush(stdout); mat->release();
    if (plane) { printf("Releasing rigid plane...\n"); fflush(stdout); plane->release(); }
    if (rigidMat) { printf("Releasing rigid material...\n"); fflush(stdout); rigidMat->release(); }
    printf("Releasing scene...\n"); fflush(stdout); scene->release();
    printf("Releasing dispatcher...\n"); fflush(stdout); dsp->release();
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
        std::_Exit(pass ? 0 : 1);
    }
    return pass ? 0 : 1;
}
