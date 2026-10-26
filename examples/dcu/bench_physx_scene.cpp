// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions
// are met:
//  * Redistributions of source code must retain the above copyright
//    notice, this list of conditions and the following disclaimer.
//  * Redistributions in binary form must reproduce the above copyright
//    notice, this list of conditions and the following disclaimer in the
//    documentation and/or other materials provided with the distribution.
//  * Neither the name of NVIDIA CORPORATION nor the names of its
//    contributors may be used to endorse or promote products derived
//    from this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ''AS IS'' AND ANY
// EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
// PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
// CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
// EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
// PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
// OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//
// PhysX Scene Benchmark — PxScene::simulate() on GPU
// Same code for HCU (g++) and A800 (g++).

#include "PxPhysicsAPI.h"
#include "cudamanager/PxCudaContextManager.h"
#include "gpu/PxGpu.h"
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <cmath>
#include <ctime>
#include <cstring>
#include <algorithm>
#if defined(__linux__)
#include <dlfcn.h>
#endif
#if defined(__HIPCC__)
#include <hip/hip_runtime.h>
#endif

using namespace physx;

struct SummaryOptions
{
    int steps = 300;
    int dumpEvery = 0;
    const char* summaryCsv = nullptr;
    bool profileContactFrame = false;
    bool traceContactFrame = false;
    bool profileControlProbe = false;
    int stage2IndexP54 = -1; // Explicit ordinary-rigid-only runtime ablation; default inactive.
    bool finalSummary = false; // P46: output only after the simulation timer stops.
};

struct FrameSummary
{
    int frame = 0;
    float minY = 0.0f;
    float maxY = 0.0f;
    double avgY = 0.0;
    double avgSpeed = 0.0;
    float maxSpeed = 0.0f;
    int nanCount = 0;
    int finiteCount = 0;
    int sleepingCount = 0;
    int above05 = 0;
    int belowGround = 0;
    double stateHash = 0.0;
};

static SummaryOptions parseArgs(int argc, char** argv)
{
    SummaryOptions opt;
    for (int i = 1; i < argc; ++i)
    {
        if (!std::strcmp(argv[i], "--steps") && i + 1 < argc)
            opt.steps = std::atoi(argv[++i]);
        else if (!std::strcmp(argv[i], "--dump-every") && i + 1 < argc)
            opt.dumpEvery = std::atoi(argv[++i]);
        else if (!std::strcmp(argv[i], "--summary") && i + 1 < argc)
            opt.summaryCsv = argv[++i];
        else if (!std::strcmp(argv[i], "--profile-contact-frame")) {
            opt.profileContactFrame = true;
            opt.traceContactFrame = true;
        }
        else if (!std::strcmp(argv[i], "--contact-frame-check"))
            opt.profileContactFrame = true;
        else if (!std::strcmp(argv[i], "--p54-stage2-index") && i + 1 < argc) {
            const char* mode = argv[++i];
            if (std::strcmp(mode, "0") && std::strcmp(mode, "1")) {
                std::fprintf(stderr, "P54 stage2 index mode must be 0 or 1\n");
                std::exit(2);
            }
            opt.stage2IndexP54 = mode[0] - '0';
        }
        else if (!std::strcmp(argv[i], "--final-summary"))
            opt.finalSummary = true;
        else if (!std::strcmp(argv[i], "--profile-control-probe"))
            opt.profileControlProbe = true;
        else if (!std::strcmp(argv[i], "--help"))
        {
            printf("Usage: bench_physx_scene [--steps N] [--dump-every N] [--summary file.csv] [--final-summary] [--p54-stage2-index 0|1] [--profile-contact-frame | --contact-frame-check]\n");
            std::exit(0);
        }
    }
    if (opt.steps <= 0)
        opt.steps = 300;
    if (opt.dumpEvery < 0)
        opt.dumpEvery = 0;
    return opt;
}

// Opt-in diagnostic only. Keep all original contact/solver flags; request points
// only for sphere-involving pairs rather than for every pair in the box tower.
struct ProfileContacts : PxSimulationEventCallback
{
    PxU32 pairs = 0, points = 0;
    void reset() { pairs = points = 0; }
    void onContact(const PxContactPairHeader&, const PxContactPair* p, PxU32 count) override
    {
        for (PxU32 i = 0; i < count; ++i)
            if ((p[i].events & (PxPairFlag::eNOTIFY_TOUCH_FOUND | PxPairFlag::eNOTIFY_TOUCH_PERSISTS)) && p[i].contactCount > 0) {
                ++pairs;
                points += p[i].contactCount;
            }
    }
    void onConstraintBreak(PxConstraintInfo*, PxU32) override {}
    void onWake(PxActor**, PxU32) override {}
    void onSleep(PxActor**, PxU32) override {}
    void onTrigger(PxTriggerPair*, PxU32) override {}
    void onAdvance(const PxRigidBody* const*, const PxTransform*, PxU32) override {}
};

static PxFilterFlags profileContactFilter(PxFilterObjectAttributes a0, PxFilterData d0,
    PxFilterObjectAttributes a1, PxFilterData d1, PxPairFlags& flags,
    const void* data, PxU32 size)
{
    PxFilterFlags result = PxDefaultSimulationFilterShader(a0, d0, a1, d1, flags, data, size);
    if (!(result & (PxFilterFlag::eSUPPRESS | PxFilterFlag::eKILL)) && (d0.word1 == 0x50443434u || d1.word1 == 0x50443434u))
        flags |= PxPairFlag::eNOTIFY_TOUCH_FOUND | PxPairFlag::eNOTIFY_TOUCH_PERSISTS | PxPairFlag::eNOTIFY_CONTACT_POINTS;
    return result;
}

struct ProfileControl
{
    typedef void (*Control)();
    typedef int (*StatusControl)();
    typedef int (*GetDeviceCount)(int*);
    Control traceStart = nullptr, traceStop = nullptr;
    StatusControl hipStart = nullptr, hipStop = nullptr, synchronize = nullptr;
    GetDeviceCount getCount = nullptr;
    const char* backend = "unavailable";
    bool resolve()
    {
#if defined(__linux__)
        // Initialize HIP while tracing is OFF, allowing DTK to load its tracing
        // library. Never dlopen a second tracer instance. The runner verifies
        // this exact backend's off/on/off behavior with a tiny API-only probe.
        getCount = reinterpret_cast<GetDeviceCount>(dlsym(RTLD_DEFAULT, "hipGetDeviceCount"));
        int devices = 0;
        if (!getCount || getCount(&devices) != 0 || devices != 1)
            return false;
        traceStart = reinterpret_cast<Control>(dlsym(RTLD_DEFAULT, "roctracer_start"));
        traceStop = reinterpret_cast<Control>(dlsym(RTLD_DEFAULT, "roctracer_stop"));
        hipStart = reinterpret_cast<StatusControl>(dlsym(RTLD_DEFAULT, "hipProfilerStart"));
        hipStop = reinterpret_cast<StatusControl>(dlsym(RTLD_DEFAULT, "hipProfilerStop"));
        synchronize = reinterpret_cast<StatusControl>(dlsym(RTLD_DEFAULT, "hipDeviceSynchronize"));
#endif
        if (traceStart && traceStop) backend = "roctracer";
        else if (hipStart && hipStop) backend = "hipProfiler";
        else return false;
        return synchronize != nullptr;
    }
    int start() { if (traceStart && traceStop) { traceStart(); return 0; } return hipStart(); }
    int stop() { if (traceStart && traceStop) { traceStop(); return 0; } return hipStop(); }
};

static FrameSummary computeFrameSummary(int frame, const std::vector<PxRigidDynamic*>& bodies)
{
    FrameSummary s;
    s.frame = frame;
    float minY = 1e30f;
    float maxY = -1e30f;
    float maxSpeed = 0.0f;
    double sumY = 0.0;
    double sumSpeed = 0.0;
    double hash = 0.0;

    for (size_t i = 0; i < bodies.size(); ++i)
    {
        PxRigidDynamic* b = bodies[i];
        if (!b)
            continue;

        const PxTransform pose = b->getGlobalPose();
        const PxVec3 p = pose.p;
        const PxVec3 v = b->getLinearVelocity();

        if (b->isSleeping())
            s.sleepingCount++;

        const bool finite =
            std::isfinite(p.x) && std::isfinite(p.y) && std::isfinite(p.z) &&
            std::isfinite(v.x) && std::isfinite(v.y) && std::isfinite(v.z);
        if (!finite)
        {
            s.nanCount++;
            continue;
        }

        const float speed = std::sqrt(v.x * v.x + v.y * v.y + v.z * v.z);
        minY = std::min(minY, p.y);
        maxY = std::max(maxY, p.y);
        maxSpeed = std::max(maxSpeed, speed);
        sumY += p.y;
        sumSpeed += speed;

        if (p.y > 0.5f)
            s.above05++;
        if (p.y < -0.01f)
            s.belowGround++;

        // Lightweight state fingerprint for locating HCU/NV divergence frames.
        hash += double(i + 1) *
                (double(p.x) * 0.13 + double(p.y) * 0.17 + double(p.z) * 0.19 +
                 double(v.x) * 0.23 + double(v.y) * 0.29 + double(v.z) * 0.31);
        s.finiteCount++;
    }

    if (s.finiteCount > 0)
    {
        s.minY = minY;
        s.maxY = maxY;
        s.avgY = sumY / double(s.finiteCount);
        s.avgSpeed = sumSpeed / double(s.finiteCount);
        s.maxSpeed = maxSpeed;
        s.stateHash = hash;
    }
    return s;
}

static void printFrameSummary(const FrameSummary& s)
{
    printf("[frame %04d] minY=%.9g maxY=%.9g avgY=%.9g avgSpeed=%.9g maxSpeed=%.9g nan=%d finite=%d sleeping=%d above05=%d belowGround=%d hash=%.17g\n",
           s.frame, s.minY, s.maxY, s.avgY, s.avgSpeed, s.maxSpeed, s.nanCount,
           s.finiteCount, s.sleepingCount, s.above05, s.belowGround, s.stateHash);
}

static void writeSummaryHeader(FILE* fp)
{
    if (fp)
        fprintf(fp, "frame,minY,maxY,avgY,avgSpeed,maxSpeed,nanCount,finiteCount,sleepingCount,above05,belowGround,stateHash\n");
}

static void writeFrameSummary(FILE* fp, const FrameSummary& s)
{
    if (!fp)
        return;
    fprintf(fp, "%d,%.9g,%.9g,%.17g,%.17g,%.9g,%d,%d,%d,%d,%d,%.17g\n",
            s.frame, s.minY, s.maxY, s.avgY, s.avgSpeed, s.maxSpeed,
            s.nanCount, s.finiteCount, s.sleepingCount, s.above05,
            s.belowGround, s.stateHash);
}

int main(int argc, char** argv)
{
    SummaryOptions opt = parseArgs(argc, argv);
    if (opt.stage2IndexP54 >= 0) {
#if defined(__linux__)
        const char* value = opt.stage2IndexP54 ? "ordinary-rigid/p54/1" : "ordinary-rigid/p54/0";
        if (setenv("PX_DCU_RIGID_STAGE2_INDEX_P54", value, 1) != 0) return 2;
        printf("p54_benchmark_marker=PX_DCU_RIGID_STAGE2_INDEX_P54_BENCH_V1 mode=%d scope=ordinary_rigid\n", opt.stage2IndexP54);
#else
        std::fprintf(stderr, "P54 requires the gfx936 HCU product\n");
        return 2;
#endif
    }

    ProfileControl profile;
    ProfileContacts contacts;
    if (opt.profileControlProbe) {
        setvbuf(stdout, nullptr, _IOLBF, 0);
        printf("profile_probe_marker=PX_DCU_RIGID_PROFILE_CONTROL_PROBE_P44_R2_V1\n");
        if (!profile.resolve() || profile.stop() != 0) return 3;
        int devices = 0;
        if (profile.getCount(&devices) != 0 || profile.start() != 0) return 3;
        const int insideStatus = profile.getCount(&devices);
        const int stopStatus = profile.stop();
        const int outsideStatus = profile.getCount(&devices);
        if (insideStatus != 0 || stopStatus != 0 || outsideStatus != 0) return 3;
        printf("profile_probe_gate=1 backend=%s expected_traced_hipGetDeviceCount_calls=1\n", profile.backend);
        return 0;
    }
    if (opt.profileContactFrame) {
        setvbuf(stdout, nullptr, _IOLBF, 0);
        printf("profile_marker=PX_DCU_RIGID_MAIN_CONTACT_FRAME_P44_R2_V1\n");
        if (opt.traceContactFrame && !profile.resolve()) {
            fprintf(stderr, "ERROR: DTK trace control or single visible device unavailable; no physics launched\n");
            return 3;
        }
        if (opt.traceContactFrame) {
            if (profile.stop() != 0) return 3;
            printf("profile_control_gate=1 trace_initially_off=1\n");
            printf("profile_control_backend=%s\n", profile.backend);
        }
        opt.steps = 1;
        opt.dumpEvery = 1;
    }
    printf("========================================\n");
    printf(" PhysX Scene Benchmark — Tower Collapse\n");
    printf("========================================\n\n");

    static PxDefaultErrorCallback gErr;
    static PxDefaultAllocator       gAlloc;
    PxFoundation* fnd = PxCreateFoundation(PX_PHYSICS_VERSION, gAlloc, gErr);
    PxPhysics*    phy = PxCreatePhysics(PX_PHYSICS_VERSION, *fnd, PxTolerancesScale());

    // GPU setup
    PxCudaContextManagerDesc gpuDesc;
    gpuDesc.deviceOrdinal = 0;
    PxCudaContextManager* gpuMgr = PxCreateCudaContextManager(*fnd, gpuDesc, nullptr, false);
    bool gpuOk = gpuMgr && gpuMgr->contextIsValid();
    printf("GPU: %s (%d CUs, %.1f GB) — %s\n",
        gpuOk ? gpuMgr->getDeviceName() : "NONE",
        gpuOk ? gpuMgr->getMultiprocessorCount() : 0,
        gpuOk ? gpuMgr->getDeviceTotalMemBytes()/(1024.0*1024.0*1024.0) : 0.0,
        gpuOk ? "GPU enabled" : "CPU only");

    // ---- CPU warm-up: initialize ROCm/PhysX internals ----
    printf("CPU warmup...\n");
    {
        PxDefaultCpuDispatcher* dspWarm = PxDefaultCpuDispatcherCreate(0);
        PxSceneDesc sdCpu(phy->getTolerancesScale());
        sdCpu.gravity = PxVec3(0, -9.81f, 0);
        sdCpu.cpuDispatcher = dspWarm;
        sdCpu.filterShader = PxDefaultSimulationFilterShader;
        PxScene* cpuScene = phy->createScene(sdCpu);
        cpuScene->simulate(1.0f/60.0f);
        cpuScene->fetchResults(true);
        cpuScene->release();
        dspWarm->release();
        printf("CPU warmup OK\n");
    }

    PxSceneDesc sd(phy->getTolerancesScale());
    sd.gravity = PxVec3(0, -9.81f, 0);
    sd.cudaContextManager = gpuMgr;
    if (gpuOk) {
        sd.flags |= PxSceneFlag::eENABLE_GPU_DYNAMICS;
        sd.gpuMaxNumPartitions = 8;
        sd.gpuDynamicsConfig.maxRigidContactCount   = 1024u * 1024u * 8u;
        sd.gpuDynamicsConfig.maxRigidPatchCount     = 1024u * 1024u;
        sd.gpuDynamicsConfig.foundLostPairsCapacity = 1024u * 1024u * 2u;
        sd.gpuDynamicsConfig.heapCapacity           = 512u * 1024u * 1024u;
        sd.gpuDynamicsConfig.collisionStackSize     = 256u * 1024u * 1024u;
        sd.gpuDynamicsConfig.tempBufferCapacity     = 128u * 1024u * 1024u;
    }
    PxDefaultCpuDispatcher* dsp = PxDefaultCpuDispatcherCreate(0);
    sd.cpuDispatcher = dsp;
    sd.filterShader  = opt.profileContactFrame ? profileContactFilter : PxDefaultSimulationFilterShader;
    if (opt.profileContactFrame)
        sd.simulationEventCallback = &contacts;
    PxScene* scene = phy->createScene(sd);

    // ---- Build Tower ----
    const int LAYERS = 20, BASE = 44;
    const float H = 0.5f;
    PxMaterial* mat = phy->createMaterial(0.5f, 0.5f, 0.3f);
    PxShape* boxShape = phy->createShape(PxBoxGeometry(H, H, H), *mat);
    scene->addActor(*PxCreatePlane(*phy, PxPlane(PxVec3(0,1,0), 0), *mat));

    std::vector<PxRigidDynamic*> bodies;
    int nBox = 0;

    for (int layer = 0; layer < LAYERS; layer++) {
        int side = BASE - layer * 2; if (side <= 0) break;
        float y = (layer * 2 + 1) * H + 0.01f;
        for (int ix = 0; ix < side; ix++) for (int iz = 0; iz < side; iz++) {
            float x = (ix - (side-1)/2.0f) * 2.0f * H;
            float z = (iz - (side-1)/2.0f) * 2.0f * H;
            PxRigidDynamic* b = phy->createRigidDynamic(PxTransform(PxVec3(x, y, z)));
            b->attachShape(*boxShape);
            PxRigidBodyExt::updateMassAndInertia(*b, 1.0f);
            scene->addActor(*b);
            bodies.push_back(b);
            nBox++;
        }
    }
    int nTotal = (int)bodies.size();
    printf("Bodies: %d (%d layers)\n", nBox, LAYERS);

    // ---- Sphere Cannon ----
    const int NSPH = 300;
    PxMaterial* matH = phy->createMaterial(0.8f, 0.8f, 0.1f);
    PxShape* sphShape = phy->createShape(PxSphereGeometry(0.6f), *matH);
    if (opt.profileContactFrame) {
        PxFilterData data;
        // PxDefaultSimulationFilterShader ignores word1; word2/word3 are
        // collision group masks and must remain unchanged.
        data.word1 = 0x50443434u;
        sphShape->setSimulationFilterData(data);
    }
    srand(42);
    for (int s = 0; s < NSPH; s++) {
        float a = rand()/(float)RAND_MAX * 6.283f;
        float v = 20 + rand()/(float)RAND_MAX * 35;
        PxRigidDynamic* sp = phy->createRigidDynamic(
            PxTransform(PxVec3((rand()%60-30), 15+rand()%10, (rand()%60-30))));
        sp->attachShape(*sphShape);
        sp->setLinearVelocity(PxVec3(cosf(a)*v, -5, sinf(a)*v));
        PxRigidBodyExt::updateMassAndInertia(*sp, 5.0f);
        scene->addActor(*sp);
        bodies.push_back(sp);
    }
    nTotal = (int)bodies.size();
    printf("Projectiles: %d | Total: %d\n\n", NSPH, nTotal);

    // ---- Diagnostic: zero-gravity zero-time step ----
    printf("Diagnostic: 0 gravity + dt=0...\n");
    scene->setGravity(PxVec3(0, 0, 0));
    scene->simulate(0.0f);
#if defined(__HIPCC__)
    hipError_t e = hipDeviceSynchronize();
    if (e != hipSuccess) {
        printf("FATAL: crash during zero-step (init phase): %s\n", hipGetErrorString(e));
        exit(1);
    }
#endif
    scene->fetchResults(true);
    printf("Diagnostic OK\n");
    scene->setGravity(PxVec3(0, -9.81f, 0));

    // ---- Settle ----
    printf("Settling...\n");
#if defined(__HIPCC__)
    for (int s = 0; s < 120; s++) {
        if (s % 10 == 0) printf("  step %d...\n", s);
        if (opt.profileContactFrame) contacts.reset();
        scene->simulate(1.0f/60.0f);
        e = hipDeviceSynchronize();
        if (e != hipSuccess) {
            printf("FATAL step %d: %s\n", s, hipGetErrorString(e));
            exit(1);
        }
        scene->fetchResults(true);
    }
#else
    for (int s = 0; s < 120; s++) { if (opt.profileContactFrame) contacts.reset(); scene->simulate(1.0f/60.0f); scene->fetchResults(true); }
#endif

    int contactWait = 0;
    if (opt.profileContactFrame) {
        // Original 120 settling steps remain untraced. Select the next frame
        // only after real sphere contact points in a completed step. Confirm
        // contacts independently in the captured frame; never collect a retry.
        while (contacts.points == 0 && contactWait < 30) {
            contacts.reset();
            scene->simulate(1.0f/60.0f);
            scene->fetchResults(true);
            ++contactWait;
        }
        printf("PROFILE_CONTACT_READY settle_steps=120 wait_steps=%d preceding_pairs=%u preceding_points=%u\n",
            contactWait, contacts.pairs, contacts.points);
        if (contacts.points == 0) {
            fprintf(stderr, "ERROR: no sphere collision contact points after settling/contact wait\n");
            return 4;
        }
    }

    // ---- Benchmark ----
    printf("Running %d simulation steps...\n", opt.steps);
    if (opt.summaryCsv)
        printf("Writing frame summary to: %s\n", opt.summaryCsv);
    if (opt.dumpEvery > 0)
        printf("Printing frame summary every %d frame(s)\n", opt.dumpEvery);
    float step = 1.0f/60.0f;

    FILE* summaryFile = nullptr;
    if (opt.summaryCsv) {
        summaryFile = std::fopen(opt.summaryCsv, "w");
        if (!summaryFile) {
            printf("ERROR: failed to open summary csv: %s\n", opt.summaryCsv);
            return 1;
        }
        writeSummaryHeader(summaryFile);
    }

    FrameSummary s0 = computeFrameSummary(0, bodies);
    writeFrameSummary(summaryFile, s0);
    if (opt.dumpEvery > 0)
        printFrameSummary(s0);

    struct timespec t0, t1;
    clock_gettime(CLOCK_MONOTONIC, &t0);
    for (int f = 0; f < opt.steps; f++) {
        if (opt.profileContactFrame) {
            contacts.reset();
            if (opt.traceContactFrame) {
                if (profile.synchronize() != 0) {
                    fprintf(stderr, "FATAL: pre-window hipDeviceSynchronize failed\n");
                    return 5;
                }
                printf("PROFILE_WINDOW_BEGIN frame=1 absolute_step=%d\n", 121 + contactWait);
                if (profile.start() != 0) {
                    fprintf(stderr, "FATAL: profile start failed\n");
                    return 5;
                }
            }
        }
        scene->simulate(step);
        scene->fetchResults(true);
        if (opt.profileContactFrame) {
            if (opt.traceContactFrame) {
                const int syncStatus = profile.synchronize();
                const int stopStatus = profile.stop();
                if (syncStatus != 0 || stopStatus != 0) {
                    fprintf(stderr, "FATAL: in-window hipDeviceSynchronize failed code=%d\n", syncStatus);
                    return 5;
                }
                printf("PROFILE_WINDOW_END frame=1 captured_frames=1 contact_pairs=%u contact_points=%u\n", contacts.pairs, contacts.points);
            } else {
                printf("CONTACT_FRAME_CHECK frame=1 contact_pairs=%u contact_points=%u\n", contacts.pairs, contacts.points);
            }
        }

        const int frame = f + 1;
        if (summaryFile || (opt.dumpEvery > 0 && frame % opt.dumpEvery == 0)) {
            FrameSummary s = computeFrameSummary(frame, bodies);
            writeFrameSummary(summaryFile, s);
            if (opt.dumpEvery > 0 && frame % opt.dumpEvery == 0)
                printFrameSummary(s);
        }
    }
    clock_gettime(CLOCK_MONOTONIC, &t1);

    if (summaryFile)
        std::fclose(summaryFile);

    double elapsed = (t1.tv_sec - t0.tv_sec) + (t1.tv_nsec - t0.tv_nsec) * 1e-9;
    double avgMs = elapsed * 1000.0 / double(opt.steps);

    // ---- Stats ----
    FrameSummary finalSummary = computeFrameSummary(opt.steps, bodies);
    if (opt.finalSummary) {
        printf("final_summary_marker=PX_DCU_RIGID_FINAL_SUMMARY_P46_V1 timing=outside_simulation_loop\n");
        printFrameSummary(finalSummary);
    }

    printf("\n=== Results ===\n");
    printf("PERF marker=PX_DCU_BENCH_SIMULATION_TIMING_V1 simulation_ms=%.6f ms_per_step=%.6f steps_per_second=%.6f\n", elapsed * 1000.0, avgMs, 1000.0 / avgMs);
    printf("Avg frame time: %.2f ms (%.0f FPS)\n", avgMs, 1000.0f/avgMs);
    printf("Above ground:   %d / %d\n", finalSummary.above05, nTotal);
    printf("Height range:   [%.1f, %.1f]\n", finalSummary.minY, finalSummary.maxY);
    printf("NaN/Inf bodies: %d\n", finalSummary.nanCount);
    printf("State hash:     %.17g\n", finalSummary.stateHash);
    const bool physicalPass = finalSummary.nanCount == 0 && finalSummary.belowGround == 0;
    const bool contactPass = !opt.profileContactFrame || (contacts.pairs > 0 && contacts.points > 0);
    printf("VERDICT: %s\n", physicalPass && contactPass ? "PASS" : "FAIL");

    scene->release();
    dsp->release();
    if (gpuMgr) gpuMgr->release();
    phy->release();
    fnd->release();
    return opt.profileContactFrame && !(physicalPass && contactPass) ? 5 : 0;
}
