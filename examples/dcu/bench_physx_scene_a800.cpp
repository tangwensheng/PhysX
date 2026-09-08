// PhysX Scene Benchmark — A800 GPU
// Same scene as DCU version, but with eENABLE_GPU_DYNAMICS for NVIDIA.

#include "PxPhysicsAPI.h"
#include "cudamanager/PxCudaContextManager.h"
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <cmath>
#include <ctime>
#include <cstring>
#include <algorithm>

using namespace physx;

struct SummaryOptions
{
    int steps = 300;
    int dumpEvery = 0;
    const char* summaryCsv = nullptr;
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
        else if (!std::strcmp(argv[i], "--help"))
        {
            printf("Usage: bench_physx_scene [--steps N] [--dump-every N] [--summary file.csv]\n");
            std::exit(0);
        }
    }
    if (opt.steps <= 0)
        opt.steps = 300;
    if (opt.dumpEvery < 0)
        opt.dumpEvery = 0;
    return opt;
}

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

        // Lightweight state fingerprint for locating DCU/NV divergence frames.
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
    printf("========================================\n");
    printf(" PhysX A800 GPU — Tower Collapse\n");
    printf("========================================\n\n");

    static PxDefaultErrorCallback gErr;
    static PxDefaultAllocator       gAlloc;
    PxFoundation* fnd = PxCreateFoundation(PX_PHYSICS_VERSION, gAlloc, gErr);
    PxPhysics*    phy = PxCreatePhysics(PX_PHYSICS_VERSION, *fnd, PxTolerancesScale());

    PxCudaContextManagerDesc gpuDesc;
    gpuDesc.deviceOrdinal = 0;
    PxCudaContextManager* gpuMgr = PxCreateCudaContextManager(*fnd, gpuDesc, nullptr, false);
    bool gpuOk = gpuMgr && gpuMgr->contextIsValid();
    printf("GPU: %s (%d CUs, %.1f GB) — %s\n",
        gpuOk ? gpuMgr->getDeviceName() : "NONE",
        gpuOk ? gpuMgr->getMultiprocessorCount() : 0,
        gpuOk ? gpuMgr->getDeviceTotalMemBytes()/(1024.0*1024.0*1024.0) : 0.0,
        gpuOk ? "GPU enabled" : "CPU only");

    PxSceneDesc sd(phy->getTolerancesScale());
    sd.gravity = PxVec3(0, -9.81f, 0);
    sd.cudaContextManager = gpuMgr;
    if (gpuOk) {
        sd.flags |= PxSceneFlag::eENABLE_GPU_DYNAMICS;
        sd.gpuMaxNumPartitions = 8;
    }
    PxDefaultCpuDispatcher* dsp = PxDefaultCpuDispatcherCreate(0);
    sd.cpuDispatcher = dsp;
    sd.filterShader = PxDefaultSimulationFilterShader;
    PxScene* scene = phy->createScene(sd);

    const int LAYERS=20, BASE=44;
    const float H=0.5f;
    PxMaterial* mat = phy->createMaterial(0.5f, 0.5f, 0.3f);
    PxShape* boxShape = phy->createShape(PxBoxGeometry(H,H,H), *mat);
    scene->addActor(*PxCreatePlane(*phy, PxPlane(PxVec3(0,1,0), 0), *mat));
    std::vector<PxRigidDynamic*> bodies; int nBox=0;

    for(int layer=0; layer<LAYERS; layer++){
        int side=BASE-layer*2; if(side<=0)break;
        float y=(layer*2+1)*H+0.01f;
        for(int ix=0; ix<side; ix++) for(int iz=0; iz<side; iz++){
            float x=(ix-(side-1)/2.0f)*2.0f*H, z=(iz-(side-1)/2.0f)*2.0f*H;
            PxRigidDynamic* b=phy->createRigidDynamic(PxTransform(PxVec3(x,y,z)));
            b->attachShape(*boxShape); PxRigidBodyExt::updateMassAndInertia(*b,1.0f);
            scene->addActor(*b); bodies.push_back(b); nBox++;
        }
    }
    printf("Bodies: %d (%d layers)\n", nBox, LAYERS);

    const int NSPH=300;
    PxMaterial* matH=phy->createMaterial(0.8f,0.8f,0.1f);
    PxShape* sphShape=phy->createShape(PxSphereGeometry(0.6f),*matH);
    srand(42);
    for(int s=0; s<NSPH; s++){
        float a=rand()/(float)RAND_MAX*6.283f, v=20+rand()/(float)RAND_MAX*35;
        PxRigidDynamic* sp=phy->createRigidDynamic(PxTransform(PxVec3((rand()%60-30),15+rand()%10,(rand()%60-30))));
        sp->attachShape(*sphShape); sp->setLinearVelocity(PxVec3(cosf(a)*v,-5,sinf(a)*v));
        PxRigidBodyExt::updateMassAndInertia(*sp,5.0f); scene->addActor(*sp); bodies.push_back(sp);
    }
    int nTotal=(int)bodies.size();
    printf("Projectiles: %d | Total: %d\n\n", NSPH, nTotal);

    printf("Settling...\n");
    for(int s=0; s<120; s++){ scene->simulate(1.0f/60.0f); scene->fetchResults(true); }

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
        scene->simulate(step);
        scene->fetchResults(true);

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

    printf("\n=== Results ===\n");
    printf("Avg frame time: %.2f ms (%.0f FPS)\n", avgMs, 1000.0f/avgMs);
    printf("Above ground:   %d / %d\n", finalSummary.above05, nTotal);
    printf("Height range:   [%.1f, %.1f]\n", finalSummary.minY, finalSummary.maxY);
    printf("NaN/Inf bodies: %d\n", finalSummary.nanCount);
    printf("State hash:     %.17g\n", finalSummary.stateHash);

    scene->release(); phy->release(); fnd->release(); dsp->release();
    return 0;
}
