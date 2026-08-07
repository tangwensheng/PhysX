// PhysX DCU smoke test: reduced-coordinate articulation in a GPU scene.

#include "PxPhysicsAPI.h"
#include "cudamanager/PxCudaContextManager.h"
#include "gpu/PxGpu.h"
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
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

static bool finiteQuat(const PxQuat& q)
{
    return std::isfinite(q.x) && std::isfinite(q.y) &&
        std::isfinite(q.z) && std::isfinite(q.w);
}

static bool validLinkState(const PxArticulationLink& link)
{
    const PxTransform pose = link.getGlobalPose();
    const PxVec3 linearVelocity = link.getLinearVelocity();
    const PxVec3 angularVelocity = link.getAngularVelocity();
    return finiteVec(pose.p) && finiteQuat(pose.q) &&
        finiteVec(linearVelocity) && finiteVec(angularVelocity) &&
        pose.p.magnitudeSquared() < 10000.0f &&
        linearVelocity.magnitudeSquared() < 10000.0f &&
        angularVelocity.magnitudeSquared() < 10000.0f;
}

int main(int argc, char** argv)
{
    int steps = 180;
    bool useTgs = false;
    bool printAllSteps = false;
    bool cleanExit = false;
    int cleanExitWaitSeconds = 5;
    float driveTarget = 0.0f;
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "tgs") == 0 || std::strcmp(argv[i], "TGS") == 0) {
            useTgs = true;
        } else if (std::strcmp(argv[i], "pgs") == 0 || std::strcmp(argv[i], "PGS") == 0) {
            useTgs = false;
        } else if (std::strcmp(argv[i], "--print-all-steps") == 0) {
            printAllSteps = true;
        } else if (std::strcmp(argv[i], "cleanexit") == 0 || std::strcmp(argv[i], "--cleanexit") == 0) {
            cleanExit = true;
        } else if (std::strcmp(argv[i], "--cleanexit-wait") == 0 && i + 1 < argc) {
            cleanExit = true;
            cleanExitWaitSeconds = std::atoi(argv[++i]);
            if (cleanExitWaitSeconds < 0)
                cleanExitWaitSeconds = 0;
        } else if (std::strcmp(argv[i], "--drive-target") == 0 && i + 1 < argc) {
            driveTarget = std::strtof(argv[++i], nullptr);
        } else {
            const int parsedSteps = std::atoi(argv[i]);
            if (parsedSteps > 0)
                steps = parsedSteps;
        }
    }

    printf("========================================\n");
    printf(" PhysX DCU Smoke - Articulation (%s)\n", useTgs ? "TGS" : "PGS");
    printf("========================================\n\n");

    static SmokeErrorCallback gErr;
    static PxDefaultAllocator gAlloc;
    PxFoundation* fnd = PxCreateFoundation(PX_PHYSICS_VERSION, gAlloc, gErr);
    if (!fnd) {
        printf("FAIL: PxCreateFoundation returned null\n");
        return 2;
    }

    PxPhysics* phy = PxCreatePhysics(PX_PHYSICS_VERSION, *fnd, PxTolerancesScale());
    if (!phy) {
        printf("FAIL: PxCreatePhysics returned null\n");
        fnd->release();
        return 2;
    }

    PxCudaContextManagerDesc gpuDesc;
    gpuDesc.deviceOrdinal = 0;
    PxCudaContextManager* gpuMgr = PxCreateCudaContextManager(*fnd, gpuDesc, nullptr, false);
    const bool gpuOk = gpuMgr && gpuMgr->contextIsValid();
    printf("GPU: %s - %s\n", gpuOk ? gpuMgr->getDeviceName() : "NONE", gpuOk ? "enabled" : "disabled");
    if (!gpuOk) {
        if (gpuMgr)
            gpuMgr->release();
        phy->release();
        fnd->release();
        return 2;
    }

    PxSceneDesc sd(phy->getTolerancesScale());
    sd.gravity = PxVec3(0.0f, -9.81f, 0.0f);
    sd.cudaContextManager = gpuMgr;
    PxDefaultCpuDispatcher* dsp = PxDefaultCpuDispatcherCreate(0);
    sd.cpuDispatcher = dsp;
    sd.filterShader = PxDefaultSimulationFilterShader;
    sd.flags |= PxSceneFlag::eENABLE_GPU_DYNAMICS;
    sd.solverType = useTgs ? PxSolverType::eTGS : PxSolverType::ePGS;
    sd.gpuMaxNumPartitions = 8;
    sd.gpuDynamicsConfig.maxRigidContactCount = 1024u * 1024u * 8u;
    sd.gpuDynamicsConfig.maxRigidPatchCount = 1024u * 1024u;
    sd.gpuDynamicsConfig.foundLostPairsCapacity = 1024u * 1024u * 2u;
    sd.gpuDynamicsConfig.heapCapacity = 512u * 1024u * 1024u;
    sd.gpuDynamicsConfig.collisionStackSize = 256u * 1024u * 1024u;
    sd.gpuDynamicsConfig.tempBufferCapacity = 128u * 1024u * 1024u;

    printf("Creating scene with solver=%s...\n", useTgs ? "TGS" : "PGS");
    fflush(stdout);
    PxScene* scene = phy->createScene(sd);
    if (!scene) {
        printf("FAIL: createScene returned null\n");
        dsp->release();
        gpuMgr->release();
        phy->release();
        fnd->release();
        return 3;
    }
    printf("Scene solver type: %s\n", scene->getSolverType() == PxSolverType::eTGS ? "TGS" : "PGS");

    PxMaterial* mat = phy->createMaterial(0.5f, 0.5f, 0.2f);
    PxRigidStatic* ground = PxCreatePlane(*phy, PxPlane(0, 1, 0, 0), *mat);
    scene->addActor(*ground);

    PxArticulationReducedCoordinate* art = phy->createArticulationReducedCoordinate();
    if (!art) {
        printf("FAIL: createArticulationReducedCoordinate returned null\n");
        ground->release();
        mat->release();
        scene->release();
        dsp->release();
        gpuMgr->release();
        phy->release();
        fnd->release();
        return 4;
    }
    art->setArticulationFlag(PxArticulationFlag::eFIX_BASE, true);

    PxShape* baseShape = phy->createShape(PxBoxGeometry(0.25f, 0.25f, 0.25f), *mat);
    PxShape* linkShape = phy->createShape(PxCapsuleGeometry(0.12f, 0.45f), *mat);
    std::vector<PxArticulationLink*> links;
    links.reserve(5);
    std::vector<PxArticulationJointReducedCoordinate*> joints;
    joints.reserve(4);

    PxArticulationLink* base = art->createLink(nullptr, PxTransform(PxVec3(0.0f, 4.0f, 0.0f)));
    if (!base) {
        printf("FAIL: create base link returned null\n");
        baseShape->release();
        linkShape->release();
        art->release();
        ground->release();
        mat->release();
        scene->release();
        dsp->release();
        gpuMgr->release();
        phy->release();
        fnd->release();
        return 4;
    }
    base->attachShape(*baseShape);
    PxRigidBodyExt::updateMassAndInertia(*base, 1.0f);
    links.push_back(base);

    PxArticulationLink* prev = base;
    for (int i = 0; i < 4; ++i) {
        PxArticulationLink* link = art->createLink(prev, PxTransform(PxVec3(0.0f, 3.5f - 0.55f * i, 0.0f)));
        if (!link) {
            printf("FAIL: create child link %d returned null\n", i + 1);
            baseShape->release();
            linkShape->release();
            art->release();
            ground->release();
            mat->release();
            scene->release();
            dsp->release();
            gpuMgr->release();
            phy->release();
            fnd->release();
            return 4;
        }
        link->attachShape(*linkShape);
        PxRigidBodyExt::updateMassAndInertia(*link, 1.0f);

        PxArticulationJointReducedCoordinate* joint = link->getInboundJoint();
        joint->setJointType(PxArticulationJointType::eREVOLUTE);
        joint->setParentPose(PxTransform(PxVec3(0.0f, -0.3f, 0.0f)));
        joint->setChildPose(PxTransform(PxVec3(0.0f, 0.3f, 0.0f)));
        joint->setMotion(PxArticulationAxis::eTWIST, PxArticulationMotion::eLIMITED);
        joint->setLimitParams(PxArticulationAxis::eTWIST, PxArticulationLimit(-PxPi / 4.0f, PxPi / 4.0f));
        joint->setDriveParams(PxArticulationAxis::eTWIST,
            PxArticulationDrive(10.0f, 1.0f, 100.0f, PxArticulationDriveType::eFORCE));
        joint->setDriveTarget(PxArticulationAxis::eTWIST, (i & 1) ? -driveTarget : driveTarget);

        links.push_back(link);
        joints.push_back(joint);
        prev = link;
    }

    baseShape->release();
    linkShape->release();
    mat->release();

    printf("Adding articulation with %zu links to scene...\n", links.size());
    fflush(stdout);
    if (!scene->addArticulation(*art)) {
        printf("FAIL: addArticulation returned false\n");
        art->release();
        ground->release();
        scene->release();
        dsp->release();
        gpuMgr->release();
        phy->release();
        fnd->release();
        return 5;
    }

    int completedSteps = 0;
    int firstBadStep = 0;
    int firstBadLink = -1;
    float observedMinY = 1e30f;
    float observedMaxY = -1e30f;
    float observedMaxLinearSpeed = 0.0f;
    float observedMaxAngularSpeed = 0.0f;
    float observedMaxJointPosition = 0.0f;
    float observedMaxJointVelocity = 0.0f;

    printf("Articulation added. Starting simulation...\n");
    fflush(stdout);
    for (int i = 0; i < steps; ++i) {
        if (printAllSteps || i < 5 || i == steps - 1) {
            printf("Simulate step %d/%d...\n", i + 1, steps);
            fflush(stdout);
        }
        scene->simulate(1.0f / 60.0f);
        if (printAllSteps || i < 5 || i == steps - 1) {
            printf("Fetch step %d/%d...\n", i + 1, steps);
            fflush(stdout);
        }
        scene->fetchResults(true);
        completedSteps = i + 1;

        float stepMinY = 1e30f;
        float stepMaxY = -1e30f;
        float stepMaxLinearSpeed = 0.0f;
        float stepMaxAngularSpeed = 0.0f;
        float stepMaxJointPosition = 0.0f;
        float stepMaxJointVelocity = 0.0f;
        for (size_t linkIndex = 0; linkIndex < links.size(); ++linkIndex) {
            PxArticulationLink* link = links[linkIndex];
            const PxTransform pose = link->getGlobalPose();
            const float linearSpeed = link->getLinearVelocity().magnitude();
            const float angularSpeed = link->getAngularVelocity().magnitude();
            stepMinY = PxMin(stepMinY, pose.p.y);
            stepMaxY = PxMax(stepMaxY, pose.p.y);
            stepMaxLinearSpeed = PxMax(stepMaxLinearSpeed, linearSpeed);
            stepMaxAngularSpeed = PxMax(stepMaxAngularSpeed, angularSpeed);
            if (firstBadStep == 0 && !validLinkState(*link)) {
                firstBadStep = completedSteps;
                firstBadLink = int(linkIndex);
            }
        }
        for (size_t jointIndex = 0; jointIndex < joints.size(); ++jointIndex) {
            const float jointPosition = joints[jointIndex]->getJointPosition(PxArticulationAxis::eTWIST);
            const float jointVelocity = joints[jointIndex]->getJointVelocity(PxArticulationAxis::eTWIST);
            stepMaxJointPosition = PxMax(stepMaxJointPosition, PxAbs(jointPosition));
            stepMaxJointVelocity = PxMax(stepMaxJointVelocity, PxAbs(jointVelocity));
            if (firstBadStep == 0 && (!std::isfinite(jointPosition) || !std::isfinite(jointVelocity))) {
                firstBadStep = completedSteps;
                firstBadLink = int(jointIndex + 1);
            }
        }

        observedMinY = PxMin(observedMinY, stepMinY);
        observedMaxY = PxMax(observedMaxY, stepMaxY);
        observedMaxLinearSpeed = PxMax(observedMaxLinearSpeed, stepMaxLinearSpeed);
        observedMaxAngularSpeed = PxMax(observedMaxAngularSpeed, stepMaxAngularSpeed);
        observedMaxJointPosition = PxMax(observedMaxJointPosition, stepMaxJointPosition);
        observedMaxJointVelocity = PxMax(observedMaxJointVelocity, stepMaxJointVelocity);

        if (printAllSteps) {
            printf("Step state %d/%d: height=[%.6f, %.6f] maxLinear=%.6f maxAngular=%.6f maxJointPos=%.6f maxJointVel=%.6f\n",
                completedSteps, steps, stepMinY, stepMaxY, stepMaxLinearSpeed, stepMaxAngularSpeed,
                stepMaxJointPosition, stepMaxJointVelocity);
            fflush(stdout);
        }

        if (firstBadStep != 0 || gErr.getFatalErrorCount() != 0) {
            printf("Stopping after step %d: badLink=%d fatalErrors=%u\n",
                completedSteps, firstBadLink, gErr.getFatalErrorCount());
            fflush(stdout);
            break;
        }
    }

    int badLinks = 0;
    float finalMinY = 1e30f;
    float finalMaxY = -1e30f;
    float finalMaxLinearSpeed = 0.0f;
    float finalMaxAngularSpeed = 0.0f;
    float finalMaxJointPosition = 0.0f;
    float finalMaxJointVelocity = 0.0f;
    for (PxArticulationLink* link : links) {
        const PxTransform pose = link->getGlobalPose();
        const float linearSpeed = link->getLinearVelocity().magnitude();
        const float angularSpeed = link->getAngularVelocity().magnitude();
        if (!validLinkState(*link))
            ++badLinks;
        finalMinY = PxMin(finalMinY, pose.p.y);
        finalMaxY = PxMax(finalMaxY, pose.p.y);
        finalMaxLinearSpeed = PxMax(finalMaxLinearSpeed, linearSpeed);
        finalMaxAngularSpeed = PxMax(finalMaxAngularSpeed, angularSpeed);
    }
    for (PxArticulationJointReducedCoordinate* joint : joints) {
        const float jointPosition = joint->getJointPosition(PxArticulationAxis::eTWIST);
        const float jointVelocity = joint->getJointVelocity(PxArticulationAxis::eTWIST);
        if (!std::isfinite(jointPosition) || !std::isfinite(jointVelocity))
            ++badLinks;
        finalMaxJointPosition = PxMax(finalMaxJointPosition, PxAbs(jointPosition));
        finalMaxJointVelocity = PxMax(finalMaxJointVelocity, PxAbs(jointVelocity));
    }

    const bool driveRequested = PxAbs(driveTarget) > 1e-6f;
    const bool driveResponded = !driveRequested ||
        (observedMaxJointPosition > 0.01f && observedMaxJointVelocity > 0.01f);
    const bool limitsRespected = observedMaxJointPosition <= PxPi / 4.0f + 0.05f;
    const bool settled = !driveRequested ||
        (PxAbs(finalMaxJointPosition - PxAbs(driveTarget)) < 0.15f &&
         finalMaxJointVelocity < 0.2f && finalMaxLinearSpeed < 0.2f &&
         finalMaxAngularSpeed < 0.2f);
    const bool pass = completedSteps == steps && firstBadStep == 0 && badLinks == 0 &&
        gErr.getFatalErrorCount() == 0 && driveResponded && limitsRespected && settled &&
        finalMinY > -1.0f && finalMaxY < 10.0f;

    printf("Solver: %s\n", useTgs ? "TGS" : "PGS");
    printf("Drive target magnitude: %.6f\n", PxAbs(driveTarget));
    printf("Steps: %d/%d\n", completedSteps, steps);
    printf("Articulation links: %zu\n", links.size());
    printf("Observed height range: [%.6f, %.6f]\n", observedMinY, observedMaxY);
    printf("Final height range: [%.6f, %.6f]\n", finalMinY, finalMaxY);
    printf("Observed max linear speed: %.6f\n", observedMaxLinearSpeed);
    printf("Observed max angular speed: %.6f\n", observedMaxAngularSpeed);
    printf("Observed max joint position: %.6f\n", observedMaxJointPosition);
    printf("Observed max joint velocity: %.6f\n", observedMaxJointVelocity);
    printf("Final max linear speed: %.6f\n", finalMaxLinearSpeed);
    printf("Final max angular speed: %.6f\n", finalMaxAngularSpeed);
    printf("Final max joint position: %.6f\n", finalMaxJointPosition);
    printf("Final max joint velocity: %.6f\n", finalMaxJointVelocity);
    printf("Drive response: %s\n", driveResponded ? "PASS" : "FAIL");
    printf("Joint limits: %s\n", limitsRespected ? "PASS" : "FAIL");
    printf("Settled state: %s\n", settled ? "PASS" : "FAIL");
    printf("Bad links: %d\n", badLinks);
    printf("First bad step/link: %d/%d\n", firstBadStep, firstBadLink);
    printf("PhysX errors: %u (fatal: %u)\n", gErr.getErrorCount(), gErr.getFatalErrorCount());
    printf("VERDICT: %s\n", pass ? "PASS" : "FAIL");
    fflush(stdout);

    printf("Releasing articulation...\n");
    fflush(stdout);
    art->release();
    printf("Releasing ground...\n");
    fflush(stdout);
    ground->release();
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
        printf("Clean exit after explicit PhysX release (--cleanexit), waiting %d seconds for DCU/HSA cleanup...\n",
            cleanExitWaitSeconds);
        fflush(stdout);
        if (cleanExitWaitSeconds > 0)
            std::this_thread::sleep_for(std::chrono::seconds(cleanExitWaitSeconds));
        printf("Clean exit wait finished.\n");
        fflush(stdout);
        std::_Exit(pass ? 0 : 1);
    }
    return pass ? 0 : 1;
}
