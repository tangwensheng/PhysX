// PhysX CPU vs GPU comparison — 1 frame, compare all body poses
#include "PxPhysicsAPI.h"
#include "cudamanager/PxCudaContextManager.h"
#include "gpu/PxGpu.h"
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <cmath>
#include <cstring>

using namespace physx;

// Build identical scene, return bodies and scene
static void buildScene(PxPhysics* phy, PxScene* scene, std::vector<PxRigidDynamic*>& bodies)
{
    const int LAYERS = 5, BASE = 10;
    const float H = 0.5f;
    PxMaterial* mat = phy->createMaterial(0.5f, 0.5f, 0.3f);
    PxShape* boxShape = phy->createShape(PxBoxGeometry(H, H, H), *mat);
    scene->addActor(*PxCreatePlane(*phy, PxPlane(PxVec3(0,1,0), 0), *mat));

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
        }
    }

    const int NSPH = 50;
    PxMaterial* matH = phy->createMaterial(0.8f, 0.8f, 0.1f);
    PxShape* sphShape = phy->createShape(PxSphereGeometry(0.6f), *matH);
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
}

int main()
{
    static PxDefaultErrorCallback gErr;
    static PxDefaultAllocator       gAlloc;
    PxFoundation* fnd = PxCreateFoundation(PX_PHYSICS_VERSION, gAlloc, gErr);
    PxPhysics*    phy = PxCreatePhysics(PX_PHYSICS_VERSION, *fnd, PxTolerancesScale());

    // GPU setup
    PxCudaContextManagerDesc gpuDesc;
    gpuDesc.deviceOrdinal = 0;
    PxCudaContextManager* gpuMgr = PxCreateCudaContextManager(*fnd, gpuDesc, nullptr, false);
    bool gpuOk = gpuMgr && gpuMgr->contextIsValid();
    printf("GPU: %s\n", gpuOk ? gpuMgr->getDeviceName() : "NONE");

    // ---- CPU Run ----
    printf("\n=== CPU Run ===\n");
    {
        PxSceneDesc sd(phy->getTolerancesScale());
        sd.gravity = PxVec3(0, -9.81f, 0);
        PxDefaultCpuDispatcher* dsp1 = PxDefaultCpuDispatcherCreate(0);
        sd.cpuDispatcher = dsp1;
        sd.filterShader = PxDefaultSimulationFilterShader;
        PxScene* scene = phy->createScene(sd);

        std::vector<PxRigidDynamic*> bodies;
        buildScene(phy, scene, bodies);
        printf("Bodies: %d\n", (int)bodies.size());

        for (int f = 0; f < 10; f++) {
            scene->simulate(1.0f/60.0f);
            scene->fetchResults(true);
        }

        // Save CPU poses
        std::vector<PxTransform> cpuPoses(bodies.size());
        for (size_t i = 0; i < bodies.size(); i++)
            cpuPoses[i] = bodies[i]->getGlobalPose();

        scene->release();
        dsp1->release();

        // ---- GPU Run ----
        printf("\n=== GPU Run ===\n");
        {
            PxSceneDesc sd2(phy->getTolerancesScale());
            sd2.gravity = PxVec3(0, -9.81f, 0);
            sd2.cudaContextManager = gpuMgr;
            if (gpuOk) {
                sd2.flags |= PxSceneFlag::eENABLE_GPU_DYNAMICS;
                sd2.gpuMaxNumPartitions = 8;
                sd2.gpuDynamicsConfig.maxRigidContactCount   = 1024u * 1024u * 4u;
                sd2.gpuDynamicsConfig.maxRigidPatchCount     = 1024u * 512u;
                sd2.gpuDynamicsConfig.heapCapacity           = 256u * 1024u * 1024u;
                sd2.gpuDynamicsConfig.foundLostPairsCapacity = 512u * 1024u;
                sd2.gpuDynamicsConfig.collisionStackSize     = 128u * 1024u * 1024u;
                sd2.gpuDynamicsConfig.tempBufferCapacity     = 64u * 1024u * 1024u;
            }
            PxDefaultCpuDispatcher* dsp2 = PxDefaultCpuDispatcherCreate(0);
            sd2.cpuDispatcher = dsp2;
            sd2.filterShader = PxDefaultSimulationFilterShader;
            PxScene* scene2 = phy->createScene(sd2);

            std::vector<PxRigidDynamic*> bodies2;
            buildScene(phy, scene2, bodies2);
            printf("Bodies: %d\n", (int)bodies2.size());

            for (int f = 0; f < 10; f++) {
                scene2->simulate(1.0f/60.0f);
                scene2->fetchResults(true);
            }

            // Compare
            printf("\n=== Comparison (CPU vs GPU) ===\n");
            float maxPosDiff = 0, maxRotDiff = 0;
            int suspectCount = 0;
            for (size_t i = 0; i < bodies.size(); i++) {
                PxTransform gpuPose = bodies2[i]->getGlobalPose();
                PxVec3 dp = gpuPose.p - cpuPoses[i].p;
                float posDiff = dp.magnitude();
                float rotDiff = (gpuPose.q.getNormalized().dot(cpuPoses[i].q.getNormalized()));
                rotDiff = 1.0f - fabsf(rotDiff);

                if (posDiff > maxPosDiff) maxPosDiff = posDiff;
                if (rotDiff > maxRotDiff) maxRotDiff = rotDiff;
                if (posDiff > 0.01f || rotDiff > 0.001f) suspectCount++;
            }

            printf("Max position diff: %.6f\n", maxPosDiff);
            printf("Max rotation diff: %.6f\n", maxRotDiff);
            printf("Suspect bodies (>1cm or >0.001 rot): %d / %d\n", suspectCount, (int)bodies.size());

            if (maxPosDiff < 0.01f && maxRotDiff < 0.002f)
                printf("VERDICT: CPU and GPU match — no silent data corruption.\n");
            else
                printf("VERDICT: MISMATCH detected — data corruption present.\n");

            scene2->release();
            dsp2->release();
        }
    }

    if (gpuMgr) gpuMgr->release();
    phy->release();
    fnd->release();
    return 0;
}
