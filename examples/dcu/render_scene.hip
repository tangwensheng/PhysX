// PhysX Tower Collapse — Cinematic Renderer
// Glowing projectiles, motion trails, delayed collapse, camera choreography

#include "PxPhysicsAPI.h"
#include "foundation/PxVec3.h"
#include "foundation/PxVec4.h"
#include "foundation/PxMat33.h"
#include <hip/hip_runtime.h>
#include <cstdio>
#include <cmath>
#include <vector>
#include <cstdlib>
#include <deque>

using namespace physx;

const int IMG_W=800,IMG_H=600,IMG_PX=IMG_W*IMG_H;
const int TRAIL_LEN=5; // motion trail length

#define SHAPE_SPHERE 0
#define SHAPE_BOX    1

struct GpuShape {
    float px,py,pz, r0,r1,r2,r3, cr,cg,cb;
    int type;
    float rot00,rot01,rot02,rot10,rot11,rot12,rot20,rot21,rot22;
    float glow; // 1=sphere (always lit), 0=box
};

extern "C" __global__ void render_frame(
    const GpuShape* shapes, int n, PxVec3* fb, int w, int h,
    float cpX,float cpY,float cpZ, float cdX,float cdY,float cdZ,
    float crX,float crY,float crZ, float cuX,float cuY,float cuZ,
    float lx,float ly,float lz)
{
    int px=blockIdx.x*blockDim.x+threadIdx.x, py=blockIdx.y*blockDim.y+threadIdx.y;
    if(px>=w||py>=h)return;

    float u=(px/(float)w-0.5f)*2.0f;
    float v=-(py/(float)h-0.5f)*2.0f*(float)h/(float)w;
    float dx=cdX+crX*u+cuX*v, dy=cdY+crY*u+cuY*v, dz=cdZ+crZ*u+cuZ*v;
    float rl=rsqrtf(dx*dx+dy*dy+dz*dz); dx*=rl;dy*=rl;dz*=rl;

    float tMin=1e20f,cr=0.05f,cg=0.08f,cb=0.15f,hitGlow=0;
    float hitNx=0,hitNy=0,hitNz=0,hitPx=0,hitPy=0,hitPz=0;

    for(int s=0;s<n;s++){
        GpuShape sh=shapes[s];
        float ox=cpX-sh.px,oy=cpY-sh.py,oz=cpZ-sh.pz,t;

        if(sh.type==SHAPE_SPHERE){
            float b=ox*dx+oy*dy+oz*dz, c=ox*ox+oy*oy+oz*oz-sh.r0*sh.r0;
            float disc=b*b-c;
            if(disc>0){t=b-sqrtf(disc);if(t>0.001f&&t<tMin){
                tMin=t;cr=sh.cr;cg=sh.cg;cb=sh.cb;hitGlow=1;
                float invR=1.0f/sh.r0;
                hitNx=(cpX+t*dx-sh.px)*invR;hitNy=(cpY+t*dy-sh.py)*invR;hitNz=(cpZ+t*dz-sh.pz)*invR;
                hitPx=cpX+t*dx;hitPy=cpY+t*dy;hitPz=cpZ+t*dz;
            }}
        }else{
            float lx_=ox*sh.rot00+oy*sh.rot10+oz*sh.rot20;
            float ly_=ox*sh.rot01+oy*sh.rot11+oz*sh.rot21;
            float lz_=ox*sh.rot02+oy*sh.rot12+oz*sh.rot22;
            float ldx=dx*sh.rot00+dy*sh.rot10+dz*sh.rot20;
            float ldy=dx*sh.rot01+dy*sh.rot11+dz*sh.rot21;
            float ldz=dx*sh.rot02+dy*sh.rot12+dz*sh.rot22;
            float tx1=(-sh.r0-lx_)/ldx,tx2=(sh.r0-lx_)/ldx;
            float ty1=(-sh.r1-ly_)/ldy,ty2=(sh.r1-ly_)/ldy;
            float tz1=(-sh.r2-lz_)/ldz,tz2=(sh.r2-lz_)/ldz;
            float tn=fmaxf(fmaxf(fminf(tx1,tx2),fminf(ty1,ty2)),fminf(tz1,tz2));
            float tf=fminf(fminf(fmaxf(tx1,tx2),fmaxf(ty1,ty2)),fmaxf(tz1,tz2));
            if(tn<tf&&tf>0.001f&&tn<tMin){
                tMin=tn;cr=sh.cr;cg=sh.cg;cb=sh.cb;hitGlow=0;
                float lnx=(tn==fminf(tx1,tx2))?(ldx>0?-1:1):0;
                float lny=(tn==fminf(ty1,ty2))?(ldy>0?-1:1):0;
                float lnz=(tn==fminf(tz1,tz2))?(ldz>0?-1:1):0;
                hitNx=lnx*sh.rot00+lny*sh.rot01+lnz*sh.rot02;
                hitNy=lnx*sh.rot10+lny*sh.rot11+lnz*sh.rot12;
                hitNz=lnx*sh.rot20+lny*sh.rot21+lnz*sh.rot22;
                hitPx=cpX+tn*dx;hitPy=cpY+tn*dy;hitPz=cpZ+tn*dz;
            }
        }
    }

    // Shadow
    if(tMin<1e19f){
        float sx=hitPx+hitNx*0.01f,sy=hitPy+hitNy*0.01f,sz=hitPz+hitNz*0.01f;
        float slen=rsqrtf(lx*lx+ly*ly+lz*lz);
        float slx=lx*slen,sly=ly*slen,slz=lz*slen, sr=0.3f;
        for(int s=0;s<n&&sr<1.0f;s++){
            GpuShape sh=shapes[s];
            float ox=sx-sh.px,oy=sy-sh.py,oz=sz-sh.pz;
            if(sh.type==SHAPE_SPHERE){
                float b=ox*slx+oy*sly+oz*slz,c=ox*ox+oy*oy+oz*oz-sh.r0*sh.r0;
                float disc=b*b-c;
                if(disc>0){float t=b-sqrtf(disc);if(t>0.001f)sr+=0.5f;}
            }
        }
        cr*=fminf(sr,1.0f);cg*=fminf(sr,1.0f);cb*=fminf(sr,1.0f);
    }

    float lam=fmaxf(0.15f,hitNy*0.6f+0.4f);
    float bright=lam*(1.0f-hitGlow)+hitGlow*0.8f;
    fb[py*w+px]=PxVec3(cr*bright,cg*bright,cb*bright);
}

void write_ppm(const char* fn, PxVec3* px, int w, int h){
    FILE* f=fopen(fn,"wb");
    fprintf(f,"P6\n%d %d\n255\n",w,h);
    for(int i=0;i<w*h;i++){
        unsigned char r=(unsigned char)(fminf(fmaxf(px[i].x,0),1)*255);
        unsigned char g=(unsigned char)(fminf(fmaxf(px[i].y,0),1)*255);
        unsigned char b=(unsigned char)(fminf(fmaxf(px[i].z,0),1)*255);
        fwrite(&r,1,1,f);fwrite(&g,1,1,f);fwrite(&b,1,1,f);
    }
    fclose(f);
}

int main()
{
    printf("========================================\n");
    printf(" PhysX Tower Collapse — Cinematic\n");
    printf("========================================\n\n");

    static PxDefaultErrorCallback err;static PxDefaultAllocator alloc;
    PxFoundation* fnd=PxCreateFoundation(PX_PHYSICS_VERSION,alloc,err);
    PxPhysics* phy=PxCreatePhysics(PX_PHYSICS_VERSION,*fnd,PxTolerancesScale());
    PxSceneDesc sd(phy->getTolerancesScale());
    sd.gravity=PxVec3(0,-9.81f,0);
    PxDefaultCpuDispatcher* dsp=PxDefaultCpuDispatcherCreate(4);
    sd.cpuDispatcher=dsp;sd.filterShader=PxDefaultSimulationFilterShader;
    PxScene* scene=phy->createScene(sd);

    PxMaterial* mat=phy->createMaterial(0.5f,0.5f,0.0f); // restitution=0 for stable stack
    PxMaterial* matH=phy->createMaterial(0.8f,0.8f,0.1f);
    scene->addActor(*PxCreatePlane(*phy,PxPlane(PxVec3(0,1,0),0),*mat));

    PxVec3 cols[]={{0.9f,0.2f,0.2f},{0.2f,0.8f,0.2f},{0.2f,0.2f,0.9f},
                   {0.9f,0.9f,0.2f},{0.9f,0.5f,0.2f},{0.5f,0.2f,0.9f}};

    // --- Tower (kinematic initially — frozen until impact) ---
    const int LAYERS=14,BASE=30;const float BH=0.5f;
    std::vector<PxRigidDynamic*> bodies;std::vector<PxVec3> bodyCols;
    int nBox=0;

    for(int layer=0;layer<LAYERS;layer++){
        int side=BASE-layer*2;if(side<=0)break;
        float y=(layer*2+1)*BH+0.01f;
        for(int ix=0;ix<side;ix++)for(int iz=0;iz<side;iz++){
            float x=(ix-(side-1)/2.0f)*2.0f*BH,z=(iz-(side-1)/2.0f)*2.0f*BH;
            PxRigidDynamic* b=phy->createRigidDynamic(PxTransform(PxVec3(x,y,z)));
            PxShape* sh=phy->createShape(PxBoxGeometry(BH,BH,BH),*mat);
            b->attachShape(*sh);PxRigidBodyExt::updateMassAndInertia(*b,1.0f);
            scene->addActor(*b);bodies.push_back(b);bodyCols.push_back(cols[layer%6]);nBox++;
        }
    }
    printf("Tower: %d boxes\n",nBox);

    // --- Settle: simulate 120 frames to let tower stabilize under gravity ---
    printf("Settling...\n");
    float settleStep=1.0f/60.0f;
    for(int s=0;s<120;s++){scene->simulate(settleStep);scene->fetchResults(true);}
    printf("Tower stable. Spawning projectiles...\n\n");

    // --- Projectiles: spawn far away, aimed at tower ---
    const int NSPH=1;
    float sphRadius=1.5f;
    PxShape* sphSh=phy->createShape(PxSphereGeometry(sphRadius),*matH);
    struct SphereInfo{PxRigidDynamic* body;PxVec3 trail[TRAIL_LEN];};
    std::vector<SphereInfo> sins;
    float sx=40, sy=15, sz=8;
    float vx=-35, vy=3, vz=-5;
    PxRigidDynamic* sp=phy->createRigidDynamic(PxTransform(PxVec3(sx,sy,sz)));
    sp->attachShape(*sphSh);
    sp->setLinearVelocity(PxVec3(vx,vy,vz));
    PxRigidBodyExt::updateMassAndInertia(*sp,500.0f);
    scene->addActor(*sp);bodies.push_back(sp);bodyCols.push_back(PxVec3(1.0f,0.15f,0.05f)); // red
    sins.push_back({sp,{}});
    printf("Cannonball: r=1.5m 500kg v=(%.0f,%.0f,%.0f)\n",vx,vy,vz);
    int nTotal=(int)bodies.size();
    printf("Projectile: 1 | Total: %d\n\n",nTotal);

    // --- GPU Setup ---
    GpuShape *d_s,*h_s=new GpuShape[nTotal+TRAIL_LEN*NSPH];
    PxVec3 *d_fb,*h_fb=new PxVec3[IMG_PX];
    hipMalloc(&d_s,(nTotal+TRAIL_LEN*NSPH)*sizeof(GpuShape));
    hipMalloc(&d_fb,IMG_PX*sizeof(PxVec3));
    dim3 block(16,16),grid((IMG_W+15)/16,(IMG_H+15)/16);

    float simT=0,maxT=10.0f,step=1.0f/60.0f;
    int frame=0,impactFrame=-1;
    bool towerAwake=false;
    char fn[64];
    printf("Simulating + Rendering...\n");

    while(simT<maxT){
        // Check for first impact: any projectile within 2m of tower
        if(!towerAwake){
            for(int i=nBox;i<nTotal;i++){
                PxVec3 p=bodies[i]->getGlobalPose().p;
                if(p.y<10&&fabsf(p.x)<15&&fabsf(p.z)<15){ // entered tower zone
                    towerAwake=true;impactFrame=frame;
                    printf("  IMPACT at frame %d!\n",frame);
                    break;
                }
            }
        }

        scene->simulate(step);scene->fetchResults(true);simT+=step;

        // Update trails
        for(int i=0;i<NSPH;i++){
            PxVec3 p=bodies[nBox+i]->getGlobalPose().p;
            for(int t=TRAIL_LEN-1;t>0;t--) sins[i].trail[t]=sins[i].trail[t-1];
            sins[i].trail[0]=p;
        }

        // Build GPU shapes
        int gIdx=0;
        for(int i=0;i<nTotal;i++){
            PxTransform t=bodies[i]->getGlobalPose();PxVec3 p=t.p;PxQuat q=t.q;
            h_s[gIdx].px=p.x;h_s[gIdx].py=p.y;h_s[gIdx].pz=p.z;
            h_s[gIdx].cr=bodyCols[i].x;h_s[gIdx].cg=bodyCols[i].y;h_s[gIdx].cb=bodyCols[i].z;
            if(i<nBox){
                h_s[gIdx].type=SHAPE_BOX;h_s[gIdx].r0=BH;h_s[gIdx].r1=BH;h_s[gIdx].r2=BH;
                h_s[gIdx].glow=0;float x=q.x,y=q.y,z=q.z,w=q.w;
                h_s[gIdx].rot00=1-2*(y*y+z*z);h_s[gIdx].rot01=2*(x*y-w*z);h_s[gIdx].rot02=2*(x*z+w*y);
                h_s[gIdx].rot10=2*(x*y+w*z);h_s[gIdx].rot11=1-2*(x*x+z*z);h_s[gIdx].rot12=2*(y*z-w*x);
                h_s[gIdx].rot20=2*(x*z-w*y);h_s[gIdx].rot21=2*(y*z+w*x);h_s[gIdx].rot22=1-2*(x*x+y*y);
            }else{
                h_s[gIdx].type=SHAPE_SPHERE;h_s[gIdx].r0=sphRadius;h_s[gIdx].glow=1;
            }
            gIdx++;

            // Motion trail ghosts (for projectiles only)
            if(i>=nBox && i-nBox<(int)sins.size()){
                int si=i-nBox;
                for(int t=0;t<TRAIL_LEN;t++){
                    PxVec3 tp=sins[si].trail[t];
                    if(tp.x==0&&tp.y==0&&tp.z==0)continue; // not filled yet
                    float fade=0.15f/(t+1);
                    h_s[gIdx].px=tp.x;h_s[gIdx].py=tp.y;h_s[gIdx].pz=tp.z;
                    h_s[gIdx].type=SHAPE_SPHERE;h_s[gIdx].r0=0.8f;h_s[gIdx].glow=1;
                    h_s[gIdx].cr=bodyCols[i].x;h_s[gIdx].cg=bodyCols[i].y;h_s[gIdx].cb=bodyCols[i].z;
                    gIdx++;
                }
            }
        }

        hipMemcpy(d_s,h_s,gIdx*sizeof(GpuShape),hipMemcpyHostToDevice);

        // Camera: fixed side view, then slow orbit after impact
        float cpX,cpY,cpZ;
        if(impactFrame>=0&&frame>impactFrame+30){
            float orbitAngle=(frame-impactFrame-30)*0.004f;
            float radius=40;
            cpX=30*cosf(orbitAngle);cpZ=30*sinf(orbitAngle);cpY=12+simT*0.3f;
        }else{
            // Fixed view from collision side: camera at (25,15,35) looking at tower
            cpX=25;cpZ=35;cpY=15;
        }
        PxVec3 camP(cpX,cpY,cpZ),camT(0,5,0);
        PxVec3 camD=camT-camP;camD=camD*(1.0f/sqrtf(camD.magnitudeSquared()));
        PxVec3 camR(-camD.z,0,camD.x);camR=camR*(1.0f/sqrtf(camR.magnitudeSquared()));
        PxVec3 camU=camR.cross(camD);
        PxVec3 light(10,30,20);

        render_frame<<<grid,block>>>(d_s,gIdx,d_fb,IMG_W,IMG_H,
            camP.x,camP.y,camP.z,camD.x,camD.y,camD.z,camR.x,camR.y,camR.z,camU.x,camU.y,camU.z,light.x,light.y,light.z);
        hipDeviceSynchronize();
        hipMemcpy(h_fb,d_fb,IMG_PX*sizeof(PxVec3),hipMemcpyDeviceToHost);

        snprintf(fn,sizeof(fn),"frame_%03d.ppm",frame++);
        write_ppm(fn,h_fb,IMG_W,IMG_H);
        if(frame%60==0)printf("  %d frames...\n",frame);
    }
    printf("\nDone: %d frames.\nffmpeg -framerate 30 -i frame_%%03d.ppm tower.mp4\n",frame);

    delete[]h_s;delete[]h_fb;hipFree(d_s);hipFree(d_fb);
    scene->release();phy->release();fnd->release();dsp->release();
    return 0;
}
