// A800 GPU Pipeline Benchmark — matches bench_gpu_pipeline.hip
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>

// Stage 1: Broadphase
extern "C" __global__ void bp_stage(const unsigned int* boxes, int n, unsigned int* overlaps, int* count)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int lane = tid & 31; // warp=32
    if (tid >= n) return;
    unsigned int myBox = boxes[tid];
    int found = 0;
    for (int tile = 0; tile < n; tile += 32) {
        int j = tile + lane;
        if (j < n) { unsigned mask = __ballot_sync(0xFFFFFFFF, myBox > boxes[j]); found += __popc(mask); }
    }
    if (found > 0) { int idx = atomicAdd(count, 1); overlaps[idx] = tid; }
}

// Stage 2: Narrowphase
extern "C" __global__ void np_stage(const float* positions, float* contacts, int n, const unsigned int* overlaps, int overlapCount, float contactDist)
{
    __shared__ float shX[256],shY[256],shZ[256];
    int tid=blockIdx.x*blockDim.x+threadIdx.x,lane=threadIdx.x;
    if(tid>=overlapCount)return;
    int idx=overlaps[tid];
    float px=positions[idx*3],py=positions[idx*3+1],pz=positions[idx*3+2],cx=0,cy=0,cz=0,cc=0;
    for(int tile=0;tile<overlapCount;tile+=256){
        int j=tile+lane,jdx=(j<overlapCount)?overlaps[j]:0;
        shX[lane]=positions[jdx*3];shY[lane]=positions[jdx*3+1];shZ[lane]=positions[jdx*3+2];
        __syncthreads();
        for(int k=0;k<256&&tile+k<overlapCount;k++){
            float dx=shX[k]-px,dy=shY[k]-py,dz=shZ[k]-pz,dsq=dx*dx+dy*dy+dz*dz+1e-8f;
            if(dsq<contactDist*contactDist){float inv=rsqrtf(dsq),overlap=contactDist-(1.0f/inv);
                if(overlap>0){cx+=dx*overlap*inv;cy+=dy*overlap*inv;cz+=dz*overlap*inv;cc+=1;}}
        }
        __syncthreads();
    }
    contacts[tid*4]=cx;contacts[tid*4+1]=cy;contacts[tid*4+2]=cz;contacts[tid*4+3]=cc;
}

// Stage 3: Solver
extern "C" __global__ void solver_stage(float* vel, float* pos, const float* contacts, int n, float dt)
{
    __shared__ float sLam[256];
    int tid=blockIdx.x*blockDim.x+threadIdx.x,lane=threadIdx.x;
    if(tid>=n)return;
    float vx=vel[tid*2],vy=vel[tid*2+1],cx=contacts[tid*4],cy=contacts[tid*4+1],cz=contacts[tid*4+2],lam=0;
    for(int iter=0;iter<4;iter++){
        sLam[lane]=lam;__syncthreads();
        float accX=cx,accY=cy;
        for(int k=0;k<256&&k<n;k++){float lk=sLam[k],nx=contacts[k*4],ny=contacts[k*4+1],nz=contacts[k*4+2];
            float dot=cx*nx+cy*ny+cz*nz;accX-=nx*lk*dot;accY-=ny*lk*dot;}
        float diag=cx*cx+cy*cy+cz*cz+1e-6f;lam+=accX*cx+accY*cy/(diag*4.0f);__syncthreads();
    }
    vel[tid*2]=vx+lam*cx;vel[tid*2+1]=vy+lam*cy;
    pos[tid*2]=pos[tid*2]+vel[tid*2]*dt;pos[tid*2+1]=pos[tid*2+1]+vel[tid*2+1]*dt;
}

// Stage 4: Integration
extern "C" __global__ void integrate_stage(float* vel, float* pos, int n, float dt, float gravity)
{
    int tid=blockIdx.x*blockDim.x+threadIdx.x;
    if(tid>=n)return;
    vel[tid*2+1]-=gravity*dt;pos[tid*2]+=vel[tid*2]*dt;pos[tid*2+1]+=vel[tid*2+1]*dt;
    if(pos[tid*2+1]<0){pos[tid*2+1]=0;vel[tid*2+1]*=-0.3f;}
}

int main()
{
    printf("=== A800 GPU Pipeline ===\n");
    cudaDeviceProp p;cudaGetDeviceProperties(&p,0);
    printf("Device: %s, %d SMs\n\n",p.name,p.multiProcessorCount);

    const int N=65536,BLK=256,RND=100;
    unsigned int *d_boxes,*d_overlaps,*d_count;
    float *d_pos,*d_vel,*d_contacts;
    cudaMalloc(&d_boxes,N*sizeof(unsigned));
    cudaMalloc(&d_overlaps,N*sizeof(unsigned));
    cudaMalloc(&d_count,sizeof(int));
    cudaMalloc(&d_pos,N*3*sizeof(float));
    cudaMalloc(&d_vel,N*2*sizeof(float));
    cudaMalloc(&d_contacts,N*4*sizeof(float));

    unsigned*h_boxes=new unsigned[N];float*h_vp=new float[N*4];
    for(int i=0;i<N;i++){h_boxes[i]=rand()%65536;
        h_vp[i*3]=(rand()%2000)/10.0f-100;h_vp[i*3+1]=(rand()%300)/10.0f;h_vp[i*3+2]=(rand()%2000)/10.0f-100;}
    cudaMemcpy(d_boxes,h_boxes,N*sizeof(unsigned),cudaMemcpyHostToDevice);
    cudaMemcpy(d_pos,h_vp,N*3*sizeof(float),cudaMemcpyHostToDevice);
    cudaMemset(d_vel,0,N*2*sizeof(float));

    cudaEvent_t s1,s2;cudaEventCreate(&s1);cudaEventCreate(&s2);
    float ms,totalMs,bpMs,npMs,solvMs,intMs;

    printf("Running %d frames...\n",RND);

    cudaEventRecord(s1);
    for(int r=0;r<RND;r++){
        cudaMemset(d_count,0,sizeof(int));
        bp_stage<<<(N+255)/256,256>>>(d_boxes,N,d_overlaps,d_count);
        int h_count;cudaMemcpy(&h_count,d_count,sizeof(int),cudaMemcpyDeviceToHost);int np_n=(h_count>0&&h_count<N)?h_count:N;np_stage<<<(np_n+255)/256,256>>>(d_pos,d_contacts,N,d_overlaps,np_n,2.0f);
        solver_stage<<<(N+255)/256,256>>>(d_vel,d_pos,d_contacts,N,1.0f/60.0f);
        integrate_stage<<<(N+255)/256,256>>>(d_vel,d_pos,N,1.0f/60.0f,9.81f);
    }
    cudaEventRecord(s2);cudaDeviceSynchronize();cudaEventElapsedTime(&totalMs,s1,s2);

    cudaMemset(d_count,0,sizeof(int));
    cudaEventRecord(s1);for(int r=0;r<RND;r++){cudaMemset(d_count,0,sizeof(int));bp_stage<<<(N+255)/256,256>>>(d_boxes,N,d_overlaps,d_count);}
    cudaEventRecord(s2);cudaDeviceSynchronize();cudaEventElapsedTime(&bpMs,s1,s2);

    cudaEventRecord(s1);for(int r=0;r<RND;r++)int h_count;cudaMemcpy(&h_count,d_count,sizeof(int),cudaMemcpyDeviceToHost);int np_n=(h_count>0&&h_count<N)?h_count:N;np_stage<<<(np_n+255)/256,256>>>(d_pos,d_contacts,N,d_overlaps,np_n,2.0f);
    cudaEventRecord(s2);cudaDeviceSynchronize();cudaEventElapsedTime(&npMs,s1,s2);

    cudaEventRecord(s1);for(int r=0;r<RND;r++)solver_stage<<<(N+255)/256,256>>>(d_vel,d_pos,d_contacts,N,1.0f/60.0f);
    cudaEventRecord(s2);cudaDeviceSynchronize();cudaEventElapsedTime(&solvMs,s1,s2);

    cudaEventRecord(s1);for(int r=0;r<RND;r++)integrate_stage<<<(N+255)/256,256>>>(d_vel,d_pos,N,1.0f/60.0f,9.81f);
    cudaEventRecord(s2);cudaDeviceSynchronize();cudaEventElapsedTime(&intMs,s1,s2);

    printf("\n=== Per-Stage ===\n");
    printf("Broadphase:    %7.1f us\n",bpMs/RND*1000);
    printf("Narrowphase:   %7.1f us\n",npMs/RND*1000);
    printf("Solver:        %7.1f us\n",solvMs/RND*1000);
    printf("Integration:   %7.1f us\n",intMs/RND*1000);
    printf("Full frame:    %7.1f us  (%.0f FPS)\n",totalMs/RND*1000,1e6f/(totalMs/RND*1000));

    delete[]h_boxes;delete[]h_vp;
    cudaFree(d_boxes);cudaFree(d_overlaps);cudaFree(d_count);
    cudaFree(d_pos);cudaFree(d_vel);cudaFree(d_contacts);
    cudaEventDestroy(s1);cudaEventDestroy(s2);
    return 0;
}
