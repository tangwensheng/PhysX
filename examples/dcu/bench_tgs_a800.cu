// A800 CUDA Benchmark — TGS Solver (equivalent pattern)
// Compile: nvcc -O3 bench_tgs_a800.cu -o bench_tgs_a800
// Run:     ./bench_tgs_a800

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>

// v0: tiled + shared memory (original)
extern "C" __global__ void tgs_solver(
    float* vel, float* pos, const float* cfm,
    const float* jac, const float* rhs, int n, float dt)
{
    __shared__ float sJ[256*4];
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int lane = threadIdx.x;
    if (tid >= n) return;

    float cf = cfm[tid], rh = rhs[tid], lam = 0;
    float vx = vel[tid*2], vy = vel[tid*2+1];
    float jx = jac[tid*8+0], jy = jac[tid*8+1], jz = jac[tid*8+2];
    float jx2=jac[tid*8+4], jy2=jac[tid*8+5], jz2=jac[tid*8+6];

    for (int tile = 0; tile < n; tile += 256) {
        int j = tile + lane;
        if (j < n) {
            sJ[lane*4+0] = jac[j*8+0]; sJ[lane*4+1] = jac[j*8+1];
            sJ[lane*4+2] = jac[j*8+2]; sJ[lane*4+3] = jac[j*8+4];
        }
        __syncthreads();
        for (int k = 0; k < 256 && tile + k < n; k++) {
            float jxL=sJ[k*4+0], jyL=sJ[k*4+1], jzL=sJ[k*4+2], jx2L=sJ[k*4+3];
            float d = jx*jxL + jy*jyL + jz*jzL + jx2*jx2L + cf;
            float corr = (rh - d * lam) / (d + cf + 1e-6f);
            lam += corr * 0.5f; rh -= corr * d;
        }
        __syncthreads();
    }
    vel[tid*2]   = vx + lam * jx;   vel[tid*2+1] = vy + lam * jy;
    pos[tid*2]   = pos[tid*2]   + vel[tid*2]   * dt;
    pos[tid*2+1] = pos[tid*2+1] + vel[tid*2+1] * dt;
}

// Jacobi: fully parallel, each contact solved independently
extern "C" __global__ void jacobi_solver(
    float* vel, float* pos, const float* cfm,
    const float* jac, const float* rhs, int n, float dt, int iters)
{
    __shared__ float sLam[256], sJx[256], sJy[256], sJz[256], sJx2[256];
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int lane = threadIdx.x;
    if (tid >= n) return;

    float jx=jac[tid*8+0],jy=jac[tid*8+1],jz=jac[tid*8+2],jx2=jac[tid*8+4];
    float cf=cfm[tid],rh=rhs[tid],lam=0;
    float vx=vel[tid*2],vy=vel[tid*2+1];

    if(lane<n){sJx[lane]=jac[lane*8+0];sJy[lane]=jac[lane*8+1];sJz[lane]=jac[lane*8+2];sJx2[lane]=jac[lane*8+4];}
    __syncthreads();

    for(int iter=0;iter<iters;iter++){
        sLam[lane]=lam;
        __syncthreads();
        float delta=rh;
        for(int k=0;k<n&&k<256;k++)
            delta-=(jx*sJx[k]+jy*sJy[k]+jz*sJz[k]+jx2*sJx2[k]+cf)*sLam[k];
        delta=-delta/(jx*jx+jy*jy+jz*jz+jx2*jx2+cf+1e-6f);
        lam+=delta;rh-=delta*(jx*jx+jy*jy+jz*jz+jx2*jx2+cf);
        __syncthreads();
    }

    vel[tid*2]=vx+lam*jx;vel[tid*2+1]=vy+lam*jy;
    pos[tid*2]+=vel[tid*2]*dt;pos[tid*2+1]+=vel[tid*2+1]*dt;
}

// v2: block-level sparse solver — each block solves 256 contacts
extern "C" __global__ void tgs_solver_v2(
    float* vel, float* pos, const float* cfm,
    const float* jac, const float* rhs, int n, float dt)
{
    __shared__ float sLam[256], sRh[256], sCfm[256];
    __shared__ float sJx[256], sJy[256], sJz[256], sJx2[256];
    int lane = threadIdx.x;
    int idx = blockIdx.x * 256 + lane;
    if (idx >= n) return;

    sLam[lane]=0; sRh[lane]=rhs[idx]; sCfm[lane]=cfm[idx];
    sJx[lane]=jac[idx*8+0]; sJy[lane]=jac[idx*8+1];
    sJz[lane]=jac[idx*8+2]; sJx2[lane]=jac[idx*8+4];
    __syncthreads();

    for (int pass=0; pass<4; pass++) {
        for (int k=0; k<256; k++) {
            float d = jac[idx*8+0]*sJx[k] + jac[idx*8+1]*sJy[k] + jac[idx*8+2]*sJz[k] + jac[idx*8+4]*sJx2[k] + sCfm[lane];
            float corr = (sRh[k] - d * sLam[k]) / (d + sCfm[lane] + 1e-6f);
            sLam[k] += corr;
        }
        __syncthreads();
    }

    vel[idx*2] += sLam[lane] * sJx[lane];
    vel[idx*2+1] += sLam[lane] * sJy[lane];
    pos[idx*2] += vel[idx*2] * dt;
    pos[idx*2+1] += vel[idx*2+1] * dt;
}

// v1: flat global reads — no __syncthreads
extern "C" __global__ void tgs_solver_v1(
    float* vel, float* pos, const float* cfm,
    const float* jac, const float* rhs, int n, float dt)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n) return;
    float cf = cfm[tid], rh = rhs[tid], lam = 0;
    float vx = vel[tid*2], vy = vel[tid*2+1];
    float jx = jac[tid*8+0], jy = jac[tid*8+1], jz = jac[tid*8+2];
    float jx2=jac[tid*8+4], jy2=jac[tid*8+5], jz2=jac[tid*8+6];
    for (int j = 0; j < n; j++) {
        float jxL=jac[j*8+0], jyL=jac[j*8+1], jzL=jac[j*8+2], jx2L=jac[j*8+4];
        float d = jx*jxL + jy*jyL + jz*jzL + jx2*jx2L + cf;
        float corr = (rh - d * lam) / (d + cf + 1e-6f);
        lam += corr * 0.5f; rh -= corr * d;
    }
    vel[tid*2]=vx+lam*jx;vel[tid*2+1]=vy+lam*jy;
    pos[tid*2]=pos[tid*2]+vel[tid*2]*dt;pos[tid*2+1]=pos[tid*2+1]+vel[tid*2+1]*dt;
}

int main()
{
    printf("=== A800 TGS Solver (equivalent) ===\n");
    cudaDeviceProp p; cudaGetDeviceProperties(&p, 0);
    printf("Device: %s, %d SMs\n\n", p.name, p.multiProcessorCount);

    const int N = 65536, BLK = 256, RND = 50;
    float *d_vel, *d_pos, *d_cfm, *d_jac, *d_rhs;
    cudaMalloc(&d_vel, N*2*sizeof(float));
    cudaMalloc(&d_pos, N*2*sizeof(float));
    cudaMalloc(&d_cfm, N*sizeof(float));
    cudaMalloc(&d_jac, N*8*sizeof(float));
    cudaMalloc(&d_rhs, N*sizeof(float));

    float *h = new float[N*8];
    for(int i=0;i<N*8;i++) h[i]=(rand()%1000)/100.0f;
    cudaMemcpy(d_vel, h, N*2*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_pos, h, N*2*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_cfm, h, N*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_jac, h, N*8*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_rhs, h, N*sizeof(float), cudaMemcpyHostToDevice);

    cudaEvent_t s1, s2;
    cudaEventCreate(&s1); cudaEventCreate(&s2);

    cudaEventRecord(s1);
    for(int r=0;r<RND;r++)
        tgs_solver<<<(N+255)/256, 256>>>(d_vel, d_pos, d_cfm, d_jac, d_rhs, N, 1.0f/60.0f);
    cudaEventRecord(s2);
    cudaDeviceSynchronize();
    float ms;
    cudaEventElapsedTime(&ms, s1, s2);
    printf("tgs_solver_v0 (tiled+smem): %.1f us\n", ms/RND*1000);

    cudaEventRecord(s1);
    for(int r=0;r<RND;r++)
        tgs_solver_v1<<<(N+255)/256, 256>>>(d_vel, d_pos, d_cfm, d_jac, d_rhs, N, 1.0f/60.0f);
    cudaEventRecord(s2);
    cudaDeviceSynchronize();
    cudaEventElapsedTime(&ms, s1, s2);
    printf("tgs_solver_v1 (flat O(N^2)): %.1f us\n", ms/RND*1000);

    cudaEventRecord(s1);
    for(int r=0;r<RND;r++)
        tgs_solver_v2<<<(N+255)/256, 256>>>(d_vel, d_pos, d_cfm, d_jac, d_rhs, N, 1.0f/60.0f);
    cudaEventRecord(s2);
    cudaDeviceSynchronize();
    cudaEventElapsedTime(&ms, s1, s2);
    printf("tgs_solver_v2 (block sparse): %.1f us\n", ms/RND*1000);

    cudaEventRecord(s1);
    for(int r=0;r<RND;r++)
        jacobi_solver<<<(N+255)/256,256>>>(d_vel,d_pos,d_cfm,d_jac,d_rhs,N,1.0f/60.0f,8);
    cudaEventRecord(s2);
    cudaDeviceSynchronize();
    cudaEventElapsedTime(&ms,s1,s2);
    cudaEventRecord(s1);
    for(int r=0;r<RND;r++)
        jacobi_solver<<<(N+255)/256,256>>>(d_vel,d_pos,d_cfm,d_jac,d_rhs,N,1.0f/60.0f,1);
    cudaEventRecord(s2); cudaDeviceSynchronize();
    cudaEventElapsedTime(&ms,s1,s2);
    printf("jacobi (1 iter baseline): %.1f us\n", ms/RND*1000);

    cudaEventRecord(s1);
    for(int r=0;r<RND;r++)
        jacobi_solver<<<(N+255)/256,256>>>(d_vel,d_pos,d_cfm,d_jac,d_rhs,N,1.0f/60.0f,8);
    cudaEventRecord(s2); cudaDeviceSynchronize();
    cudaEventElapsedTime(&ms,s1,s2);
    printf("jacobi_solver (8 iters): %.1f us\n", ms/RND*1000);

    delete[] h;
    cudaFree(d_vel); cudaFree(d_pos); cudaFree(d_cfm); cudaFree(d_jac); cudaFree(d_rhs);
    cudaEventDestroy(s1); cudaEventDestroy(s2);
    return 0;
}
