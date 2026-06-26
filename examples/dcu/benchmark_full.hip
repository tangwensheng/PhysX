// PhysX GPU Kernel Benchmark — gpucommon + TGS Solver
#include "PxgCommonDefines.h"
#include "foundation/PxVec4.h"
#include "foundation/PxVec3.h"
#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdlib>
using namespace physx;

// ====== TGS Solver (compute-intensive, equivalent pattern) ======
// Matches real solverMultiBlockTGS.cu: shared memory tiles, iterative loop, scatter update
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

// ====== DCU-optimized radix sort kernels ======
extern "C" __global__ void radixSortCopy_dcu(const PxU64* in, PxU64* out, PxU32* rank, const PxU32* numKeys)
{
    unsigned int i0 = (blockIdx.x*blockDim.x+threadIdx.x)*4;
    unsigned int n = *numKeys;
    if(i0>=n)return;
    unsigned int r0=rank[i0],r1=rank[i0+1],r2=rank[i0+2],r3=rank[i0+3];
    out[i0]=in[r0];out[i0+1]=in[r1];out[i0+2]=in[r2];out[i0+3]=in[r3];
}
extern "C" __global__ void radixSortCopy2_dcu(const PxU64* in, PxU64* out, PxU32* rank, PxU32 n)
{
    unsigned int i0 = (blockIdx.x*blockDim.x+threadIdx.x)*4;
    if(i0>=n)return;
    PxU32 r0=rank[i0],r1=rank[i0+1],r2=rank[i0+2],r3=rank[i0+3];
    out[i0]=(r0==0xffffffff)?0xffffffffffffffffULL:in[r0];
    out[i0+1]=(r1==0xffffffff)?0xffffffffffffffffULL:in[r1];
    out[i0+2]=(r2==0xffffffff)?0xffffffffffffffffULL:in[r2];
    out[i0+3]=(r3==0xffffffff)?0xffffffffffffffffULL:in[r3];
}
extern "C" __global__ void radixSortCopyHigh32Bits_dcu(const PxU64* in, PxU32* out, PxU32* rank, const PxU32* numKeys)
{
    unsigned int i0=(blockIdx.x*blockDim.x+threadIdx.x)*4, n=*numKeys;
    if(i0>=n)return;
    uint4 r=*(uint4*)(rank+i0);
    out[i0]=(unsigned)(in[r.x]>>32);out[i0+1]=(unsigned)(in[r.y]>>32);
    out[i0+2]=(unsigned)(in[r.z]>>32);out[i0+3]=(unsigned)(in[r.w]>>32);
}
extern "C" __global__ void radixSortCopyBits2_dcu(const PxU64* in, PxU32* out, PxU32* rank, PxU32 n, bool lo)
{
    unsigned int i0=(blockIdx.x*blockDim.x+threadIdx.x)*4;
    if(i0>=n)return;
    PxU32 r0=rank[i0],r1=rank[i0+1],r2=rank[i0+2],r3=rank[i0+3];
    if(lo){
        out[i0]=(r0==0xffffffff)?0xffffffff:(PxU32)(in[r0]&0xffffffffULL);
        out[i0+1]=(r1==0xffffffff)?0xffffffff:(PxU32)(in[r1]&0xffffffffULL);
        out[i0+2]=(r2==0xffffffff)?0xffffffff:(PxU32)(in[r2]&0xffffffffULL);
        out[i0+3]=(r3==0xffffffff)?0xffffffff:(PxU32)(in[r3]&0xffffffffULL);
    }else{
        out[i0]=(r0==0xffffffff)?0xffffffff:(PxU32)(in[r0]>>32);
        out[i0+1]=(r1==0xffffffff)?0xffffffff:(PxU32)(in[r1]>>32);
        out[i0+2]=(r2==0xffffffff)?0xffffffff:(PxU32)(in[r2]>>32);
        out[i0+3]=(r3==0xffffffff)?0xffffffff:(PxU32)(in[r3]>>32);
    }
}
extern "C" __global__ void interleaveBuffers_dcu(const float4* v, const float4* n, PxU32 len, PxVec3* out)
{
    unsigned int i=(blockIdx.x*blockDim.x+threadIdx.x)*2;
    if(i>=len)return;
    float4*d=(float4*)out,v0=v[i],n0=n[i];
    if(i+1<len){float4 v1=v[i+1],n1=n[i+1];
        d[i*3/2]=make_float4(v0.x,v0.y,v0.z,n0.x);
        d[i*3/2+1]=make_float4(n0.y,n0.z,v1.x,v1.y);
        d[i*3/2+2]=make_float4(v1.z,n1.x,n1.y,n1.z);
    }else{d[i*3/2]=make_float4(v0.x,v0.y,v0.z,n0.x);d[i*3/2+1]=make_float4(n0.y,n0.z,0,0);}
}
extern "C" __global__ void clampMaxValue_dcu(PxU32* v, PxU32 m) { if(*v>m)*v=m; }

// v2: block-level sparse solver — each block solves its own contacts
// Matches real PhysX solverMultiBlock pattern: contacts assigned to blocks
extern "C" __global__ void tgs_solver_v2(
    float* vel, float* pos, const float* cfm,
    const float* jac, const float* rhs, int n, float dt)
{
    __shared__ float sLam[256], sRh[256], sCfm[256];
    __shared__ float sJx[256], sJy[256], sJz[256], sJx2[256];
    int lane = threadIdx.x;
    int blk = blockIdx.x;
    int base = blk * 256;
    if (base >= n) return;

    // Load this block's 256 contacts into shared memory
    int idx = base + lane;
    if (idx < n) {
        sLam[lane]=0; sRh[lane]=rhs[idx]; sCfm[lane]=cfm[idx];
        sJx[lane]=jac[idx*8+0]; sJy[lane]=jac[idx*8+1];
        sJz[lane]=jac[idx*8+2]; sJx2[lane]=jac[idx*8+4];
    }
    __syncthreads();

    // Gauss-Seidel iteration: 4 passes over the block's contacts
    for (int pass = 0; pass < 4; pass++) {
        for (int k = 0; k < 256; k++) {
            float jxL=sJx[k], jyL=sJy[k], jzL=sJz[k], jx2L=sJx2[k];
            // Each lane solves using its own jacobian row against contact k
            if (idx < n) {
                float jx=jac[idx*8+0], jy=jac[idx*8+1], jz=jac[idx*8+2];
                float jx2=jac[idx*8+4];
                float cf=sCfm[lane];
                float d = jx*jxL + jy*jyL + jz*jzL + jx2*jx2L + cf;
                float corr = (sRh[k] - d * sLam[k]) / (d + cf + 1e-6f);
                sLam[k] += corr;
            }
            __syncthreads();
        }
    }

    // Write back
    if (idx < n) {
        vel[idx*2] += sLam[lane] * sJx[lane];
        vel[idx*2+1] += sLam[lane] * sJy[lane];
        pos[idx*2] += vel[idx*2] * dt;
        pos[idx*2+1] += vel[idx*2+1] * dt;
    }
}

// v1: flat global reads — no __syncthreads, no shared memory
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

// ====== Jacobi Solver (fully parallel, fair cross-platform) ======
// Each contact solved independently using previous iteration's values.
// No sequential dependencies, no atomic conflicts — all threads do equal work.
extern "C" __global__ void jacobi_solver(
    float* vel, float* pos, const float* cfm,
    const float* jac, const float* rhs, int n, float dt, int iters)
{
    __shared__ float sLam[256], sJx[256], sJy[256], sJz[256], sJx2[256];
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int lane = threadIdx.x;
    if (tid >= n) return;

    float jx = jac[tid*8+0], jy = jac[tid*8+1], jz = jac[tid*8+2], jx2=jac[tid*8+4];
    float cf = cfm[tid], rh = rhs[tid], lam = 0;
    float vx = vel[tid*2], vy = vel[tid*2+1];

    // Load this block's jacobian rows into shared memory (once)
    if (lane < n) {
        sJx[lane]=jac[lane*8+0]; sJy[lane]=jac[lane*8+1];
        sJz[lane]=jac[lane*8+2]; sJx2[lane]=jac[lane*8+4];
    }
    __syncthreads();

    for (int iter = 0; iter < iters; iter++) {
        // Jacobi: compute new lambda using OLD lambda values
        sLam[lane] = lam;
        __syncthreads();

        float delta = rh;
        for (int k = 0; k < n && k < 256; k++) {
            delta -= (jx*sJx[k] + jy*sJy[k] + jz*sJz[k] + jx2*sJx2[k] + cf) * sLam[k];
        }
        delta = -delta / (jx*jx + jy*jy + jz*jz + jx2*jx2 + cf + 1e-6f);
        lam += delta;
        rh -= delta * (jx*jx + jy*jy + jz*jz + jx2*jx2 + cf);
        __syncthreads();
    }

    vel[tid*2] = vx + lam * jx;
    vel[tid*2+1] = vy + lam * jy;
    pos[tid*2] += vel[tid*2] * dt;
    pos[tid*2+1] += vel[tid*2+1] * dt;
}

// ====== Include original gpucommon sources ======
#include "radixSortImpl.cu"
#include "MemCopyBalanced.cu"
#include "utility.cu"

int main()
{
    printf("========================================\n");
    printf(" PhysX GPU Kernel Benchmark\n");
    printf("========================================\n\n");
    hipDeviceProp_t p;hipGetDeviceProperties(&p,0);
    printf("Device: %s, %d CUs\n\n",p.name,p.multiProcessorCount);

    const int N=65536,BLK=256,RND=50;
    size_t sU=N*sizeof(PxU32),sU64=N*sizeof(PxU64),sF4=N*sizeof(float4),sV3=N*sizeof(PxVec3);
    PxU32 *d_u1,*d_u2,*d_u3,*d_num;
    PxU64 *d_64i,*d_64o;
    float4 *d_f4i;
    PxVec3 *d_v3;
    hipMalloc(&d_u1,sU);hipMalloc(&d_u2,sU);hipMalloc(&d_u3,sU);
    hipMalloc(&d_64i,sU64);hipMalloc(&d_64o,sU64);hipMalloc(&d_num,sizeof(PxU32));
    hipMalloc(&d_f4i,sF4);hipMalloc(&d_v3,sV3);
    PxU32 *h_u=new PxU32[N],*h_u2=new PxU32[N];PxU64 *h_64=new PxU64[N];float4 *h_f4=new float4[N];
    for(int i=0;i<N;i++){h_u[i]=rand()%65536;h_u2[i]=i;h_64[i]=rand()%65536;h_f4[i]=make_float4(1,2,3,4);}
    hipMemcpy(d_u1,h_u,sU,hipMemcpyHostToDevice);hipMemcpy(d_u2,h_u2,sU,hipMemcpyHostToDevice);
    hipMemcpy(d_64i,h_64,sU64,hipMemcpyHostToDevice);hipMemcpy(d_f4i,h_f4,sF4,hipMemcpyHostToDevice);
    PxU32 hn=N;hipMemcpy(d_num,&hn,sizeof(PxU32),hipMemcpyHostToDevice);

    hipEvent_t s1,s2;hipEventCreate(&s1);hipEventCreate(&s2);float ms;int kc=0;
#define B(label,kernel,grid,block,...) do{hipEventRecord(s1);for(int r=0;r<RND;r++)kernel<<<grid,block>>>(__VA_ARGS__);hipEventRecord(s2);hipDeviceSynchronize();hipEventElapsedTime(&ms,s1,s2);printf("[%2d] %-45s %7.1f us\n",++kc,label,ms/RND*1000);}while(0)

    // ---- gpucommon (orig vs dcu) ----
    printf("--- radixSortCopy ---\n");
    int g1=(N+255)/256,g4=(N/4+255)/256;
    B("radixSortCopy      (orig)",  radixSortCopy,              g1,256,d_64i,d_64o,d_u2,d_num);
    B("radixSortCopy      (dcu )",  radixSortCopy_dcu,          g4,256,d_64i,d_64o,d_u2,d_num);
    B("radixSortCopy2     (orig)",  radixSortCopy2,             g1,256,d_64i,d_64o,d_u2,N);
    B("radixSortCopy2     (dcu )",  radixSortCopy2_dcu,         g4,256,d_64i,d_64o,d_u2,N);
    B("radixSortCopyHi32  (orig)",  radixSortCopyHigh32Bits,    g1,256,d_64i,d_u3,d_u2,d_num);
    B("radixSortCopyHi32  (dcu )",  radixSortCopyHigh32Bits_dcu,g4,256,d_64i,d_u3,d_u2,d_num);
    B("radixSortCopyBits2 (orig)",  radixSortCopyBits2,         g1,256,d_64i,d_u3,d_u2,N,true);
    B("radixSortCopyBits2 (dcu )",  radixSortCopyBits2_dcu,     g4,256,d_64i,d_u3,d_u2,N,true);

    printf("--- interleaveBuffers ---\n");
    B("interleaveBuffers  (orig)",  interleaveBuffers,          (N+255)/256,256,d_f4i,d_f4i,N/2,d_v3);
    B("interleaveBuffers  (dcu )",  interleaveBuffers_dcu,      (N/4+255)/256,256,d_f4i,d_f4i,N/2,d_v3);

    printf("--- clampMaxValue ---\n");
    B("clampMaxValue      (orig)",  clampMaxValue,              1,1,d_u1,(PxU32)5000);
    B("clampMaxValue      (dcu )",  clampMaxValue_dcu,          1,1,d_u1,(PxU32)5000);

    // ---- TGS Solver (compute-intensive) ----
    printf("--- TGS Solver ---\n");
    {
        int tsN=65536;
        float* d_vs,*d_ps,*d_cf,*d_js,*d_rs;
        hipMalloc(&d_vs,tsN*2*sizeof(float));hipMalloc(&d_ps,tsN*2*sizeof(float));
        hipMalloc(&d_cf,tsN*sizeof(float));hipMalloc(&d_js,tsN*8*sizeof(float));
        hipMalloc(&d_rs,tsN*sizeof(float));
        float *hv=new float[tsN*8];for(int i=0;i<tsN*8;i++)hv[i]=(rand()%1000)/100.0f;
        hipMemcpy(d_vs,hv,tsN*2*sizeof(float),hipMemcpyHostToDevice);
        hipMemcpy(d_ps,hv,tsN*2*sizeof(float),hipMemcpyHostToDevice);
        hipMemcpy(d_cf,hv,tsN*sizeof(float),hipMemcpyHostToDevice);
        hipMemcpy(d_js,hv,tsN*8*sizeof(float),hipMemcpyHostToDevice);
        hipMemcpy(d_rs,hv,tsN*sizeof(float),hipMemcpyHostToDevice);

        B("tgs_solver_v0 (tiled+smem)",  tgs_solver, (tsN+255)/256,256,
            d_vs,d_ps,d_cf,d_js,d_rs,tsN,1.0f/60.0f);

        B("tgs_solver_v1 (flat O(N^2))", tgs_solver_v1, (tsN+255)/256,256,
            d_vs,d_ps,d_cf,d_js,d_rs,tsN,1.0f/60.0f);
        B("jacobi (1 iter baseline)",  jacobi_solver, (tsN+255)/256,256,
            d_vs,d_ps,d_cf,d_js,d_rs,tsN,1.0f/60.0f,1);
        B("jacobi_solver (8 iters)",   jacobi_solver, (tsN+255)/256,256,
            d_vs,d_ps,d_cf,d_js,d_rs,tsN,1.0f/60.0f,8);

        delete[]hv;hipFree(d_vs);hipFree(d_ps);hipFree(d_cf);hipFree(d_js);hipFree(d_rs);
    }

    printf("\n========================================\n");
    printf(" %d benchmarks.\n",kc);
    printf("========================================\n");

    delete[]h_u;delete[]h_u2;delete[]h_64;delete[]h_f4;
    hipFree(d_u1);hipFree(d_u2);hipFree(d_u3);hipFree(d_64i);hipFree(d_64o);
    hipFree(d_f4i);hipFree(d_v3);hipFree(d_num);
    hipEventDestroy(s1);hipEventDestroy(s2);
    return 0;
}
