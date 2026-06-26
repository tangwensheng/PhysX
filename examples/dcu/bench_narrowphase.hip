// Benchmark: gpunarrowphase compute-intensive kernels
#include "PxgCommonDefines.h"
#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdlib>
using namespace physx;

// Include narrowphase source files
#include "cudaGJKEPA.cu"
#include "pairManagement.cu"
#include "compressOutputContacts.cu"

int main()
{
    printf("=== gpunarrowphase kernels ===\n");
    hipDeviceProp_t p;hipGetDeviceProperties(&p,0);printf("Device: %s\n\n",p.name);
    const int N=65536,RND=50;
    hipEvent_t s1,s2;hipEventCreate(&s1);hipEventCreate(&s2);float ms;int kc=0;
#define B(label,kernel,grid,block,...) do{hipEventRecord(s1);for(int r=0;r<RND;r++)kernel<<<grid,block>>>(__VA_ARGS__);hipEventRecord(s2);hipDeviceSynchronize();hipEventElapsedTime(&ms,s1,s2);printf("[%2d] %-45s %7.1f us\n",++kc,label,ms/RND*1000);}while(0)

    // Common buffers
    size_t sU=N*sizeof(PxU32), sF4=N*sizeof(float4);
    PxU32 *d_u1,*d_u2,*d_u3;
    float4 *d_f4i,*d_f4o;
    hipMalloc(&d_u1,sU);hipMalloc(&d_u2,sU);hipMalloc(&d_u3,sU);
    hipMalloc(&d_f4i,sF4);hipMalloc(&d_f4o,sF4);
    PxU32 *h_u=new PxU32[N];for(int i=0;i<N;i++)h_u[i]=rand()%65536;
    hipMemcpy(d_u1,h_u,sU,hipMemcpyHostToDevice);

    // ---- cudaGJKEPA.cu: prepareLostFoundPairs ----
    printf("--- cudaGJKEPA.cu ---\n");
    B("prepareLostFoundPairs_Stg1", prepareLostFoundPairs_Stage1, (N+255)/256,256,d_u1,d_u2,d_u3,N);

    // ---- pairManagement.cu: initializeManifolds ----
    printf("--- pairManagement.cu ---\n");
    float4 *h_f4=new float4[N];for(int i=0;i<N;i++)h_f4[i]=make_float4(i,i,i,i);
    hipMemcpy(d_f4i,h_f4,sF4,hipMemcpyHostToDevice);
    B("initializeManifolds",        initializeManifolds,        (N+255)/256,256,d_f4o,d_f4i,64,(PxU32)4);

    // ---- compressOutputContacts.cu ----
    printf("--- compressOutputContacts.cu ---\n");
    size_t bufSz=1024*1024;
    void *d_cmOut,*d_blockNums;hipMalloc(&d_cmOut,bufSz);hipMalloc(&d_blockNums,bufSz);
    hipMemset(d_cmOut,0,bufSz);hipMemset(d_blockNums,0,bufSz);
    B("compressContactStage1",       compressContactStage1,      1,128,
        (PxsContactManagerOutput*)d_cmOut,(PxU32)64,(PxU32*)d_blockNums);
    hipFree(d_cmOut);hipFree(d_blockNums);

    // ---- Output ----
    printf("\n%d kernels\n",kc);
    delete[]h_u;delete[]h_f4;
    hipFree(d_u1);hipFree(d_u2);hipFree(d_u3);hipFree(d_f4i);hipFree(d_f4o);
    hipEventDestroy(s1);hipEventDestroy(s2);return 0;
}
