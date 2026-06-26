// Benchmark: gpusimulationcontroller + gpunarrowphase + gpusolver kernels
// Each module #included separately to avoid cuh conflicts.
// gpusimulationcontroller has the most simple-arg kernels (~12).

#include "PxgCommonDefines.h"
#include "foundation/PxVec3.h"
#include "foundation/PxVec4.h"
#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdlib>
using namespace physx;

// ====== gpusimulationcontroller ======
#include "algorithms.cu"
#include "SDFConstruction.cu"
#include "sparseGridStandalone.cu"

// ====== gpunarrowphase ======
#include "cudaGJKEPA.cu"

// ====== gpusolver ======
#include "constraintBlockPrePrep.cu"

int main()
{
    printf("========================================\n");
    printf(" PhysX GPU Kernel Benchmark — 3 modules\n");
    printf("========================================\n\n");
    hipDeviceProp_t p;hipGetDeviceProperties(&p,0);
    printf("Device: %s, %d CUs\n\n",p.name,p.multiProcessorCount);

    const int N=65536,BLK=256,RND=50;
    PxU32 *d_u1,*d_u2,*d_u3,*d_num,*d_total;
    float4 *d_f4i,*d_f4o;
    hipMalloc(&d_u1,N*sizeof(PxU32));hipMalloc(&d_u2,N*sizeof(PxU32));hipMalloc(&d_u3,N*sizeof(PxU32));
    hipMalloc(&d_num,sizeof(PxU32));hipMalloc(&d_total,sizeof(PxU32));
    hipMalloc(&d_f4i,N*sizeof(float4));hipMalloc(&d_f4o,N*sizeof(float4));

    PxU32 *h_u=new PxU32[N],*h_u2=new PxU32[N];float4 *h_f4=new float4[N];
    for(int i=0;i<N;i++){h_u[i]=rand()%65536;h_u2[i]=i;h_f4[i]=make_float4(1,2,3,4);}
    hipMemcpy(d_u1,h_u,N*sizeof(PxU32),hipMemcpyHostToDevice);
    hipMemcpy(d_u2,h_u2,N*sizeof(PxU32),hipMemcpyHostToDevice);
    hipMemcpy(d_f4i,h_f4,N*sizeof(float4),hipMemcpyHostToDevice);

    hipEvent_t s1,s2;hipEventCreate(&s1);hipEventCreate(&s2);float ms;int kc=0;
#define B(label,kernel,grid,block,...) do{hipEventRecord(s1);for(int r=0;r<RND;r++)kernel<<<grid,block>>>(__VA_ARGS__);hipEventRecord(s2);hipDeviceSynchronize();hipEventElapsedTime(&ms,s1,s2);printf("[%2d] %-45s %7.1f us\n",++kc,label,ms/RND*1000);}while(0)

    printf("--- gpusimulationcontroller/algorithms.cu ---\n");
    hipMemset(d_total,0,sizeof(PxU32));
    B("scanPerBlockKernel",      scanPerBlockKernel,      (N+1023)/1024,1024,d_u1,d_u2,d_u3,N,0,d_total);
    B("addBlockSumsKernel",      addBlockSumsKernel,      (N+1023)/1024,1024,d_u3,d_u2,N,d_total);
    B("reorderKernel",           reorderKernel,           (N+255)/256,256,d_f4i,d_f4o,N,d_u2);

    printf("--- gpusimulationcontroller/SDFConstruction.cu ---\n");
    int nTri=16384;
    PxVec3*d_tv;float4*d_tl,*d_tu;PxI32*d_keys;PxReal*d_deltas;PxU32*d_ti;
    hipMalloc(&d_tv,nTri*3*sizeof(PxVec3));hipMalloc(&d_ti,nTri*3*sizeof(PxU32));
    hipMalloc(&d_tl,nTri*sizeof(float4));hipMalloc(&d_tu,nTri*sizeof(float4));
    hipMalloc(&d_keys,nTri*sizeof(PxI32));hipMalloc(&d_deltas,nTri*sizeof(PxReal));
    hipMemcpy(d_tv,h_f4,nTri*3*sizeof(PxVec3),hipMemcpyHostToDevice);hipMemset(d_ti,0,nTri*3*sizeof(PxU32));
    B("bvhComputeTriangleBounds",bvhComputeTriangleBounds,(nTri+255)/256,256,d_tv,d_ti,nTri,d_tl,d_tu,0.01f);
    B("bvhComputeTotalBounds",   bvhComputeTotalBounds,   (nTri+255)/256,256,d_tl,d_tu,d_tv,d_tv+N/4,nTri);
    B("bvhCalculateKeyDeltas",   bvhCalculateKeyDeltas,   (nTri+255)/256,256,d_keys,d_deltas,nTri);
    hipFree(d_tv);hipFree(d_ti);hipFree(d_tl);hipFree(d_tu);hipFree(d_keys);hipFree(d_deltas);

    printf("--- gpusimulationcontroller/sparseGridStandalone.cu ---\n");
    PxReal*d_dens;hipMalloc(&d_dens,N*sizeof(PxReal));hipMemset(d_dens,0,N*sizeof(PxReal));
    B("sg_SortedArrayToDelta",   sg_SparseGridSortedArrayToDelta,(N+255)/256,256,d_u1,d_u1,d_u2,N);
    B("sg_MarkSubgridEndIndices",sg_MarkSubgridEndIndices,(N+255)/256,256,d_u1,N/64,d_u2);
    B("sg_ClearDensity",         sg_SparseGridClearDensity,(N+255)/256,256,d_dens,0.0f,d_u1,(PxU32)64);
    hipFree(d_dens);

    printf("--- gpunarrowphase/cudaGJKEPA.cu ---\n");
    B("prepareLostFoundPairs_Stg1",prepareLostFoundPairs_Stage1,(N+255)/256,256,d_u1,d_u2,d_u3,N);

    printf("--- gpusolver/constraintBlockPrePrep.cu ---\n");
    B("clearFrictionPatchCounts",clearFrictionPatchCounts,(256+255)/256,256,d_u1,d_u2,(PxU32)256);

    printf("\n========================================\n");
    printf(" %d kernels across 3 modules.\n",kc);
    printf("========================================\n");

    delete[]h_u;delete[]h_u2;delete[]h_f4;
    hipFree(d_u1);hipFree(d_u2);hipFree(d_u3);hipFree(d_num);hipFree(d_total);
    hipFree(d_f4i);hipFree(d_f4o);
    hipEventDestroy(s1);hipEventDestroy(s2);
    return 0;
}
