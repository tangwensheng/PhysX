// Benchmark: gpusimulationcontroller kernels
#include "PxgCommonDefines.h"
#include "foundation/PxVec3.h"
#include "foundation/PxVec4.h"
#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdlib>
using namespace physx;

#include "SDFConstruction.cu"
#include "sparseGridStandalone.cu"

int main()
{
    printf("=== gpusimulationcontroller kernels ===\n");
    hipDeviceProp_t p;hipGetDeviceProperties(&p,0);printf("Device: %s\n\n",p.name);
    const int N=65536,RND=50;
    hipEvent_t s1,s2;hipEventCreate(&s1);hipEventCreate(&s2);float ms;int kc=0;
#define B(label,kernel,grid,block,...) do{hipEventRecord(s1);for(int r=0;r<RND;r++)kernel<<<grid,block>>>(__VA_ARGS__);hipEventRecord(s2);hipDeviceSynchronize();hipEventElapsedTime(&ms,s1,s2);printf("[%2d] %-45s %7.1f us\n",++kc,label,ms/RND*1000);}while(0)

    PxU32 *d_u1,*d_u2;
    hipMalloc(&d_u1,N*sizeof(PxU32));hipMalloc(&d_u2,N*sizeof(PxU32));
    PxU32 *h_u=new PxU32[N];for(int i=0;i<N;i++)h_u[i]=rand()%65536;
    hipMemcpy(d_u1,h_u,N*sizeof(PxU32),hipMemcpyHostToDevice);

    printf("--- SDFConstruction.cu ---\n");
    int nTri=16384;
    PxVec3*d_tv;float4*d_tl,*d_tu;PxI32*d_keys;PxReal*d_deltas;PxU32*d_ti;
    hipMalloc(&d_tv,nTri*3*sizeof(PxVec3));hipMalloc(&d_ti,nTri*3*sizeof(PxU32));
    hipMalloc(&d_tl,nTri*sizeof(float4));hipMalloc(&d_tu,nTri*sizeof(float4));
    hipMalloc(&d_keys,nTri*sizeof(PxI32));hipMalloc(&d_deltas,nTri*sizeof(PxReal));
    // Valid triangle indices: 0,1,2 for each triangle
    PxU32 *h_ti=new PxU32[nTri*3];for(int i=0;i<nTri*3;i++)h_ti[i]=i;
    hipMemcpy(d_ti,h_ti,nTri*3*sizeof(PxU32),hipMemcpyHostToDevice);
    PxVec3 *h_tv=new PxVec3[nTri*3];for(int i=0;i<nTri*3;i++)h_tv[i]=PxVec3(i%7,i%11,i%13);
    hipMemcpy(d_tv,h_tv,nTri*3*sizeof(PxVec3),hipMemcpyHostToDevice);
    B("bvhComputeTriangleBounds",bvhComputeTriangleBounds,(nTri+255)/256,256,d_tv,d_ti,nTri,d_tl,d_tu,0.01f);
    B("bvhComputeTotalBounds",   bvhComputeTotalBounds,   (nTri+255)/256,256,d_tl,d_tu,d_tv,d_tv+N/4,nTri);
    B("bvhCalculateKeyDeltas",   bvhCalculateKeyDeltas,   (nTri+255)/256,256,d_keys,d_deltas,nTri);
    delete[]h_ti;delete[]h_tv;
    hipFree(d_tv);hipFree(d_ti);hipFree(d_tl);hipFree(d_tu);hipFree(d_keys);hipFree(d_deltas);

    printf("--- sparseGridStandalone.cu ---\n");
    PxReal*d_dens;hipMalloc(&d_dens,N*sizeof(PxReal));hipMemset(d_dens,0,N*sizeof(PxReal));
    B("sg_SortedArrayToDelta",   sg_SparseGridSortedArrayToDelta,(N+255)/256,256,d_u1,d_u1,d_u2,N);
    B("sg_MarkSubgridEndIndices",sg_MarkSubgridEndIndices,(N+255)/256,256,d_u1,N/64,d_u2);
    B("sg_ClearDensity",         sg_SparseGridClearDensity,(N+255)/256,256,d_dens,0.0f,d_u1,(PxU32)64);
    hipFree(d_dens);

    printf("\n%d kernels\n",kc);
    delete[]h_u;hipFree(d_u1);hipFree(d_u2);
    hipEventDestroy(s1);hipEventDestroy(s2);return 0;
}
