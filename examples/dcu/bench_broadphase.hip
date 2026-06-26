// Benchmark: gpubroadphase kernels
#include "PxgCommonDefines.h"
#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdlib>
using namespace physx;

#include "broadphase.cu"
#include "aggregate.cu"

extern "C" __global__ void bpSignalComplete(PxU32* sig);

int main()
{
    printf("=== gpubroadphase kernels ===\n");
    hipDeviceProp_t p;hipGetDeviceProperties(&p,0);printf("Device: %s\n\n",p.name);
    const int RND=100;
    hipEvent_t s1,s2;hipEventCreate(&s1);hipEventCreate(&s2);float ms;int kc=0;
#define B(label,kernel,grid,block,...) do{hipEventRecord(s1);for(int r=0;r<RND;r++)kernel<<<grid,block>>>(__VA_ARGS__);hipEventRecord(s2);hipDeviceSynchronize();hipEventElapsedTime(&ms,s1,s2);printf("[%2d] %-40s %7.1f us\n",++kc,label,ms/RND*1000);}while(0)

    PxU32 *d_u;hipMalloc(&d_u,4);
    B("bpSignalComplete",    bpSignalComplete,   1,1,d_u);

    printf("\n%d kernels\n",kc);
    hipFree(d_u);hipEventDestroy(s1);hipEventDestroy(s2);
}
