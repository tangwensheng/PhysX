#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#define HIP_CHECK(x) do { hipError_t e=(x); if(e!=hipSuccess){ fprintf(stderr,"HIP error: %s -> %s\n",#x,hipGetErrorString(e)); return 2; } } while(0)

__global__ void packetShapeKernel(unsigned int* out, unsigned int seed)
{
    extern __shared__ unsigned int smem[];
    const int tid = threadIdx.x + blockDim.x * threadIdx.y;
    unsigned int acc = seed ^ unsigned(tid * 747796405u);
    unsigned int local[1024];
#pragma unroll 1
    for(int i=0;i<1024;i++){ acc = acc * 1664525u + 1013904223u + unsigned(i); local[i] = acc; }
    if(tid < 1024) smem[tid] = acc;
    __syncthreads();
    if(tid == 0) out[blockIdx.x & 1023] = acc ^ local[seed & 1023] ^ smem[0];
}

int main(int argc, char** argv)
{
    int iters = 100000;
    int syncEvery = 1;
    int eventEvery = 0;
    int eventSync = 0;
    int d2dEvery = 1;
    int progressEvery = 1000;
    for(int i=1;i<argc;i++){
        if(!strcmp(argv[i],"--iters") && i+1<argc) iters = atoi(argv[++i]);
        else if(!strcmp(argv[i],"--sync-every") && i+1<argc) syncEvery = atoi(argv[++i]);
        else if(!strcmp(argv[i],"--event-every") && i+1<argc) eventEvery = atoi(argv[++i]);
        else if(!strcmp(argv[i],"--event-sync") && i+1<argc) eventSync = atoi(argv[++i]);
        else if(!strcmp(argv[i],"--d2d-every") && i+1<argc) d2dEvery = atoi(argv[++i]);
        else if(!strcmp(argv[i],"--progress-every") && i+1<argc) progressEvery = atoi(argv[++i]);
    }

    hipDeviceProp_t prop{};
    HIP_CHECK(hipGetDeviceProperties(&prop, 0));
    printf("[QUEUE REPRO] device0 name=%s arch=%s iters=%d syncEvery=%d eventEvery=%d eventSync=%d d2dEvery=%d\n", prop.name, prop.gcnArchName, iters, syncEvery, eventEvery, eventSync, d2dEvery);

    unsigned int* out = nullptr;
    char* a = nullptr;
    char* b = nullptr;
    HIP_CHECK(hipMalloc(&out, 4096));
    HIP_CHECK(hipMalloc(&a, 256 * 1024 * 1024));
    HIP_CHECK(hipMalloc(&b, 256 * 1024 * 1024));
    HIP_CHECK(hipMemset(out, 0, 4096));
    HIP_CHECK(hipMemset(a, 0x5a, 4096));
    HIP_CHECK(hipMemset(b, 0xa5, 4096));

    hipStream_t stream{};
    HIP_CHECK(hipStreamCreateWithFlags(&stream, hipStreamNonBlocking));
    hipEvent_t ev{};
    if(eventEvery > 0) HIP_CHECK(hipEventCreateWithFlags(&ev, hipEventDisableTiming));

    for(int i=0;i<iters;i++){
        hipLaunchKernelGGL(packetShapeKernel, dim3(2,1,1), dim3(32,2,1), 4096, stream, out, unsigned(i));
        HIP_CHECK(hipGetLastError());
        if(d2dEvery > 0 && ((i+1) % d2dEvery) == 0)
            HIP_CHECK(hipMemcpyDtoDAsync((hipDeviceptr_t)b, (hipDeviceptr_t)a, 16, stream));
        if(eventEvery > 0 && ((i+1) % eventEvery) == 0){
            HIP_CHECK(hipEventRecord(ev, stream));
            if(eventSync) HIP_CHECK(hipEventSynchronize(ev));
        }
        if(syncEvery > 0 && ((i+1) % syncEvery) == 0)
            HIP_CHECK(hipStreamSynchronize(stream));
        if(progressEvery > 0 && ((i+1) % progressEvery) == 0){ printf("[QUEUE REPRO] iter=%d/%d\n", i+1, iters); fflush(stdout); }
    }

    HIP_CHECK(hipDeviceSynchronize());
    if(eventEvery > 0) HIP_CHECK(hipEventDestroy(ev));
    HIP_CHECK(hipStreamDestroy(stream));
    HIP_CHECK(hipFree(b));
    HIP_CHECK(hipFree(a));
    HIP_CHECK(hipFree(out));
    printf("[QUEUE REPRO] VERDICT: PASS\n");
    return 0;
}
