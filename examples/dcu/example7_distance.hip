// Example 7: GPU Parallel Distance Computation
// Tests: PxVec3 math + shared memory for convex-convex distance
// Modules: gpunarrowphase patterns (convex collision distance)

#include "foundation/PxVec3.h"
#include "PxgCommonDefines.h"
#include <hip/hip_runtime.h>
#include <cstdio>
#include <cmath>
#include <cstdlib>

using namespace physx;

// Sphere descriptor for GPU
struct GpuSphere { PxVec3 center; float radius; };

// GPU kernel: compute closest distance between N spheres (parallel)
extern "C" __global__ void sphere_distance(
    const GpuSphere* spheres, int n,
    const PxVec3* query_points,
    float* distances)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    PxVec3 q = query_points[i];
    PxVec3 c = spheres[i].center;
    float r = spheres[i].radius;
    float d = (q - c).magnitude() - r;
    distances[i] = (d > 0) ? d : 0;
}

// CPU reference
void sphere_distance_cpu(const GpuSphere* s, int n, const PxVec3* q, float* d)
{
    for (int i = 0; i < n; i++) {
        float dd = (q[i] - s[i].center).magnitude() - s[i].radius;
        d[i] = (dd > 0) ? dd : 0;
    }
}

int main()
{
    printf("========================================\n");
    printf(" Example 7: Parallel Distance (narrowphase)\n");
    printf("========================================\n\n");

    const int N = 65536;  // 64K spheres
    GpuSphere *h_spheres = new GpuSphere[N];
    PxVec3    *h_query   = new PxVec3[N];

    srand(123);
    for (int i = 0; i < N; i++) {
        h_spheres[i] = {PxVec3((rand()%200)-100.0f, (rand()%200)-100.0f, (rand()%200)-100.0f),
                        (rand()%50) / 10.0f + 0.5f};
        h_query[i]   = PxVec3((rand()%200)-100.0f, (rand()%200)-100.0f, (rand()%200)-100.0f);
    }

    // GPU
    GpuSphere *d_s; PxVec3 *d_q; float *d_d;
    hipMalloc(&d_s, N * sizeof(GpuSphere));
    hipMalloc(&d_q, N * sizeof(PxVec3));
    hipMalloc(&d_d, N * sizeof(float));
    hipMemcpy(d_s, h_spheres, N * sizeof(GpuSphere), hipMemcpyHostToDevice);
    hipMemcpy(d_q, h_query,  N * sizeof(PxVec3),    hipMemcpyHostToDevice);

    hipEvent_t start, stop;
    hipEventCreate(&start); hipEventCreate(&stop);

    hipEventRecord(start);
    int blk = 256, grd = (N + blk - 1) / blk;
    sphere_distance<<<grd, blk>>>(d_s, N, d_q, d_d);
    hipEventRecord(stop);
    hipDeviceSynchronize();

    float ms;
    hipEventElapsedTime(&ms, start, stop);

    float *h_dist_gpu = new float[N];
    hipMemcpy(h_dist_gpu, d_d, N * sizeof(float), hipMemcpyDeviceToHost);

    // CPU reference
    float *h_dist_cpu = new float[N];
    sphere_distance_cpu(h_spheres, N, h_query, h_dist_cpu);

    // Accuracy
    double maxErr = 0, sumErr = 0;
    for (int i = 0; i < N; i++) {
        double e = fabs(h_dist_gpu[i] - h_dist_cpu[i]);
        sumErr += e; if (e > maxErr) maxErr = e;
    }

    printf("Distance compute: %.2f ms\n", ms);
    printf("Max error:  %.6e\n", maxErr);
    printf("Avg error:  %.6e\n", sumErr / N);

    bool pass = (maxErr < 1e-4f);
    printf("\n%s\n", pass ? "[PASS] GPU distance matches CPU" : "[FAIL]");

    delete[] h_spheres; delete[] h_query; delete[] h_dist_gpu; delete[] h_dist_cpu;
    hipFree(d_s); hipFree(d_q); hipFree(d_d);
    hipEventDestroy(start); hipEventDestroy(stop);
    return pass ? 0 : 1;
}
