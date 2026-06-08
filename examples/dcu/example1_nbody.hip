// Example 1: N-Body Gravitational Simulation
//
// 16K particles compute pairwise gravitational forces entirely on GPU.
// Validates: GPU compute throughput, PxVec3 math, atomic operations.

#include "foundation/PxVec3.h"
#include "PxgCommonDefines.h"
#include <hip/hip_runtime.h>
#include <cstdio>
#include <cmath>

using namespace physx;

const int N = 16384;      // particles
const float G = 0.01f;    // gravitational constant
const float DT = 0.01f;   // time step
const float EPS = 0.1f;   // softening (prevents singularity)

// ---- GPU Kernel: all-pairs gravity ----
extern "C" __global__ void nbody_gravity(
    const PxVec3* pos_in, const float* mass_in,
    PxVec3* vel_out, PxVec3* pos_out, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    PxVec3 pi = pos_in[i];
    float  mi = mass_in[i];
    PxVec3 acc(0, 0, 0);

    for (int j = 0; j < n; j++) {
        if (i == j) continue;
        PxVec3 d = pos_in[j] - pi;
        float distSq = d.magnitudeSquared() + EPS * EPS;
        float invDist = 1.0f / sqrtf(distSq);
        float f = G * mass_in[j] * invDist * invDist * invDist;
        acc = acc + d * f;
    }

    PxVec3 vel = vel_out[i] + acc * DT;
    vel_out[i] = vel;
    pos_out[i] = pi + vel * DT;
}

// ---- CPU reference: single-precision all-pairs gravity ----
void nbody_cpu(const PxVec3* pos, const float* mass,
               PxVec3* vel, PxVec3* pos_out, int n)
{
    for (int i = 0; i < n; i++) {
        PxVec3 pi = pos[i];
        PxVec3 acc(0, 0, 0);
        for (int j = 0; j < n; j++) {
            if (i == j) continue;
            PxVec3 d = pos[j] - pi;
            float distSq = d.magnitudeSquared() + EPS * EPS;
            float invDist = 1.0f / sqrtf(distSq);
            float f = G * mass[j] * invDist * invDist * invDist;
            acc = acc + d * f;
        }
        PxVec3 vv = vel[i] + acc * DT;
        vel[i] = vv;
        pos_out[i] = pi + vv * DT;
    }
}

int main()
{
    printf("========================================\n");
    printf(" Example 1: N-Body Gravity (%d bodies)\n", N);
    printf("========================================\n\n");

    // Allocate GPU memory
    PxVec3 *d_pos, *d_vel, *d_pos2;
    float  *d_mass;
    hipMalloc(&d_pos,  N * sizeof(PxVec3));
    hipMalloc(&d_vel,  N * sizeof(PxVec3));
    hipMalloc(&d_pos2, N * sizeof(PxVec3));
    hipMalloc(&d_mass, N * sizeof(float));

    // Initialize: random positions in a sphere, random masses
    PxVec3 *h_pos  = new PxVec3[N];
    PxVec3 *h_vel  = new PxVec3[N];
    float  *h_mass = new float[N];

    srand(42);  // fixed seed for reproducibility
    for (int i = 0; i < N; i++) {
        float theta = rand() / (float)RAND_MAX * 6.28318f;
        float phi   = rand() / (float)RAND_MAX * 3.14159f;
        float r     = rand() / (float)RAND_MAX * 50.0f;
        h_pos[i]  = PxVec3(r * sinf(phi) * cosf(theta), r * sinf(phi) * sinf(theta), r * cosf(phi));
        h_vel[i]  = PxVec3(0, 0, 0);
        h_mass[i] = 0.1f + rand() / (float)RAND_MAX * 10.0f;
    }

    hipMemcpy(d_pos,  h_pos,  N * sizeof(PxVec3), hipMemcpyHostToDevice);
    hipMemcpy(d_vel,  h_vel,  N * sizeof(PxVec3), hipMemcpyHostToDevice);
    hipMemcpy(d_mass, h_mass, N * sizeof(float),  hipMemcpyHostToDevice);

    // Benchmark GPU
    hipEvent_t start, stop;
    hipEventCreate(&start); hipEventCreate(&stop);

    hipEventRecord(start);
    int block = 256, grid = (N + block - 1) / block;
    nbody_gravity<<<grid, block>>>(d_pos, d_mass, d_vel, d_pos2, N);
    hipEventRecord(stop);
    hipDeviceSynchronize();

    float ms_gpu;
    hipEventElapsedTime(&ms_gpu, start, stop);

    // Download GPU results
    PxVec3 *h_pos_gpu = new PxVec3[N];
    PxVec3 *h_vel_gpu = new PxVec3[N];
    hipMemcpy(h_pos_gpu, d_pos2, N * sizeof(PxVec3), hipMemcpyDeviceToHost);
    hipMemcpy(h_vel_gpu, d_vel,  N * sizeof(PxVec3), hipMemcpyDeviceToHost);

    // CPU reference
    PxVec3 *h_pos_cpu = new PxVec3[N];
    PxVec3 *h_vel_cpu = new PxVec3[N];
    memcpy(h_pos_cpu, h_pos, N * sizeof(PxVec3));
    memcpy(h_vel_cpu, h_vel, N * sizeof(PxVec3));
    nbody_cpu(h_pos_cpu, h_mass, h_vel_cpu, h_pos_cpu, N);

    // Compare GPU vs CPU
    double maxDiff = 0, sumDiff = 0;
    for (int i = 0; i < N; i++) {
        PxVec3 diff = h_pos_gpu[i] - h_pos_cpu[i];
        double d = sqrt(diff.magnitudeSquared());
        sumDiff += d;
        if (d > maxDiff) maxDiff = d;
    }
    double avgDiff = sumDiff / N;

    printf("GPU time:      %.2f ms\n", ms_gpu);
    printf("Max position diff:  %.6e\n", maxDiff);
    printf("Avg position diff:  %.6e\n", avgDiff);

    bool pass = (maxDiff < 1e-3f) && (avgDiff < 1e-4f);
    if (pass) {
        printf("\n[PASS] GPU N-body matches CPU reference (max err < 0.001)\n");
    } else {
        printf("\n[FAIL] Max error = %.6e exceeds tolerance\n", maxDiff);
    }

    // Cleanup
    delete[] h_pos; delete[] h_vel; delete[] h_mass;
    delete[] h_pos_gpu; delete[] h_vel_gpu; delete[] h_pos_cpu; delete[] h_vel_cpu;
    hipFree(d_pos); hipFree(d_vel); hipFree(d_pos2); hipFree(d_mass);
    hipEventDestroy(start); hipEventDestroy(stop);

    return pass ? 0 : 1;
}
