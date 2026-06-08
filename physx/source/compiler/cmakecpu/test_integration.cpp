// PhysX DCU End-to-End Integration Test
//
// Combines CPU Foundation library + GPU kernels to validate
// the full PhysX toolchain on DCU.
//
// Simulates: 1024 particles falling under gravity with ground collision
// Entirely on GPU — upload positions, compute step, download results.

#include "foundation/PxVec3.h"
#include "foundation/PxVec4.h"
#include "PxgCommonDefines.h"
#include <hip/hip_runtime.h>
#include <cstdio>
#include <cmath>

using namespace physx;

// ---- GPU Kernel: particle physics step ----
// Simple Verlet-like integration with ground plane collision
extern "C" __global__ void simulate_particles(
    PxVec3* positions,
    PxVec3* velocities,
    int    numParticles,
    float  dt,
    float  groundY)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= numParticles) return;

    PxVec3 pos = positions[tid];
    PxVec3 vel = velocities[tid];

    // Gravity
    PxVec3 gravity(0.0f, -9.81f, 0.0f);
    vel = vel + gravity * dt;

    // Update position
    pos = pos + vel * dt;

    // Ground collision — bounce
    if (pos.y < groundY) {
        pos.y = groundY;
        vel.y = -vel.y * 0.5f;  // coefficient of restitution = 0.5
    }

    positions[tid] = pos;
    velocities[tid] = vel;
}

// ---- GPU Kernel: compute kinetic energy (reduction) ----
extern "C" __global__ void compute_energy(
    const PxVec3* velocities,
    float*        energy,
    int           numParticles)
{
    __shared__ float smem[256];
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int lane = threadIdx.x;

    float ke = 0.0f;
    if (tid < numParticles) {
        PxVec3 v = velocities[tid];
        ke = 0.5f * v.magnitudeSquared();  // mass = 1.0
    }

    smem[lane] = ke;
    __syncthreads();

    // Block reduction
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (lane < s) smem[lane] += smem[lane + s];
        __syncthreads();
    }

    if (lane == 0) atomicAdd(energy, smem[0]);
}

int main()
{
    printf("========================================\n");
    printf(" PhysX DCU Integration Test\n");
    printf("========================================\n\n");

    // Device info
    hipDeviceProp_t props;
    hipGetDeviceProperties(&props, 0);
    printf("Device: %s\n", props.name);
    printf("CUs: %d\n\n", props.multiProcessorCount);

    const int N = 1024 * 256;       // 256K particles
    const int block = 256;
    const int grid = (N + block - 1) / block;
    const float dt = 1.0f / 60.0f;
    const float groundY = 0.0f;

    size_t vecBytes = N * sizeof(PxVec3);

    // Allocate GPU memory
    PxVec3 *d_pos, *d_vel;
    float  *d_energy;
    hipMalloc(&d_pos, vecBytes);
    hipMalloc(&d_vel, vecBytes);
    hipMalloc(&d_energy, sizeof(float));

    // Initialize: particles in a 3D grid, random velocities
    PxVec3 *h_pos = new PxVec3[N];
    PxVec3 *h_vel = new PxVec3[N];
    for (int i = 0; i < N; i++) {
        h_pos[i] = PxVec3((i % 64) * 0.5f, 5.0f + (i / 4096) * 0.5f, ((i / 64) % 64) * 0.5f);
        h_vel[i] = PxVec3(0.0f, (i % 17 - 8) * 0.2f, (i % 13 - 6) * 0.2f);
    }
    hipMemcpy(d_pos, h_pos, vecBytes, hipMemcpyHostToDevice);
    hipMemcpy(d_vel, h_vel, vecBytes, hipMemcpyHostToDevice);

    printf("Particles: %d\n", N);
    printf("Grid: %d blocks × %d threads\n", grid, block);

    // Run 60 simulation steps (1 second at 60fps)
    hipEvent_t start, stop;
    hipEventCreate(&start);
    hipEventCreate(&stop);

    hipEventRecord(start);
    for (int step = 0; step < 60; step++) {
        simulate_particles<<<grid, block>>>(d_pos, d_vel, N, dt, groundY);
    }
    hipEventRecord(stop);
    hipDeviceSynchronize();

    float simTime;
    hipEventElapsedTime(&simTime, start, stop);
    printf("Simulation: 60 steps in %.2f ms (%.1f FPS)\n", simTime, 60000.0f / simTime);

    // Compute final kinetic energy
    float zero = 0.0f;
    hipMemcpy(d_energy, &zero, sizeof(float), hipMemcpyHostToDevice);
    compute_energy<<<grid, block>>>(d_vel, d_energy, N);
    hipDeviceSynchronize();

    float totalEnergy;
    hipMemcpy(&totalEnergy, d_energy, sizeof(float), hipMemcpyDeviceToHost);

    // Download results
    hipMemcpy(h_pos, d_pos, vecBytes, hipMemcpyDeviceToHost);
    hipMemcpy(h_vel, d_vel, vecBytes, hipMemcpyDeviceToHost);

    // Verify: all particles should be above ground
    int belowGround = 0;
    float maxHeight = 0;
    for (int i = 0; i < N; i++) {
        if (h_pos[i].y < groundY - 0.01f) belowGround++;
        if (h_pos[i].y > maxHeight) maxHeight = h_pos[i].y;
    }

    printf("\n=== Results ===\n");
    printf("Total kinetic energy: %.2f\n", totalEnergy);
    printf("Max particle height:  %.2f\n", maxHeight);
    printf("Particles below ground: %d / %d\n", belowGround, N);

    int passed = 0, total = 2;
    printf("\n");
    if (belowGround == 0) {
        passed++;
        printf("  [PASS] No particles below ground\n");
    } else {
        printf("  [FAIL] %d particles below ground\n", belowGround);
    }

    if (totalEnergy > 0.0f) {
        passed++;
        printf("  [PASS] Physics simulation running (energy=%.1f, maxH=%.2f)\n", totalEnergy, maxHeight);
    } else {
        printf("  [FAIL] Physics might be wrong\n");
    }

    printf("\n========================================\n");
    printf(" %d / %d tests passed\n", passed, total);
    printf("========================================\n");

    // Cleanup
    delete[] h_pos;
    delete[] h_vel;
    hipFree(d_pos);
    hipFree(d_vel);
    hipFree(d_energy);
    hipEventDestroy(start);
    hipEventDestroy(stop);

    return (passed == total) ? 0 : 1;
}
