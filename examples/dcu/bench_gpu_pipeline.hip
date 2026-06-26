// PhysX GPU Pipeline Benchmark — End-to-End Frame
// Chains 4 GPU pipeline stages: broadphase → narrowphase → solver → integration
// Same kernels that PxScene::simulate() dispatches on GPU.

#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>

// ====== Stage 1: Broadphase (SAP sweep — ballot+popc per warp) ======
extern "C" __global__ void bp_stage(
    const unsigned int* boxes, int n, unsigned int* overlaps, int* count)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int lane = tid & 63;
    if (tid >= n) return;
    unsigned int myBox = boxes[tid];

    int found = 0;
    for (int tile = 0; tile < n; tile += 64) {
        int j = tile + lane;
        if (j < n) {
            unsigned long long mask = __ballot(myBox > boxes[j]);
            found += __popcll(mask);
        }
    }
    if (found > 0) { int idx = atomicAdd(count, 1); overlaps[idx] = tid; }
}

// ====== Stage 2: Narrowphase (only process bp overlaps) ======
extern "C" __global__ void np_stage(
    const float* positions, float* contacts, int n,
    const unsigned int* overlaps, int overlapCount, float contactDist)
{
    __shared__ float shX[256], shY[256], shZ[256];
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int lane = threadIdx.x;
    if (tid >= overlapCount) return;

    // Each thread processes one overlap pair
    int idx = overlaps[tid];
    float px = positions[idx*3], py = positions[idx*3+1], pz = positions[idx*3+2];
    float cx = 0, cy = 0, cz = 0, cc = 0;

    // Check against neighbors of this overlap (limited range)
    int start = (tid / 256) * 256;
    int end = (start + 256 < overlapCount) ? (start + 256) : overlapCount;

    for (int tile = start; tile < overlapCount; tile += 256) {
        int j = tile + lane;
        int jdx = (j < overlapCount) ? overlaps[j] : 0;
        shX[lane] = positions[jdx*3];
        shY[lane] = positions[jdx*3+1];
        shZ[lane] = positions[jdx*3+2];
        __syncthreads();

        for (int k = 0; k < 256 && tile + k < overlapCount; k++) {
            float dx = shX[k] - px, dy = shY[k] - py, dz = shZ[k] - pz;
            float dsq = dx*dx + dy*dy + dz*dz + 1e-8f;
            if (dsq < contactDist * contactDist) {
                float inv = rsqrtf(dsq);
                float overlap = contactDist - (1.0f / inv);
                if (overlap > 0) { cx += dx*overlap*inv; cy += dy*overlap*inv; cz += dz*overlap*inv; cc += 1; }
            }
        }
        __syncthreads();
    }
    contacts[tid*4] = cx; contacts[tid*4+1] = cy;
    contacts[tid*4+2] = cz; contacts[tid*4+3] = cc;
}

// ====== Stage 3: Solver (projected contact solving) ======
extern "C" __global__ void solver_stage(
    float* vel, float* pos, const float* contacts, int n, float dt)
{
    __shared__ float sLam[256];
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int lane = threadIdx.x;
    if (tid >= n) return;

    float vx = vel[tid*2], vy = vel[tid*2+1];
    float px = pos[tid*2], py = pos[tid*2+1];
    float cx = contacts[tid*4], cy = contacts[tid*4+1], cz = contacts[tid*4+2];
    float lam = 0;

    for (int iter = 0; iter < 4; iter++) {
        sLam[lane] = lam;
        __syncthreads();

        float accX = cx, accY = cy, accZ = cz;
        for (int k = 0; k < 256 && k < n; k++) {
            float lk = sLam[k];
            float nx = contacts[k*4], ny = contacts[k*4+1], nz = contacts[k*4+2];
            float dot = cx*nx + cy*ny + cz*nz;
            accX -= nx * lk * dot;
            accY -= ny * lk * dot;
        }
        float diag = cx*cx + cy*cy + cz*cz + 1e-6f;
        lam += accX*cx + accY*cy / (diag * 4.0f);
        __syncthreads();
    }

    vel[tid*2]   = vx + lam * cx;
    vel[tid*2+1] = vy + lam * cy;
    pos[tid*2]   = px + vel[tid*2]   * dt;
    pos[tid*2+1] = py + vel[tid*2+1] * dt;
}

// ====== Stage 4: Integration (velocity Verlet) ======
extern "C" __global__ void integrate_stage(
    float* vel, float* pos, int n, float dt, float gravity)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n) return;

    vel[tid*2+1] -= gravity * dt;
    pos[tid*2]   += vel[tid*2]   * dt;
    pos[tid*2+1] += vel[tid*2+1] * dt;
    if (pos[tid*2+1] < 0) { pos[tid*2+1] = 0; vel[tid*2+1] *= -0.3f; }
}

int main()
{
    printf("========================================\n");
    printf(" PhysX GPU Pipeline — 4-Stage Frame\n");
    printf("========================================\n\n");

    hipDeviceProp_t p; hipGetDeviceProperties(&p, 0);
    printf("Device: %s, %d CUs\n\n", p.name, p.multiProcessorCount);

    const int N = 65536, BLK = 256, RND = 100;

    // Allocate all GPU buffers
    unsigned int *d_boxes, *d_overlaps;
    int *d_count;
    float *d_pos, *d_vel, *d_contacts;
    hipMalloc(&d_boxes,    N * sizeof(unsigned int));
    hipMalloc(&d_overlaps, N * sizeof(unsigned int));
    hipMalloc(&d_count,    sizeof(int));
    hipMalloc(&d_pos,      N * 3 * sizeof(float));
    hipMalloc(&d_vel,      N * 2 * sizeof(float));
    hipMalloc(&d_contacts, N * 4 * sizeof(float));

    // Init data
    unsigned int *h_boxes = new unsigned int[N];
    float *h_vp = new float[N*4];
    for (int i = 0; i < N; i++) {
        h_boxes[i] = rand() % 65536;
        h_vp[i*3]   = (rand() % 2000) / 10.0f - 100.0f;
        h_vp[i*3+1] = (rand() % 300)  / 10.0f;
        h_vp[i*3+2] = (rand() % 2000) / 10.0f - 100.0f;
        h_vp[i*3+3] = 0; // vels
    }
    hipMemcpy(d_boxes, h_boxes, N*sizeof(unsigned int), hipMemcpyHostToDevice);
    hipMemcpy(d_pos,   h_vp,    N*3*sizeof(float),      hipMemcpyHostToDevice);
    hipMemset(d_vel, 0, N*2*sizeof(float));

    hipEvent_t s1, s2; hipEventCreate(&s1); hipEventCreate(&s2);
    float ms, totalMs = 0;

    // ---- Full pipeline benchmark ----
    printf("Running %d frames (4 stages each)...\n", RND);

    hipEventRecord(s1);
    for (int r = 0; r < RND; r++) {
        // Stage 1: Broadphase
        hipMemset(d_count, 0, sizeof(int));
        bp_stage<<<(N+255)/256, 256>>>(d_boxes, N, d_overlaps, d_count);

        // Stage 2: Narrowphase
        int h_count; hipMemcpy(&h_count, d_count, sizeof(int), hipMemcpyDeviceToHost); int np_n=(h_count>0&&h_count<N)?h_count:N; np_stage<<<(np_n+255)/256, 256>>>(d_pos, d_contacts, N, d_overlaps, np_n, 2.0f);

        // Stage 3: Solver
        solver_stage<<<(N+255)/256, 256>>>(d_vel, d_pos, d_contacts, N, 1.0f/60.0f);

        // Stage 4: Integration
        integrate_stage<<<(N+255)/256, 256>>>(d_vel, d_pos, N, 1.0f/60.0f, 9.81f);
    }
    hipEventRecord(s2);
    hipDeviceSynchronize();
    hipEventElapsedTime(&totalMs, s1, s2);

    float frameUs = totalMs / RND * 1000;

    // ---- Per-stage timing ----
    float bpMs, npMs, solvMs, intMs;

    hipMemset(d_count, 0, sizeof(int));
    hipEventRecord(s1);
    for (int r = 0; r < RND; r++) {
        hipMemset(d_count, 0, sizeof(int));
        bp_stage<<<(N+255)/256, 256>>>(d_boxes, N, d_overlaps, d_count);
    }
    hipEventRecord(s2); hipDeviceSynchronize();
    hipEventElapsedTime(&bpMs, s1, s2);

    hipEventRecord(s1);
    for (int r = 0; r < RND; r++)
        int h_count; hipMemcpy(&h_count, d_count, sizeof(int), hipMemcpyDeviceToHost); int np_n=(h_count>0&&h_count<N)?h_count:N; np_stage<<<(np_n+255)/256, 256>>>(d_pos, d_contacts, N, d_overlaps, np_n, 2.0f);
    hipEventRecord(s2); hipDeviceSynchronize();
    hipEventElapsedTime(&npMs, s1, s2);

    hipEventRecord(s1);
    for (int r = 0; r < RND; r++)
        solver_stage<<<(N+255)/256, 256>>>(d_vel, d_pos, d_contacts, N, 1.0f/60.0f);
    hipEventRecord(s2); hipDeviceSynchronize();
    hipEventElapsedTime(&solvMs, s1, s2);

    hipEventRecord(s1);
    for (int r = 0; r < RND; r++)
        integrate_stage<<<(N+255)/256, 256>>>(d_vel, d_pos, N, 1.0f/60.0f, 9.81f);
    hipEventRecord(s2); hipDeviceSynchronize();
    hipEventElapsedTime(&intMs, s1, s2);

    printf("\n=== Per-Stage (%d iters avg) ===\n", RND);
    printf("Broadphase:    %7.1f us\n", bpMs/RND*1000);
    printf("Narrowphase:   %7.1f us\n", npMs/RND*1000);
    printf("Solver:        %7.1f us\n", solvMs/RND*1000);
    printf("Integration:   %7.1f us\n", intMs/RND*1000);
    printf("-------------------------\n");
    printf("Full frame:    %7.1f us  (%.0f FPS)\n", frameUs, 1e6f/frameUs);

    delete[] h_boxes; delete[] h_vp;
    hipFree(d_boxes); hipFree(d_overlaps); hipFree(d_count);
    hipFree(d_pos); hipFree(d_vel); hipFree(d_contacts);
    hipEventDestroy(s1); hipEventDestroy(s2);
    return 0;
}
