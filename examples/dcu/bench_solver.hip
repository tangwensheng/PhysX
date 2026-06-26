// PhysX Solver Pattern — sparse contacts, multi-block parallel, atomic reduction
// Matches real solverMultiBlockTGS structure. DCU + A800 fair benchmark.

#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>

// ====== Sparse Contact Solver (real PhysX pattern) ======
//
// N contacts distributed across blocks. Each block solves its own batch
// of 256 contacts using projected Gauss-Seidel. Global residual accumulated
// via atomicAdd for convergence check.
//
// Structure:
//   1. Cooperative load: contacts → shared memory
//   2. Block-level solve: 4 passes of projection
//   3. Atomic reduction: per-block residual → global residual
//   4. Scatter write: apply velocity/position updates

struct Contact { float px,py,pz; float nx,ny,nz; float dist; float cfm; };

extern "C" __global__ void solver_block(
    const Contact* contacts, float* vel, float* pos, float* globalResidual,
    int n, float dt, int iters)
{
    __shared__ Contact sCon[256];
    __shared__ float    sLam[256];
    __shared__ float    sRes[256];

    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int lane = threadIdx.x;
    int blkBase = blockIdx.x * 256;
    if (blkBase >= n) return;

    // ---- Phase 1: load contacts cooperatively ----
    int ci = blkBase + lane;
    if (ci < n) sCon[lane] = contacts[ci];
    else { sCon[lane].dist = 0; sCon[lane].cfm = 1e10f; } // inactive
    __syncthreads();

    // ---- Phase 2: block-level projected solver ----
    for (int pass = 0; pass < iters; pass++) {
        // Load velocity for our contact
        float vx = vel[ci*2], vy = vel[ci*2+1];
        float px = pos[ci*2], py = pos[ci*2+1];
        float lam = 0;

        if (ci < n) {
            Contact c = sCon[lane];
            // Project velocity onto constraint normal
            float vn = vx*c.nx + vy*c.ny + vx*c.nz;  // use vx for vz placeholder
            // Resolve penetration
            float pen = c.dist;
            if (pen < 0) {
                float dlambda = (-pen - lam * c.cfm) / (1.0f + c.cfm);
                lam += dlambda > 0 ? dlambda : 0;
            }
            // Write lambda to shared memory for interleaved GS
            sLam[lane] = lam;
        }
        __syncthreads();

        // Interleave: each contact reads neighbor lambdas for coupled solve
        float residual = 0;
        if (ci < n) {
            float sumLam = 0;
            #pragma unroll 4
            for (int k = 0; k < 256; k++) {
                if (blkBase + k >= n) break;
                Contact ck = sCon[k];
                Contact ci_con = sCon[lane];
                float dot = ci_con.nx*ck.nx + ci_con.ny*ck.ny + ci_con.nz*ck.nz;
                sumLam += dot * sLam[k];
            }
            // Residual = how much constraint is violated
            Contact c = sCon[lane];
            float vn = vel[ci*2]*c.nx + vel[ci*2+1]*c.ny;
            residual = fabsf(vn + c.dist + sumLam);
        }
        sRes[lane] = residual;
        __syncthreads();
    }

    // ---- Phase 3: write back velocity + position ----
    if (ci < n) {
        Contact c = sCon[lane];
        float lam = sLam[lane];
        vel[ci*2]   += lam * c.nx;
        vel[ci*2+1] += lam * c.ny;
        pos[ci*2]   += vel[ci*2]   * dt;
        pos[ci*2+1] += vel[ci*2+1] * dt;
    }

    // ---- Phase 4: block residual reduction ----
    for (int s = 128; s > 0; s >>= 1) {
        if (lane < s && ci + s < n) sRes[lane] += sRes[lane + s];
        __syncthreads();
    }
    if (lane == 0 && blkBase < n) atomicAdd(globalResidual, sRes[0]);
}

int main()
{
    printf("========================================\n");
    printf(" PhysX Sparse Contact Solver\n");
    printf("========================================\n\n");

    hipDeviceProp_t p; hipGetDeviceProperties(&p, 0);
    printf("Device: %s, %d CUs\n\n", p.name, p.multiProcessorCount);

    const int N = 65536, BLK = 256, GRD = N/BLK, RND = 50, ITERS = 4;

    Contact *d_con; float *d_vel, *d_pos, *d_res;
    hipMalloc(&d_con, N * sizeof(Contact));
    hipMalloc(&d_vel, N * 2 * sizeof(float));
    hipMalloc(&d_pos, N * 2 * sizeof(float));
    hipMalloc(&d_res, sizeof(float));

    // Random contacts with penetration
    Contact *h_con = new Contact[N];
    float *h_vp = new float[N*4];
    for (int i = 0; i < N; i++) {
        h_con[i] = {
            (rand()%1000)/10.0f, (rand()%1000)/10.0f, (rand()%1000)/10.0f,  // position
            (rand()%200-100)/100.0f, (rand()%200-100)/100.0f, (rand()%200-100)/100.0f, // normal
            (rand()%100-200)/100.0f,  // penetration (negative = overlapping)
            (rand()%10)/100.0f + 0.001f // cfm
        };
        // normalize normal
        float len = sqrtf(h_con[i].nx*h_con[i].nx + h_con[i].ny*h_con[i].ny + h_con[i].nz*h_con[i].nz);
        h_con[i].nx /= len; h_con[i].ny /= len; h_con[i].nz /= len;
        h_vp[i*4+0] = (rand()%100-50)/10.0f;
        h_vp[i*4+1] = (rand()%100-50)/10.0f;
        h_vp[i*4+2] = (rand()%1000)/10.0f;
        h_vp[i*4+3] = (rand()%1000)/10.0f;
    }
    hipMemcpy(d_con, h_con, N*sizeof(Contact), hipMemcpyHostToDevice);
    hipMemcpy(d_vel, h_vp, N*2*sizeof(float), hipMemcpyHostToDevice);
    hipMemcpy(d_pos, h_vp+2, N*2*sizeof(float), hipMemcpyHostToDevice);

    hipEvent_t s1, s2; hipEventCreate(&s1); hipEventCreate(&s2);
    float ms;

    // Benchmark
    hipMemset(d_res, 0, sizeof(float));
    hipDeviceSynchronize();

    hipEventRecord(s1);
    for (int r = 0; r < RND; r++) {
        hipMemset(d_res, 0, sizeof(float));
        solver_block<<<GRD, BLK>>>(d_con, d_vel, d_pos, d_res, N, 1.0f/60.0f, ITERS);
    }
    hipEventRecord(s2);
    hipDeviceSynchronize();
    hipEventElapsedTime(&ms, s1, s2);

    float h_res;
    hipMemcpy(&h_res, d_res, sizeof(float), hipMemcpyDeviceToHost);
    printf("solver_block (%d iters): %.1f us (avg)  residual=%.3f\n", ITERS, ms/RND*1000, h_res);

    printf("\nRun: hipprof ./build/bench_solver\n");

    delete[] h_con; delete[] h_vp;
    hipFree(d_con); hipFree(d_vel); hipFree(d_pos); hipFree(d_res);
    hipEventDestroy(s1); hipEventDestroy(s2);
    return 0;
}
