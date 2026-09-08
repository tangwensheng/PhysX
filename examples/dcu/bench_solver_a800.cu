// A800 Sparse Contact Solver — matches bench_solver.hip
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>

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

    int ci = blkBase + lane;
    if (ci < n) sCon[lane] = contacts[ci];
    else { sCon[lane].dist = 0; sCon[lane].cfm = 1e10f; }
    __syncthreads();

    for (int pass = 0; pass < iters; pass++) {
        float lam = 0;
        if (ci < n) {
            Contact c = sCon[lane];
            float pen = c.dist;
            if (pen < 0) {
                float dlambda = (-pen - lam * c.cfm) / (1.0f + c.cfm);
                lam += dlambda > 0 ? dlambda : 0;
            }
            sLam[lane] = lam;
        }
        __syncthreads();

        float residual = 0;
        if (ci < n) {
            float sumLam = 0;
            for (int k = 0; k < 256; k++) {
                if (blkBase + k >= n) break;
                Contact ck = sCon[k];
                Contact ci_con = sCon[lane];
                float dot = ci_con.nx*ck.nx + ci_con.ny*ck.ny + ci_con.nz*ck.nz;
                sumLam += dot * sLam[k];
            }
            Contact c = sCon[lane];
            float vn = vel[ci*2]*c.nx + vel[ci*2+1]*c.ny;
            residual = fabsf(vn + c.dist + sumLam);
        }
        sRes[lane] = residual;
        __syncthreads();
    }

    if (ci < n) {
        Contact c = sCon[lane];
        float lam = sLam[lane];
        vel[ci*2]   += lam * c.nx;
        vel[ci*2+1] += lam * c.ny;
        pos[ci*2]   += vel[ci*2]   * dt;
        pos[ci*2+1] += vel[ci*2+1] * dt;
    }

    for (int s = 128; s > 0; s >>= 1) {
        if (lane < s && ci + s < n) sRes[lane] += sRes[lane + s];
        __syncthreads();
    }
    if (lane == 0 && blkBase < n) atomicAdd(globalResidual, sRes[0]);
}

int main()
{
    printf("=== A800 Sparse Contact Solver ===\n");
    cudaDeviceProp p; cudaGetDeviceProperties(&p, 0);
    printf("Device: %s, %d SMs\n\n", p.name, p.multiProcessorCount);

    const int N = 65536, BLK = 256, GRD = N/BLK, RND = 50, ITERS = 4;
    Contact *d_con; float *d_vel, *d_pos, *d_res;
    cudaMalloc(&d_con, N*sizeof(Contact));
    cudaMalloc(&d_vel, N*2*sizeof(float));
    cudaMalloc(&d_pos, N*2*sizeof(float));
    cudaMalloc(&d_res, sizeof(float));

    Contact *h_con = new Contact[N];
    float *h_vp = new float[N*4];
    for (int i = 0; i < N; i++) {
        h_con[i] = {(rand()%1000)/10.0f,(rand()%1000)/10.0f,(rand()%1000)/10.0f,
                    (rand()%200-100)/100.0f,(rand()%200-100)/100.0f,(rand()%200-100)/100.0f,
                    (rand()%100-200)/100.0f,(rand()%10)/100.0f+0.001f};
        float len=sqrtf(h_con[i].nx*h_con[i].nx+h_con[i].ny*h_con[i].ny+h_con[i].nz*h_con[i].nz);
        h_con[i].nx/=len;h_con[i].ny/=len;h_con[i].nz/=len;
        h_vp[i*4+0]=(rand()%100-50)/10.0f;h_vp[i*4+1]=(rand()%100-50)/10.0f;
        h_vp[i*4+2]=(rand()%1000)/10.0f;h_vp[i*4+3]=(rand()%1000)/10.0f;
    }
    cudaMemcpy(d_con,h_con,N*sizeof(Contact),cudaMemcpyHostToDevice);
    cudaMemcpy(d_vel,h_vp,N*2*sizeof(float),cudaMemcpyHostToDevice);
    cudaMemcpy(d_pos,h_vp+2,N*2*sizeof(float),cudaMemcpyHostToDevice);

    cudaEvent_t s1,s2;cudaEventCreate(&s1);cudaEventCreate(&s2);
    float ms;

    cudaMemset(d_res,0,sizeof(float));cudaDeviceSynchronize();

    cudaEventRecord(s1);
    for(int r=0;r<RND;r++){
        cudaMemset(d_res,0,sizeof(float));
        solver_block<<<GRD,BLK>>>(d_con,d_vel,d_pos,d_res,N,1.0f/60.0f,ITERS);
    }
    cudaEventRecord(s2);cudaDeviceSynchronize();
    cudaEventElapsedTime(&ms,s1,s2);

    float h_res;cudaMemcpy(&h_res,d_res,sizeof(float),cudaMemcpyDeviceToHost);
    printf("solver_block (%d iters): %.1f us  residual=%.3f\n",ITERS,ms/RND*1000,h_res);

    delete[]h_con;delete[]h_vp;
    cudaFree(d_con);cudaFree(d_vel);cudaFree(d_pos);cudaFree(d_res);
    cudaEventDestroy(s1);cudaEventDestroy(s2);
    return 0;
}
