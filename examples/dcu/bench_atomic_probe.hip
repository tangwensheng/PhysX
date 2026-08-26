// Standalone probe: does atomicAdd trip this platform's PCIe AtomicOp support?
//
// Motivation: dmesg reports "hycu: Found atomic err(0x1) in pcie:UR_ATOMIC_OPCODE"
// when running the PhysX complex-trimesh workload. The PhysX kernels only ever
// perform atomics on plain device memory, so this probe isolates each memory kind
// and reports which one (if any) actually triggers the fault.
//
// Modes (pick one per run so a device drop unambiguously identifies the culprit):
//   --mode device   atomicAdd on hipMalloc memory                     (expected safe)
//   --mode mapped   atomicAdd on hipHostMalloc(...Mapped) memory      (PCIe atomic)
//   --mode managed  atomicAdd on hipMallocManaged memory
//   --mode store    plain store to mapped memory, no atomic           (control)
//
// Widths: --width 32 (default) or 64, since 64-bit PCIe atomics are a separate
// capability bit from 32-bit ones.

#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#define HIP_CHECK(x) do { \
    hipError_t err__ = (x); \
    if (err__ != hipSuccess) { \
        std::fprintf(stderr, "HIP error %s:%d: %s -> %s\n", __FILE__, __LINE__, #x, hipGetErrorString(err__)); \
        return 2; \
    } \
} while (0)

__global__ void atomicAdd32(unsigned int* counter, unsigned int perThread)
{
    atomicAdd(counter, perThread);
}

__global__ void atomicAdd64(unsigned long long* counter, unsigned long long perThread)
{
    atomicAdd(counter, perThread);
}

// Control: the same access pattern without any atomic. Mirrors what
// bpSignalComplete does in PhysX (a plain store into host-mapped memory).
__global__ void plainStore32(unsigned int* target, unsigned int value)
{
    if (threadIdx.x == 0 && blockIdx.x == 0)
        *target = value;
}

// Warp-aggregated variant: what PhysX's midphase actually does, i.e. one atomic
// per warp rather than one per thread.
__global__ void atomicAdd32Aggregated(unsigned int* counter, unsigned int perThread)
{
    const unsigned int lane = threadIdx.x & (warpSize - 1);
    unsigned int contribution = perThread * warpSize;
    if (lane == 0)
        atomicAdd(counter, contribution);
}

// Closer to what PhysX actually does: the atomic sits behind a force-inlined
// __device__ helper, is reached from a nested call, and only a subset of the
// wavefront is active when it executes. Inlining does not change the memory
// transaction an atomic produces, but the divergence and nesting do change how
// the compiler schedules it.
__forceinline__ __device__ void bumpCounter(unsigned int* counter, unsigned int value)
{
    atomicAdd(counter, value);
}

__forceinline__ __device__ void innerTraversalStep(unsigned int* counter, unsigned int depth, bool active)
{
    // Mimic a ballot-driven warp-aggregated allocation inside a traversal loop.
    const unsigned long long mask = __ballot(active ? 1 : 0);
    const unsigned int validCount = (unsigned int)__popcll(mask);
    const unsigned int lane = threadIdx.x & (warpSize - 1);

    if (lane == 0 && validCount > 0)
        bumpCounter(counter, validCount * depth);
}

__global__ void atomicDivergentNested(unsigned int* counter, unsigned int levels)
{
    // Only part of each wavefront stays active, and the atomic is issued from
    // inside a loop nested two calls deep.
    for (unsigned int d = 1; d <= levels; ++d)
    {
        const bool active = ((threadIdx.x + d) & 3u) == 0u;
        innerTraversalStep(counter, 1u, active);
    }
}


int main(int argc, char** argv)
{
    int device = 0;
    const char* mode = "device";
    int width = 32;
    int blocks = 64;
    int threads = 256;
    int iters = 10000;
    int progressEvery = 1000;
    int aggregated = 0;
    int divergent = 0;
    unsigned int levels = 8;

    for (int i = 1; i < argc; ++i)
    {
        auto need = [&](const char* name) -> const char* {
            if (i + 1 >= argc) { std::fprintf(stderr, "missing value for %s\n", name); std::exit(1); }
            return argv[++i];
        };
        if (!std::strcmp(argv[i], "--device")) device = std::atoi(need("--device"));
        else if (!std::strcmp(argv[i], "--mode")) mode = need("--mode");
        else if (!std::strcmp(argv[i], "--width")) width = std::atoi(need("--width"));
        else if (!std::strcmp(argv[i], "--blocks")) blocks = std::atoi(need("--blocks"));
        else if (!std::strcmp(argv[i], "--threads")) threads = std::atoi(need("--threads"));
        else if (!std::strcmp(argv[i], "--iters")) iters = std::atoi(need("--iters"));
        else if (!std::strcmp(argv[i], "--progress-every")) progressEvery = std::atoi(need("--progress-every"));
        else if (!std::strcmp(argv[i], "--levels")) levels = (unsigned int)std::atoi(need("--levels"));
        else if (!std::strcmp(argv[i], "--aggregated")) aggregated = 1;
        else if (!std::strcmp(argv[i], "--divergent")) divergent = 1;
        else if (!std::strcmp(argv[i], "--help"))
        {
            std::printf("usage: %s [--device N] [--mode device|mapped|managed|store]\n"
                        "          [--width 32|64] [--blocks N] [--threads N] [--iters N]\n"
                        "          [--aggregated] [--divergent] [--levels N]\n"
                        "          [--progress-every N]\n", argv[0]);
            return 0;
        }
        else { std::fprintf(stderr, "unknown arg: %s\n", argv[i]); return 1; }
    }

    HIP_CHECK(hipSetDevice(device));

    hipDeviceProp_t props;
    HIP_CHECK(hipGetDeviceProperties(&props, device));
    std::printf("device %d: %s (gcn=%s, warpSize=%d)\n", device, props.name, props.gcnArchName, props.warpSize);
    std::printf("mode=%s width=%d blocks=%d threads=%d iters=%d aggregated=%d\n",
                mode, width, blocks, threads, iters, aggregated, divergent, levels);

    const bool isMapped  = !std::strcmp(mode, "mapped");
    const bool isManaged = !std::strcmp(mode, "managed");
    const bool isStore   = !std::strcmp(mode, "store");
    const bool isDevice  = !std::strcmp(mode, "device");
    if (!isMapped && !isManaged && !isStore && !isDevice)
    {
        std::fprintf(stderr, "bad --mode: %s\n", mode);
        return 1;
    }

    // 'store' reuses the mapped allocation; it is the no-atomic control for it.
    const bool wantMapped = isMapped || isStore;

    void* hostPtr = nullptr;   // host-side pointer, mapped/managed only
    void* devPtr  = nullptr;   // pointer handed to the kernel
    const size_t bytes = (width == 64) ? sizeof(unsigned long long) : sizeof(unsigned int);

    if (wantMapped)
    {
        HIP_CHECK(hipHostMalloc(&hostPtr, bytes, hipHostMallocMapped));
        std::memset(hostPtr, 0, bytes);
        HIP_CHECK(hipHostGetDevicePointer(&devPtr, hostPtr, 0));
        std::printf("host-mapped alloc: host=%p device=%p\n", hostPtr, devPtr);
    }
    else if (isManaged)
    {
        HIP_CHECK(hipMallocManaged(&devPtr, bytes));
        std::memset(devPtr, 0, bytes);
        hostPtr = devPtr;
        std::printf("managed alloc: %p\n", devPtr);
    }
    else
    {
        HIP_CHECK(hipMalloc(&devPtr, bytes));
        HIP_CHECK(hipMemset(devPtr, 0, bytes));
        std::printf("device alloc: %p\n", devPtr);
    }

    std::printf("launching...\n");
    std::fflush(stdout);

    for (int it = 0; it < iters; ++it)
    {
        if (isStore)
        {
            hipLaunchKernelGGL(plainStore32, dim3(blocks), dim3(threads), 0, 0,
                               reinterpret_cast<unsigned int*>(devPtr), (unsigned int)(it + 1));
        }
        else if (width == 64)
        {
            hipLaunchKernelGGL(atomicAdd64, dim3(blocks), dim3(threads), 0, 0,
                               reinterpret_cast<unsigned long long*>(devPtr), 1ull);
        }
        else if (divergent)
        {
            hipLaunchKernelGGL(atomicDivergentNested, dim3(blocks), dim3(threads), 0, 0,
                               reinterpret_cast<unsigned int*>(devPtr), levels);
        }
        else if (aggregated)
        {
            hipLaunchKernelGGL(atomicAdd32Aggregated, dim3(blocks), dim3(threads), 0, 0,
                               reinterpret_cast<unsigned int*>(devPtr), 1u);
        }
        else
        {
            hipLaunchKernelGGL(atomicAdd32, dim3(blocks), dim3(threads), 0, 0,
                               reinterpret_cast<unsigned int*>(devPtr), 1u);
        }

        hipError_t launchErr = hipGetLastError();
        if (launchErr != hipSuccess)
        {
            std::fprintf(stderr, "launch failed at iter %d: %s\n", it, hipGetErrorString(launchErr));
            return 3;
        }

        HIP_CHECK(hipDeviceSynchronize());

        if (progressEvery > 0 && ((it + 1) % progressEvery) == 0)
        {
            std::printf("  iter %d/%d ok\n", it + 1, iters);
            std::fflush(stdout);
        }
    }

    // Read the counter back and check it against the expected total.
    unsigned long long got = 0;
    if (wantMapped || isManaged)
    {
        got = (width == 64) ? *reinterpret_cast<unsigned long long*>(hostPtr)
                            : *reinterpret_cast<unsigned int*>(hostPtr);
    }
    else
    {
        if (width == 64)
        {
            unsigned long long tmp = 0;
            HIP_CHECK(hipMemcpy(&tmp, devPtr, bytes, hipMemcpyDeviceToHost));
            got = tmp;
        }
        else
        {
            unsigned int tmp = 0;
            HIP_CHECK(hipMemcpy(&tmp, devPtr, bytes, hipMemcpyDeviceToHost));
            got = tmp;
        }
    }

    int rc = 0;
    if (isStore)
    {
        const unsigned long long expected = (unsigned long long)iters;
        std::printf("result: got=%llu expected=%llu\n", got, expected);
        rc = (got == expected) ? 0 : 4;
    }
    else
    {
        // Every mode adds one per active thread per iteration, just via different
        // paths: plain and aggregated both total blocks*threads, while divergent
        // has each wavefront contribute popc(mask) once per level.
        unsigned long long perIter = (unsigned long long)blocks * (unsigned long long)threads;
        if (divergent)
        {
            // Per level, one atomic per wavefront adding popc(active mask).
            // Exactly a quarter of the lanes match ((tid + d) & 3) == 0.
            perIter = (unsigned long long)blocks * (unsigned long long)threads / 4ull
                    * (unsigned long long)levels;
        }
        unsigned long long expected = perIter * (unsigned long long)iters;
        if (width == 32) expected &= 0xFFFFFFFFull;   // 32-bit counter wraps
        std::printf("result: got=%llu expected=%llu\n", got, expected);
        rc = (got == expected) ? 0 : 4;
    }

    std::printf("VERDICT: %s\n", rc == 0 ? "PASS" : "FAIL");

    if (wantMapped)       HIP_CHECK(hipHostFree(hostPtr));
    else if (isManaged)   HIP_CHECK(hipFree(devPtr));
    else                  HIP_CHECK(hipFree(devPtr));

    return rc;
}
