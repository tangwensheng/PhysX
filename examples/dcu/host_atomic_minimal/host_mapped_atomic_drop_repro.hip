#include <hip/hip_runtime.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <unistd.h>

#define HIP_CHECK(call)                                                        \
    do                                                                         \
    {                                                                          \
        const hipError_t error = (call);                                        \
        if (error != hipSuccess)                                                \
        {                                                                      \
            std::fprintf(stderr, "%s failed: %s\n", #call,                   \
                hipGetErrorString(error));                                      \
            return 2;                                                          \
        }                                                                      \
    } while (0)

__global__ void updateHighWaterMark(
    unsigned int* hostMappedCounter,
    const unsigned int* deviceCandidate)
{
    if (blockIdx.x == 0 && threadIdx.x == 0)
        atomicMax(hostMappedCounter, *deviceCandidate);
}

int main(int argc, char** argv)
{
    const char* confirmation = std::getenv("DCU_ALLOW_HOST_ATOMIC_DROP");
    if (!confirmation || std::strcmp(confirmation, "I_UNDERSTAND") != 0)
    {
        std::fprintf(stderr,
            "Refusing to launch a GPU atomic against mapped host memory.\n"
            "This test may SIGKILL the process and reset the selected DCU.\n"
            "Set DCU_ALLOW_HOST_ATOMIC_DROP=I_UNDERSTAND to run it.\n");
        return 64;
    }

    const unsigned int holdSeconds = argc > 1
        ? static_cast<unsigned int>(std::strtoul(argv[1], nullptr, 10))
        : 20u;

    int logicalDevice = 0;
    hipDeviceProp_t properties{};
    HIP_CHECK(hipGetDevice(&logicalDevice));
    HIP_CHECK(hipGetDeviceProperties(&properties, logicalDevice));

    unsigned int* hostCounter = nullptr;
    unsigned int* gpuMappedPointer = nullptr;
    unsigned int* deviceCandidate = nullptr;
    HIP_CHECK(hipHostMalloc(reinterpret_cast<void**>(&hostCounter),
        sizeof(*hostCounter), hipHostMallocMapped | hipHostMallocPortable));
    HIP_CHECK(hipMalloc(reinterpret_cast<void**>(&deviceCandidate),
        sizeof(*deviceCandidate)));

    const unsigned int expected = 1234u;
    *hostCounter = 0;
    HIP_CHECK(hipHostGetDevicePointer(reinterpret_cast<void**>(&gpuMappedPointer),
        hostCounter, 0));
    HIP_CHECK(hipMemcpy(deviceCandidate, &expected, sizeof(expected),
        hipMemcpyHostToDevice));

    std::printf("[REPRO] logical_device=%d name=%s arch=%s\n",
        logicalDevice, properties.name, properties.gcnArchName);
    std::printf("[REPRO] atomic_requester=DCU kernel\n");
    std::printf("[REPRO] atomic_target_cpu_host_backing=%p gpu_mapped_address=%p\n",
        static_cast<void*>(hostCounter), static_cast<void*>(gpuMappedPointer));
    std::printf("[REPRO] atomic_value_gpu_device_buffer=%p value=%u\n",
        static_cast<void*>(deviceCandidate), expected);
    std::printf("[REPRO] path=DCU load(deviceCandidate in VRAM) -> "
        "atomicMax(mapped CPU RAM) -> PCIe AtomicOp\n");
    std::fflush(stdout);

    updateHighWaterMark<<<1, 1>>>(gpuMappedPointer, deviceCandidate);
    HIP_CHECK(hipGetLastError());

    const hipError_t syncResult = hipDeviceSynchronize();
    std::printf("[REPRO] sync=%d (%s) phase=hold seconds=%u\n",
        static_cast<int>(syncResult), hipGetErrorString(syncResult), holdSeconds);
    std::fflush(stdout);

    sleep(holdSeconds);

    const unsigned int observed = *hostCounter;
    const bool committed = observed == expected;
    std::printf("[REPRO] survived hold counter=%u expected=%u\n",
        observed, expected);
    std::printf("[REPRO] verdict=%s\n", committed
        ? "ATOMIC_COMMITTED_NO_DEVICE_DROP"
        : "ATOMIC_REJECTED_WITHOUT_DEVICE_DROP");
    std::fflush(stdout);
    HIP_CHECK(hipFree(deviceCandidate));
    HIP_CHECK(hipHostFree(hostCounter));
    if (syncResult != hipSuccess)
        return 3;
    return committed ? 0 : 5;
}
