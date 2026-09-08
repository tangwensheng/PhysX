#include <hip/hip_runtime.h>

#include <cstdio>

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
    unsigned int* deviceCounter,
    const unsigned int* deviceCandidate)
{
    if (blockIdx.x == 0 && threadIdx.x == 0)
        atomicMax(deviceCounter, *deviceCandidate);
}

int main()
{
    int logicalDevice = 0;
    hipDeviceProp_t properties{};
    HIP_CHECK(hipGetDevice(&logicalDevice));
    HIP_CHECK(hipGetDeviceProperties(&properties, logicalDevice));

    unsigned int* deviceCounter = nullptr;
    unsigned int* deviceCandidate = nullptr;
    unsigned int* hostReadback = nullptr;
    hipStream_t stream = nullptr;

    HIP_CHECK(hipMalloc(reinterpret_cast<void**>(&deviceCounter),
        sizeof(*deviceCounter)));
    HIP_CHECK(hipMalloc(reinterpret_cast<void**>(&deviceCandidate),
        sizeof(*deviceCandidate)));
    HIP_CHECK(hipHostMalloc(reinterpret_cast<void**>(&hostReadback),
        sizeof(*hostReadback), hipHostMallocMapped | hipHostMallocPortable));
    HIP_CHECK(hipStreamCreate(&stream));

    const unsigned int expected = 1234u;
    *hostReadback = 0;
    HIP_CHECK(hipMemsetAsync(deviceCounter, 0, sizeof(*deviceCounter), stream));
    HIP_CHECK(hipMemcpyAsync(deviceCandidate, &expected, sizeof(expected),
        hipMemcpyHostToDevice, stream));

    std::printf("[FIXED] logical_device=%d name=%s arch=%s\n",
        logicalDevice, properties.name, properties.gcnArchName);
    std::printf("[FIXED] atomic_requester=DCU kernel\n");
    std::printf("[FIXED] atomic_target_gpu_device_buffer=%p\n",
        static_cast<void*>(deviceCounter));
    std::printf("[FIXED] atomic_value_gpu_device_buffer=%p value=%u\n",
        static_cast<void*>(deviceCandidate), expected);
    std::printf("[FIXED] host_readback_cpu_pinned=%p\n",
        static_cast<void*>(hostReadback));
    std::printf("[FIXED] path=DCU load(deviceCandidate in VRAM) -> "
        "atomicMax(deviceCounter in VRAM) -> DtoH copy -> CPU\n");
    std::fflush(stdout);

    updateHighWaterMark<<<1, 1, 0, stream>>>(deviceCounter, deviceCandidate);
    HIP_CHECK(hipGetLastError());
    HIP_CHECK(hipMemcpyDtoHAsync(hostReadback,
        reinterpret_cast<hipDeviceptr_t>(deviceCounter), sizeof(*hostReadback), stream));
    HIP_CHECK(hipStreamSynchronize(stream));

    std::printf("[FIXED] readback=%u expected=%u verdict=%s\n",
        *hostReadback, expected, *hostReadback == expected ? "PASS" : "FAIL");

    const bool passed = *hostReadback == expected;
    HIP_CHECK(hipStreamDestroy(stream));
    HIP_CHECK(hipHostFree(hostReadback));
    HIP_CHECK(hipFree(deviceCandidate));
    HIP_CHECK(hipFree(deviceCounter));
    return passed ? 0 : 4;
}
