#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#define HIP_CHECK(x) do { \
    hipError_t err__ = (x); \
    if (err__ != hipSuccess) { \
        std::fprintf(stderr, "HIP error %s:%d: %s -> %s\n", __FILE__, __LINE__, #x, hipGetErrorString(err__)); \
        return 2; \
    } \
} while (0)

static size_t parseSize(const char* s)
{
    char* end = nullptr;
    double v = std::strtod(s, &end);
    size_t mul = 1;
    if (end && *end)
    {
        if (!std::strcmp(end, "k") || !std::strcmp(end, "K") || !std::strcmp(end, "kb") || !std::strcmp(end, "KB")) mul = 1024ull;
        else if (!std::strcmp(end, "m") || !std::strcmp(end, "M") || !std::strcmp(end, "mb") || !std::strcmp(end, "MB")) mul = 1024ull * 1024ull;
        else if (!std::strcmp(end, "g") || !std::strcmp(end, "G") || !std::strcmp(end, "gb") || !std::strcmp(end, "GB")) mul = 1024ull * 1024ull * 1024ull;
    }
    return size_t(v * double(mul));
}

static int parseInt(const char* s) { return std::atoi(s); }

int main(int argc, char** argv)
{
    int device = 0;
    size_t bytes = 16;
    size_t allocBytes = 256ull * 1024ull * 1024ull;
    int iters = 1000000;
    int asyncCopy = 0;
    int syncEach = 1;
    int rotateOffset = 0;
    int progressEvery = 1000;

    for (int i = 1; i < argc; ++i)
    {
        auto need = [&](const char* name) -> const char* {
            if (i + 1 >= argc) { std::fprintf(stderr, "missing value for %s\n", name); std::exit(1); }
            return argv[++i];
        };
        if (!std::strcmp(argv[i], "--device")) device = parseInt(need(argv[i]));
        else if (!std::strcmp(argv[i], "--bytes")) bytes = parseSize(need(argv[i]));
        else if (!std::strcmp(argv[i], "--alloc-bytes")) allocBytes = parseSize(need(argv[i]));
        else if (!std::strcmp(argv[i], "--iters")) iters = parseInt(need(argv[i]));
        else if (!std::strcmp(argv[i], "--async")) asyncCopy = parseInt(need(argv[i]));
        else if (!std::strcmp(argv[i], "--sync-each")) syncEach = parseInt(need(argv[i]));
        else if (!std::strcmp(argv[i], "--rotate-offset")) rotateOffset = parseInt(need(argv[i]));
        else if (!std::strcmp(argv[i], "--progress-every")) progressEvery = parseInt(need(argv[i]));
        else if (!std::strcmp(argv[i], "--help") || !std::strcmp(argv[i], "-h")) {
            std::printf("Usage: %s [--device N] [--bytes 16|4k|1m] [--alloc-bytes 256m] [--iters N] [--async 0|1] [--sync-each 0|1] [--rotate-offset 0|1]\n", argv[0]);
            return 0;
        } else {
            std::fprintf(stderr, "unknown option: %s\n", argv[i]);
            return 1;
        }
    }

    int deviceCount = 0;
    HIP_CHECK(hipGetDeviceCount(&deviceCount));
    if (device < 0 || device >= deviceCount)
    {
        std::fprintf(stderr, "invalid device=%d deviceCount=%d\n", device, deviceCount);
        return 1;
    }
    HIP_CHECK(hipSetDevice(device));

    hipDeviceProp_t prop{};
    HIP_CHECK(hipGetDeviceProperties(&prop, device));
    std::printf("[D2D REPRO] device=%d name=%s arch=%s\n", device, prop.name, prop.gcnArchName);
    std::printf("[D2D REPRO] bytes=%zu allocBytes=%zu iters=%d async=%d syncEach=%d rotateOffset=%d\n",
                bytes, allocBytes, iters, asyncCopy, syncEach, rotateOffset);

    if (bytes == 0 || allocBytes < bytes * 2)
    {
        std::fprintf(stderr, "alloc-bytes must be >= bytes*2\n");
        return 1;
    }

    void* src = nullptr;
    void* dst = nullptr;
    HIP_CHECK(hipMalloc(&src, allocBytes));
    HIP_CHECK(hipMalloc(&dst, allocBytes));

    std::vector<unsigned char> init(bytes);
    for (size_t i = 0; i < bytes; ++i)
        init[i] = static_cast<unsigned char>((i * 131u + 17u) & 0xffu);
    HIP_CHECK(hipMemcpy(src, init.data(), bytes, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(dst, init.data(), bytes, hipMemcpyHostToDevice));

    hipStream_t stream = nullptr;
    HIP_CHECK(hipStreamCreateWithFlags(&stream, hipStreamNonBlocking));

    const size_t usable = allocBytes - bytes;
    const size_t stride = 4096;
    for (int i = 0; i < iters; ++i)
    {
        size_t off = rotateOffset ? (size_t(i) * stride) % usable : 0;
        char* s = static_cast<char*>(src) + off;
        char* d = static_cast<char*>(dst) + ((off + bytes + 128) % usable);

        if (asyncCopy)
            HIP_CHECK(hipMemcpyDtoDAsync(reinterpret_cast<hipDeviceptr_t>(d), reinterpret_cast<hipDeviceptr_t>(s), bytes, stream));
        else
            HIP_CHECK(hipMemcpyDtoD(reinterpret_cast<hipDeviceptr_t>(d), reinterpret_cast<hipDeviceptr_t>(s), bytes));

        if (syncEach)
            HIP_CHECK(hipStreamSynchronize(stream));

        if (progressEvery > 0 && ((i + 1) % progressEvery) == 0)
            std::printf("[D2D REPRO] iter=%d/%d\n", i + 1, iters);
    }

    HIP_CHECK(hipStreamSynchronize(stream));
    HIP_CHECK(hipStreamDestroy(stream));
    HIP_CHECK(hipFree(dst));
    HIP_CHECK(hipFree(src));
    HIP_CHECK(hipDeviceSynchronize());

    std::printf("[D2D REPRO] VERDICT: PASS\n");
    return 0;
}
