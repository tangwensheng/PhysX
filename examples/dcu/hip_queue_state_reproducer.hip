#include <hip/hip_runtime.h>

#include <cerrno>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <thread>
#include <chrono>

namespace
{

enum class Mode
{
    eNONE,
    eD2D
};

struct Options
{
    Mode mode = Mode::eNONE;
    size_t stackMiB = 256;
    unsigned holdSeconds = 40;
};

struct Layout
{
    size_t arenaBytes = 0;
    size_t pairOffset = 0;
    size_t counterOffsets[5] = {};
    size_t stackOffsets[2] = {};
};

constexpr size_t kNumTests = 3;
// convexNpCommon.h documents ConvexMeshPair as 56 bytes on the target ABI.
// Keep this standalone reproducer independent from PhysX headers.
constexpr size_t kConvexMeshPairBytes = 56;
constexpr const char* kBuildMarker = "HIP_QUEUE_STATE_CONTROL_V2_PAIR56_DIRECT_HIPCC";

const char* modeName(Mode mode)
{
    switch (mode)
    {
    case Mode::eNONE:
        return "none";
    case Mode::eD2D:
        return "d2d";
    }
    return "invalid";
}

bool parseUnsigned(const char* text, unsigned long long maximum, unsigned long long& value)
{
    if (!text || !text[0])
        return false;

    errno = 0;
    char* end = nullptr;
    const unsigned long long parsed = std::strtoull(text, &end, 10);
    if (errno != 0 || end == text || *end != '\0' || parsed > maximum)
        return false;

    value = parsed;
    return true;
}

bool parseOptions(int argc, char** argv, Options& options)
{
    for (int index = 1; index < argc; ++index)
    {
        if (std::strcmp(argv[index], "--mode") == 0 && index + 1 < argc)
        {
            const char* value = argv[++index];
            if (std::strcmp(value, "none") == 0)
                options.mode = Mode::eNONE;
            else if (std::strcmp(value, "d2d") == 0)
                options.mode = Mode::eD2D;
            else
                return false;
        }
        else if (std::strcmp(argv[index], "--stack-mib") == 0 && index + 1 < argc)
        {
            unsigned long long value = 0;
            if (!parseUnsigned(argv[++index], 4096, value) || value == 0)
                return false;
            options.stackMiB = static_cast<size_t>(value);
        }
        else if (std::strcmp(argv[index], "--hold-seconds") == 0 && index + 1 < argc)
        {
            unsigned long long value = 0;
            if (!parseUnsigned(argv[++index], 600, value))
                return false;
            options.holdSeconds = static_cast<unsigned>(value);
        }
        else
        {
            return false;
        }
    }
    return true;
}

size_t alignUp(size_t value, size_t alignment)
{
    return (value + alignment - 1) & ~(alignment - 1);
}

size_t reserve(size_t& cursor, size_t alignment, size_t bytes)
{
    cursor = alignUp(cursor, alignment);
    const size_t offset = cursor;
    cursor += bytes;
    return offset;
}

Layout makeLayout(size_t stackBytes)
{
    Layout layout;
    size_t cursor = 0;

    // Reproduce the call-5 mIntermStackAlloc suballocation order in
    // PxgNarrowphaseCore::runModifiableContactManagersConvexMesh().
    layout.pairOffset = reserve(cursor, 256, kNumTests * kConvexMeshPairBytes);
    layout.counterOffsets[0] = reserve(cursor, alignof(uint32_t), sizeof(uint32_t));
    layout.counterOffsets[1] = reserve(cursor, alignof(uint32_t), sizeof(uint32_t));
    layout.counterOffsets[2] = reserve(cursor, 256, sizeof(uint32_t));
    for (unsigned index = 0; index < 6; ++index)
        reserve(cursor, 256, sizeof(void*));
    layout.counterOffsets[3] = reserve(cursor, alignof(uint32_t), sizeof(uint32_t));
    layout.stackOffsets[0] = reserve(cursor, 256, stackBytes);
    layout.stackOffsets[1] = reserve(cursor, 256, stackBytes);
    layout.counterOffsets[4] = reserve(cursor, 256, sizeof(uint32_t));
    layout.arenaBytes = alignUp(cursor, 256);
    return layout;
}

void printHipResult(const char* phase, const char* operation, hipError_t result)
{
    std::printf("phase=%s operation=%s result=%d error=%s\n",
        phase, operation, static_cast<int>(result), hipGetErrorString(result));
    std::fflush(stdout);
}

} // namespace

int main(int argc, char** argv)
{
    Options options;
    if (!parseOptions(argc, argv, options))
    {
        std::fprintf(stderr,
            "usage: %s --mode none|d2d --stack-mib N --hold-seconds N\n", argv[0]);
        return 2;
    }

    if (options.stackMiB > std::numeric_limits<size_t>::max() / (1024u * 1024u))
    {
        std::fprintf(stderr, "stack size overflow\n");
        return 2;
    }

    const size_t stackBytes = options.stackMiB * 1024u * 1024u;
    const Layout layout = makeLayout(stackBytes);
    std::printf(
        "phase=setup marker=%s mode=%s logicalDevice=0 numTests=%zu convexMeshPairBytes=%zu stackMiB=%zu stackBytes=%zu arenaBytes=%zu exactPhysXStackMiB=256\n",
        kBuildMarker, modeName(options.mode), kNumTests, kConvexMeshPairBytes,
        options.stackMiB, stackBytes, layout.arenaBytes);
    std::fflush(stdout);

    hipError_t result = hipSetDevice(0);
    if (result != hipSuccess)
    {
        printHipResult("setup", "hipSetDevice", result);
        std::printf("VERDICT=SETUP_FAILED\n");
        return 10;
    }

    hipStream_t stream = nullptr;
    result = hipStreamCreateWithFlags(&stream, hipStreamNonBlocking);
    if (result != hipSuccess)
    {
        printHipResult("setup", "hipStreamCreateWithFlags", result);
        std::printf("VERDICT=SETUP_FAILED\n");
        return 12;
    }

    void* arena = nullptr;
    result = hipMalloc(&arena, layout.arenaBytes);
    if (result != hipSuccess)
    {
        printHipResult("allocated", "hipMalloc-arena", result);
        hipStreamDestroy(stream);
        std::printf("VERDICT=SETUP_FAILED\n");
        return 13;
    }

    std::printf(
        "phase=allocated arena=%p arenaBytes=%zu pairOffset=%zu stack0Offset=%zu stack1Offset=%zu counterOffsets=%zu,%zu,%zu,%zu,%zu\n",
        arena, layout.arenaBytes, layout.pairOffset, layout.stackOffsets[0], layout.stackOffsets[1],
        layout.counterOffsets[0], layout.counterOffsets[1], layout.counterOffsets[2],
        layout.counterOffsets[3], layout.counterOffsets[4]);
    std::fflush(stdout);

    uint32_t* counters[5] = {};
    for (unsigned index = 0; index < 5; ++index)
        counters[index] = reinterpret_cast<uint32_t*>(static_cast<unsigned char*>(arena) + layout.counterOffsets[index]);

    hipError_t memsetResults[5] = {};
    memsetResults[0] = hipMemsetD32Async(reinterpret_cast<hipDeviceptr_t>(counters[0]), 0, 1, stream);
    memsetResults[1] = hipMemsetD32Async(reinterpret_cast<hipDeviceptr_t>(counters[1]), 0, 1, stream);
    memsetResults[2] = hipMemsetD32Async(reinterpret_cast<hipDeviceptr_t>(counters[2]), 0, 1, stream);
    memsetResults[3] = hipMemsetD32Async(reinterpret_cast<hipDeviceptr_t>(counters[3]), 0, 1, stream);
    memsetResults[4] = hipMemsetD32Async(reinterpret_cast<hipDeviceptr_t>(counters[4]), 0, 1, stream);

    const hipError_t syncResult = hipStreamSynchronize(stream);
    uint32_t values[5] = {0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu};
    hipError_t copyResults[5] = {};
    for (unsigned index = 0; index < 5; ++index)
    {
        if (syncResult == hipSuccess)
            copyResults[index] = hipMemcpyDtoH(&values[index], reinterpret_cast<hipDeviceptr_t>(counters[index]), sizeof(uint32_t));
        else
            copyResults[index] = syncResult;
    }

    bool prefixValid = syncResult == hipSuccess;
    for (unsigned index = 0; index < 5; ++index)
        prefixValid = prefixValid && memsetResults[index] == hipSuccess && copyResults[index] == hipSuccess && values[index] == 0;

    std::printf(
        "phase=prefix memsetResults=%d,%d,%d,%d,%d sync=%d copyResults=%d,%d,%d,%d,%d values=%u,%u,%u,%u,%u zeros=%u\n",
        static_cast<int>(memsetResults[0]), static_cast<int>(memsetResults[1]),
        static_cast<int>(memsetResults[2]), static_cast<int>(memsetResults[3]),
        static_cast<int>(memsetResults[4]), static_cast<int>(syncResult),
        static_cast<int>(copyResults[0]), static_cast<int>(copyResults[1]),
        static_cast<int>(copyResults[2]), static_cast<int>(copyResults[3]),
        static_cast<int>(copyResults[4]), values[0], values[1], values[2], values[3], values[4],
        prefixValid ? 1u : 0u);
    std::fflush(stdout);

    if (!prefixValid)
    {
        std::printf("VERDICT=PREFIX_FAILED\n");
        std::fflush(stdout);
        hipFree(arena);
        hipStreamDestroy(stream);
        return 20;
    }

    hipError_t finalSubmitResult = hipSuccess;
    hipError_t finalSyncResult = hipSuccess;
    if (options.mode == Mode::eD2D)
    {
        unsigned char* sourceBytes = static_cast<unsigned char*>(arena) + layout.stackOffsets[0];
        const hipDeviceptr_t source = reinterpret_cast<hipDeviceptr_t>(sourceBytes);
        const hipDeviceptr_t destination = reinterpret_cast<hipDeviceptr_t>(sourceBytes + 16);
        finalSubmitResult = hipMemcpyDtoDAsync(destination, source, 16, stream);
        if (finalSubmitResult == hipSuccess)
            finalSyncResult = hipStreamSynchronize(stream);
    }

    std::printf("phase=final mode=%s submitted=%u submitResult=%d syncResult=%d bytes=%u\n",
        modeName(options.mode), options.mode == Mode::eNONE ? 0u : 1u,
        static_cast<int>(finalSubmitResult), static_cast<int>(finalSyncResult),
        options.mode == Mode::eD2D ? 16u : 0u);
    std::fflush(stdout);

    if (finalSubmitResult != hipSuccess || finalSyncResult != hipSuccess)
    {
        std::printf("VERDICT=FINAL_COMMAND_FAILED\n");
        std::fflush(stdout);
        hipFree(arena);
        hipStreamDestroy(stream);
        return 30;
    }

    std::printf("phase=hold seconds=%u\n", options.holdSeconds);
    std::fflush(stdout);
    std::this_thread::sleep_for(std::chrono::seconds(options.holdSeconds));
    std::printf("phase=hold-complete seconds=%u\n", options.holdSeconds);
    std::fflush(stdout);

    if (options.mode == Mode::eNONE)
    {
        std::printf("phase=end cleanup=skipped reason=control-no-final-command\n");
        std::printf("VERDICT=CONTROL_STABLE\n");
        std::fflush(stdout);
        std::_Exit(0);
    }

    const hipError_t freeResult = hipFree(arena);
    const hipError_t streamDestroyResult = hipStreamDestroy(stream);
    std::printf("phase=cleanup freeResult=%d streamDestroyResult=%d\n",
        static_cast<int>(freeResult), static_cast<int>(streamDestroyResult));
    std::fflush(stdout);

    if (freeResult != hipSuccess || streamDestroyResult != hipSuccess)
    {
        std::printf("VERDICT=CLEANUP_FAILED\n");
        return 40;
    }

    std::printf("phase=end\n");
    std::printf("VERDICT=TRIGGER_COMMAND_STABLE\n");
    std::fflush(stdout);
    return 0;
}
