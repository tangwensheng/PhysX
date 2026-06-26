// GPU module table — populated by static constructors from .cu files
extern "C" {
    // These are populated by HIP's __hipRegisterFatBinary etc.
    // Placeholder: actual registration happens via hipcc static init
    // when .cu files are linked into the same binary.
    static void* gModuleTable[128] = {nullptr};
    static unsigned int gModuleCount = 0;

    unsigned int PxGpuGetCudaModuleTableSize() { return gModuleCount; }
    void** PxGpuGetCudaModuleTable() { return gModuleTable; }
}
