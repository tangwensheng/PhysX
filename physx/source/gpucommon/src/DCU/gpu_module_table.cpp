// GPU module table — intercepts __hipRegisterFatBinary to capture fat
// binary data for hipModuleLoadData().
//
// HipContextManager::loadGpuLibrary() will call hipModuleLoadData() with
// each captured fat binary pointer, creating a hipModule_t for each.
// hipModuleGetFunction() can then resolve kernel names from those modules.

#include <cstdio>
#include <cstdlib>

static const int MAX_MODULES = 128;
static const void* gFatBinData[MAX_MODULES];
static int   gNumModules = 0;

// Intercept HIP's __hipRegisterFatBinary to capture the raw fat binary
// pointer.  The HIP runtime's own registration is also fine — we just
// need a copy of the pointer for our manual hipModuleLoadData.
extern "C" {

void** __hipRegisterFatBinary(const void* data) {
	fprintf(stderr, "[DCU GPU] __hipRegisterFatBinary data=%p cnt=%d\n", data, gNumModules);
	if (gNumModules >= MAX_MODULES) return nullptr;
	gFatBinData[gNumModules] = data;
	// Return a handle that identifies this module index
	return (void**)(size_t)(gNumModules++);
}

void __hipRegisterFunction(void**, const void*, const char*, const char*, const void*, const void*) {
	// Not used by DCU path — KernelWrangler resolves via hipModuleGetFunction
}

void __hipUnregisterFatBinary(void** fp) {
	if (fp) {
		int idx = (int)(size_t)fp;
		if (idx >= 0 && idx < MAX_MODULES) gFatBinData[idx] = nullptr;
	}
}

unsigned int PxGpuGetCudaModuleTableSize() {
	return (unsigned int)gNumModules;
}

void** PxGpuGetCudaModuleTable() {
	// Not used by DCU KernelWrangler — just satisfy link references
	static void* dummy = nullptr;
	return &dummy;
}

// --- Accessor for HipContextManager ---
const void* PxGpuGetFatBinaryData(int index) {
	if (index >= 0 && index < gNumModules) return gFatBinData[index];
	return nullptr;
}

int PxGpuGetNumFatBinaries() {
	return gNumModules;
}

} // extern "C"
