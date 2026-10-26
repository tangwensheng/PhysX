// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions
// are met:
//  * Redistributions of source code must retain the above copyright
//    notice, this list of conditions and the following disclaimer.
//  * Redistributions in binary form must reproduce the above copyright
//    notice, this list of conditions and the following disclaimer in the
//    documentation and/or other materials provided with the distribution.
//  * Neither the name of NVIDIA CORPORATION nor the names of its
//    contributors may be used to endorse or promote products derived
//    from this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ''AS IS'' AND ANY
// EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
// PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
// CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
// EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
// PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
// OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

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
	fprintf(stderr, "[HCU device] __hipRegisterFatBinary data=%p cnt=%d\n", data, gNumModules);
	if (gNumModules >= MAX_MODULES) return nullptr;
	gFatBinData[gNumModules] = data;
	// Return a handle that identifies this module index
	return (void**)(size_t)(gNumModules++);
}

void __hipRegisterFunction(void**, const void*, const char*, const char*, const void*, const void*) {
	// Not used by HCU path — KernelWrangler resolves via hipModuleGetFunction
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
	// Not used by HCU KernelWrangler — just satisfy link references
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
