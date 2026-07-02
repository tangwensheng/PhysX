// CudaWrappersDCU — implements CUDA Driver API functions using HIP
//
// This file is compiled with access to <hip/hip_runtime.h> and performs
// the opaque-pointer casts between CUDA types (CUstream_st*, etc.) and
// HIP types (ihipStream_t*, etc.).
//
// The stub <cuda.h> provides only declarations — keeping HIP vector types
// (float4 etc.) out of the global namespace so PhysX GPU headers can safely
// forward-declare their own "struct float4".

#include <hip/hip_runtime.h>

// ---- Stream management ----
extern "C" CUresult cuStreamSynchronize(CUstream hStream) {
	return (CUresult)hipStreamSynchronize((hipStream_t)hStream);
}

extern "C" CUresult cuStreamQuery(CUstream hStream) {
	return (CUresult)hipStreamQuery((hipStream_t)hStream);
}

// ---- Stream priority ----
extern "C" CUresult cuCtxGetStreamPriorityRange(int* leastPriority, int* greatestPriority) {
	return (CUresult)hipDeviceGetStreamPriorityRange(leastPriority, greatestPriority);
}

// ---- Synchronous memory copy ----
extern "C" CUresult cuMemcpyDtoH(void* dstHost, CUdeviceptr srcDevice, size_t ByteCount) {
	return (CUresult)hipMemcpyDtoH(dstHost, (hipDeviceptr_t)srcDevice, ByteCount);
}

extern "C" CUresult cuMemcpyHtoD(CUdeviceptr dstDevice, const void* srcHost, size_t ByteCount) {
	return (CUresult)hipMemcpyHtoD((hipDeviceptr_t)dstDevice, (void*)srcHost, ByteCount);
}

// ---- Texture / Array (SDF) — runtime stubs ----
extern "C" CUresult cuArray3DCreate(CUarray* pHandle, const CUDA_ARRAY3D_DESCRIPTOR* pAllocateArray) {
	(void)pHandle; (void)pAllocateArray;
	return CUDA_ERROR_UNKNOWN;
}
extern "C" CUresult cuMemcpy3D(const CUDA_MEMCPY3D* pCopy) {
	(void)pCopy;
	return CUDA_ERROR_UNKNOWN;
}
extern "C" CUresult cuMemcpy3DAsync(const CUDA_MEMCPY3D* pCopy, CUstream hStream) {
	(void)pCopy; (void)hStream;
	return CUDA_ERROR_UNKNOWN;
}
extern "C" CUresult cuArrayDestroy(CUarray hArray) {
	(void)hArray;
	return CUDA_ERROR_UNKNOWN;
}

extern "C" CUresult cuTexObjectCreate(CUtexObject* pTexObject,
	const void* pResDesc, const void* pTexDesc, const void* pResViewDesc) {
	(void)pTexObject; (void)pResDesc; (void)pTexDesc; (void)pResViewDesc;
	return CUDA_ERROR_UNKNOWN;
}

extern "C" CUresult cuTexObjectDestroy(CUtexObject texObject) {
	(void)texObject;
	return CUDA_ERROR_UNKNOWN;
}
