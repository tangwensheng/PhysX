// HipContextManager — full HIP PxCudaContext / PxCudaContextManager implementation
// Compiled with g++ + __HIP_PLATFORM_AMD__ + PX_SUPPORT_GPU_PHYSX

#include <hip/hip_runtime.h>
#include <cstring>
#include "foundation/PxPreprocessor.h"
#include "foundation/PxErrorCallback.h"
#include "foundation/PxThread.h"
#include "foundation/PxArray.h"
#include "cudamanager/PxCudaContextManager.h"
#include "cudamanager/PxCudaContext.h"

namespace physx {

// GPU module table — loaded via dlopen at init
#include <dlfcn.h>
static void* gGpuLib = nullptr;
static PxU32  gModCount = 0;
static void** gModTable = nullptr;

static void loadGpuLibrary() {
    if (gGpuLib) return;
    // Try to dlopen our HIP GPU shared library
    const char* paths[] = {
        "./libPhysXGpuDCU.so",
        "../build_dcu/libPhysXGpuDCU.so",
        "/public/home/tangwsh/PhysX/build_dcu/libPhysXGpuDCU.so",
        nullptr
    };
    for (int i = 0; paths[i]; i++) {
        gGpuLib = dlopen(paths[i], RTLD_NOW | RTLD_GLOBAL);
        if (gGpuLib) {
            auto getSize = (PxU32(*)())dlsym(gGpuLib, "PxGpuGetCudaModuleTableSize");
            auto getTable = (void**(*)())dlsym(gGpuLib, "PxGpuGetCudaModuleTable");
            if (getSize && getTable) {
                gModCount = getSize();
                gModTable = getTable();
            }
            return;
        }
    }
    // Fallback: empty table
    gModCount = 0;
    static void* emptyTable[1] = {nullptr};
    gModTable = emptyTable;
}

#define HIP_CALL(call) do{ hipError_t _e=(call); if(_e!=hipSuccess) return PxCUresult((PxU32)_e); }while(0)

class HipCtx : public PxCudaContext
{
    hipError_t mLast;
    bool       mSync, mAbort;
public:
    HipCtx(PxDeviceAllocatorCallback* cb, bool sync)
        : mLast(hipSuccess), mSync(sync), mAbort(false) { mAllocatorCallback = cb; }
    void release() override { delete this; }

    PxCUresult memAlloc(CUdeviceptr* dptr, size_t bytes) override {
        if(mAbort){*dptr=0;return PxCUresult(mLast);}
        void* p=nullptr; mLast=hipMalloc(&p,bytes); *dptr=(CUdeviceptr)p; return PxCUresult(mLast);
    }
    PxCUresult memFree(CUdeviceptr dptr) override {
        if(!(void*)dptr) return PxCUresult(mLast);
        mLast=hipFree((void*)dptr); return PxCUresult(mLast);
    }
    PxCUresult memHostAlloc(void** pp, size_t bytes, unsigned int) override {
        mLast=hipHostMalloc(pp,bytes,hipHostMallocMapped|hipHostMallocPortable); return PxCUresult(mLast);
    }
    PxCUresult memFreeHost(void* p) override {
        mLast=hipHostFree(p); return PxCUresult(mLast);
    }
    PxCUresult memHostGetDevicePointer(CUdeviceptr* pdptr, void* p, unsigned int) override {
        if(!p){*pdptr=0; return PxCUresult(hipSuccess);}
        mLast=hipHostGetDevicePointer((void**)pdptr,p,0); return PxCUresult(mLast);
    }
    PxCUresult moduleLoadDataEx(CUmodule* mod, const void* image, unsigned int, PxCUjit_option*, void**) override {
        mLast=hipModuleLoadData((hipModule_t*)mod,image); return PxCUresult(mLast);
    }
    PxCUresult moduleGetFunction(CUfunction* func, CUmodule mod, const char* name) override {
        mLast=hipModuleGetFunction((hipFunction_t*)func,(hipModule_t)mod,name); return PxCUresult(mLast);
    }
    PxCUresult moduleUnload(CUmodule mod) override {
        mLast=hipModuleUnload((hipModule_t)mod); return PxCUresult(mLast);
    }
    PxCUresult streamCreate(CUstream* s, unsigned int) override {
        mLast=hipStreamCreate((hipStream_t*)s); return PxCUresult(mLast);
    }
    PxCUresult streamCreateWithPriority(CUstream* s, unsigned int, int) override {
        mLast=hipStreamCreate((hipStream_t*)s); return PxCUresult(mLast);
    }
    PxCUresult streamFlush(CUstream s) override {
        return PxCUresult((PxU32)hipStreamQuery((hipStream_t)s));
    }
    PxCUresult streamWaitEvent(CUstream s, CUevent e, unsigned int) override {
        mLast=hipStreamWaitEvent((hipStream_t)s,(hipEvent_t)e,0); return PxCUresult(mLast);
    }
    PxCUresult streamWaitEvent(CUstream s, CUevent e) override {
        return streamWaitEvent(s,e,0);
    }
    PxCUresult streamDestroy(CUstream s) override {
        if(!s) return PxCUresult(mLast);
        mLast=hipStreamDestroy((hipStream_t)s); return PxCUresult(mLast);
    }
    PxCUresult streamSynchronize(CUstream s) override {
        mLast=hipStreamSynchronize((hipStream_t)s); return PxCUresult(mLast);
    }
    PxCUresult eventCreate(CUevent* e, unsigned int) override {
        mLast=hipEventCreate((hipEvent_t*)e); return PxCUresult(mLast);
    }
    PxCUresult eventRecord(CUevent e, CUstream s) override {
        mLast=hipEventRecord((hipEvent_t)e,(hipStream_t)s); return PxCUresult(mLast);
    }
    PxCUresult eventQuery(CUevent e) override {
        mLast=hipEventQuery((hipEvent_t)e); return PxCUresult(mLast);
    }
    PxCUresult eventSynchronize(CUevent e) override {
        mLast=hipEventSynchronize((hipEvent_t)e); return PxCUresult(mLast);
    }
    PxCUresult eventDestroy(CUevent e) override {
        mLast=hipEventDestroy((hipEvent_t)e); return PxCUresult(mLast);
    }

    PxCUresult launchKernel(CUfunction f,
        unsigned int gx, unsigned int gy, unsigned int gz,
        unsigned int bx, unsigned int by, unsigned int bz,
        unsigned int shared, CUstream stream,
        PxCudaKernelParam* params, size_t paramSize, void** extra,
        const char*, int) override
    {
        if(mAbort) return PxCUresult(mLast);
        void* kp[32]; int n=(int)(paramSize/sizeof(PxCudaKernelParam));
        for(int i=0;i<n&&i<32;i++) kp[i]=params[i].data;
        mLast=hipModuleLaunchKernel((hipFunction_t)f,gx,gy,gz,bx,by,bz,shared,(hipStream_t)stream,kp,extra);
        if(mSync) mLast=hipStreamSynchronize((hipStream_t)stream);
        return PxCUresult(mLast);
    }

    PxCUresult launchKernel(CUfunction f,
        PxU32 gx, PxU32 gy, PxU32 gz,
        PxU32 bx, PxU32 by, PxU32 bz,
        PxU32 shared, CUstream stream,
        void** params, void** extra,
        const char*, int) override
    {
        if(mAbort) return PxCUresult(mLast);
        mLast=hipModuleLaunchKernel((hipFunction_t)f,gx,gy,gz,bx,by,bz,shared,(hipStream_t)stream,params,extra);
        if(mSync) mLast=hipStreamSynchronize((hipStream_t)stream);
        return PxCUresult(mLast);
    }

    PxCUresult memcpyDtoH(void* d, CUdeviceptr s, size_t b) override {
        mLast=hipMemcpyDtoH(d,(hipDeviceptr_t)s,b); return PxCUresult(mLast);
    }
    PxCUresult memcpyDtoHAsync(void* d, CUdeviceptr s, size_t b, CUstream st) override {
        mLast=hipMemcpyDtoHAsync(d,(hipDeviceptr_t)s,b,(hipStream_t)st); return PxCUresult(mLast);
    }
    PxCUresult memcpyHtoD(CUdeviceptr d, const void* s, size_t b) override {
        mLast=hipMemcpyHtoD((hipDeviceptr_t)d,(void*)s,b); return PxCUresult(mLast);
    }
    PxCUresult memcpyHtoDAsync(CUdeviceptr d, const void* s, size_t b, CUstream st) override {
        mLast=hipMemcpyHtoDAsync((hipDeviceptr_t)d,(void*)s,b,(hipStream_t)st); return PxCUresult(mLast);
    }
    PxCUresult memcpyDtoDAsync(CUdeviceptr d, CUdeviceptr s, size_t b, CUstream st) override {
        mLast=hipMemcpyDtoDAsync((hipDeviceptr_t)d,(hipDeviceptr_t)s,b,(hipStream_t)st); return PxCUresult(mLast);
    }
    PxCUresult memcpyDtoD(CUdeviceptr d, CUdeviceptr s, size_t b) override {
        mLast=hipMemcpyDtoD((hipDeviceptr_t)d,(hipDeviceptr_t)s,b); return PxCUresult(mLast);
    }
    PxCUresult memcpyPeerAsync(CUdeviceptr, CUcontext, CUdeviceptr, CUcontext, size_t, CUstream) override {
        return PxCUresult(hipSuccess);
    }
    PxCUresult memsetD32Async(CUdeviceptr d, unsigned int v, size_t n, CUstream s) override {
        mLast=hipMemsetD32Async((hipDeviceptr_t)d,v,n,(hipStream_t)s); return PxCUresult(mLast);
    }
    PxCUresult memsetD8Async(CUdeviceptr d, unsigned char v, size_t n, CUstream s) override {
        mLast=hipMemsetD8Async((hipDeviceptr_t)d,v,n,(hipStream_t)s); return PxCUresult(mLast);
    }
    PxCUresult memsetD32(CUdeviceptr d, unsigned int v, size_t n) override {
        mLast=hipMemsetD32((hipDeviceptr_t)d,v,n); return PxCUresult(mLast);
    }
    PxCUresult memsetD16(CUdeviceptr d, unsigned short v, size_t n) override {
        mLast=hipMemsetD16((hipDeviceptr_t)d,v,n); return PxCUresult(mLast);
    }
    PxCUresult memsetD8(CUdeviceptr d, unsigned char v, size_t n) override {
        mLast=hipMemsetD8((hipDeviceptr_t)d,v,n); return PxCUresult(mLast);
    }
    PxCUresult getLastError() override { return PxCUresult(mAbort?hipErrorOutOfMemory:mLast); }
    void setAbortMode(bool a) override { mAbort=a; }
    bool isInAbortMode() override { return mAbort; }
};


class HipCtxMgr : public PxCudaContextManager
{
    bool mValid;
    hipDevice_t mDev; hipCtx_t mCtx;
    HipCtx* mHCtx;
    PxArray<hipModule_t> mMods;
    int mOrd; char mName[128];
    size_t mTM; int mMC, mMT, mSM;
    PxU32 mTLS;

    // Override internal alloc/free methods (pure virtual in base)
    void* allocDeviceBufferInternal(PxU64 b, const char*, PxI32) override {
        void* p; hipMalloc(&p,(size_t)b); return p;
    }
    void* allocPinnedHostBufferInternal(PxU64 b, const char*, PxI32) override {
        void* p; hipHostMalloc(&p,(size_t)b,hipHostMallocMapped|hipHostMallocPortable); return p;
    }
    void freeDeviceBufferInternal(void* p) override { if(p) hipFree(p); }
    void freePinnedHostBufferInternal(void* p) override { if(p) hipHostFree(p); }
    void clearDeviceBufferAsyncInternal(void* p, PxU32 b, CUstream s, PxI32 v) override {
        hipMemsetD32Async((hipDeviceptr_t)p,(unsigned)v,b/4,(hipStream_t)s);
    }
    void copyDToHAsyncInternal(void* d, const void* s, PxU32 b, CUstream st) override {
        hipMemcpyDtoHAsync(d,(hipDeviceptr_t)s,b,(hipStream_t)st);
    }
    void copyHToDAsyncInternal(void* d, const void* s, PxU32 b, CUstream st) override {
        hipMemcpyHtoDAsync((hipDeviceptr_t)d,(void*)s,b,(hipStream_t)st);
    }
    void copyDToDAsyncInternal(void* d, const void* s, PxU32 b, CUstream st) override {
        hipMemcpyDtoDAsync((hipDeviceptr_t)d,(hipDeviceptr_t)s,b,(hipStream_t)st);
    }
    void copyDToHInternal(void* d, const void* s, PxU32 b) override {
        hipMemcpyDtoH(d,(hipDeviceptr_t)s,b);
    }
    void copyHToDInternal(void* d, const void* s, PxU32 b) override {
        hipMemcpyHtoD((hipDeviceptr_t)d,(void*)s,b);
    }
    void memsetD8AsyncInternal(void* d, const PxU8& v, PxU32 b, CUstream s) override {
        hipMemsetD8Async((hipDeviceptr_t)d,v,b,(hipStream_t)s);
    }
    void memsetD32AsyncInternal(void* d, const PxU32& v, PxU32 n, CUstream s) override {
        hipMemsetD32Async((hipDeviceptr_t)d,v,n,(hipStream_t)s);
    }

public:
    HipCtxMgr(const PxCudaContextManagerDesc& d, PxProfilerCallback* = nullptr, bool sync = false)
        : mValid(false), mHCtx(nullptr)
    {
        mName[0]=0; int ord=d.deviceOrdinal<0?0:d.deviceOrdinal;
        if(hipInit(0)!=hipSuccess)return;
        if(hipDeviceGet(&mDev,ord)!=hipSuccess)return;
        if(hipCtxCreate(&mCtx,0,mDev)!=hipSuccess)return;
        mHCtx=new HipCtx(d.deviceAllocator,sync);
        hipDeviceProp_t pr; hipGetDeviceProperties(&pr,mDev);
        mOrd=ord; strncpy(mName,pr.name,127);
        mTM=pr.totalGlobalMem; mMC=pr.multiProcessorCount; mMT=pr.maxThreadsPerBlock; mSM=pr.sharedMemPerBlock;
        loadGpuLibrary();
        PxU32 n=gModCount; void** tbl=gModTable;
        mMods.resize(n,nullptr);
        for(PxU32 i=0;i<n;i++){if(tbl[i]){hipModule_t mo=nullptr; hipError_t st=hipModuleLoadData(&mo,tbl[i]); if(st==hipSuccess||st==hipErrorNoBinaryForGpu) mMods[i]=mo;}}
        mTLS=PxTlsAlloc(); mValid=true;
    }

    ~HipCtxMgr() override {
        if(mHCtx){for(PxU32 i=0;i<mMods.size();i++) if(mMods[i]) hipModuleUnload(mMods[i]); mHCtx->release();}
        if(mCtx) hipCtxDestroy(mCtx); PxTlsFree(mTLS);
    }

    void release() override { delete this; }
    void acquireContext() override { hipCtxPushCurrent(mCtx); PxTlsSetValue(mTLS,PxTlsGetValue(mTLS)+1); }
    void releaseContext() override { PxU32 r=PxTlsGetValue(mTLS); if(--r==0){hipCtx_t c;hipCtxPopCurrent(&c);} PxTlsSetValue(mTLS,r); }
    bool tryAcquireContext() override { acquireContext(); return true; }
    bool contextIsValid() const override { return mValid; }
    bool supportsArchSM10() const override { return mValid; }
    bool supportsArchSM11() const override { return mValid; }
    bool supportsArchSM12() const override { return mValid; }
    bool supportsArchSM13() const override { return mValid; }
    bool supportsArchSM20() const override { return mValid; }
    bool supportsArchSM30() const override { return mValid; }
    bool supportsArchSM35() const override { return mValid; }
    bool supportsArchSM50() const override { return mValid; }
    bool supportsArchSM52() const override { return mValid; }
    bool supportsArchSM60() const override { return mValid; }
    bool isIntegrated() const override { return false; }
    bool canMapHostMemory() const override { return true; }
    int getDriverVersion() const override { return 12000; }
    size_t getDeviceTotalMemBytes() const override { return mTM; }
    int getMultiprocessorCount() const override { return mMC; }
    int getSharedMemPerBlock() const override { return mSM; }
    int getSharedMemPerMultiprocessor() const override { return mSM; }
    unsigned int getMaxThreadsPerBlock() const override { return (unsigned int)mMT; }
    unsigned int getClockRate() const override { return 1400; }
    const char* getDeviceName() const override { return mValid?mName:"Invalid"; }
    CUdevice getDevice() const override { return (CUdevice)(mValid?mOrd:-1); }
    CUcontext getContext() override { return (CUcontext)mCtx; }
    PxCudaContext* getCudaContext() override { return mHCtx; }
    CUmodule* getCuModules() override { return (CUmodule*)mMods.begin(); }
    void setUsingConcurrentStreams(bool) override {}
    bool getUsingConcurrentStreams() const override { return true; }
    void getDeviceMemoryInfo(size_t& f, size_t& t) const override { hipMemGetInfo(&f,&t); }
    CUdeviceptr getMappedDevicePtr(void* pb) override { CUdeviceptr d=0; hipHostGetDevicePointer((void**)&d,pb,0); return d; }
};

extern "C" PxCudaContextManager* PxCreateCudaContextManager(PxFoundation&, const PxCudaContextManagerDesc& d, PxProfilerCallback*, bool s)
{ return new HipCtxMgr(d,nullptr,s); }

// GPU module loader stubs (replaces PxGpu.cpp / PxPhysXGpuModuleLoader.cpp)
void PxLoadPhysxGPUModule(const char*) {}
void PxUnloadPhysxGPUModule() {}
typedef PxCudaContextManager* (*PxCreatePhysXGpu_Func)(PxFoundation&, const PxCudaContextManagerDesc&, PxProfilerCallback*, bool);
PxCreatePhysXGpu_Func g_PxCreatePhysXGpu_Func = PxCreateCudaContextManager;

} // namespace physx
