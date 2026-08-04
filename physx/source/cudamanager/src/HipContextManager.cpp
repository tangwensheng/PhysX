// HipContextManager — full HIP PxCudaContext / PxCudaContextManager for DCU
#include <hip/hip_runtime.h>
#include <cstring>
#include <cstdlib>
#include <cstdio>
#include <dlfcn.h>
#include "foundation/PxPreprocessor.h"
#include "foundation/PxErrorCallback.h"
#include "foundation/PxThread.h"
#include "foundation/PxArray.h"
#include "cudamanager/PxCudaContextManager.h"
#include "cudamanager/PxCudaContext.h"
#include "foundation/PxFoundation.h"

// Global scope for cross-TU access (unsigned avoids PxU32 dependency)
unsigned int g_DCU_ModuleCount = 0;

namespace physx {

static PxArray<hipModule_t> gModules;
static bool gLoaded = false;

#include <dirent.h>

static void loadGpuLibrary() {
    if(gLoaded) return;
    gLoaded = true;

    const char* dirs[] = {
        "./kernels", "../build_dcu/kernels",
        "/public/home/tangwsh/PhysX/build_dcu/kernels", nullptr};

    for(int d = 0; dirs[d]; d++) {
        DIR* dp = opendir(dirs[d]);
        if(!dp) continue;
        int n = 0;
        struct dirent* e;
        while((e = readdir(dp))) {
            const char* nm = e->d_name;
            int len = strlen(nm);
            if(len < 6 || strcmp(nm+len-6, ".hsaco")) continue;

            char path[512];
            snprintf(path, sizeof(path), "%s/%s", dirs[d], nm);
            FILE* fp = fopen(path, "rb");
            if(!fp) continue;
            fseek(fp, 0, SEEK_END);
            size_t sz = ftell(fp); fseek(fp, 0, SEEK_SET);
            void* buf = malloc(sz);
            fread(buf, 1, sz, fp); fclose(fp);

            hipModule_t mod = nullptr;
            hipError_t r = hipModuleLoadData(&mod, buf);
            free(buf);
            if(r == hipSuccess && mod) {
                gModules.pushBack(mod);
                if(n == 0) fprintf(stderr, "[DCU GPU] First hsaco: %s\n", path);
                n++;
            }
        }
        closedir(dp);
        if(n > 0) {
            g_DCU_ModuleCount = n;
            fprintf(stderr, "[DCU GPU] Loaded %d hsaco modules from %s\n", n, dirs[d]);
            return;
        }
    }

    // Fallback: hipModuleLoad .so
    fprintf(stderr, "[DCU GPU] No hsaco dir, trying .so...\n");
    const char* paths[] = {
        "./libPhysXGpuDCU.so", "../build_dcu/libPhysXGpuDCU.so",
        "/public/home/tangwsh/PhysX/build_dcu/libPhysXGpuDCU.so", nullptr};
    for(int i=0; paths[i]; i++) {
        FILE* fp = fopen(paths[i], "r");
        if(!fp) continue; fclose(fp);
        hipModule_t mod = nullptr;
        if(hipModuleLoad(&mod, paths[i]) == hipSuccess && mod) {
            gModules.pushBack(mod);
            g_DCU_ModuleCount = 1;
            fprintf(stderr, "[DCU GPU] Loaded .so: %s\n", paths[i]);
            return;
        }
    }
    fprintf(stderr, "[DCU GPU] No module found\n");
}

class HipCtx : public PxCudaContext {
    hipError_t mLast; bool mSync, mAbort;
    int mKernelCount;
public:
    HipCtx(PxDeviceAllocatorCallback* cb, bool sync)
        : mLast(hipSuccess), mSync(sync), mAbort(false), mKernelCount(0)
    { mAllocatorCallback = cb; }
    void release() override {
        fprintf(stderr, "[DCU PROFILER] Total kernel launches: %d\n", mKernelCount);
        delete this;
    }
    int totalKernels() const { return mKernelCount; }

    PxCUresult memAlloc(CUdeviceptr* d, size_t b) override {
        if(mAbort){*d=0; return PxCUresult(mLast);}
        void* p=nullptr; mLast=hipMalloc(&p,b); *d=(CUdeviceptr)p; return PxCUresult(mLast);
    }
    PxCUresult memFree(CUdeviceptr d) override {
        if(!(void*)d) return PxCUresult(mLast);
        mLast=hipFree((void*)d); return PxCUresult(mLast);
    }
    PxCUresult memHostAlloc(void** p,size_t b,unsigned) override {
        mLast=hipHostMalloc(p,b,hipHostMallocMapped|hipHostMallocPortable); return PxCUresult(mLast);
    }
    PxCUresult memFreeHost(void* p) override { mLast=hipHostFree(p); return PxCUresult(mLast); }
    PxCUresult memHostGetDevicePointer(CUdeviceptr* d,void* p,unsigned) override {
        if(!p){*d=0; return PxCUresult(hipSuccess);}
        mLast=hipHostGetDevicePointer((void**)d,p,0); return PxCUresult(mLast);
    }
    PxCUresult moduleLoadDataEx(CUmodule* m,const void* img,unsigned,PxCUjit_option*,void**) override {
        mLast=hipModuleLoadData((hipModule_t*)m,img); return PxCUresult(mLast);
    }
    PxCUresult moduleGetFunction(CUfunction* f,CUmodule m,const char* n) override {
        mLast=hipModuleGetFunction((hipFunction_t*)f,(hipModule_t)m,n);
        return PxCUresult(mLast);
    }
    PxCUresult moduleUnload(CUmodule m) override {
        mLast=hipModuleUnload((hipModule_t)m); return PxCUresult(mLast);
    }
    PxCUresult streamCreate(CUstream* s,unsigned) override {
        mLast=hipStreamCreate((hipStream_t*)s); return PxCUresult(mLast);
    }
    PxCUresult streamCreateWithPriority(CUstream* s,unsigned,int) override {
        mLast=hipStreamCreate((hipStream_t*)s); return PxCUresult(mLast);
    }
    PxCUresult streamFlush(CUstream s) override {
        return PxCUresult((PxU32)hipStreamQuery((hipStream_t)s));
    }
    PxCUresult streamWaitEvent(CUstream s,CUevent e,unsigned) override {
        mLast=hipStreamWaitEvent((hipStream_t)s,(hipEvent_t)e,0); return PxCUresult(mLast);
    }
    PxCUresult streamWaitEvent(CUstream s,CUevent e) override { return streamWaitEvent(s,e,0); }
    PxCUresult streamDestroy(CUstream s) override {
        if(!s) return PxCUresult(mLast);
        mLast=hipStreamDestroy((hipStream_t)s); return PxCUresult(mLast);
    }
    PxCUresult streamSynchronize(CUstream s) override {
        mLast=hipStreamSynchronize((hipStream_t)s); return PxCUresult(mLast);
    }
    PxCUresult eventCreate(CUevent* e,unsigned) override {
        mLast=hipEventCreate((hipEvent_t*)e); return PxCUresult(mLast);
    }
    PxCUresult eventRecord(CUevent e,CUstream s) override {
        mLast=hipEventRecord((hipEvent_t)e,(hipStream_t)s); return PxCUresult(mLast);
    }
    PxCUresult eventQuery(CUevent e) override { mLast=hipEventQuery((hipEvent_t)e); return PxCUresult(mLast); }
    PxCUresult eventSynchronize(CUevent e) override { mLast=hipEventSynchronize((hipEvent_t)e); return PxCUresult(mLast); }
    PxCUresult eventDestroy(CUevent e) override { mLast=hipEventDestroy((hipEvent_t)e); return PxCUresult(mLast); }

    PxCUresult launchKernel(CUfunction f,
        unsigned gx,unsigned gy,unsigned gz, unsigned bx,unsigned by,unsigned bz,
        unsigned sh, CUstream s, PxCudaKernelParam* p, size_t ps, void** ex,
        const char* name, int line) override
    {
        if(mAbort) return PxCUresult(mLast);
        void* kp[32]; int n=(int)(ps/sizeof(PxCudaKernelParam));
        for(int i=0;i<n&&i<32;i++) kp[i]=p[i].data;
        mKernelCount++;
        if(std::getenv("PX_DCU_TRACE_KERNELS"))
            fprintf(stderr, "[DCU KERNEL %d] %s:%d grid=(%u,%u,%u) block=(%u,%u,%u) sh=%u paramBytes=%zu\n", mKernelCount, name ? name : "<unnamed>", line, gx, gy, gz, bx, by, bz, sh, ps);
        mLast=hipModuleLaunchKernel((hipFunction_t)f,gx,gy,gz,bx,by,bz,sh,(hipStream_t)s,kp,ex);
        if(mLast == hipSuccess && std::getenv("PX_DCU_SYNC_KERNELS"))
        {
            mLast = hipStreamSynchronize((hipStream_t)s);
            if(mLast != hipSuccess)
                fprintf(stderr, "[DCU KERNEL ERROR %d] %s:%d sync failed: %s (%d)\n", mKernelCount, name ? name : "<unnamed>", line, hipGetErrorString(mLast), int(mLast));
        }
        return PxCUresult(mLast);
    }

    PxCUresult launchKernel(CUfunction f,
        PxU32 gx,PxU32 gy,PxU32 gz, PxU32 bx,PxU32 by,PxU32 bz,
        PxU32 sh, CUstream s, void** params, void** ex, const char* name, int line) override
    {
        if(mAbort) return PxCUresult(mLast);
        mKernelCount++;
        if(std::getenv("PX_DCU_TRACE_KERNELS"))
            fprintf(stderr, "[DCU KERNEL %d] %s:%d grid=(%u,%u,%u) block=(%u,%u,%u) sh=%u params=%p\n", mKernelCount, name ? name : "<unnamed>", line, gx, gy, gz, bx, by, bz, sh, params);
        mLast=hipModuleLaunchKernel((hipFunction_t)f,gx,gy,gz,bx,by,bz,sh,(hipStream_t)s,params,ex);
        if(mLast == hipSuccess && std::getenv("PX_DCU_SYNC_KERNELS"))
        {
            mLast = hipStreamSynchronize((hipStream_t)s);
            if(mLast != hipSuccess)
                fprintf(stderr, "[DCU KERNEL ERROR %d] %s:%d sync failed: %s (%d)\n", mKernelCount, name ? name : "<unnamed>", line, hipGetErrorString(mLast), int(mLast));
        }
        return PxCUresult(mLast);
    }

    PxCUresult memcpyDtoH(void* d,CUdeviceptr s,size_t b) override { mLast=hipMemcpyDtoH(d,(hipDeviceptr_t)s,b); return PxCUresult(mLast); }
    PxCUresult memcpyDtoHAsync(void* d,CUdeviceptr s,size_t b,CUstream st) override { mLast=hipMemcpyDtoHAsync(d,(hipDeviceptr_t)s,b,(hipStream_t)st); return PxCUresult(mLast); }
    PxCUresult memcpyHtoD(CUdeviceptr d,const void* s,size_t b) override { mLast=hipMemcpyHtoD((hipDeviceptr_t)d,(void*)s,b); return PxCUresult(mLast); }
    PxCUresult memcpyHtoDAsync(CUdeviceptr d,const void* s,size_t b,CUstream st) override { mLast=hipMemcpyHtoDAsync((hipDeviceptr_t)d,(void*)s,b,(hipStream_t)st); return PxCUresult(mLast); }
    PxCUresult memcpyDtoDAsync(CUdeviceptr d,CUdeviceptr s,size_t b,CUstream st) override { mLast=hipMemcpyDtoDAsync((hipDeviceptr_t)d,(hipDeviceptr_t)s,b,(hipStream_t)st); return PxCUresult(mLast); }
    PxCUresult memcpyDtoD(CUdeviceptr d,CUdeviceptr s,size_t b) override { mLast=hipMemcpyDtoD((hipDeviceptr_t)d,(hipDeviceptr_t)s,b); return PxCUresult(mLast); }
    PxCUresult memcpyPeerAsync(CUdeviceptr,CUcontext,CUdeviceptr,CUcontext,size_t,CUstream) override { return PxCUresult(hipSuccess); }
    PxCUresult memsetD32Async(CUdeviceptr d,unsigned v,size_t n,CUstream s) override { mLast=hipMemsetD32Async((hipDeviceptr_t)d,v,n,(hipStream_t)s); return PxCUresult(mLast); }
    PxCUresult memsetD8Async(CUdeviceptr d,unsigned char v,size_t n,CUstream s) override { mLast=hipMemsetD8Async((hipDeviceptr_t)d,v,n,(hipStream_t)s); return PxCUresult(mLast); }
    PxCUresult memsetD32(CUdeviceptr d,unsigned v,size_t n) override { mLast=hipMemsetD32((hipDeviceptr_t)d,v,n); return PxCUresult(mLast); }
    PxCUresult memsetD16(CUdeviceptr d,unsigned short v,size_t n) override { mLast=hipMemsetD16((hipDeviceptr_t)d,v,n); return PxCUresult(mLast); }
    PxCUresult memsetD8(CUdeviceptr d,unsigned char v,size_t n) override { mLast=hipMemsetD8((hipDeviceptr_t)d,v,n); return PxCUresult(mLast); }
    PxCUresult getLastError() override { return PxCUresult(mAbort?hipErrorOutOfMemory:mLast); }
    void setAbortMode(bool a) override { mAbort=a; }
    bool isInAbortMode() override { return mAbort; }
};

class HipCtxMgr : public PxCudaContextManager {
    bool mValid; hipDevice_t mDev; hipCtx_t mCtx; HipCtx* mHCtx;
    PxArray<hipModule_t> mMods; int mOrd; char mName[128];
    size_t mTM; int mMC,mMT,mSM; PxU32 mTLS;

    void* allocDeviceBufferInternal(PxU64 b,const char*,PxI32) override { void* p; hipMalloc(&p,(size_t)b); return p; }
    void* allocPinnedHostBufferInternal(PxU64 b,const char*,PxI32) override { void* p; hipHostMalloc(&p,(size_t)b,hipHostMallocMapped|hipHostMallocPortable); return p; }
    void freeDeviceBufferInternal(void* p) override { if(p) hipFree(p); }
    void freePinnedHostBufferInternal(void* p) override { if(p) hipHostFree(p); }
    void clearDeviceBufferAsyncInternal(void* p,PxU32 b,CUstream s,PxI32 v) override { hipMemsetD32Async((hipDeviceptr_t)p,(unsigned)v,b/4,(hipStream_t)s); }
    void copyDToHAsyncInternal(void* d,const void* s,PxU32 b,CUstream st) override { hipMemcpyDtoHAsync(d,(hipDeviceptr_t)s,b,(hipStream_t)st); }
    void copyHToDAsyncInternal(void* d,const void* s,PxU32 b,CUstream st) override { hipMemcpyHtoDAsync((hipDeviceptr_t)d,(void*)s,b,(hipStream_t)st); }
    void copyDToDAsyncInternal(void* d,const void* s,PxU32 b,CUstream st) override { hipMemcpyDtoDAsync((hipDeviceptr_t)d,(hipDeviceptr_t)s,b,(hipStream_t)st); }
    void copyDToHInternal(void* d,const void* s,PxU32 b) override { hipMemcpyDtoH(d,(hipDeviceptr_t)s,b); }
    void copyHToDInternal(void* d,const void* s,PxU32 b) override { hipMemcpyHtoD((hipDeviceptr_t)d,(void*)s,b); }
    void memsetD8AsyncInternal(void* d,const PxU8& v,PxU32 b,CUstream s) override { hipMemsetD8Async((hipDeviceptr_t)d,v,b,(hipStream_t)s); }
    void memsetD32AsyncInternal(void* d,const PxU32& v,PxU32 n,CUstream s) override { hipMemsetD32Async((hipDeviceptr_t)d,v,n,(hipStream_t)s); }

public:
    HipCtxMgr(const PxCudaContextManagerDesc& d, PxProfilerCallback* = nullptr, bool sync = false) : mValid(false), mHCtx(nullptr) {
        mName[0]=0; int ord=d.deviceOrdinal<0?0:d.deviceOrdinal;
        if(hipInit(0)!=hipSuccess)return;
        if(hipDeviceGet(&mDev,ord)!=hipSuccess)return;
        if(hipCtxCreate(&mCtx,0,mDev)!=hipSuccess)return;
        mHCtx = new HipCtx(d.deviceAllocator, sync);
        hipDeviceProp_t pr; hipGetDeviceProperties(&pr,mDev);
        mOrd=ord; strncpy(mName,pr.name,127);
        mTM=pr.totalGlobalMem; mMC=pr.multiProcessorCount; mMT=pr.maxThreadsPerBlock; mSM=pr.sharedMemPerBlock;
        loadGpuLibrary();
        mMods.resize(gModules.size());
        for(PxU32 i = 0; i < gModules.size(); i++)
            mMods[i] = gModules[i];
        mTLS=PxTlsAlloc(); mValid=true;
    }

    ~HipCtxMgr() override {
        // Sync device before cleanup to prevent segfault from pending GPU work
        if(mValid) hipDeviceSynchronize();
        if(mHCtx){ for(PxU32 i=0;i<mMods.size();i++) if(mMods[i]) hipModuleUnload(mMods[i]); mHCtx->release(); }
        if(mCtx) hipCtxDestroy(mCtx); PxTlsFree(mTLS);
    }

    void release() override { delete this; }
    void acquireContext() override { hipCtxPushCurrent(mCtx); PxTlsSetValue(mTLS,PxTlsGetValue(mTLS)+1); }
    void releaseContext() override { PxU32 r=PxTlsGetValue(mTLS); if(--r==0){hipCtx_t c;hipCtxPopCurrent(&c);} PxTlsSetValue(mTLS,r); }
    bool tryAcquireContext() override { acquireContext(); return true; }
    bool contextIsValid() const override { return mValid; }
    bool supportsArchSM10()const override{return mValid;} bool supportsArchSM11()const override{return mValid;}
    bool supportsArchSM12()const override{return mValid;} bool supportsArchSM13()const override{return mValid;}
    bool supportsArchSM20()const override{return mValid;} bool supportsArchSM30()const override{return mValid;}
    bool supportsArchSM35()const override{return mValid;} bool supportsArchSM50()const override{return mValid;}
    bool supportsArchSM52()const override{return mValid;} bool supportsArchSM60()const override{return mValid;}
    bool isIntegrated() const override { return false; } bool canMapHostMemory() const override { return true; }
    int getDriverVersion() const override { return 12000; }
    size_t getDeviceTotalMemBytes() const override { return mTM; }
    int getMultiprocessorCount() const override { return mMC; }
    int getSharedMemPerBlock() const override { return mSM; }
    int getSharedMemPerMultiprocessor() const override { return mSM; }
    unsigned getMaxThreadsPerBlock() const override { return (unsigned)mMT; }
    unsigned getClockRate() const override { return 1400; }
    const char* getDeviceName() const override { return mValid?mName:"Invalid"; }
    CUdevice getDevice() const override { return (CUdevice)(mValid?mOrd:-1); }
    CUcontext getContext() override { return (CUcontext)mCtx; }
    PxCudaContext* getCudaContext() override { return mHCtx; }
    CUmodule* getCuModules() override { return (CUmodule*)mMods.begin(); }
    PxU32     getCuModuleCount()          { return mMods.size(); }
    void setUsingConcurrentStreams(bool) override {}
    bool getUsingConcurrentStreams() const override { return true; }
    void getDeviceMemoryInfo(size_t& f, size_t& t) const override { hipMemGetInfo(&f,&t); }
    CUdeviceptr getMappedDevicePtr(void* pb) override { CUdeviceptr d=0; hipHostGetDevicePointer((void**)&d,pb,0); return d; }
    int getKernelLaunchCount() const { return mHCtx ? mHCtx->totalKernels() : 0; }
};

// GPU stubs (replaces PxGpu.cpp)
void PxLoadPhysxGPUModule(const char*) {}
void PxUnloadPhysxGPUModule() {}
class PxProfilerCallback;
typedef PxCudaContextManager* (*PxCreatePhysXGpu_Func)(PxFoundation&,const PxCudaContextManagerDesc&,PxProfilerCallback*,bool);
PxCreatePhysXGpu_Func g_PxCreatePhysXGpu_Func = nullptr;

} // namespace physx

extern "C" physx::PxCudaContextManager* PxCreateCudaContextManager(
    physx::PxFoundation&, const physx::PxCudaContextManagerDesc& d,
    physx::PxProfilerCallback*, bool s)
{ return new physx::HipCtxMgr(d,nullptr,s); }
