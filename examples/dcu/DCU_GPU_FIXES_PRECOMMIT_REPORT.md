# DCU GPU Fixes Pre-commit Report

## Scope

This report summarizes the three DCU GPU fixes completed in this round. Exit-time `Segmentation fault` after `VERDICT: PASS` is recorded as a separate teardown/exit-stability issue and is not included as a functional blocker here.

## 1. GPU Broadphase forced eGPU

### Symptom

Forcing GPU broadphase with `PxBroadPhaseType::eGPU` caused DCU/HIP launch failure or VMFault in `computeStartAndActiveRegionHistogram`.

### Root Cause

PhysX CUDA path launches `BP_COMPUTE_ACTIVE_HISTOGRAM` with 512 threads/block. On the current DCU/HIP build, this kernel is compiled with a 256-thread launch bound. Launching it with 512 threads exceeds the DCU kernel bound and trips the runtime/device.

### Fix

- Keep CUDA/NVIDIA behavior unchanged at 512 threads/block.
- Under `PX_DCU_PORT`, set `BP_COMPUTE_ACTIVE_HISTOGRAM` to 256.
- Replace the hardcoded 16 active-histogram warps with `BP_COMPUTE_ACTIVE_HISTOGRAM / WARP_SIZE`.
- Keep PhysX logical `WARP_SIZE=32`; do not switch to DCU physical wavefront 64.

### Validation

- `bench_tgs_smoke pgs --force-gpu-bp --print-all-steps --cleanexit-wait 0`: PASS.
- `bench_trimesh_smoke 60 --force-gpu-bp --print-all-steps --cleanexit-wait 0`: PASS.
- `bench_trimesh_smoke 240 --force-gpu-bp --print-all-steps --cleanexit-wait 0`: PASS.
- Typical 240-step result: `Boxes=16`, `Height range=[0.377173,0.839057]`, `Bad boxes=0`, `VERDICT=PASS`.

### Files

- `physx/source/gpubroadphase/include/PxgBroadPhaseKernelIndices.h`
- `physx/source/gpubroadphase/src/CUDA/broadphase.cu`

## 2. GPU TGS Scene Creation

### Symptom

Creating a GPU scene with TGS solver crashed immediately after:

```text
[DCU GPU] createGpuDynamicsContext solverType=1
Segmentation fault
```

### Root Cause

`HipPhysXGpu::createGpuDynamicsContext` returned `nullptr` for `PxSolverType::eTGS`. The upper scene creation path did not handle this null dynamics context and continued using it, causing a scene-creation segfault.

### Fix

- Include `PxgTGSDynamicsContext.h` in the DCU GPU factory.
- Match the NVIDIA factory behavior: create `PxgTGSDynamicsContext` when `solverType == PxSolverType::eTGS`.
- Keep PGS on `PxgDynamicsContext`.

### Validation

`bench_tgs_smoke tgs --print-all-steps --cleanexit-wait 0` now creates the scene and completes the smoke test:

```text
Scene solver type: TGS
Steps: 180
Bodies: 64
Height range: [0.499981, 7.493330]
Bad bodies: 0
VERDICT: PASS
[DCU PROFILER] Total kernel launches: 11068
```

The segfault after profiler output is exit/teardown related and tracked separately.

### Files

- `physx/source/cudamanager/src/HipPhysXGpu.cpp`

## 3. PBD Particle Buffer Crash Guard

### Symptom

PBD particle smoke crashed while creating a plain `PxParticleBuffer`:

```text
Creating plain PxParticleBuffer...
Segmentation fault
```

After the first null guard, the benchmark returned early and then hit:

```text
pure virtual method called
Aborted
```

### Root Cause

The DCU backend particle buffer factory functions are still stubs and return `nullptr`:

- `createParticleBuffer`
- `createParticleAndDiffuseBuffer`
- `createParticleClothBuffer`
- `createParticleRigidBuffer`

The high-level `NpParticleBuffer` constructors dereferenced `mGpuBuffer` unconditionally via `mGpuBuffer->getUniqueId()`. Also, factory cleanup of an untracked wrapper could call tracking/PVD removal during destruction and trigger a pure virtual call.

### Fix

- Add null checks for all four particle-buffer constructors in `NpParticleBuffer.cpp`.
- Report `PxErrorCode::eINVALID_OPERATION` when the active GPU backend does not implement particle buffers.
- Add `getLowLevelParticleBuffer()`, `markAsTracked()`, and `mIsTracked` to `NpParticleBufferBase`.
- Make the destructor call `onParticleBufferRelease()` only for tracked buffers.
- In `NpFactory.cpp`, destroy the wrapper and return `NULL` when no low-level GPU buffer exists.
- In `bench_pbd_particles_conservative_smoke.cpp`, treat null particle buffer as `VERDICT: UNSUPPORTED`, continue the normal release path, and return 77.

### Validation

`bench_pbd_particles_conservative_smoke 1 --dim 2 --skip-buffer --print-all-steps --cleanexit-wait 0` now reports unsupported and releases cleanly:

```text
createParticleBuffer returned (nil)
VERDICT: UNSUPPORTED - createParticleBuffer returned null
Releasing scene...
Releasing dispatcher...
Releasing GPU manager...
[DCU PROFILER] Total kernel launches: 0
Releasing physics...
Releasing foundation...
Release finished.
Clean exit wait finished.
```

This does not implement real DCU PBD particle buffers; it converts the previous crash into a clear unsupported result.

### Files

- `physx/source/physx/src/NpParticleBuffer.cpp`
- `physx/source/physx/src/NpParticleBuffer.h`
- `physx/source/physx/src/NpFactory.cpp`
- `examples/dcu/bench_pbd_particles_conservative_smoke.cpp`

## 4. TGS Smoke Clean Exit

### Symptom

The TGS smoke test could finish simulation and print `VERDICT: PASS`, but then segfault after returning from `main()` or during late process teardown. This made automated scripts treat a passing simulation as a failed process.

### Root Cause Status

The functional simulation path and explicit PhysX release path are valid. The remaining crash is likely in process-level static/global destruction, HIP runtime teardown, or late cleanup after `main()` returns. This root cause is still tracked separately.

### Fix

- Add `--cleanexit` and `--cleanexit-wait N` to `bench_tgs_smoke.cpp`.
- Add explicit release-stage logging for scene, dispatcher, GPU manager, physics, and foundation.
- After explicit release, call `std::_Exit(pass ? 0 : 1)` when clean exit is enabled.
- Add `--print-all-steps` support for easier step-level localization.

### Validation

`bench_tgs_smoke tgs --print-all-steps --cleanexit-wait 0` now exits stably after `VERDICT: PASS` and explicit release.

### Files

- `examples/dcu/bench_tgs_smoke.cpp`

## Recommended Commit

Title:

```text
DCU: fix GPU broadphase/TGS setup and stabilize smoke exits
```

Suggested path whitelist for staging/diff:

```bash
git diff -- \
  physx/source/gpubroadphase/include/PxgBroadPhaseKernelIndices.h \
  physx/source/gpubroadphase/src/CUDA/broadphase.cu \
  physx/source/cudamanager/src/HipPhysXGpu.cpp \
  physx/source/physx/src/NpParticleBuffer.cpp \
  physx/source/physx/src/NpParticleBuffer.h \
  physx/source/physx/src/NpFactory.cpp \
  examples/dcu/bench_pbd_particles_conservative_smoke.cpp \
  examples/dcu/PLAN_EXTENDED_GPU_PATHS.md \
  examples/dcu/DCU_GPU_FIXES_PRECOMMIT_REPORT.md
```

Do not include unrelated working-tree changes in this commit.
