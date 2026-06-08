# NVIDIA PhysX

<details>
<summary>Copyright & License</summary>

Copyright (c) 2008-2025 NVIDIA Corporation. All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions
are met:
 * Redistributions of source code must retain the above copyright
   notice, this list of conditions and the following disclaimer.
 * Redistributions in binary form must reproduce the above copyright
   notice, this list of conditions and the following disclaimer in the
   documentation and/or other materials provided with the distribution.
 * Neither the name of NVIDIA CORPORATION nor the names of its
   contributors may be used to endorse or promote products derived
   from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS "AS IS" AND ANY
EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

## Hygon DCU / HIP Port

PhysX GPU modules have been ported to **Hygon DCU** (gfx936/gfx938) with HIP.

### Quick Start

```bash
# Build GPU libraries
./build_dcu.sh gfx936

# Build CPU libraries (Foundation, PhysXSDK, etc.)
./build_cpu.sh

# Build and run examples
cd examples/dcu && ./build.sh
```

### Build Requirements (DCU)

| Component | Version |
|-----------|---------|
| DTK (DCU Toolkit) | 2604 | 
| CMake | >= 3.21 |
| g++ | >= 8.0 |

### Port Summary

| Layer | Modules | Status |
|-------|---------|--------|
| GPU Kernel | PhysXCommonGpu, BroadphaseGpu, ArticulationGpu, SolverGpu, SimulationControllerGpu, NarrowphaseGpu | 100% (69 .cu files) |
| CPU Library | Foundation, Common, LowLevel, LowLevelAABB, LowLevelDynamics, SceneQuery, SimulationController, PhysXSDK | 100% |

### Key Adaptations

- WARP_SIZE = 64 (wavefront)
- FULL_MASK = 0xffffffffffffffff (64-bit)
- CUDA intrinsics → HIP macros (`shuffle.cuh`, `atomic.cuh`, `warpHelpers.cuh`)
- Named barriers → `__syncthreads()`
- PTX inline assembly → HIP equivalents
- 3D texture sampling → stubbed (SDF collision path)

### Example Results (BW200, UBB BW1000, 80 CUs)

| Example | Performance | Accuracy vs CPU |
|---------|------------|-----------------|
| N-Body (16K) | sub-ms per step | max err < 1e-3 |
| Particles (1M) | 3000+ FPS | max err < 0.01 |
| Raycast | 100M+ rays/s | max err < 1e-3 |

### Adapted Files

- **New**: `physx/source/gpucommon/include/PxgHIPCompat.h` — HIP compatibility layer
- **New**: `physx/source/gpucommon/src/DCU/` — runtime, stubs, tests
- **New**: `physx/source/compiler/cmakehip/` — HIP CMake build
- **New**: `physx/source/compiler/cmakecpu/` — CPU library CMake build
- **Modified**: ~55 .cu/.cuh files — `cuda.h` → conditional, intrinsic macros, PTX asm

</details>

Please also see license files in the root folder and in the respective subfolders.

This repo contains:

| Directory | Description |
|---|---|
| [`ovphysx/`](ovphysx/) | ovphysx — C API with Python bindings for USD physics simulation with DLPack tensor interop (`pip install ovphysx`) |
| [`physx/`](physx/) | PhysX SDK — real-time physics simulation engine |
| [`omni/`](omni/) | Omniverse PhysX extensions for Kit-based applications |

Additional simulation libraries:

| Directory | Description |
|---|---|
| [`blast/`](blast/) | Blast SDK — destruction and fracture simulation |
| [`flow/`](flow/) | Flow SDK — fluid and fire simulation |

## Support

* Please use GitHub [Discussions](https://github.com/NVIDIA-Omniverse/PhysX/discussions/) for questions and comments.
* GitHub [Issues](https://github.com/NVIDIA-Omniverse/PhysX/issues) should only be used for bug reports or documentation issues.
* You can also ask questions in the NVIDIA Omniverse #physics [Discord Channel](https://discord.com/invite/XWQNJDNuaC).
