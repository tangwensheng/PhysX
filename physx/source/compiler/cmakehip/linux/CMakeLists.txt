# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#  * Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#  * Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in the
#    documentation and/or other materials provided with the distribution.
#  * Neither the name of NVIDIA CORPORATION nor the names of its
#    contributors may be used to endorse or promote products derived
#    from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ''AS IS'' AND ANY
# EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
# PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
# PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
# OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

# PhysX HCU/HIP — Linux Platform Configuration
#
# Mirrors cmakegpu/linux/CMakeLists.txt but uses hipcc instead of nvcc.
# This is the platform-specific configuration for HIP GPU builds on Linux.

# Use PhysX C++ flags (cached from main PhysX CMake build)
if(DEFINED PHYSX_CXX_FLAGS)
    set(CMAKE_CXX_FLAGS "${PHYSX_CXX_FLAGS}")
    set(CMAKE_CXX_FLAGS_DEBUG   "${PHYSX_CXX_FLAGS_DEBUG}")
    set(CMAKE_CXX_FLAGS_CHECKED "${PHYSX_CXX_FLAGS_CHECKED}")
    set(CMAKE_CXX_FLAGS_PROFILE "${PHYSX_CXX_FLAGS_PROFILE}")
    set(CMAKE_CXX_FLAGS_RELEASE "${PHYSX_CXX_FLAGS_RELEASE}")
endif()

# ---- HIP Compiler Flags (equivalent to CUDA flags in cmakegpu/linux) ----
# Fast math + flush denormals (matching NVIDIA -use_fast_math -ftz=true)
set(HIP_FAST_MATH_FLAGS "-ffast-math -fgpu-flush-denormals-to-zero")

# Suppress warnings
set(HIP_SUPPRESS_WARNINGS "-Wno-unused-function -Wno-inconsistent-missing-override")

# Force-include PxgHIPCompat.h to all HIP translation units
# This provides: HIP intrinsics, PhysX foundation types, CUDA compatibility
set(HIP_INCLUDE_FLAGS "-include PxgHIPCompat.h -DPX_DCU_PORT")

set(PHYSX_HIP_FLAGS "${HIP_INCLUDE_FLAGS} ${HIP_FAST_MATH_FLAGS} ${HIP_SUPPRESS_WARNINGS}")

# Debug/checked/profile/release variants
set(PHYSX_HIP_FLAGS_DEBUG   "-g -O0")
set(PHYSX_HIP_FLAGS_CHECKED "-lineinfo -O2")
set(PHYSX_HIP_FLAGS_PROFILE "-lineinfo -O2")
set(PHYSX_HIP_FLAGS_RELEASE "-O3")

# Set HIP flags
set(CMAKE_HIP_FLAGS "${PHYSX_HIP_FLAGS}")
set(CMAKE_HIP_FLAGS_DEBUG   "${PHYSX_HIP_FLAGS_DEBUG}")
set(CMAKE_HIP_FLAGS_CHECKED "${PHYSX_HIP_FLAGS_CHECKED}")
set(CMAKE_HIP_FLAGS_PROFILE "${PHYSX_HIP_FLAGS_PROFILE}")
set(CMAKE_HIP_FLAGS_RELEASE "${PHYSX_HIP_FLAGS_RELEASE}")

# ---- GPU Architecture ----
# Default HCU architectures
if(NOT DEFINED DCU_ARCHITECTURES)
    set(DCU_ARCHITECTURES "gfx936" CACHE STRING "HCU device architectures")
endif()
set(CMAKE_HIP_ARCHITECTURES "${DCU_ARCHITECTURES}")

# ---- Static library defines ----
if(PX_GENERATE_GPU_STATIC_LIBRARIES)
    set(PHYSXGPU_LIBTYPE_DEFS "PX_PHYSX_GPU_STATIC;" CACHE INTERNAL "GPU lib type defs")
endif()
