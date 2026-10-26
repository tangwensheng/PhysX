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

cmake_minimum_required(VERSION 3.21)
project(PhysXGpuHIP LANGUAGES CXX HIP)
set(CMAKE_HIP_FLAGS "${CMAKE_HIP_FLAGS} -include PxgHIPCompat.h -DPX_HCU_PORT -DNDEBUG -ffast-math -fgpu-flush-denormals-to-zero")
if(NOT DEFINED PHYSX_ROOT_DIR)
    get_filename_component(PHYSX_ROOT_DIR "${CMAKE_SOURCE_DIR}/../../.." ABSOLUTE)
endif()
if(NOT DEFINED HCU_ARCH)
    set(HCU_ARCH "gfx936" CACHE STRING "HCU device architecture")
endif()
set(CMAKE_HIP_ARCHITECTURES "${HCU_ARCH}")
add_compile_definitions(PX_HCU_PORT PX_PHYSX_GPU_STATIC PX_SUPPORT_GPU_PHYSX)
set(PHYSX_INCLUDE_DIRS
    ${PHYSX_ROOT_DIR}/include ${PHYSX_ROOT_DIR}/include/geometry
    ${PHYSX_ROOT_DIR}/source/common/include ${PHYSX_ROOT_DIR}/source/common/src
    ${PHYSX_ROOT_DIR}/source/foundation/include
    ${PHYSX_ROOT_DIR}/source/gpucommon/include ${PHYSX_ROOT_DIR}/source/gpucommon/src/CUDA
    ${PHYSX_ROOT_DIR}/source/gpucommon/src/HCU/stubs
    ${PHYSX_ROOT_DIR}/source/gpunarrowphase/include ${PHYSX_ROOT_DIR}/source/gpunarrowphase/src/CUDA
    ${PHYSX_ROOT_DIR}/source/gpusolver/include ${PHYSX_ROOT_DIR}/source/gpusolver/src/CUDA
    ${PHYSX_ROOT_DIR}/source/gpusimulationcontroller/include ${PHYSX_ROOT_DIR}/source/gpusimulationcontroller/src/CUDA
    ${PHYSX_ROOT_DIR}/source/gpubroadphase/include ${PHYSX_ROOT_DIR}/source/gpubroadphase/src/CUDA
    ${PHYSX_ROOT_DIR}/source/gpuarticulation/include ${PHYSX_ROOT_DIR}/source/gpuarticulation/src/CUDA
    ${PHYSX_ROOT_DIR}/source/lowlevel/api/include ${PHYSX_ROOT_DIR}/source/lowlevel/software/include
    ${PHYSX_ROOT_DIR}/source/lowlevel/common/include ${PHYSX_ROOT_DIR}/source/lowleveldynamics/include
    ${PHYSX_ROOT_DIR}/source/lowleveldynamics/shared ${PHYSX_ROOT_DIR}/source/lowlevelaabb/include
    ${PHYSX_ROOT_DIR}/source/geomutils/src ${PHYSX_ROOT_DIR}/source/geomutils/src/mesh
    ${PHYSX_ROOT_DIR}/source/geomutils/include ${PHYSX_ROOT_DIR}/source/cudamanager/include
    ${PHYSX_ROOT_DIR}/source/simulationcontroller/include ${PHYSX_ROOT_DIR}/source/scenequery/include
)

# Collect ALL .cu files (including convex modules for HCU)
file(GLOB GPU_ALL_CU_FILES
    ${PHYSX_ROOT_DIR}/source/gpucommon/src/CUDA/*.cu
    ${PHYSX_ROOT_DIR}/source/gpubroadphase/src/CUDA/*.cu
    ${PHYSX_ROOT_DIR}/source/gpuarticulation/src/CUDA/*.cu
    ${PHYSX_ROOT_DIR}/source/gpusolver/src/CUDA/*.cu
    ${PHYSX_ROOT_DIR}/source/gpusimulationcontroller/src/CUDA/*.cu
    ${PHYSX_ROOT_DIR}/source/gpunarrowphase/src/CUDA/*.cu
)

# Build shared library (for tests, static linkage)
add_library(PhysXGpuHCU SHARED ${GPU_ALL_CU_FILES})
target_include_directories(PhysXGpuHCU PRIVATE ${PHYSX_INCLUDE_DIRS})
set_source_files_properties(${GPU_ALL_CU_FILES} PROPERTIES LANGUAGE HIP)
option(PX_HCU_VOLUME_O3_P33 "Opt-in gfx936 softBodyGM translation-unit O3 candidate" OFF)
if(PX_HCU_VOLUME_O3_P33)
    if(NOT HCU_ARCH STREQUAL "gfx936")
        message(FATAL_ERROR "PX_HCU_VOLUME_O3_P33 is validated only for gfx936")
    endif()
    set_property(SOURCE ${PHYSX_ROOT_DIR}/source/gpusimulationcontroller/src/CUDA/softBodyGM.cu
        APPEND PROPERTY COMPILE_OPTIONS -O3 -DPX_HCU_VOLUME_O3_P33=1)
endif()
option(PX_HCU_VOLUME_SOFTBODY_O3_P36 "Opt-in gfx936 softBody translation-unit O3 candidate on P33" OFF)
if(PX_HCU_VOLUME_SOFTBODY_O3_P36)
    if(NOT HCU_ARCH STREQUAL "gfx936" OR NOT PX_HCU_VOLUME_O3_P33)
        message(FATAL_ERROR "PX_HCU_VOLUME_SOFTBODY_O3_P36 requires gfx936 and PX_HCU_VOLUME_O3_P33=ON")
    endif()
    set_property(SOURCE ${PHYSX_ROOT_DIR}/source/gpusimulationcontroller/src/CUDA/softBody.cu
        APPEND PROPERTY COMPILE_OPTIONS -O3 -DPX_HCU_VOLUME_SOFTBODY_O3_P36=1)
endif()
option(PX_HCU_VOLUME_RIGID_DELTA_O3_P37 "Opt-in gfx936 rigidDeltaAccum translation-unit O3 candidate on P33/P36" OFF)
option(PX_HCU_RIGID_DELTA_ACCUM_O3_P43 "Opt-in gfx936 rigidDeltaAccum translation-unit O3 candidate for ordinary rigid-body path" OFF)
# Reset this TU's source-local options before applying either scoped product.
# P37 is the historical Volume chain; P43 is independent and must be measured
# with the Volume chain disabled so the ordinary-rigid product has one owner.
set_property(SOURCE ${PHYSX_ROOT_DIR}/source/gpusimulationcontroller/src/CUDA/rigidDeltaAccum.cu
    PROPERTY COMPILE_OPTIONS "")
if(PX_HCU_VOLUME_RIGID_DELTA_O3_P37 AND PX_HCU_RIGID_DELTA_ACCUM_O3_P43)
    message(FATAL_ERROR "PX_HCU_VOLUME_RIGID_DELTA_O3_P37 and PX_HCU_RIGID_DELTA_ACCUM_O3_P43 are mutually exclusive")
endif()
if(PX_HCU_VOLUME_RIGID_DELTA_O3_P37)
    if(NOT HCU_ARCH STREQUAL "gfx936" OR NOT PX_HCU_VOLUME_O3_P33 OR NOT PX_HCU_VOLUME_SOFTBODY_O3_P36)
        message(FATAL_ERROR "PX_HCU_VOLUME_RIGID_DELTA_O3_P37 requires gfx936 and P33/P36=ON")
    endif()
    set_property(SOURCE ${PHYSX_ROOT_DIR}/source/gpusimulationcontroller/src/CUDA/rigidDeltaAccum.cu
        APPEND PROPERTY COMPILE_OPTIONS -O3 -DPX_HCU_VOLUME_RIGID_DELTA_O3_P37=1)
endif()
option(PX_HCU_VOLUME_RADIX_O3_P38 "Opt-in gfx936 radixSortImpl translation-unit O3 candidate on P33/P36/P37" OFF)
if(PX_HCU_VOLUME_RADIX_O3_P38)
    if(NOT HCU_ARCH STREQUAL "gfx936" OR NOT PX_HCU_VOLUME_O3_P33 OR NOT PX_HCU_VOLUME_SOFTBODY_O3_P36 OR NOT PX_HCU_VOLUME_RIGID_DELTA_O3_P37)
        message(FATAL_ERROR "PX_HCU_VOLUME_RADIX_O3_P38 requires gfx936 and P33/P36/P37=ON")
    endif()
    set_property(SOURCE ${PHYSX_ROOT_DIR}/source/gpucommon/src/CUDA/radixSortImpl.cu
        APPEND PROPERTY COMPILE_OPTIONS -O3 -DPX_HCU_VOLUME_RADIX_O3_P38=1)
endif()
option(PX_HCU_VOLUME_SOFTBODY_PRIMITIVES_O3_P39 "Opt-in gfx936 softbodyPrimitives translation-unit O3 candidate on P33/P36/P37/P38" OFF)
if(PX_HCU_VOLUME_SOFTBODY_PRIMITIVES_O3_P39)
    if(NOT HCU_ARCH STREQUAL "gfx936" OR NOT PX_HCU_VOLUME_O3_P33 OR NOT PX_HCU_VOLUME_SOFTBODY_O3_P36 OR NOT PX_HCU_VOLUME_RIGID_DELTA_O3_P37 OR NOT PX_HCU_VOLUME_RADIX_O3_P38)
        message(FATAL_ERROR "PX_HCU_VOLUME_SOFTBODY_PRIMITIVES_O3_P39 requires gfx936 and P33/P36/P37/P38=ON")
    endif()
    set_property(SOURCE ${PHYSX_ROOT_DIR}/source/gpunarrowphase/src/CUDA/softbodyPrimitives.cu
        APPEND PROPERTY COMPILE_OPTIONS -O3 -DPX_HCU_VOLUME_SOFTBODY_PRIMITIVES_O3_P39=1)
endif()
if(PX_HCU_RIGID_DELTA_ACCUM_O3_P43)
    if(NOT HCU_ARCH STREQUAL "gfx936")
        message(FATAL_ERROR "PX_HCU_RIGID_DELTA_ACCUM_O3_P43 is validated only for gfx936")
    endif()
    set_property(SOURCE ${PHYSX_ROOT_DIR}/source/gpusimulationcontroller/src/CUDA/rigidDeltaAccum.cu
        APPEND PROPERTY COMPILE_OPTIONS -O3 -DPX_HCU_RIGID_DELTA_ACCUM_O3_P43=1)
endif()
# P45 changes only this translation unit's optimization level. Physics stays intact.
option(PX_HCU_RIGID_GJKEPA_O3_P45 "Opt-in gfx936 cudaGJKEPA translation-unit O3 candidate for ordinary rigid-body path" OFF)
set_property(SOURCE ${PHYSX_ROOT_DIR}/source/gpunarrowphase/src/CUDA/cudaGJKEPA.cu
    PROPERTY COMPILE_OPTIONS "")
if(PX_HCU_RIGID_GJKEPA_O3_P45)
    if(NOT HCU_ARCH STREQUAL "gfx936")
        message(FATAL_ERROR "PX_HCU_RIGID_GJKEPA_O3_P45 is validated only for gfx936")
    endif()
    set_property(SOURCE ${PHYSX_ROOT_DIR}/source/gpunarrowphase/src/CUDA/cudaGJKEPA.cu
        APPEND PROPERTY COMPILE_OPTIONS -O3 -DPX_HCU_RIGID_GJKEPA_O3_P45=1)
endif()
option(PX_HCU_SDF_CONVEX_MIDPHASE_O3_P41 "Opt-in gfx936 convexMeshMidphase translation-unit O3 candidate for T2 SDF PGS" OFF)
# Always reset this TU's source-local options before applying the product
# selection. The ABBA runner reconfigures the same build tree OFF -> ON ->
# ON -> OFF; an APPEND-only property can otherwise leave the P41 define in a
# later OFF configure on toolchains/generators that retain source properties.
set_property(SOURCE ${PHYSX_ROOT_DIR}/source/gpunarrowphase/src/CUDA/convexMeshMidphase.cu
    PROPERTY COMPILE_OPTIONS "")
if(PX_HCU_SDF_CONVEX_MIDPHASE_O3_P41)
    if(NOT HCU_ARCH STREQUAL "gfx936")
        message(FATAL_ERROR "PX_HCU_SDF_CONVEX_MIDPHASE_O3_P41 is validated only for gfx936")
    endif()
    set_property(SOURCE ${PHYSX_ROOT_DIR}/source/gpunarrowphase/src/CUDA/convexMeshMidphase.cu
        APPEND PROPERTY COMPILE_OPTIONS -O3 -DPX_HCU_SDF_CONVEX_MIDPHASE_O3_P41=1)
endif()
option(PX_HCU_RIGID_CUDABOX_O3_P42 "Opt-in gfx936 cudaBox translation-unit O3 candidate for ordinary rigid-body path" OFF)
# Reset this TU's source-local options before applying the product selection.
# The same build tree is reconfigured repeatedly by scoped experiments.
set_property(SOURCE ${PHYSX_ROOT_DIR}/source/gpunarrowphase/src/CUDA/cudaBox.cu
    PROPERTY COMPILE_OPTIONS "")
if(PX_HCU_RIGID_CUDABOX_O3_P42)
    if(NOT HCU_ARCH STREQUAL "gfx936")
        message(FATAL_ERROR "PX_HCU_RIGID_CUDABOX_O3_P42 is validated only for gfx936")
    endif()
    set_property(SOURCE ${PHYSX_ROOT_DIR}/source/gpunarrowphase/src/CUDA/cudaBox.cu
        APPEND PROPERTY COMPILE_OPTIONS -O3 -DPX_HCU_RIGID_CUDABOX_O3_P42=1)
endif()
# P47: ordinary-rigid cudaBox O3 on the P43/P45 product; no physics changes.
# P42 remains the historical standalone experiment. Never combine both owners.
option(PX_HCU_RIGID_CUDABOX_O3_P47 "Opt-in gfx936 cudaBox TU O3 candidate on ordinary-rigid P43/P45" OFF)
if(PX_HCU_RIGID_CUDABOX_O3_P47)
    if(NOT HCU_ARCH STREQUAL "gfx936" OR NOT PX_HCU_RIGID_DELTA_ACCUM_O3_P43 OR NOT PX_HCU_RIGID_GJKEPA_O3_P45 OR PX_HCU_RIGID_CUDABOX_O3_P42)
        message(FATAL_ERROR "PX_HCU_RIGID_CUDABOX_O3_P47 requires gfx936, P43/P45=ON and P42=OFF")
    endif()
    set_property(SOURCE ${PHYSX_ROOT_DIR}/source/gpunarrowphase/src/CUDA/cudaBox.cu
        APPEND PROPERTY COMPILE_OPTIONS -O3 -DPX_HCU_RIGID_CUDABOX_O3_P47=1)
endif()
# P48: TU-local compiler candidate; all kernels in this module share O3.
option(PX_HCU_RIGID_SOLVER_MULTIBLOCK_O3_P48 "Opt-in gfx936 solverMultiBlock TU O3 on ordinary-rigid P43/P45/P47" OFF)
set_property(SOURCE ${PHYSX_ROOT_DIR}/source/gpusolver/src/CUDA/solverMultiBlock.cu
    PROPERTY COMPILE_OPTIONS "")
if(PX_HCU_RIGID_SOLVER_MULTIBLOCK_O3_P48)
    if(NOT HCU_ARCH STREQUAL "gfx936" OR NOT PX_HCU_RIGID_DELTA_ACCUM_O3_P43 OR NOT PX_HCU_RIGID_GJKEPA_O3_P45 OR NOT PX_HCU_RIGID_CUDABOX_O3_P47 OR PX_HCU_RIGID_CUDABOX_O3_P42)
        message(FATAL_ERROR "PX_HCU_RIGID_SOLVER_MULTIBLOCK_O3_P48 requires gfx936, P43/P45/P47=ON and P42=OFF")
    endif()
    set_property(SOURCE ${PHYSX_ROOT_DIR}/source/gpusolver/src/CUDA/solverMultiBlock.cu
        APPEND PROPERTY COMPILE_OPTIONS -O3 -DPX_HCU_RIGID_SOLVER_MULTIBLOCK_O3_P48=1)
endif()
# P49: ordinary-rigid pair-management TU compiler candidate; preserve serial compaction.
option(PX_HCU_RIGID_PAIR_MANAGEMENT_O3_P49 "Opt-in gfx936 pairManagement TU O3 on ordinary-rigid P43/P45/P47/P48" OFF)
set_property(SOURCE ${PHYSX_ROOT_DIR}/source/gpunarrowphase/src/CUDA/pairManagement.cu
    PROPERTY COMPILE_OPTIONS "")
if(PX_HCU_RIGID_PAIR_MANAGEMENT_O3_P49)
    if(NOT HCU_ARCH STREQUAL "gfx936" OR NOT PX_HCU_RIGID_DELTA_ACCUM_O3_P43 OR NOT PX_HCU_RIGID_GJKEPA_O3_P45 OR NOT PX_HCU_RIGID_CUDABOX_O3_P47 OR NOT PX_HCU_RIGID_SOLVER_MULTIBLOCK_O3_P48 OR PX_HCU_RIGID_CUDABOX_O3_P42)
        message(FATAL_ERROR "PX_HCU_RIGID_PAIR_MANAGEMENT_O3_P49 requires gfx936, P43/P45/P47/P48=ON and P42=OFF")
    endif()
    set_property(SOURCE ${PHYSX_ROOT_DIR}/source/gpunarrowphase/src/CUDA/pairManagement.cu
        APPEND PROPERTY COMPILE_OPTIONS -O3 -DPX_HCU_RIGID_PAIR_MANAGEMENT_O3_P49=1)
endif()
# P59: dedicated gfx936 ordinary-rigid product; all kernels in this TU use O3.
# This is a compiler-only candidate, not a change to contact preparation algorithms.
option(PX_HCU_RIGID_CONTACT_PREP_O3_P59 "Opt-in gfx936 constraintBlockPrep TU O3 on ordinary-rigid P43/P45/P47/P48/P49" OFF)
set_property(SOURCE ${PHYSX_ROOT_DIR}/source/gpusolver/src/CUDA/constraintBlockPrep.cu
    PROPERTY COMPILE_OPTIONS "")
if(PX_HCU_RIGID_CONTACT_PREP_O3_P59)
    if(NOT HCU_ARCH STREQUAL "gfx936" OR NOT PX_HCU_RIGID_DELTA_ACCUM_O3_P43 OR NOT PX_HCU_RIGID_GJKEPA_O3_P45 OR NOT PX_HCU_RIGID_CUDABOX_O3_P47 OR NOT PX_HCU_RIGID_SOLVER_MULTIBLOCK_O3_P48 OR NOT PX_HCU_RIGID_PAIR_MANAGEMENT_O3_P49 OR PX_HCU_RIGID_CUDABOX_O3_P42)
        message(FATAL_ERROR "P59 requires gfx936, P43/P45/P47/P48/P49=ON and P42=OFF; use only the ordinary-rigid candidate product")
    endif()
    set_property(SOURCE ${PHYSX_ROOT_DIR}/source/gpusolver/src/CUDA/constraintBlockPrep.cu
        APPEND PROPERTY COMPILE_OPTIONS -O3 -DPX_HCU_RIGID_CONTACT_PREP_O3_P59=1)
endif()
# P54 retains the serial algorithm and exposes a benchmark-only runtime ablation.
option(PX_HCU_RIGID_STAGE2_INDEX_P54 "Opt-in gfx936 ordinary-rigid Stage2 bounded-index candidate" OFF)
if(PX_HCU_RIGID_STAGE2_INDEX_P54)
    if(NOT HCU_ARCH STREQUAL "gfx936" OR NOT PX_HCU_RIGID_GJKEPA_O3_P45 OR NOT PX_HCU_RIGID_PAIR_MANAGEMENT_O3_P49)
        message(FATAL_ERROR "P54 requires gfx936 and the P49 ON product (all existing O3 options retained)")
    endif()
    set_property(SOURCE ${PHYSX_ROOT_DIR}/source/gpunarrowphase/src/CUDA/cudaGJKEPA.cu
        APPEND PROPERTY COMPILE_OPTIONS -DPX_HCU_RIGID_STAGE2_INDEX_P54=1)
endif()
# P60: preserve serial lost/found compaction and modulo indexing; batch flag reads only.
option(PX_HCU_RIGID_STAGE2_FLAGS4_P60 "Opt-in gfx936 ordinary-rigid Stage2 four-flag read candidate" OFF)
if(PX_HCU_RIGID_STAGE2_FLAGS4_P60)
    if(NOT HCU_ARCH STREQUAL "gfx936" OR NOT PX_HCU_RIGID_STAGE2_INDEX_P54 OR NOT PX_HCU_RIGID_CONTACT_PREP_O3_P59 OR NOT PX_HCU_RIGID_DELTA_ACCUM_O3_P43 OR NOT PX_HCU_RIGID_GJKEPA_O3_P45 OR NOT PX_HCU_RIGID_CUDABOX_O3_P47 OR NOT PX_HCU_RIGID_SOLVER_MULTIBLOCK_O3_P48 OR NOT PX_HCU_RIGID_PAIR_MANAGEMENT_O3_P49 OR PX_HCU_RIGID_CUDABOX_O3_P42)
        message(FATAL_ERROR "P60 requires gfx936, P43/P45/P47/P48/P49/P54/P59=ON and P42=OFF; ordinary-rigid dedicated candidate only")
    endif()
    set_property(SOURCE ${PHYSX_ROOT_DIR}/source/gpunarrowphase/src/CUDA/cudaGJKEPA.cu
        APPEND PROPERTY COMPILE_OPTIONS -DPX_HCU_RIGID_STAGE2_FLAGS4_P60=1)
endif()
# P64: split only the existing P54 bounded-index serial loop into two affine halves.
option(PX_HCU_RIGID_STAGE2_BOUNDED_SPLIT_P64 "Opt-in gfx936 ordinary-rigid Stage2 bounded-index split candidate" OFF)
if(PX_HCU_RIGID_STAGE2_BOUNDED_SPLIT_P64)
    if(NOT HCU_ARCH STREQUAL "gfx936" OR NOT PX_HCU_RIGID_STAGE2_INDEX_P54 OR NOT PX_HCU_RIGID_STAGE2_FLAGS4_P60 OR NOT PX_HCU_RIGID_CONTACT_PREP_O3_P59 OR NOT PX_HCU_RIGID_DELTA_ACCUM_O3_P43 OR NOT PX_HCU_RIGID_GJKEPA_O3_P45 OR NOT PX_HCU_RIGID_CUDABOX_O3_P47 OR NOT PX_HCU_RIGID_SOLVER_MULTIBLOCK_O3_P48 OR NOT PX_HCU_RIGID_PAIR_MANAGEMENT_O3_P49 OR PX_HCU_RIGID_CUDABOX_O3_P42)
        message(FATAL_ERROR "P64 requires gfx936, P43/P45/P47/P48/P49/P54/P59/P60=ON and P42=OFF; ordinary-rigid dedicated candidate only")
    endif()
    set_property(SOURCE ${PHYSX_ROOT_DIR}/source/gpunarrowphase/src/CUDA/cudaGJKEPA.cu
        APPEND PROPERTY COMPILE_OPTIONS -DPX_HCU_RIGID_STAGE2_BOUNDED_SPLIT_P64=1)
endif()
# P65: batch four flag reads inside the already bounded, serial P64 halves.
option(PX_HCU_RIGID_STAGE2_BOUNDED_PREFETCH_P65 "Opt-in gfx936 ordinary-rigid Stage2 bounded-split flag prefetch candidate" OFF)
if(PX_HCU_RIGID_STAGE2_BOUNDED_PREFETCH_P65)
    if(NOT HCU_ARCH STREQUAL "gfx936" OR NOT PX_HCU_RIGID_STAGE2_INDEX_P54 OR NOT PX_HCU_RIGID_STAGE2_FLAGS4_P60 OR NOT PX_HCU_RIGID_STAGE2_BOUNDED_SPLIT_P64 OR NOT PX_HCU_RIGID_CONTACT_PREP_O3_P59 OR NOT PX_HCU_RIGID_DELTA_ACCUM_O3_P43 OR NOT PX_HCU_RIGID_GJKEPA_O3_P45 OR NOT PX_HCU_RIGID_CUDABOX_O3_P47 OR NOT PX_HCU_RIGID_SOLVER_MULTIBLOCK_O3_P48 OR NOT PX_HCU_RIGID_PAIR_MANAGEMENT_O3_P49 OR PX_HCU_RIGID_CUDABOX_O3_P42)
        message(FATAL_ERROR "P65 requires gfx936, P43/P45/P47/P48/P49/P54/P59/P60/P64=ON and P42=OFF; ordinary-rigid dedicated candidate only")
    endif()
    set_property(SOURCE ${PHYSX_ROOT_DIR}/source/gpunarrowphase/src/CUDA/cudaGJKEPA.cu
        APPEND PROPERTY COMPILE_OPTIONS -DPX_HCU_RIGID_STAGE2_BOUNDED_PREFETCH_P65=1)
endif()
# P68: widen only the bounded serial flag-read group from four to eight entries.
option(PX_HCU_RIGID_STAGE2_BOUNDED_PREFETCH8_P68 "Opt-in gfx936 ordinary-rigid Stage2 bounded-split eight-flag prefetch candidate" OFF)
if(PX_HCU_RIGID_STAGE2_BOUNDED_PREFETCH8_P68)
    if(NOT HCU_ARCH STREQUAL "gfx936" OR NOT PX_HCU_RIGID_STAGE2_INDEX_P54 OR NOT PX_HCU_RIGID_STAGE2_FLAGS4_P60 OR NOT PX_HCU_RIGID_STAGE2_BOUNDED_SPLIT_P64 OR NOT PX_HCU_RIGID_STAGE2_BOUNDED_PREFETCH_P65 OR NOT PX_HCU_RIGID_CONTACT_PREP_O3_P59 OR NOT PX_HCU_RIGID_DELTA_ACCUM_O3_P43 OR NOT PX_HCU_RIGID_GJKEPA_O3_P45 OR NOT PX_HCU_RIGID_CUDABOX_O3_P47 OR NOT PX_HCU_RIGID_SOLVER_MULTIBLOCK_O3_P48 OR NOT PX_HCU_RIGID_PAIR_MANAGEMENT_O3_P49 OR PX_HCU_RIGID_CUDABOX_O3_P42)
        message(FATAL_ERROR "P68 requires gfx936, P43/P45/P47/P48/P49/P54/P59/P60/P64/P65=ON and P42=OFF; ordinary-rigid dedicated candidate only")
    endif()
    set_property(SOURCE ${PHYSX_ROOT_DIR}/source/gpunarrowphase/src/CUDA/cudaGJKEPA.cu
        APPEND PROPERTY COMPILE_OPTIONS -DPX_HCU_RIGID_STAGE2_BOUNDED_PREFETCH8_P68=1)
endif()
# P70: widen the existing P68 batch from eight to sixteen flags.
option(PX_HCU_RIGID_STAGE2_BOUNDED_PREFETCH16_P70 "Opt-in gfx936 ordinary-rigid Stage2 bounded-split sixteen-flag prefetch candidate" OFF)
if(PX_HCU_RIGID_STAGE2_BOUNDED_PREFETCH16_P70)
    if(NOT HCU_ARCH STREQUAL "gfx936" OR NOT PX_HCU_RIGID_STAGE2_INDEX_P54 OR NOT PX_HCU_RIGID_STAGE2_FLAGS4_P60 OR NOT PX_HCU_RIGID_STAGE2_BOUNDED_SPLIT_P64 OR NOT PX_HCU_RIGID_STAGE2_BOUNDED_PREFETCH_P65 OR NOT PX_HCU_RIGID_STAGE2_BOUNDED_PREFETCH8_P68 OR NOT PX_HCU_RIGID_CONTACT_PREP_O3_P59 OR NOT PX_HCU_RIGID_DELTA_ACCUM_O3_P43 OR NOT PX_HCU_RIGID_GJKEPA_O3_P45 OR NOT PX_HCU_RIGID_CUDABOX_O3_P47 OR NOT PX_HCU_RIGID_SOLVER_MULTIBLOCK_O3_P48 OR NOT PX_HCU_RIGID_PAIR_MANAGEMENT_O3_P49 OR PX_HCU_RIGID_CUDABOX_O3_P42)
        message(FATAL_ERROR "P70 requires gfx936, P43/P45/P47/P48/P49/P54/P59/P60/P64/P65/P68=ON and P42=OFF; ordinary-rigid dedicated candidate only")
    endif()
    set_property(SOURCE ${PHYSX_ROOT_DIR}/source/gpunarrowphase/src/CUDA/cudaGJKEPA.cu
        APPEND PROPERTY COMPILE_OPTIONS -DPX_HCU_RIGID_STAGE2_BOUNDED_PREFETCH16_P70=1)
endif()
# P72: retain P68's eight-flag batch and prefetch manager-output metadata/pointers.
option(PX_HCU_RIGID_STAGE2_METADATA_PREFETCH_P72 "Opt-in gfx936 ordinary-rigid Stage2 bounded-split metadata prefetch candidate" OFF)
if(PX_HCU_RIGID_STAGE2_METADATA_PREFETCH_P72)
    if(NOT HCU_ARCH STREQUAL "gfx936" OR NOT PX_HCU_RIGID_STAGE2_INDEX_P54 OR NOT PX_HCU_RIGID_STAGE2_FLAGS4_P60 OR NOT PX_HCU_RIGID_STAGE2_BOUNDED_SPLIT_P64 OR NOT PX_HCU_RIGID_STAGE2_BOUNDED_PREFETCH_P65 OR NOT PX_HCU_RIGID_STAGE2_BOUNDED_PREFETCH8_P68 OR PX_HCU_RIGID_STAGE2_BOUNDED_PREFETCH16_P70 OR NOT PX_HCU_RIGID_CONTACT_PREP_O3_P59 OR NOT PX_HCU_RIGID_DELTA_ACCUM_O3_P43 OR NOT PX_HCU_RIGID_GJKEPA_O3_P45 OR NOT PX_HCU_RIGID_CUDABOX_O3_P47 OR NOT PX_HCU_RIGID_SOLVER_MULTIBLOCK_O3_P48 OR NOT PX_HCU_RIGID_PAIR_MANAGEMENT_O3_P49 OR PX_HCU_RIGID_CUDABOX_O3_P42)
        message(FATAL_ERROR "P72 requires gfx936, P43/P45/P47/P48/P49/P54/P59/P60/P64/P65/P68=ON, P70=OFF and P42=OFF; ordinary-rigid dedicated candidate only")
    endif()
    set_property(SOURCE ${PHYSX_ROOT_DIR}/source/gpunarrowphase/src/CUDA/cudaGJKEPA.cu
        APPEND PROPERTY COMPILE_OPTIONS -DPX_HCU_RIGID_STAGE2_METADATA_PREFETCH_P72=1)
endif()
# P73: retain P68's eight-flag batch and skip groups whose flags are all zero.
option(PX_HCU_RIGID_STAGE2_EMPTY_BATCH_SKIP_P73 "Opt-in gfx936 ordinary-rigid Stage2 empty eight-flag batch skip candidate" OFF)
if(PX_HCU_RIGID_STAGE2_EMPTY_BATCH_SKIP_P73)
    if(NOT HCU_ARCH STREQUAL "gfx936" OR NOT PX_HCU_RIGID_STAGE2_INDEX_P54 OR NOT PX_HCU_RIGID_STAGE2_FLAGS4_P60 OR NOT PX_HCU_RIGID_STAGE2_BOUNDED_SPLIT_P64 OR NOT PX_HCU_RIGID_STAGE2_BOUNDED_PREFETCH_P65 OR NOT PX_HCU_RIGID_STAGE2_BOUNDED_PREFETCH8_P68 OR PX_HCU_RIGID_STAGE2_BOUNDED_PREFETCH16_P70 OR PX_HCU_RIGID_STAGE2_METADATA_PREFETCH_P72 OR NOT PX_HCU_RIGID_CONTACT_PREP_O3_P59 OR NOT PX_HCU_RIGID_DELTA_ACCUM_O3_P43 OR NOT PX_HCU_RIGID_GJKEPA_O3_P45 OR NOT PX_HCU_RIGID_CUDABOX_O3_P47 OR NOT PX_HCU_RIGID_SOLVER_MULTIBLOCK_O3_P48 OR NOT PX_HCU_RIGID_PAIR_MANAGEMENT_O3_P49 OR PX_HCU_RIGID_CUDABOX_O3_P42)
        message(FATAL_ERROR "P73 requires gfx936, P43/P45/P47/P48/P49/P54/P59/P60/P64/P65/P68=ON, P70/P72=OFF and P42=OFF; ordinary-rigid dedicated candidate only")
    endif()
    set_property(SOURCE ${PHYSX_ROOT_DIR}/source/gpunarrowphase/src/CUDA/cudaGJKEPA.cu
        APPEND PROPERTY COMPILE_OPTIONS -DPX_HCU_RIGID_STAGE2_EMPTY_BATCH_SKIP_P73=1)
endif()
# P74: build an eight-bit lane mask on the P73 base and emit only set lanes.
option(PX_HCU_RIGID_STAGE2_SET_BIT_COMPACT_P74 "Opt-in gfx936 ordinary-rigid Stage2 set-bit compaction candidate" OFF)
if(PX_HCU_RIGID_STAGE2_SET_BIT_COMPACT_P74)
    if(NOT HCU_ARCH STREQUAL "gfx936" OR NOT PX_HCU_RIGID_STAGE2_INDEX_P54 OR NOT PX_HCU_RIGID_STAGE2_FLAGS4_P60 OR NOT PX_HCU_RIGID_STAGE2_BOUNDED_SPLIT_P64 OR NOT PX_HCU_RIGID_STAGE2_BOUNDED_PREFETCH_P65 OR NOT PX_HCU_RIGID_STAGE2_BOUNDED_PREFETCH8_P68 OR NOT PX_HCU_RIGID_STAGE2_EMPTY_BATCH_SKIP_P73 OR PX_HCU_RIGID_STAGE2_BOUNDED_PREFETCH16_P70 OR PX_HCU_RIGID_STAGE2_METADATA_PREFETCH_P72 OR NOT PX_HCU_RIGID_CONTACT_PREP_O3_P59 OR NOT PX_HCU_RIGID_DELTA_ACCUM_O3_P43 OR NOT PX_HCU_RIGID_GJKEPA_O3_P45 OR NOT PX_HCU_RIGID_CUDABOX_O3_P47 OR NOT PX_HCU_RIGID_SOLVER_MULTIBLOCK_O3_P48 OR NOT PX_HCU_RIGID_PAIR_MANAGEMENT_O3_P49 OR PX_HCU_RIGID_CUDABOX_O3_P42)
        message(FATAL_ERROR "P74 requires gfx936, P43/P45/P47/P48/P49/P54/P59/P60/P64/P65/P68/P73=ON, P70/P72=OFF and P42=OFF; ordinary-rigid dedicated candidate only")
    endif()
    set_property(SOURCE ${PHYSX_ROOT_DIR}/source/gpunarrowphase/src/CUDA/cudaGJKEPA.cu
        APPEND PROPERTY COMPILE_OPTIONS -DPX_HCU_RIGID_STAGE2_SET_BIT_COMPACT_P74=1)
endif()
# P75: batch two eight-flag groups and skip a fully empty sixteen-lane window.
option(PX_HCU_RIGID_STAGE2_DUAL_GROUP_SKIP_P75 "Opt-in gfx936 ordinary-rigid Stage2 dual-group empty-window skip candidate" OFF)
if(PX_HCU_RIGID_STAGE2_DUAL_GROUP_SKIP_P75)
    if(NOT HCU_ARCH STREQUAL "gfx936" OR NOT PX_HCU_RIGID_STAGE2_INDEX_P54 OR NOT PX_HCU_RIGID_STAGE2_FLAGS4_P60 OR NOT PX_HCU_RIGID_STAGE2_BOUNDED_SPLIT_P64 OR NOT PX_HCU_RIGID_STAGE2_BOUNDED_PREFETCH_P65 OR NOT PX_HCU_RIGID_STAGE2_BOUNDED_PREFETCH8_P68 OR NOT PX_HCU_RIGID_STAGE2_EMPTY_BATCH_SKIP_P73 OR PX_HCU_RIGID_STAGE2_SET_BIT_COMPACT_P74 OR PX_HCU_RIGID_STAGE2_BOUNDED_PREFETCH16_P70 OR PX_HCU_RIGID_STAGE2_METADATA_PREFETCH_P72 OR NOT PX_HCU_RIGID_CONTACT_PREP_O3_P59 OR NOT PX_HCU_RIGID_DELTA_ACCUM_O3_P43 OR NOT PX_HCU_RIGID_GJKEPA_O3_P45 OR NOT PX_HCU_RIGID_CUDABOX_O3_P47 OR NOT PX_HCU_RIGID_SOLVER_MULTIBLOCK_O3_P48 OR NOT PX_HCU_RIGID_PAIR_MANAGEMENT_O3_P49 OR PX_HCU_RIGID_CUDABOX_O3_P42)
        message(FATAL_ERROR "P75 requires gfx936, P43/P45/P47/P48/P49/P54/P59/P60/P64/P65/P68/P73=ON, P70/P72/P74=OFF and P42=OFF; ordinary-rigid dedicated candidate only")
    endif()
    set_property(SOURCE ${PHYSX_ROOT_DIR}/source/gpunarrowphase/src/CUDA/cudaGJKEPA.cu
        APPEND PROPERTY COMPILE_OPTIONS -DPX_HCU_RIGID_STAGE2_DUAL_GROUP_SKIP_P75=1)
endif()
# P76: pack the three copied output-count fields into one aligned 32-bit transfer.
option(PX_HCU_RIGID_STAGE2_PACKED_OUTPUT_COUNTS_P76 "Opt-in gfx936 ordinary-rigid Stage2 packed output-count transfer candidate" OFF)
if(PX_HCU_RIGID_STAGE2_PACKED_OUTPUT_COUNTS_P76)
    if(NOT HCU_ARCH STREQUAL "gfx936" OR NOT PX_HCU_RIGID_STAGE2_INDEX_P54 OR NOT PX_HCU_RIGID_STAGE2_FLAGS4_P60 OR NOT PX_HCU_RIGID_STAGE2_BOUNDED_SPLIT_P64 OR NOT PX_HCU_RIGID_STAGE2_BOUNDED_PREFETCH_P65 OR NOT PX_HCU_RIGID_STAGE2_BOUNDED_PREFETCH8_P68 OR NOT PX_HCU_RIGID_STAGE2_EMPTY_BATCH_SKIP_P73 OR PX_HCU_RIGID_STAGE2_SET_BIT_COMPACT_P74 OR PX_HCU_RIGID_STAGE2_DUAL_GROUP_SKIP_P75 OR PX_HCU_RIGID_STAGE2_BOUNDED_PREFETCH16_P70 OR PX_HCU_RIGID_STAGE2_METADATA_PREFETCH_P72 OR NOT PX_HCU_RIGID_CONTACT_PREP_O3_P59 OR NOT PX_HCU_RIGID_DELTA_ACCUM_O3_P43 OR NOT PX_HCU_RIGID_GJKEPA_O3_P45 OR NOT PX_HCU_RIGID_CUDABOX_O3_P47 OR NOT PX_HCU_RIGID_SOLVER_MULTIBLOCK_O3_P48 OR NOT PX_HCU_RIGID_PAIR_MANAGEMENT_O3_P49 OR PX_HCU_RIGID_CUDABOX_O3_P42)
        message(FATAL_ERROR "P76 requires gfx936, P43/P45/P47/P48/P49/P54/P59/P60/P64/P65/P68/P73=ON, P70/P72/P74/P75=OFF and P42=OFF; ordinary-rigid dedicated candidate only")
    endif()
    set_property(SOURCE ${PHYSX_ROOT_DIR}/source/gpunarrowphase/src/CUDA/cudaGJKEPA.cu
        APPEND PROPERTY COMPILE_OPTIONS -DPX_HCU_RIGID_STAGE2_PACKED_OUTPUT_COUNTS_P76=1)
endif()
# P78: constrain the existing 128-thread contact-prePrep kernel to two blocks per CU.
option(PX_HCU_RIGID_CONTACT_PREP_LAUNCH_BOUNDS_P78 "Opt-in gfx936 contact-prePrep launch-bounds candidate" OFF)
if(PX_HCU_RIGID_CONTACT_PREP_LAUNCH_BOUNDS_P78)
    if(NOT HCU_ARCH STREQUAL "gfx936" OR NOT PX_HCU_RIGID_CONTACT_PREP_O3_P59 OR NOT PX_HCU_RIGID_DELTA_ACCUM_O3_P43 OR NOT PX_HCU_RIGID_GJKEPA_O3_P45 OR NOT PX_HCU_RIGID_CUDABOX_O3_P47 OR NOT PX_HCU_RIGID_SOLVER_MULTIBLOCK_O3_P48 OR NOT PX_HCU_RIGID_PAIR_MANAGEMENT_O3_P49 OR NOT PX_HCU_RIGID_STAGE2_EMPTY_BATCH_SKIP_P73 OR PX_HCU_RIGID_CUDABOX_O3_P42)
        message(FATAL_ERROR "P78 requires gfx936, P43/P45/P47/P48/P49/P59/P73=ON and P42=OFF; ordinary-rigid current-product candidate only")
    endif()
    set_property(SOURCE ${PHYSX_ROOT_DIR}/source/gpusolver/src/CUDA/constraintBlockPrePrep.cu
        APPEND PROPERTY COMPILE_OPTIONS -DPX_HCU_RIGID_CONTACT_PREP_LAUNCH_BOUNDS_P78=1)
endif()
option(PX_HCU_VOLUME_SOFTBODY_GM_O2_P40 "Opt-in gfx936 softBodyGM O2 comparison on P33/P36/P37/P38/P39" OFF)
if(PX_HCU_VOLUME_SOFTBODY_GM_O2_P40)
    if(NOT HCU_ARCH STREQUAL "gfx936" OR NOT PX_HCU_VOLUME_O3_P33 OR NOT PX_HCU_VOLUME_SOFTBODY_O3_P36 OR NOT PX_HCU_VOLUME_RIGID_DELTA_O3_P37 OR NOT PX_HCU_VOLUME_RADIX_O3_P38 OR NOT PX_HCU_VOLUME_SOFTBODY_PRIMITIVES_O3_P39)
        message(FATAL_ERROR "PX_HCU_VOLUME_SOFTBODY_GM_O2_P40 requires gfx936 and P33/P36/P37/P38/P39=ON")
    endif()
    # Append after P33's -O3: only this TU's effective level becomes -O2.
    set_property(SOURCE ${PHYSX_ROOT_DIR}/source/gpusimulationcontroller/src/CUDA/softBodyGM.cu
        APPEND PROPERTY COMPILE_OPTIONS -O2 -DPX_HCU_VOLUME_SOFTBODY_GM_O2_P40=1)
endif()
# HCU: cudaBox.cu still requires the hipcc register-loss codegen workaround.
set_source_files_properties(
    ${PHYSX_ROOT_DIR}/source/gpunarrowphase/src/CUDA/cudaBox.cu
    PROPERTIES COMPILE_FLAGS "-mllvm -amdgpu-early-inline-all=0 -mllvm -inline-threshold=0"
)

# ---- POST_BUILD: extract .hsaco from each .o file via Python ----
add_custom_command(TARGET PhysXGpuHCU POST_BUILD
    COMMAND python3 ${CMAKE_CURRENT_SOURCE_DIR}/extract_hsaco.py
        ${CMAKE_CURRENT_BINARY_DIR}/CMakeFiles/PhysXGpuHCU.dir
        ${CMAKE_CURRENT_BINARY_DIR}/kernels
    COMMENT "Extracting .hsaco from .o files"
)

# Tests
add_executable(test_physx_hcu ${PHYSX_ROOT_DIR}/source/gpucommon/src/HCU/test_gpucommon.hip)
target_include_directories(test_physx_hcu PRIVATE ${PHYSX_INCLUDE_DIRS})
target_link_libraries(test_physx_hcu PRIVATE PhysXGpuHCU)
add_executable(test_functional ${PHYSX_ROOT_DIR}/source/gpucommon/src/HCU/test_functional.hip)
target_include_directories(test_functional PRIVATE ${PHYSX_INCLUDE_DIRS})
target_link_libraries(test_functional PRIVATE PhysXGpuHCU)
