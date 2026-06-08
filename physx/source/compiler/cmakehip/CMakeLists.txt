# PhysX DCU/HIP GPU Build
#
# Standalone CMake project to build ALL PhysX GPU libraries with hipcc.
#
# Usage:
#   cd physx/source/compiler/cmakehip
#   mkdir build && cd build
#   cmake .. -DPHYSX_ROOT_DIR=/path/to/PhysX -DDCU_ARCH=gfx936
#   cmake --build .

cmake_minimum_required(VERSION 3.21)

# HIP language support
project(PhysXGpuHIP LANGUAGES CXX HIP)

# ---- HIP compiler flags (MUST come before any target) ----
set(CMAKE_HIP_FLAGS "${CMAKE_HIP_FLAGS} -include PxgHIPCompat.h -DPX_DCU_PORT -ffast-math -fgpu-flush-denormals-to-zero")
set(CMAKE_HIP_FLAGS_DEBUG "-g -O0")
set(CMAKE_HIP_FLAGS_RELEASE "-O3")

# ---- PhysX Root ----
if(NOT DEFINED PHYSX_ROOT_DIR)
    get_filename_component(PHYSX_ROOT_DIR "${CMAKE_SOURCE_DIR}/../../.." ABSOLUTE)
endif()
message(STATUS "PhysX root: ${PHYSX_ROOT_DIR}")

# ---- DCU Architecture ----
if(NOT DEFINED DCU_ARCH)
    set(DCU_ARCH "gfx936" CACHE STRING "DCU GPU architecture")
endif()
set(CMAKE_HIP_ARCHITECTURES "${DCU_ARCH}")
message(STATUS "DCU architecture: ${DCU_ARCH}")

# ---- Global compile definitions ----
add_compile_definitions(
    PX_DCU_PORT
    PX_PHYSX_GPU_STATIC
    PX_SUPPORT_GPU_PHYSX
)

# ---- Include paths ----
set(PHYSX_INCLUDE_DIRS
    ${PHYSX_ROOT_DIR}/include
    ${PHYSX_ROOT_DIR}/include/geometry
    ${PHYSX_ROOT_DIR}/source/common/include
    ${PHYSX_ROOT_DIR}/source/common/src
    ${PHYSX_ROOT_DIR}/source/foundation/include
    ${PHYSX_ROOT_DIR}/source/gpucommon/include
    ${PHYSX_ROOT_DIR}/source/gpucommon/src/CUDA
    ${PHYSX_ROOT_DIR}/source/gpucommon/src/DCU/stubs
    ${PHYSX_ROOT_DIR}/source/gpunarrowphase/include
    ${PHYSX_ROOT_DIR}/source/gpunarrowphase/src/CUDA
    ${PHYSX_ROOT_DIR}/source/gpusolver/include
    ${PHYSX_ROOT_DIR}/source/gpusolver/src/CUDA
    ${PHYSX_ROOT_DIR}/source/gpusimulationcontroller/include
    ${PHYSX_ROOT_DIR}/source/gpusimulationcontroller/src/CUDA
    ${PHYSX_ROOT_DIR}/source/gpubroadphase/include
    ${PHYSX_ROOT_DIR}/source/gpubroadphase/src/CUDA
    ${PHYSX_ROOT_DIR}/source/gpuarticulation/include
    ${PHYSX_ROOT_DIR}/source/gpuarticulation/src/CUDA
    ${PHYSX_ROOT_DIR}/source/lowlevel/api/include
    ${PHYSX_ROOT_DIR}/source/lowlevel/software/include
    ${PHYSX_ROOT_DIR}/source/lowlevel/common/include
    ${PHYSX_ROOT_DIR}/source/lowleveldynamics/include
    ${PHYSX_ROOT_DIR}/source/lowleveldynamics/shared
    ${PHYSX_ROOT_DIR}/source/lowlevelaabb/include
    ${PHYSX_ROOT_DIR}/source/geomutils/src
    ${PHYSX_ROOT_DIR}/source/geomutils/src/mesh
    ${PHYSX_ROOT_DIR}/source/geomutils/include
    ${PHYSX_ROOT_DIR}/source/cudamanager/include
    ${PHYSX_ROOT_DIR}/source/simulationcontroller/include
    ${PHYSX_ROOT_DIR}/source/scenequery/include
)

# ---- Helper: add a GPU module library ----
macro(add_gpu_module TARGET MODULE_DIR)
    file(GLOB MODULE_SOURCES "${PHYSX_ROOT_DIR}/source/${MODULE_DIR}/src/CUDA/*.cu")
    if(MODULE_SOURCES)
        set_source_files_properties(${MODULE_SOURCES} PROPERTIES LANGUAGE HIP)
        add_library(${TARGET} STATIC ${MODULE_SOURCES})
        target_include_directories(${TARGET} PRIVATE ${PHYSX_INCLUDE_DIRS})
        message(STATUS "${TARGET}: ${MODULE_DIR} (${MODULE_SOURCES})")
    else()
        message(WARNING "${TARGET}: no .cu files found in ${MODULE_DIR}")
    endif()
endmacro()

# ---- Build all GPU modules ----
add_gpu_module(PhysXCommonGpu           gpucommon)
add_gpu_module(PhysXBroadphaseGpu       gpubroadphase)
add_gpu_module(PhysXArticulationGpu     gpuarticulation)
add_gpu_module(PhysXSolverGpu           gpusolver)
add_gpu_module(PhysXSimulationControllerGpu gpusimulationcontroller)
add_gpu_module(PhysXNarrowphaseGpu      gpunarrowphase)

# ---- Integration test ----
add_executable(test_physx_dcu
    ${PHYSX_ROOT_DIR}/source/gpucommon/src/DCU/test_gpucommon.hip
)
target_include_directories(test_physx_dcu PRIVATE ${PHYSX_INCLUDE_DIRS})
target_link_libraries(test_physx_dcu PRIVATE PhysXCommonGpu)

# ---- Functional test (real kernel testing) ----
add_executable(test_functional
    ${PHYSX_ROOT_DIR}/source/gpucommon/src/DCU/test_functional.hip
)
target_include_directories(test_functional PRIVATE ${PHYSX_INCLUDE_DIRS})
target_link_libraries(test_functional PRIVATE PhysXCommonGpu PhysXBroadphaseGpu)

message(STATUS "=== PhysX DCU Build Configuration Complete ===")
