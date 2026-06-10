# PhysX DCU Examples
cmake_minimum_required(VERSION 3.21)
project(PhysXDcuExamples LANGUAGES CXX HIP)

if(NOT DEFINED PHYSX_ROOT_DIR)
    get_filename_component(PHYSX_ROOT_DIR "${CMAKE_SOURCE_DIR}/../../physx" ABSOLUTE)
endif()

# HIP compiler is passed via -DCMAKE_HIP_COMPILER from build.sh
set(DCU_ARCH "gfx936" CACHE STRING "DCU architecture")
set(CMAKE_HIP_ARCHITECTURES ${DCU_ARCH})
set(CMAKE_HIP_FLAGS "-include PxgHIPCompat.h -DPX_DCU_PORT -ffast-math -fgpu-flush-denormals-to-zero")

set(CMAKE_CXX_FLAGS "-O2 -fPIC")

# Include paths
set(EXAMPLES_INC
    ${PHYSX_ROOT_DIR}/source/gpucommon/include
    ${PHYSX_ROOT_DIR}/source/gpucommon/src/CUDA
    ${PHYSX_ROOT_DIR}/source/gpucommon/src/DCU/stubs
    ${PHYSX_ROOT_DIR}/include
    ${PHYSX_ROOT_DIR}/include/geometry
    ${PHYSX_ROOT_DIR}/source/foundation/include
    ${PHYSX_ROOT_DIR}/source/common/include
    ${PHYSX_ROOT_DIR}/source/common/src
    ${PHYSX_ROOT_DIR}/source/gpucommon/include
    ${PHYSX_ROOT_DIR}/source/gpucommon/src/CUDA
    ${PHYSX_ROOT_DIR}/source/gpucommon/src/DCU/stubs
    ${PHYSX_ROOT_DIR}/source/gpunarrowphase/include
    ${PHYSX_ROOT_DIR}/source/gpunarrowphase/src/CUDA
    ${PHYSX_ROOT_DIR}/source/gpusolver/include
    ${PHYSX_ROOT_DIR}/source/gpusolver/src/CUDA
    ${PHYSX_ROOT_DIR}/source/gpusimulationcontroller/include
    ${PHYSX_ROOT_DIR}/source/gpusimulationcontroller/src/CUDA
    ${PHYSX_ROOT_DIR}/source/geomutils/src
    ${PHYSX_ROOT_DIR}/source/geomutils/include
    ${PHYSX_ROOT_DIR}/source/lowlevel/api/include
    ${PHYSX_ROOT_DIR}/source/lowlevel/software/include
    ${PHYSX_ROOT_DIR}/source/lowlevel/common/include
    ${PHYSX_ROOT_DIR}/source/lowlevel/common/include/pipeline
    ${PHYSX_ROOT_DIR}/source/lowlevel/common/include/utils
)

# === Example 1: N-Body Gravity (GPU) ===
# 16K particles gravitationally attracting each other, all-GPU computation
add_executable(example1_nbody example1_nbody.hip)
target_include_directories(example1_nbody PRIVATE ${EXAMPLES_INC})

# === Example 2: Particle-Verlet with Collision (GPU) ===
# 1M particles with Verlet integration, ground collision, validated against CPU reference
add_executable(example2_particles example2_particles.hip)
target_include_directories(example2_particles PRIVATE ${EXAMPLES_INC})

# === Example 3: PhysX Raycast with GPU BVH ===
# Uses the PhysX gpucommon library for GPU-accelerated raycasting
add_executable(example3_reduction example3_raycast.hip)
target_include_directories(example3_reduction PRIVATE ${EXAMPLES_INC})

add_executable(example4_shuffle   example4_shuffle_atomic.hip)
target_include_directories(example4_shuffle PRIVATE ${EXAMPLES_INC})

add_executable(example5_sort     example5_sort.hip)
target_include_directories(example5_sort PRIVATE ${EXAMPLES_INC})

add_executable(example6_contact  example6_contact.hip)
target_include_directories(example6_contact PRIVATE ${EXAMPLES_INC})

add_executable(example7_distance example7_distance.hip)
target_include_directories(example7_distance PRIVATE ${EXAMPLES_INC})
