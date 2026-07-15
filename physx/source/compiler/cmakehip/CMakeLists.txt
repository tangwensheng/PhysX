cmake_minimum_required(VERSION 3.21)
project(PhysXGpuHIP LANGUAGES CXX HIP)
set(CMAKE_HIP_FLAGS "${CMAKE_HIP_FLAGS} -include PxgHIPCompat.h -DPX_DCU_PORT -DNDEBUG -ffast-math -fgpu-flush-denormals-to-zero")
if(NOT DEFINED PHYSX_ROOT_DIR)
    get_filename_component(PHYSX_ROOT_DIR "${CMAKE_SOURCE_DIR}/../../.." ABSOLUTE)
endif()
if(NOT DEFINED DCU_ARCH)
    set(DCU_ARCH "gfx936" CACHE STRING "DCU GPU architecture")
endif()
set(CMAKE_HIP_ARCHITECTURES "${DCU_ARCH}")
add_compile_definitions(PX_DCU_PORT PX_PHYSX_GPU_STATIC PX_SUPPORT_GPU_PHYSX)
set(PHYSX_INCLUDE_DIRS
    ${PHYSX_ROOT_DIR}/include ${PHYSX_ROOT_DIR}/include/geometry
    ${PHYSX_ROOT_DIR}/source/common/include ${PHYSX_ROOT_DIR}/source/common/src
    ${PHYSX_ROOT_DIR}/source/foundation/include
    ${PHYSX_ROOT_DIR}/source/gpucommon/include ${PHYSX_ROOT_DIR}/source/gpucommon/src/CUDA
    ${PHYSX_ROOT_DIR}/source/gpucommon/src/DCU/stubs
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

# Collect ALL .cu files
file(GLOB GPU_ALL_CU_FILES
    ${PHYSX_ROOT_DIR}/source/gpucommon/src/CUDA/*.cu
    ${PHYSX_ROOT_DIR}/source/gpubroadphase/src/CUDA/*.cu
    ${PHYSX_ROOT_DIR}/source/gpuarticulation/src/CUDA/*.cu
    ${PHYSX_ROOT_DIR}/source/gpusolver/src/CUDA/*.cu
    ${PHYSX_ROOT_DIR}/source/gpusimulationcontroller/src/CUDA/*.cu
    ${PHYSX_ROOT_DIR}/source/gpunarrowphase/src/CUDA/*.cu
)
# DCU: exclude convex modules — API incompatibilities, not needed for box/sphere benchmark
list(REMOVE_ITEM GPU_ALL_CU_FILES
    ${PHYSX_ROOT_DIR}/source/gpunarrowphase/src/CUDA/convexCoreCollision.cu
    ${PHYSX_ROOT_DIR}/source/gpunarrowphase/src/CUDA/convexHFMidphase.cu
    ${PHYSX_ROOT_DIR}/source/gpunarrowphase/src/CUDA/convexHeightfield.cu
    ${PHYSX_ROOT_DIR}/source/gpunarrowphase/src/CUDA/convexMesh.cu
    ${PHYSX_ROOT_DIR}/source/gpunarrowphase/src/CUDA/convexMeshCorrelate.cu
    ${PHYSX_ROOT_DIR}/source/gpunarrowphase/src/CUDA/convexMeshMidphase.cu
    ${PHYSX_ROOT_DIR}/source/gpunarrowphase/src/CUDA/convexMeshOutput.cu
    ${PHYSX_ROOT_DIR}/source/gpunarrowphase/src/CUDA/convexMeshPostProcess.cu
)

# Build shared library (for tests, static linkage)
add_library(PhysXGpuDCU SHARED ${GPU_ALL_CU_FILES})
target_include_directories(PhysXGpuDCU PRIVATE ${PHYSX_INCLUDE_DIRS})
set_source_files_properties(${GPU_ALL_CU_FILES} PROPERTIES LANGUAGE HIP)
# DCU: disable inlining for cudaBox.cu (hipcc register-loss codegen bug)
set_source_files_properties(
    ${PHYSX_ROOT_DIR}/source/gpunarrowphase/src/CUDA/cudaBox.cu
    PROPERTIES COMPILE_FLAGS "-mllvm -amdgpu-early-inline-all=0 -mllvm -inline-threshold=0"
)

# ---- POST_BUILD: extract .hsaco from each .o file via Python ----
add_custom_command(TARGET PhysXGpuDCU POST_BUILD
    COMMAND python3 ${CMAKE_CURRENT_SOURCE_DIR}/extract_hsaco.py
        ${CMAKE_CURRENT_BINARY_DIR}/CMakeFiles/PhysXGpuDCU.dir
        ${CMAKE_CURRENT_BINARY_DIR}/kernels
    COMMENT "Extracting .hsaco from .o files"
)

# Tests
add_executable(test_physx_dcu ${PHYSX_ROOT_DIR}/source/gpucommon/src/DCU/test_gpucommon.hip)
target_include_directories(test_physx_dcu PRIVATE ${PHYSX_INCLUDE_DIRS})
target_link_libraries(test_physx_dcu PRIVATE PhysXGpuDCU)
add_executable(test_functional ${PHYSX_ROOT_DIR}/source/gpucommon/src/DCU/test_functional.hip)
target_include_directories(test_functional PRIVATE ${PHYSX_INCLUDE_DIRS})
target_link_libraries(test_functional PRIVATE PhysXGpuDCU)
