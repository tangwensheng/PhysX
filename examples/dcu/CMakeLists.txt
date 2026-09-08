cmake_minimum_required(VERSION 3.21)

if(NOT DEFINED CMAKE_HIP_ARCHITECTURES)
    set(CMAKE_HIP_ARCHITECTURES gfx936 CACHE STRING "DCU/HIP GPU architecture")
endif()

project(PhysXDcuExamples LANGUAGES CXX HIP)

set(PHYSX_DCU_HIP_FLAGS
    -include PxgHIPCompat.h
    -DPX_DCU_PORT
    -DNDEBUG
    -ffast-math
    -fgpu-flush-denormals-to-zero
)

if(NOT DEFINED PHYSX_ROOT_DIR)
    get_filename_component(PHYSX_ROOT_DIR "${CMAKE_SOURCE_DIR}/../../physx" ABSOLUTE)
endif()

set(PHYSX_INCLUDE_DIRS
    ${PHYSX_ROOT_DIR}/include
    ${PHYSX_ROOT_DIR}/include/geometry
    ${PHYSX_ROOT_DIR}/source/cudamanager/include
    ${PHYSX_ROOT_DIR}/source/common/include
    ${PHYSX_ROOT_DIR}/source/common/src
    ${PHYSX_ROOT_DIR}/source/foundation/include
    ${PHYSX_ROOT_DIR}/source/lowlevel/api/include
    ${PHYSX_ROOT_DIR}/source/lowlevel/common/include
    ${PHYSX_ROOT_DIR}/source/lowlevel/common/include/pipeline
    ${PHYSX_ROOT_DIR}/source/lowlevel/common/include/utils
    ${PHYSX_ROOT_DIR}/source/lowlevel/common/include/collision
    ${PHYSX_ROOT_DIR}/source/lowlevel/software/include
    ${PHYSX_ROOT_DIR}/source/geomutils/include
    ${PHYSX_ROOT_DIR}/source/geomutils/src
    ${PHYSX_ROOT_DIR}/source/geomutils/src/mesh
    ${PHYSX_ROOT_DIR}/source/geomutils/src/pcm
    ${PHYSX_ROOT_DIR}/source/geomutils/src/gjk
    ${PHYSX_ROOT_DIR}/source/geomutils/src/contact
    ${PHYSX_ROOT_DIR}/source/geomutils/src/convex
    ${PHYSX_ROOT_DIR}/source/geomutils/src/common
    ${PHYSX_ROOT_DIR}/source/geomutils/src/hf
    ${PHYSX_ROOT_DIR}/source/geomutils/src/intersection
    ${PHYSX_ROOT_DIR}/source/geomutils/src/distance
    ${PHYSX_ROOT_DIR}/source/geomutils/src/ccd
    ${PHYSX_ROOT_DIR}/source/geomutils/src/sweep
    ${PHYSX_ROOT_DIR}/source/geomutils/src/cooking
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
    ${PHYSX_ROOT_DIR}/source/lowleveldynamics/include
    ${PHYSX_ROOT_DIR}/source/lowleveldynamics/shared
    ${PHYSX_ROOT_DIR}/source/lowlevelaabb/include
    ${PHYSX_ROOT_DIR}/source/simulationcontroller/include
    ${PHYSX_ROOT_DIR}/source/scenequery/include
    /opt/dtk/hip/include
    /opt/dtk/include
)

# Find PhysX static libraries in build directory
if(NOT DEFINED PHYSX_BUILD_DIR)
    set(PHYSX_BUILD_DIR "${CMAKE_SOURCE_DIR}/../../build_cpu")
endif()

# Explicit library list (same as cmakecpu/CMakeLists.txt)
set(PHYSX_LIBS
    -Wl,--start-group
    ${PHYSX_BUILD_DIR}/libFoundation.a
    ${PHYSX_BUILD_DIR}/libCommon.a
    ${PHYSX_BUILD_DIR}/libPhysXSDK.a
    ${PHYSX_BUILD_DIR}/libPhysXExtensions.a
    ${PHYSX_BUILD_DIR}/libPhysXTask.a
    ${PHYSX_BUILD_DIR}/libPhysXCooking.a
    ${PHYSX_BUILD_DIR}/libLowLevel.a
    ${PHYSX_BUILD_DIR}/libLowLevelAABB.a
    ${PHYSX_BUILD_DIR}/libLowLevelDynamics.a
    ${PHYSX_BUILD_DIR}/libSceneQuery.a
    ${PHYSX_BUILD_DIR}/libSimulationController.a
    ${PHYSX_BUILD_DIR}/libGeomUtils.a
    ${PHYSX_BUILD_DIR}/libHipContext.a
    -Wl,--end-group
)

function(add_dcu_example SOURCE_FILE)
    get_filename_component(EXE_NAME ${SOURCE_FILE} NAME_WE)
    add_executable(${EXE_NAME} ${SOURCE_FILE} ${ARGN})
    target_include_directories(${EXE_NAME} PRIVATE ${PHYSX_INCLUDE_DIRS})
    target_link_directories(${EXE_NAME} PRIVATE /opt/dtk/hip/lib)
    target_link_libraries(${EXE_NAME} PRIVATE ${PHYSX_LIBS} amdhip64 pthread dl)
    target_compile_definitions(${EXE_NAME} PRIVATE __HIP_PLATFORM_AMD__=1)
    target_compile_options(${EXE_NAME} PRIVATE
        $<$<COMPILE_LANGUAGE:HIP>:${PHYSX_DCU_HIP_FLAGS}>
    )
endfunction()

file(GLOB EXAMPLE_CPP_SOURCES ${CMAKE_SOURCE_DIR}/*.cpp)
file(GLOB EXAMPLE_HIP_SOURCES ${CMAKE_SOURCE_DIR}/*.hip)
list(FILTER EXAMPLE_HIP_SOURCES EXCLUDE REGEX ".*_kernel\.hip$")

foreach(SOURCE_FILE ${EXAMPLE_CPP_SOURCES})
    get_filename_component(EXE_NAME ${SOURCE_FILE} NAME_WE)
    if(EXE_NAME STREQUAL "scene_tower")
        set(KERNEL_SOURCE ${CMAKE_SOURCE_DIR}/scene_tower_kernel.hip)
        set_source_files_properties(${KERNEL_SOURCE} PROPERTIES LANGUAGE HIP)
        add_dcu_example(${SOURCE_FILE} ${KERNEL_SOURCE})
    elseif(EXE_NAME STREQUAL "bench_deformable_surface_smoke")
        # Some existing CPU PhysXExtensions archives were built before the
        # deformable-surface extension was included. Compile this implementation
        # into the smoke target so PxDeformableSurfaceExt symbols are available.
        set(DEFORMABLE_SURFACE_EXT_SOURCE ${PHYSX_ROOT_DIR}/source/physxextensions/src/ExtDeformableSurfaceExt.cpp)
        add_dcu_example(${SOURCE_FILE} ${DEFORMABLE_SURFACE_EXT_SOURCE})
    elseif(EXE_NAME STREQUAL "bench_deformable_volume_smoke")
        # Keep the volume smoke self-contained because the current static
        # PhysXExtensions archive does not export deformable-volume helpers.
        set(DEFORMABLE_VOLUME_EXT_SOURCES
            ${PHYSX_ROOT_DIR}/source/physxextensions/src/ExtDeformableVolumeExt.cpp
            ${PHYSX_ROOT_DIR}/source/physxextensions/src/ExtRemeshingExt.cpp
            ${PHYSX_ROOT_DIR}/source/physxextensions/src/ExtTetMakerExt.cpp
            ${PHYSX_ROOT_DIR}/source/physxextensions/src/ExtTetrahedronMeshExt.cpp
        )
        add_dcu_example(${SOURCE_FILE} ${DEFORMABLE_VOLUME_EXT_SOURCES})
    elseif(EXE_NAME STREQUAL "bench_pbd_cloth_smoke")
        # The existing static PhysXExtensions archive may not include GPU
        # particle-cloth helpers, so compile the small implementation here.
        set(PBD_CLOTH_EXT_SOURCE ${PHYSX_ROOT_DIR}/source/physxextensions/src/ExtParticleExt.cpp)
        add_dcu_example(${SOURCE_FILE} ${PBD_CLOTH_EXT_SOURCE})
    else()
        add_dcu_example(${SOURCE_FILE})
    endif()
endforeach()

foreach(SOURCE_FILE ${EXAMPLE_HIP_SOURCES})
    set_source_files_properties(${SOURCE_FILE} PROPERTIES LANGUAGE HIP)
    add_dcu_example(${SOURCE_FILE})
endforeach()
