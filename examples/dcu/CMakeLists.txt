cmake_minimum_required(VERSION 3.21)
project(PhysXDcuExamples LANGUAGES CXX)

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

file(GLOB EXAMPLE_SOURCES ${CMAKE_SOURCE_DIR}/*.cpp)

foreach(SOURCE_FILE ${EXAMPLE_SOURCES})
    get_filename_component(EXE_NAME ${SOURCE_FILE} NAME_WE)
    add_executable(${EXE_NAME} ${SOURCE_FILE})
    target_include_directories(${EXE_NAME} PRIVATE ${PHYSX_INCLUDE_DIRS})
    target_link_directories(${EXE_NAME} PRIVATE /opt/dtk/hip/lib)
    target_link_libraries(${EXE_NAME} PRIVATE ${PHYSX_LIBS} amdhip64 pthread dl)
endforeach()