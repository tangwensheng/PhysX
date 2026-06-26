# PhysX CPU Library Build for DCU
#
# Builds PhysX CPU-side libraries with g++ and links GPU modules with hipcc.
#
# Usage:
#   cd physx/source/compiler/cmakecpu
#   mkdir build && cd build
#   cmake .. -DPHYSX_ROOT_DIR=/path/to/PhysX -DDCU_ARCH=gfx936
#   cmake --build .

cmake_minimum_required(VERSION 3.21)
project(PhysXCPU LANGUAGES CXX HIP)

# Find DTK Clang for HIP (same logic as build_dcu.sh)
if(NOT DEFINED CMAKE_HIP_COMPILER)
    foreach(clang_path /opt/dtk/bin/clang++ /opt/rocm/bin/clang++)
        if(EXISTS ${clang_path})
            set(CMAKE_HIP_COMPILER ${clang_path})
            break()
        endif()
    endforeach()
endif()

if(NOT DEFINED DCU_ARCH)
    set(DCU_ARCH "gfx936" CACHE STRING "DCU GPU architecture")
endif()
set(CMAKE_HIP_ARCHITECTURES ${DCU_ARCH})
set(CMAKE_HIP_FLAGS "${CMAKE_HIP_FLAGS} -include PxgHIPCompat.h -DPX_DCU_PORT -ffast-math -fgpu-flush-denormals-to-zero")

if(NOT DEFINED PHYSX_ROOT_DIR)
    get_filename_component(PHYSX_ROOT_DIR "${CMAKE_SOURCE_DIR}/../../.." ABSOLUTE)
endif()
message(STATUS "PhysX root: ${PHYSX_ROOT_DIR}")

# ---- Compile definitions ----
add_compile_definitions(
    NDEBUG
    PX_PHYSX_STATIC_LIB
    PX_PUBLIC_RELEASE
    PX_SUPPORT_GPU_PHYSX
)

# ---- Include paths ----
set(PHYSX_PUBLIC_INC   ${PHYSX_ROOT_DIR}/include)
set(PHYSX_GEOM_INC     ${PHYSX_ROOT_DIR}/include/geometry)
set(PHYSX_FOUNDATION   ${PHYSX_ROOT_DIR}/source/foundation/include)
set(PHYSX_COMMON_INC   ${PHYSX_ROOT_DIR}/source/common/include)
set(PHYSX_COMMON_SRC   ${PHYSX_ROOT_DIR}/source/common/src)
set(PHYSX_GEOMUTILS    ${PHYSX_ROOT_DIR}/source/geomutils/include)
set(PHYSX_GEOMUTILS_SRC ${PHYSX_ROOT_DIR}/source/geomutils/src)

# Auto-collect ALL subdirectories across the entire PhysX source tree.
# This prevents iterative "header not found" errors during CPU compilation.
function(collect_all_subdirs OUTVAR BASE)
    set(result)
    if(EXISTS ${BASE} AND IS_DIRECTORY ${BASE})
        file(GLOB_RECURSE all_files ${BASE}/*.h)
        foreach(f ${all_files})
            get_filename_component(d ${f} DIRECTORY)
            if(IS_DIRECTORY ${d})
                list(APPEND result ${d})
            endif()
        endforeach()
    endif()
    if(result)
        list(REMOVE_DUPLICATES result)
    endif()
    set(${OUTVAR} ${result} PARENT_SCOPE)
endfunction()

# Collect all include directories from the entire PhysX source
collect_all_subdirs(PHYSX_ALL_INCLUDES ${PHYSX_ROOT_DIR}/source)
collect_all_subdirs(PHYSX_ALL_PUBLIC    ${PHYSX_ROOT_DIR}/include)

# ---- Foundation Library ----
file(GLOB FOUNDATION_COMMON_SRC ${PHYSX_ROOT_DIR}/source/foundation/*.cpp)
file(GLOB FOUNDATION_UNIX_SRC   ${PHYSX_ROOT_DIR}/source/foundation/unix/*.cpp)

add_library(Foundation STATIC
    ${FOUNDATION_COMMON_SRC}
    ${FOUNDATION_UNIX_SRC}
)
target_include_directories(Foundation PUBLIC
    ${PHYSX_PUBLIC_INC}
    ${PHYSX_FOUNDATION}
    ${PHYSX_COMMON}
)
target_compile_definitions(Foundation PRIVATE PX_FOUNDATION_EXPORTS)
target_compile_options(Foundation PRIVATE -fPIC -O2)

# ---- Common Library ----
file(GLOB COMMON_SRC ${PHYSX_ROOT_DIR}/source/common/src/*.cpp)
add_library(Common STATIC ${COMMON_SRC})
target_include_directories(Common PUBLIC
    ${PHYSX_PUBLIC_INC}
    ${PHYSX_FOUNDATION}
    ${PHYSX_COMMON}
)
target_link_libraries(Common PUBLIC Foundation)
target_compile_options(Common PRIVATE -fPIC -O2)

# ---- Test: link Foundation + GPU modules and run a simple scene ----
# GPU include paths
set(GPU_INCLUDES
    ${PHYSX_ROOT_DIR}/source/gpucommon/include
    ${PHYSX_ROOT_DIR}/source/gpucommon/src/CUDA
    ${PHYSX_ROOT_DIR}/source/gpucommon/src/DCU/stubs
)

# Build test executable
add_executable(test_cpu
    ${PHYSX_ROOT_DIR}/source/compiler/cmakecpu/test_cpu.cpp
)
target_include_directories(test_cpu PRIVATE
    ${PHYSX_PUBLIC_INC}
    ${PHYSX_FOUNDATION}
    ${GPU_INCLUDES}
)
target_link_libraries(test_cpu PRIVATE Foundation Common)
target_compile_options(test_cpu PRIVATE -fPIC -O2)

# ---- LowLevel Library (collision detection basics) ----
set(LOWLEVEL_SRC_DIR ${PHYSX_ROOT_DIR}/source/lowlevel)
file(GLOB_RECURSE LOWLEVEL_SRC
    ${LOWLEVEL_SRC_DIR}/api/src/*.cpp
    ${LOWLEVEL_SRC_DIR}/common/src/*.cpp
    ${LOWLEVEL_SRC_DIR}/software/src/*.cpp
)
add_library(LowLevel STATIC ${LOWLEVEL_SRC})
target_include_directories(LowLevel PUBLIC
    ${PHYSX_PUBLIC_INC} ${PHYSX_FOUNDATION} ${PHYSX_COMMON_INC} ${PHYSX_COMMON_SRC} ${PHYSX_GEOM_INC} ${PHYSX_ALL_INCLUDES} ${PHYSX_ALL_PUBLIC}
    ${LOWLEVEL_SRC_DIR}/api/include
    ${LOWLEVEL_SRC_DIR}/common/include
    ${LOWLEVEL_SRC_DIR}/software/include
    ${LOWLEVEL_SRC_DIR}/common/include/pipeline
)
target_link_libraries(LowLevel PUBLIC Foundation Common)
target_compile_options(LowLevel PRIVATE -fPIC -O2 -Wno-unused-parameter)

# ---- LowLevelAABB Library (bounding volumes) ----
set(LOWLEVELAABB_SRC_DIR ${PHYSX_ROOT_DIR}/source/lowlevelaabb)
file(GLOB_RECURSE LOWLEVELAABB_SRC
    ${LOWLEVELAABB_SRC_DIR}/src/*.cpp
)
add_library(LowLevelAABB STATIC ${LOWLEVELAABB_SRC})
target_include_directories(LowLevelAABB PUBLIC
    ${PHYSX_PUBLIC_INC} ${PHYSX_FOUNDATION} ${PHYSX_COMMON_INC} ${PHYSX_COMMON_SRC} ${PHYSX_GEOM_INC} ${PHYSX_ALL_INCLUDES} ${PHYSX_ALL_PUBLIC}
    ${LOWLEVELAABB_SRC_DIR}/include
    ${PHYSX_ROOT_DIR}/source/geomutils/include
    ${PHYSX_ROOT_DIR}/source/geomutils/src
    ${PHYSX_ROOT_DIR}/source/geomutils/src/mesh
)
target_link_libraries(LowLevelAABB PUBLIC Foundation Common LowLevel GeomUtils)
target_compile_options(LowLevelAABB PRIVATE -fPIC -O2 -Wno-unused-parameter)

# ---- LowLevelDynamics Library (solver + constraints) ----
set(LOWLEVELDYN_SRC_DIR ${PHYSX_ROOT_DIR}/source/lowleveldynamics)
file(GLOB_RECURSE LOWLEVELDYN_SRC
    ${LOWLEVELDYN_SRC_DIR}/src/*.cpp
)
add_library(LowLevelDynamics STATIC ${LOWLEVELDYN_SRC})
target_include_directories(LowLevelDynamics PUBLIC
    ${PHYSX_PUBLIC_INC} ${PHYSX_FOUNDATION} ${PHYSX_COMMON_INC} ${PHYSX_COMMON_SRC} ${PHYSX_GEOM_INC} ${PHYSX_ALL_INCLUDES} ${PHYSX_ALL_PUBLIC}
    ${LOWLEVELDYN_SRC_DIR}/include
    ${LOWLEVELDYN_SRC_DIR}/shared
    ${LOWLEVEL_SRC_DIR}/api/include
    ${PHYSX_LOWLEVEL_COMMON_ALL}
    ${PHYSX_LOWLEVEL_SOFTWARE_ALL}
    ${LOWLEVELAABB_SRC_DIR}/include
)
target_link_libraries(LowLevelDynamics PUBLIC Foundation Common LowLevel LowLevelAABB)
target_compile_options(LowLevelDynamics PRIVATE -fPIC -O2 -Wno-unused-parameter)

# ---- SceneQuery Library ----
file(GLOB_RECURSE SCENEQUERY_SRC ${PHYSX_ROOT_DIR}/source/scenequery/src/*.cpp)
list(FILTER SCENEQUERY_SRC EXCLUDE REGEX ".*/windows/.*")
add_library(SceneQuery STATIC ${SCENEQUERY_SRC})
target_include_directories(SceneQuery PUBLIC
    ${PHYSX_PUBLIC_INC} ${PHYSX_FOUNDATION} ${PHYSX_COMMON_INC} ${PHYSX_COMMON_SRC}
    ${PHYSX_ALL_INCLUDES} ${PHYSX_ALL_PUBLIC}
    ${PHYSX_ROOT_DIR}/source/scenequery/include
)
target_link_libraries(SceneQuery PUBLIC Foundation Common LowLevel LowLevelAABB)
target_compile_options(SceneQuery PRIVATE -fPIC -O2 -Wno-unused-parameter)

# ---- SimulationController Library ----
file(GLOB_RECURSE SIMCTRL_SRC ${PHYSX_ROOT_DIR}/source/simulationcontroller/src/*.cpp)
list(FILTER SIMCTRL_SRC EXCLUDE REGEX ".*/windows/.*")
add_library(SimulationController STATIC ${SIMCTRL_SRC})
target_include_directories(SimulationController PUBLIC
    ${PHYSX_PUBLIC_INC} ${PHYSX_FOUNDATION} ${PHYSX_COMMON_INC} ${PHYSX_COMMON_SRC}
    ${PHYSX_ALL_INCLUDES} ${PHYSX_ALL_PUBLIC}
    ${PHYSX_ROOT_DIR}/source/simulationcontroller/include
)
target_link_libraries(SimulationController PUBLIC Foundation Common LowLevel LowLevelAABB LowLevelDynamics SceneQuery)
target_compile_options(SimulationController PRIVATE -fPIC -O2 -Wno-unused-parameter)

# ---- PhysX SDK (PxScene / PxRigidBody / PxPhysics etc.) ----
file(GLOB_RECURSE PHYSXSDK_SRC ${PHYSX_ROOT_DIR}/source/physx/src/*.cpp)
# Exclude OS-specific and external dependency files
list(FILTER PHYSXSDK_SRC EXCLUDE REGEX ".*/windows/.*")
list(FILTER PHYSXSDK_SRC EXCLUDE REGEX ".*omnipvd.*")
list(FILTER PHYSXSDK_SRC EXCLUDE REGEX ".*/gpu/.*")
add_library(PhysXSDK STATIC ${PHYSXSDK_SRC})
target_include_directories(PhysXSDK PUBLIC
    ${PHYSX_PUBLIC_INC} ${PHYSX_FOUNDATION} ${PHYSX_COMMON_INC} ${PHYSX_COMMON_SRC}
    ${PHYSX_ALL_INCLUDES} ${PHYSX_ALL_PUBLIC}
    ${PHYSX_ROOT_DIR}/source/physx/src
    ${PHYSX_ROOT_DIR}/source/physxgpu/include
    ${PHYSX_ROOT_DIR}/source/cudamanager/include
)
target_link_libraries(PhysXSDK PUBLIC Foundation Common LowLevel LowLevelAABB LowLevelDynamics SceneQuery SimulationController PhysXTask)
target_compile_options(PhysXSDK PRIVATE -fPIC -O2 -Wno-unused-parameter)

# ---- GeomUtils (BVH, GJK/EPA, convex hull, mesh, contact generation) ----
file(GLOB_RECURSE GEOMUTILS_CPP ${PHYSX_ROOT_DIR}/source/geomutils/src/*.cpp)
add_library(GeomUtils STATIC ${GEOMUTILS_CPP})
target_include_directories(GeomUtils PUBLIC
    ${PHYSX_PUBLIC_INC} ${PHYSX_FOUNDATION} ${PHYSX_ALL_INCLUDES} ${PHYSX_ALL_PUBLIC}
)
target_link_libraries(GeomUtils PUBLIC Foundation Common)
target_compile_options(GeomUtils PRIVATE -fPIC -O2 -Wno-unused-parameter)

# ---- PhysXTask (multithreading) ----
file(GLOB_RECURSE TASK_SRC ${PHYSX_ROOT_DIR}/source/task/src/*.cpp)
add_library(PhysXTask STATIC ${TASK_SRC})
target_include_directories(PhysXTask PUBLIC
    ${PHYSX_PUBLIC_INC} ${PHYSX_FOUNDATION} ${PHYSX_ALL_INCLUDES} ${PHYSX_ALL_PUBLIC}
    ${PHYSX_ROOT_DIR}/source/task/include
)
target_link_libraries(PhysXTask PUBLIC Foundation Common)
target_compile_options(PhysXTask PRIVATE -fPIC -O2 -Wno-unused-parameter)

# ---- PhysXCooking (geometry cooking) ----
file(GLOB_RECURSE COOKING_SRC ${PHYSX_ROOT_DIR}/source/physxcooking/src/*.cpp)
list(FILTER COOKING_SRC EXCLUDE REGEX ".*/windows/.*")
add_library(PhysXCooking STATIC ${COOKING_SRC})
target_include_directories(PhysXCooking PUBLIC
    ${PHYSX_PUBLIC_INC} ${PHYSX_FOUNDATION} ${PHYSX_ALL_INCLUDES} ${PHYSX_ALL_PUBLIC}
    ${PHYSX_ROOT_DIR}/source/physxcooking/include
)
target_link_libraries(PhysXCooking PUBLIC Foundation LowLevel)
target_compile_options(PhysXCooking PRIVATE -fPIC -O2 -Wno-unused-parameter)

# ---- PhysXExtensions ----
file(GLOB_RECURSE EXT_SRC ${PHYSX_ROOT_DIR}/source/physxextensions/src/*.cpp)
list(FILTER EXT_SRC EXCLUDE REGEX ".*/windows/.*")
list(FILTER EXT_SRC EXCLUDE REGEX ".*omnipvd.*")
add_library(PhysXExtensions STATIC ${EXT_SRC})
target_include_directories(PhysXExtensions PUBLIC
    ${PHYSX_PUBLIC_INC} ${PHYSX_FOUNDATION} ${PHYSX_ALL_INCLUDES} ${PHYSX_ALL_PUBLIC}
    ${PHYSX_ROOT_DIR}/source/physxextensions/include
    ${PHYSX_ROOT_DIR}/source/physxextensions/src
)
target_link_libraries(PhysXExtensions PUBLIC Foundation PhysXSDK PhysXCooking)
target_compile_options(PhysXExtensions PRIVATE -fPIC -O2 -Wno-unused-parameter)

# ---- PhysXVehicle ----
file(GLOB_RECURSE VEHICLE_SRC ${PHYSX_ROOT_DIR}/source/physxvehicle/src/*.cpp)
add_library(PhysXVehicle STATIC ${VEHICLE_SRC})
target_include_directories(PhysXVehicle PUBLIC
    ${PHYSX_PUBLIC_INC} ${PHYSX_FOUNDATION} ${PHYSX_ALL_INCLUDES} ${PHYSX_ALL_PUBLIC}
    ${PHYSX_ROOT_DIR}/source/physxvehicle/include
)
target_link_libraries(PhysXVehicle PUBLIC Foundation PhysXSDK PhysXExtensions)
target_compile_options(PhysXVehicle PRIVATE -fPIC -O2 -Wno-unused-parameter)

# ---- PhysXCharacterKinematic ----
file(GLOB_RECURSE CCT_SRC ${PHYSX_ROOT_DIR}/source/physxcharacterkinematic/src/*.cpp)
add_library(PhysXCharacterKinematic STATIC ${CCT_SRC})
target_include_directories(PhysXCharacterKinematic PUBLIC
    ${PHYSX_PUBLIC_INC} ${PHYSX_FOUNDATION} ${PHYSX_ALL_INCLUDES} ${PHYSX_ALL_PUBLIC}
    ${PHYSX_ROOT_DIR}/source/physxcharacterkinematic/include
)
target_link_libraries(PhysXCharacterKinematic PUBLIC Foundation PhysXSDK)
target_compile_options(PhysXCharacterKinematic PRIVATE -fPIC -O2 -Wno-unused-parameter)

# ---- HipContextManager (GPU dispatch bridge) ----
add_library(HipContext STATIC ${PHYSX_ROOT_DIR}/source/cudamanager/src/HipContextManager.cpp)
target_include_directories(HipContext PUBLIC
    ${PHYSX_PUBLIC_INC} ${PHYSX_FOUNDATION} ${PHYSX_COMMON_INC} ${PHYSX_COMMON_SRC}
    ${PHYSX_ALL_INCLUDES} ${PHYSX_ALL_PUBLIC}
    ${PHYSX_ROOT_DIR}/include/cudamanager
    ${PHYSX_ROOT_DIR}/source/gpucommon/include
    ${PHYSX_ROOT_DIR}/source/gpucommon/src/DCU/stubs
    /opt/dtk/include
)
target_compile_definitions(HipContext PRIVATE PX_DCU_PORT __HIP_PLATFORM_AMD__ PX_SUPPORT_GPU_PHYSX)
target_compile_options(HipContext PRIVATE -fPIC -O2)
target_link_directories(HipContext PUBLIC /opt/dtk/hip/lib)
target_link_libraries(HipContext PUBLIC amdhip64)

# ---- PhysX Scene Benchmark (CPU, GPU info via HipContext) ----
add_executable(bench_physx_scene ${PHYSX_ROOT_DIR}/../examples/dcu/bench_physx_scene.cpp)
target_include_directories(bench_physx_scene PRIVATE
    ${PHYSX_PUBLIC_INC} ${PHYSX_FOUNDATION} ${PHYSX_COMMON_INC} ${PHYSX_COMMON_SRC}
    ${PHYSX_ALL_INCLUDES} ${PHYSX_ALL_PUBLIC}
)
target_link_libraries(bench_physx_scene PRIVATE
    -Wl,--start-group
    Foundation Common PhysXSDK PhysXExtensions PhysXTask HipContext
    LowLevel LowLevelAABB LowLevelDynamics SceneQuery SimulationController
    GeomUtils
    -Wl,--end-group
    pthread dl
)
target_compile_options(bench_physx_scene PRIVATE -fPIC -O2)

# ---- Integration test: CPU Foundation + GPU kernels ----
set(INTEGRATION_SRC ${PHYSX_ROOT_DIR}/source/compiler/cmakecpu/test_integration.cpp)
set_source_files_properties(${INTEGRATION_SRC} PROPERTIES LANGUAGE HIP)

add_executable(test_integration ${INTEGRATION_SRC})
target_include_directories(test_integration PRIVATE
    ${PHYSX_PUBLIC_INC} ${PHYSX_FOUNDATION} ${GPU_INCLUDES}
)
target_link_libraries(test_integration PRIVATE Foundation Common)

message(STATUS "Foundation sources: ${FOUNDATION_COMMON_SRC} ...")
message(STATUS "Build configuration complete")
