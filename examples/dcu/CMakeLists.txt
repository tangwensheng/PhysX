# PhysX DCU Examples
cmake_minimum_required(VERSION 3.21)
project(PhysXDcuExamples LANGUAGES CXX HIP)

if(NOT DEFINED PHYSX_ROOT_DIR)
    get_filename_component(PHYSX_ROOT_DIR "${CMAKE_SOURCE_DIR}/../../physx" ABSOLUTE)
endif()

set(DCU_ARCH "gfx936" CACHE STRING "DCU architecture")
set(CMAKE_HIP_ARCHITECTURES ${DCU_ARCH})
set(CMAKE_HIP_FLAGS "-include PxgHIPCompat.h -DPX_DCU_PORT -ffast-math -fgpu-flush-denormals-to-zero")
set(CMAKE_CXX_FLAGS "-O2 -fPIC")

set(EXAMPLES_INC
    ${PHYSX_ROOT_DIR}/source/gpucommon/include
    ${PHYSX_ROOT_DIR}/source/gpucommon/src/CUDA
    ${PHYSX_ROOT_DIR}/source/gpucommon/src/DCU/stubs
    ${PHYSX_ROOT_DIR}/include
    ${PHYSX_ROOT_DIR}/include/geometry
    ${PHYSX_ROOT_DIR}/source/foundation/include
    ${PHYSX_ROOT_DIR}/source/common/include
    ${PHYSX_ROOT_DIR}/source/common/src
    ${PHYSX_ROOT_DIR}/source/gpunarrowphase/include
    ${PHYSX_ROOT_DIR}/source/gpunarrowphase/src/CUDA
    ${PHYSX_ROOT_DIR}/source/gpusolver/include
    ${PHYSX_ROOT_DIR}/source/gpusolver/src/CUDA
    ${PHYSX_ROOT_DIR}/source/gpusimulationcontroller/include
    ${PHYSX_ROOT_DIR}/source/gpusimulationcontroller/src/CUDA
    ${PHYSX_ROOT_DIR}/source/geomutils/src
    ${PHYSX_ROOT_DIR}/source/geomutils/src/mesh
    ${PHYSX_ROOT_DIR}/source/geomutils/src/gjk
    ${PHYSX_ROOT_DIR}/source/geomutils/src/contact
    ${PHYSX_ROOT_DIR}/source/geomutils/src/convex
    ${PHYSX_ROOT_DIR}/source/geomutils/src/common
    ${PHYSX_ROOT_DIR}/source/geomutils/src/hf
    ${PHYSX_ROOT_DIR}/source/geomutils/src/intersection
    ${PHYSX_ROOT_DIR}/source/geomutils/src/distance
    ${PHYSX_ROOT_DIR}/source/geomutils/src/ccd
    ${PHYSX_ROOT_DIR}/source/geomutils/src/sweep
    ${PHYSX_ROOT_DIR}/source/geomutils/src/pcm
    ${PHYSX_ROOT_DIR}/source/geomutils/include
    ${PHYSX_ROOT_DIR}/source/lowlevel/api/include
    ${PHYSX_ROOT_DIR}/source/lowlevel/software/include
    ${PHYSX_ROOT_DIR}/source/lowlevel/common/include
    ${PHYSX_ROOT_DIR}/source/lowlevel/common/include/pipeline
    ${PHYSX_ROOT_DIR}/source/lowlevel/common/include/utils
    ${PHYSX_ROOT_DIR}/source/lowleveldynamics/include
    ${PHYSX_ROOT_DIR}/source/lowleveldynamics/shared
    ${PHYSX_ROOT_DIR}/source/gpuarticulation/include
    ${PHYSX_ROOT_DIR}/source/gpuarticulation/src/CUDA
    ${PHYSX_ROOT_DIR}/source/gpubroadphase/include
    ${PHYSX_ROOT_DIR}/source/gpubroadphase/src/CUDA
    ${PHYSX_ROOT_DIR}/source/lowlevelaabb/include
    ${PHYSX_ROOT_DIR}/source/physxgpu/include
    ${PHYSX_ROOT_DIR}/source/cudamanager/include
    ${PHYSX_ROOT_DIR}/source/scenequery/include
    ${PHYSX_ROOT_DIR}/source/simulationcontroller/include
)

# ---- GPU Offscreen Renderer (Physics + Raycasting) ----
set(CPU_BUILD_DIR ${PHYSX_ROOT_DIR}/../build_cpu)
file(GLOB CPU_LIBS ${CPU_BUILD_DIR}/lib*.a)
add_executable(render_scene render_scene.hip)
target_include_directories(render_scene PRIVATE ${EXAMPLES_INC})
target_link_directories(render_scene PRIVATE ${CPU_BUILD_DIR})
target_link_libraries(render_scene -Wl,--start-group ${CPU_LIBS} -Wl,--end-group pthread dl)

# ---- GPU Kernel Benchmarks (one per module, #include) ----
add_executable(benchmark_full  benchmark_full.hip)
add_executable(bench_broadphase  bench_broadphase.hip)
add_executable(bench_simctrl     bench_simctrl.hip)
add_executable(bench_narrowphase bench_narrowphase.hip)
add_executable(bench_solver       bench_solver.hip)
add_executable(bench_gpu_pipeline bench_gpu_pipeline.hip)
foreach(t benchmark_full bench_broadphase bench_simctrl bench_narrowphase bench_solver bench_gpu_pipeline)
    target_include_directories(${t} PRIVATE ${EXAMPLES_INC})
endforeach()
