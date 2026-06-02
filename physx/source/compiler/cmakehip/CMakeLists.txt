## DCU/HIP Standalone Probe Build
##
## This is a MINIMAL CMake project to validate the DCU toolchain.
## It builds a single probe binary that tests:
##   - hipcc compilation
##   - HIP runtime (memory, kernel launch, sync)
##   - Device intrinsics (shuffle, ballot, atomics, shared mem)
##   - WARP_SIZE=64 correctness
##
## Usage:
##   mkdir build && cd build
##   cmake .. -DCMAKE_HIP_COMPILER=hipcc -DDCU_ARCH=gfx936
##   cmake --build .
##   ./probe_dcu
##
## NOTE: This is NOT part of the full PhysX GPU build yet.
##       It is a standalone validation step before integrating into PhysX.

cmake_minimum_required(VERSION 3.21)

# Enable HIP language support
# The user must have hipcc in PATH or set CMAKE_HIP_COMPILER
project(PhysXDcuProbe LANGUAGES CXX HIP)

# ---- DCU Architecture Setup ----
if(NOT DEFINED DCU_ARCH)
    set(DCU_ARCH "gfx936" CACHE STRING "DCU GPU architecture (gfx936 or gfx938)")
endif()

message(STATUS "DCU target architecture: ${DCU_ARCH}")

# ---- HIP Compiler Flags ----
# Equivalent to NVIDIA's: -use_fast_math -ftz=true -prec-div=false -prec-sqrt=false
set(CMAKE_HIP_FLAGS "${CMAKE_HIP_FLAGS} -ffast-math -fcuda-flush-denormals-to-zero")
set(CMAKE_HIP_FLAGS_DEBUG "-g -O0")
set(CMAKE_HIP_FLAGS_RELEASE "-O3")

# Set offload architecture
set(CMAKE_HIP_ARCHITECTURES "${DCU_ARCH}")

# ---- Probe Source Files ----
# Path to gpucommon includes (relative to this CMakeLists.txt)
set(GPUCOMMON_INCLUDE_DIR "${CMAKE_SOURCE_DIR}/../../gpucommon/include")

# Probe kernel (compiled as HIP)
set(PROBE_KERNEL_SRC "${CMAKE_SOURCE_DIR}/../../gpucommon/src/DCU/probe_kernel.hip")

# Probe host launcher (compiled as CXX)
set(PROBE_MAIN_SRC "${CMAKE_SOURCE_DIR}/../../gpucommon/src/DCU/probe_main.cpp")

# ---- Build the Probe Executable ----
add_executable(probe_dcu
    ${PROBE_KERNEL_SRC}
    ${PROBE_MAIN_SRC}
)

# Set HIP-specific properties on the kernel source
set_source_files_properties(${PROBE_KERNEL_SRC} PROPERTIES
    LANGUAGE HIP
)

target_include_directories(probe_dcu PRIVATE
    ${GPUCOMMON_INCLUDE_DIR}
)

# Link HIP runtime
target_link_libraries(probe_dcu PRIVATE hip::host)

# ---- Status Output ----
message(STATUS "HIP compiler: ${CMAKE_HIP_COMPILER}")
message(STATUS "HIP architectures: ${CMAKE_HIP_ARCHITECTURES}")
message(STATUS "Probe kernel: ${PROBE_KERNEL_SRC}")
message(STATUS "Probe launcher: ${PROBE_MAIN_SRC}")
