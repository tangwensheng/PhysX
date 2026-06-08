#!/bin/bash
# PhysX CPU Library Build for DCU
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
BUILD_DIR="$SCRIPT_DIR/build_cpu"
CMAKE_DIR="$SCRIPT_DIR/physx/source/compiler/cmakecpu"

# Find DTK Clang
for clang_path in /opt/dtk/bin/clang++ /opt/dtk/dcc/bin/clang++ /opt/rocm/bin/clang++; do
    if [ -x "$clang_path" ]; then HIP_CLANG="$clang_path"; break; fi
done

echo "========================================"
echo " PhysX CPU Library Build"
echo " HIP Clang: $HIP_CLANG"
echo "========================================"

rm -rf "$BUILD_DIR"
mkdir -p "$BUILD_DIR"
cd "$BUILD_DIR"

# Extra include paths (aggregate ALL needed subdirectories)
PHYSX="$SCRIPT_DIR/physx"
EXTRA_INCLUDES=""
for subdir in \
    $PHYSX/source/lowlevel/api/include \
    $PHYSX/source/lowlevel/common/include \
    $PHYSX/source/lowlevel/common/include/pipeline \
    $PHYSX/source/lowlevel/common/include/utils \
    $PHYSX/source/lowlevel/common/include/collision \
    $PHYSX/source/lowlevel/software/include \
    $PHYSX/source/geomutils/include \
    $PHYSX/source/geomutils/src \
    $PHYSX/source/geomutils/src/mesh \
    $PHYSX/source/geomutils/src/pcm \
    $PHYSX/source/geomutils/src/gjk \
    $PHYSX/source/geomutils/src/contact \
    $PHYSX/source/geomutils/src/convex \
    $PHYSX/source/geomutils/src/common \
    $PHYSX/source/geomutils/src/hf \
    $PHYSX/source/geomutils/src/intersection \
    $PHYSX/source/geomutils/src/distance \
    $PHYSX/source/geomutils/src/ccd \
    $PHYSX/source/geomutils/src/sweep \
    $PHYSX/source/geomutils/src/cooking \
    $PHYSX/source/common/src \
    $PHYSX/include/geometry \
    $PHYSX/source/gpucommon/include \
    $PHYSX/source/gpucommon/src/CUDA \
    $PHYSX/source/gpucommon/src/DCU/stubs \
    $PHYSX/source/gpunarrowphase/include \
    $PHYSX/source/gpunarrowphase/src/CUDA \
    $PHYSX/source/gpusolver/include \
    $PHYSX/source/gpusolver/src/CUDA \
    $PHYSX/source/gpusimulationcontroller/include \
    $PHYSX/source/gpusimulationcontroller/src/CUDA \
    $PHYSX/source/gpubroadphase/include \
    $PHYSX/source/gpuarticulation/include \
    $PHYSX/source/lowleveldynamics/include \
    $PHYSX/source/lowleveldynamics/shared \
    $PHYSX/source/lowlevelaabb/include \
    $PHYSX/source/cudamanager/include \
    $PHYSX/source/simulationcontroller/include \
    $PHYSX/source/scenequery/include \
    ; do
    if [ -d "$subdir" ]; then
        EXTRA_INCLUDES="$EXTRA_INCLUDES -I$subdir"
    fi
done

cmake "$CMAKE_DIR" \
    -DCMAKE_CXX_COMPILER="$(which g++)" \
    -DCMAKE_HIP_COMPILER="$HIP_CLANG" \
    -DCMAKE_CXX_FLAGS="$EXTRA_INCLUDES" \
    -DPHYSX_ROOT_DIR="$PHYSX"

cmake --build . -j1 2>&1 | tee build.log

echo ""
echo "Build complete."
echo "Run: ./build_cpu/test_cpu"
