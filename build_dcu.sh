#!/bin/bash
# PhysX DCU Build Script
set -e

DCU_ARCH="${1:-gfx936}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
BUILD_DIR="$SCRIPT_DIR/build_dcu"
CMAKE_DIR="$SCRIPT_DIR/physx/source/compiler/cmakehip"

echo "========================================"
echo " PhysX DCU Build"
echo " Architecture: $DCU_ARCH"
echo "========================================"

# Find DTK Clang
for clang_path in /opt/dtk/bin/clang++ /opt/dtk/dcc/bin/clang++ /opt/rocm/bin/clang++ /opt/rocm/llvm/bin/clang++; do
    if [ -x "$clang_path" ]; then HIP_CLANG="$clang_path"; break; fi
done
echo "HIP Clang: $HIP_CLANG"

# Configure (only if not already done)
if [ ! -f "$BUILD_DIR/CMakeCache.txt" ]; then
    rm -rf "$BUILD_DIR"
    mkdir -p "$BUILD_DIR"
    cd "$BUILD_DIR"
    cmake "$CMAKE_DIR" \
        -DCMAKE_HIP_COMPILER="$HIP_CLANG" \
        -DCMAKE_CXX_COMPILER="$(which g++)" \
        -DDCU_ARCH="$DCU_ARCH" \
        -DPHYSX_ROOT_DIR="$SCRIPT_DIR/physx"
else
    cd "$BUILD_DIR"
    echo "Incremental build (existing config)..."
fi

# Build changed files only
cmake --build . -j1 2>&1 | tee build.log

echo ""
echo "Build complete."
echo "Run: ./build_dcu/test_functional"
