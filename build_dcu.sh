#!/bin/bash
# PhysX DCU Build Script
#
# Builds PhysX GPU modules with hipcc on DCU hardware.
#
# Prerequisites:
#   - hipcc in PATH
#   - cmake >= 3.21
#   - Full PhysX source tree
#
# Usage:
#   cd PhysX
#   chmod +x build_dcu.sh
#   ./build_dcu.sh [gfx936|gfx938]

set -e

DCU_ARCH="${1:-gfx936}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
BUILD_DIR="$SCRIPT_DIR/build_dcu"
CMAKE_DIR="$SCRIPT_DIR/physx/source/compiler/cmakehip"

echo "========================================"
echo " PhysX DCU Build"
echo " Architecture: $DCU_ARCH"
echo " Source:       $SCRIPT_DIR"
echo " Build:        $BUILD_DIR"
echo "========================================"

# Clean build
rm -rf "$BUILD_DIR"
mkdir -p "$BUILD_DIR"
cd "$BUILD_DIR"

# Find the ROCm/DTK Clang (CMake 3.29+ requires Clang directly, not hipcc wrapper)
# DTK paths: /opt/dtk/bin/clang++ or /opt/dtk/dcc/bin/clang++
# ROCm paths: /opt/rocm/bin/clang++ or /opt/rocm/llvm/bin/clang++
for clang_path in /opt/dtk/bin/clang++ /opt/dtk/dcc/bin/clang++ /opt/rocm/bin/clang++ /opt/rocm/llvm/bin/clang++; do
    if [ -x "$clang_path" ]; then
        HIP_CLANG="$clang_path"
        break
    fi
done

if [ -z "$HIP_CLANG" ]; then
    echo "ERROR: Cannot find ROCm/DTK Clang compiler. Tried:"
    ls -la /opt/dtk/bin/clang++ /opt/rocm/bin/clang++ 2>/dev/null || true
    exit 1
fi

echo "HIP Clang: $HIP_CLANG"

# Configure
cmake "$CMAKE_DIR" \
    -DCMAKE_HIP_COMPILER="$HIP_CLANG" \
    -DCMAKE_CXX_COMPILER="$(which g++)" \
    -DCMAKE_HIP_COMPILER_ROCM_ROOT="/opt/dtk" \
    -DDCU_ARCH="$DCU_ARCH" \
    -DPHYSX_ROOT_DIR="$SCRIPT_DIR/physx"

# Build
cmake --build . -j1 2>&1 | tee build.log

echo ""
echo "========================================"
echo " Build complete!"
echo "========================================"

# Run test
if [ -f test_physx_dcu ]; then
    echo ""
    echo "Running integration test..."
    ./test_physx_dcu
fi
