#!/bin/bash
# Build PhysX DCU Examples
set -e
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
BUILD_DIR="$SCRIPT_DIR/build"
PHYSX_ROOT="$SCRIPT_DIR/../../physx"

# Find DTK Clang (same search order as build_dcu.sh)
for cp in /opt/dtk/bin/clang++ /opt/dtk/dcc/bin/clang++ /opt/rocm/bin/clang++ /opt/rocm/llvm/bin/clang++; do
    [ -x "$cp" ] && HIP_CLANG="$cp" && break
done
if [ -z "$HIP_CLANG" ]; then
    echo "ERROR: Cannot find DTK/ROCm Clang compiler"
    exit 1
fi

echo "HIP Clang: $HIP_CLANG"
echo "Building PhysX DCU Examples..."
rm -rf "$BUILD_DIR" && mkdir -p "$BUILD_DIR" && cd "$BUILD_DIR"

cmake "$SCRIPT_DIR" \
    -DCMAKE_CXX_COMPILER="$(which g++)" \
    -DCMAKE_HIP_COMPILER="$HIP_CLANG" \
    -DPHYSX_ROOT_DIR="$PHYSX_ROOT"

cmake --build . -j1

echo ""
echo "=== Running Examples ==="
# for exe in render_scene; do
#     if [ -f "$exe" ]; then
#         echo ""
#         ./"$exe"
#     fi
# done
