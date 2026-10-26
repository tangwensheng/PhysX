#!/bin/bash
# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#  * Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#  * Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in the
#    documentation and/or other materials provided with the distribution.
#  * Neither the name of NVIDIA CORPORATION nor the names of its
#    contributors may be used to endorse or promote products derived
#    from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ''AS IS'' AND ANY
# EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
# PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
# PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
# OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

# PhysX HCU Build Script
set -e

HCU_ARCH="${1:-gfx936}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
BUILD_DIR="$SCRIPT_DIR/build_hcu"
CMAKE_DIR="$SCRIPT_DIR/physx/source/compiler/cmakehip"

echo "========================================"
echo " PhysX HCU Build"
echo " Architecture: $HCU_ARCH"
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
        -DHCU_ARCH="$HCU_ARCH" \
        -DPHYSX_ROOT_DIR="$SCRIPT_DIR/physx"
else
    cd "$BUILD_DIR"
    echo "Incremental build (existing config)..."
fi

# Build changed files only
cmake --build . -j1 2>&1 | tee build.log

echo ""
echo "Build complete."
echo "Run: ./build_hcu/test_functional"
