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

# HCU Toolchain Probe — Direct build script (no CMake required)
#
# Usage:
#   chmod +x build_probe.sh
#   ./build_probe.sh [gfx936|gfx938]
#
# Prerequisites:
#   - hipcc in $PATH
#   - hiprt (HIP Runtime) installed

set -e

# Default HCU architecture
DCU_ARCH="${1:-gfx936}"

# Paths relative to this script
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROBE_DIR="$SCRIPT_DIR/../../gpucommon/src/DCU"
INCLUDE_DIR="$SCRIPT_DIR/../../gpucommon/include"
OUTPUT_BIN="./probe_dcu"

echo "=== HCU Toolchain Probe Build ==="
echo "HCU architecture: $DCU_ARCH"
echo "Kernel:   $PROBE_DIR/probe_kernel.hip"
echo "Launcher: $PROBE_DIR/probe_main.cpp"
echo "Include:  $INCLUDE_DIR"
echo "Output:   $OUTPUT_BIN"
echo ""

# Build command:
#   hipcc compiles both .hip (GPU kernels) and .cpp (host code) in one invocation
#   --offload-arch specifies the target GPU architecture
#   -DPX_DCU_PORT triggers the HCU code path in PxgCommonDefines.h
#   -ffast-math -fgpu-flush-denormals-to-zero mirrors NVIDIA's -use_fast_math -ftz=true
hipcc \
    "$PROBE_DIR/probe_kernel.hip" \
    "$PROBE_DIR/probe_main.cpp" \
    -I"$INCLUDE_DIR" \
    -DPX_DCU_PORT \
    -o "$OUTPUT_BIN" \
    --offload-arch="$DCU_ARCH" \
    -ffast-math \
    -fgpu-flush-denormals-to-zero

echo ""
echo "Build successful!"
echo ""
echo "Run:  ./probe_dcu"
echo ""
echo "Expected output:"
echo "  6 / 6 tests passed"
echo "  HCU toolchain probe: ALL TESTS PASSED"
