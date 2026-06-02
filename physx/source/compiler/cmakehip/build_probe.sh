#!/bin/bash
# DCU Toolchain Probe — Direct build script (no CMake required)
#
# Usage:
#   chmod +x build_probe.sh
#   ./build_probe.sh [gfx936|gfx938]
#
# Prerequisites:
#   - hipcc in $PATH
#   - hiprt (HIP Runtime) installed

set -e

# Default DCU architecture
DCU_ARCH="${1:-gfx936}"

# Paths relative to this script
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROBE_DIR="$SCRIPT_DIR/../../gpucommon/src/DCU"
INCLUDE_DIR="$SCRIPT_DIR/../../gpucommon/include"
OUTPUT_BIN="./probe_dcu"

echo "=== DCU Toolchain Probe Build ==="
echo "DCU architecture: $DCU_ARCH"
echo "Kernel:   $PROBE_DIR/probe_kernel.hip"
echo "Launcher: $PROBE_DIR/probe_main.cpp"
echo "Include:  $INCLUDE_DIR"
echo "Output:   $OUTPUT_BIN"
echo ""

# Build command:
#   hipcc compiles both .hip (GPU kernels) and .cpp (host code) in one invocation
#   --offload-arch specifies the target GPU architecture
#   -DPX_DCU_PORT triggers the DCU code path in PxgCommonDefines.h
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
echo "  DCU toolchain probe: ALL TESTS PASSED"
