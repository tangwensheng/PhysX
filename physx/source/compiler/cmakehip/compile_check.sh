#!/bin/bash
# PhysX DCU Port — Full Compilation Check
#
# Validates that ALL adapted .cu kernel files compile with hipcc.
# This is a syntax/header check — full linking requires the PhysX CPU library.
#
# Usage:
#   chmod +x compile_check.sh
#   ./compile_check.sh [gfx936|gfx938]

set -e

DCU_ARCH="${1:-gfx936}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PHYSX_SRC="$SCRIPT_DIR/../.."

echo "========================================"
echo " PhysX DCU Port — Compilation Check"
echo " Architecture: $DCU_ARCH"
echo " Source root:  $PHYSX_SRC"
echo "========================================"
echo ""

PASSED=0
FAILED=0
TOTAL=0

# Common include paths for PhysX GPU + CPU code
PHYSX_ROOT="$PHYSX_SRC/.."

# HIP stubs: provide cuda_runtime.h, cuda.h, sm_35_intrinsics.h, vector_types.h
# These redirect to hip/hip_runtime.h. MUST come first in include order.
STUBS_DIR="$PHYSX_SRC/gpucommon/src/DCU/stubs"

GPU_INCLUDES="
    -I$STUBS_DIR
    -I$PHYSX_ROOT/include
    -I$PHYSX_ROOT/source/common/include
    -I$PHYSX_ROOT/source/gpucommon/include
    -I$PHYSX_ROOT/source/gpucommon/src/CUDA
    -I$PHYSX_ROOT/source/gpunarrowphase/include
    -I$PHYSX_ROOT/source/gpunarrowphase/src/CUDA
    -I$PHYSX_ROOT/source/gpusolver/include
    -I$PHYSX_ROOT/source/gpusolver/src/CUDA
    -I$PHYSX_ROOT/source/gpusimulationcontroller/include
    -I$PHYSX_ROOT/source/gpusimulationcontroller/src/CUDA
    -I$PHYSX_ROOT/source/gpubroadphase/include
    -I$PHYSX_ROOT/source/gpubroadphase/src/CUDA
    -I$PHYSX_ROOT/source/gpuarticulation/include
    -I$PHYSX_ROOT/source/gpuarticulation/src/CUDA
    -I$PHYSX_ROOT/source/cudamanager/include
    -I$PHYSX_ROOT/source/foundation/include
    -I$PHYSX_ROOT/source/lowlevel/include
    -I$PHYSX_ROOT/source/lowlevel/common/include
    -I$PHYSX_ROOT/source/geomutils/include
    -I$PHYSX_ROOT/source/physicsgpu/include
    -I$PHYSX_ROOT/source/simulationcontroller/include
    -I$PHYSX_ROOT/source/scenequery/include
    -I$PHYSX_ROOT/source/lowlevel/api/include
    -I$PHYSX_ROOT/source/lowlevel/software/include
    -I$PHYSX_ROOT/source/lowleveldynamics/include
    -I$PHYSX_ROOT/source/lowleveldynamics/shared
    -I$PHYSX_ROOT/source/lowlevelaabb/include
    -I$PHYSX_ROOT/source/geomutils/src/mesh
    -I$PHYSX_ROOT/source/geomutils/src/gjk
    -I$PHYSX_ROOT/source/common/src
"

# Compile flag for hipcc — device-side only check (no linking)
# -include PxgHIPCompat.h: forces HIP compat header in every translation unit
#   This provides float4/int4/uint4 types, and maps CUDA intrinsics to HIP.
#   Eliminates the need to guard every #include <cuda.h> individually.
# -DPX_DCU_PORT: selects DCU code paths
HIP_FLAGS="-include PxgHIPCompat.h -DPX_DCU_PORT --offload-arch=$DCU_ARCH -ffast-math -fgpu-flush-denormals-to-zero -c"

compile_file() {
    local f=$1
    local name=$(basename "$f")
    TOTAL=$((TOTAL + 1))
    printf "  [%3d/%3d] %s ... " $TOTAL 0 "$name"
    if hipcc $HIP_FLAGS $GPU_INCLUDES "$f" -o /dev/null 2>&1; then
        echo "PASS"
        PASSED=$((PASSED + 1))
    else
        echo "FAIL"
        FAILED=$((FAILED + 1))
    fi
}

# ---- gpucommon (3 .cu files) ----
echo "--- gpucommon ---"
for f in $PHYSX_SRC/gpucommon/src/CUDA/*.cu; do
    compile_file "$f"
done

# ---- gpubroadphase (2 .cu files) ----
echo "--- gpubroadphase ---"
for f in $PHYSX_SRC/gpubroadphase/src/CUDA/*.cu; do
    compile_file "$f"
done

# ---- gpuarticulation (4 .cu files) ----
echo "--- gpuarticulation ---"
for f in $PHYSX_SRC/gpuarticulation/src/CUDA/*.cu; do
    compile_file "$f"
done

# ---- gpusolver (14 .cu files) ----
echo "--- gpusolver ---"
for f in $PHYSX_SRC/gpusolver/src/CUDA/*.cu; do
    compile_file "$f"
done

# ---- gpusimulationcontroller (16 .cu files) ----
echo "--- gpusimulationcontroller ---"
for f in $PHYSX_SRC/gpusimulationcontroller/src/CUDA/*.cu; do
    compile_file "$f"
done

# ---- gpunarrowphase (30 .cu files) ----
echo "--- gpunarrowphase ---"
for f in $PHYSX_SRC/gpunarrowphase/src/CUDA/*.cu; do
    compile_file "$f"
done

echo ""
echo "========================================"
echo " Results: $PASSED / $TOTAL passed"
echo "========================================"

if [ $FAILED -eq 0 ]; then
    echo "ALL KERNEL FILES COMPILE — DCU port ready for integration"
    exit 0
else
    echo "$FAILED file(s) failed — see errors above"
    exit 1
fi
