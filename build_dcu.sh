#!/bin/bash
# PhysX DCU Build Script
set -euo pipefail

BUILD_SCRIPT_MARKER="PX_DCU_BUILD_SCRIPT_R3_FOREIGN_CACHE_RECOVERY"

DCU_ARCH="${1:-gfx936}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
BUILD_DIR="$SCRIPT_DIR/build_dcu"
CMAKE_DIR="$SCRIPT_DIR/physx/source/compiler/cmakehip"

echo "========================================"
echo " PhysX DCU Build"
echo " Architecture: $DCU_ARCH"
echo "========================================"

# Find a usable HIP compiler in this container.
HIP_CLANG=""
if [[ -n "${HIPCXX:-}" ]]; then
    if [[ -x "$HIPCXX" ]]; then
        HIP_CLANG="$HIPCXX"
    else
        HIP_CLANG="$(command -v "$HIPCXX" 2>/dev/null || true)"
    fi
fi
if [[ -z "$HIP_CLANG" ]]; then
    for clang_path in /opt/dtk/bin/clang++ /opt/dtk/dcc/bin/clang++ /opt/rocm/bin/clang++ /opt/rocm/llvm/bin/clang++; do
        if [ -x "$clang_path" ]; then HIP_CLANG="$clang_path"; break; fi
    done
fi
if [[ -z "$HIP_CLANG" ]]; then
    HIP_CLANG="$(command -v clang++ 2>/dev/null || true)"
fi
if [[ -z "$HIP_CLANG" || ! -x "$HIP_CLANG" ]]; then
    echo "ERROR: no usable HIP compiler found" >&2
    echo "Checked HIPCXX, /opt/dtk/bin/clang++, /opt/dtk/dcc/bin/clang++, /opt/rocm/bin/clang++, /opt/rocm/llvm/bin/clang++, and PATH" >&2
    exit 4
fi
echo "HIP Clang: $HIP_CLANG"
printf 'build_script_marker=%s\n' "$BUILD_SCRIPT_MARKER"

reset_build_dir() {
    # BUILD_DIR is derived from SCRIPT_DIR above; only this exact generated
    # directory is ever discarded when its cache belongs to another checkout.
    rm -rf "$BUILD_DIR"
    mkdir -p "$BUILD_DIR"
}

configure_build() {
    cd "$BUILD_DIR"
    # Keep candidate switches available when a foreign cache must be rebuilt.
    # Arguments are simple -Dkey=value tokens in the delivery commands.
    local extra_args=()
    if [[ -n "${CMAKE_EXTRA_ARGS:-}" ]]; then
        read -r -a extra_args <<< "$CMAKE_EXTRA_ARGS"
    fi
    cmake "$CMAKE_DIR" \
        -DCMAKE_HIP_COMPILER="$HIP_CLANG" \
        -DCMAKE_CXX_COMPILER="$(which g++)" \
        -DDCU_ARCH="$DCU_ARCH" \
        -DPHYSX_ROOT_DIR="$SCRIPT_DIR/physx" \
        "${extra_args[@]}"
}

# Configure only when needed, and recover a cache created under another
# source/build path before cmake --build can trigger a failed auto-regenerate.
CACHE_FILE="$BUILD_DIR/CMakeCache.txt"
if [[ -f "$CACHE_FILE" ]]; then
    cache_source="$(sed -n 's/^CMAKE_HOME_DIRECTORY:INTERNAL=//p' "$CACHE_FILE" | tail -1)"
    cache_build="$(sed -n 's/^CMAKE_CACHEFILE_DIR:INTERNAL=//p' "$CACHE_FILE" | tail -1)"
    cache_physx="$(sed -n 's/^PHYSX_ROOT_DIR:[^=]*=//p' "$CACHE_FILE" | tail -1)"
    cache_hip="$(sed -n 's/^CMAKE_HIP_COMPILER:[^=]*=//p' "$CACHE_FILE" | tail -1)"
    if [[ "$cache_source" != "$CMAKE_DIR" || "$cache_build" != "$BUILD_DIR" || \
          "$cache_physx" != "$SCRIPT_DIR/physx" || "$cache_hip" != "$HIP_CLANG" || \
          ! -x "$cache_hip" ]]; then
        printf 'Discarding stale/foreign CMake cache: source=%s build=%s physx=%s hip=%s expected_hip=%s\n' \
            "${cache_source:-missing}" "${cache_build:-missing}" "${cache_physx:-missing}" \
            "${cache_hip:-missing}" "$HIP_CLANG"
        reset_build_dir
        configure_build
    else
        cd "$BUILD_DIR"
        echo "Incremental build (existing config)..."
    fi
else
    reset_build_dir
    configure_build
fi

# Build changed files only
cmake --build . -j1 2>&1 | tee build.log

python3 "$CMAKE_DIR/extract_hsaco.py" \
    "$BUILD_DIR/CMakeFiles/PhysXGpuDCU.dir" \
    "$BUILD_DIR/kernels"

KERNEL_COUNT="$(find "$BUILD_DIR/kernels" -maxdepth 1 -type f -name '*.hsaco' | wc -l)"
if [ "$KERNEL_COUNT" -eq 0 ]; then
    echo "ERROR: No DCU .hsaco modules were generated."
    exit 1
fi
echo "DCU kernel modules: $KERNEL_COUNT"

echo ""
echo "Build complete."
echo "Run: ./build_dcu/test_functional"
