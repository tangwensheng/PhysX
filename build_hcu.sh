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
set -euo pipefail

BUILD_SCRIPT_MARKER="PX_HCU_BUILD_SCRIPT_R3_FOREIGN_CACHE_RECOVERY"

HCU_ARCH="${1:-gfx936}"
CMAKE_BUILD_TYPE="${CMAKE_BUILD_TYPE:-Release}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
BUILD_DIR="$SCRIPT_DIR/build_hcu"
CMAKE_DIR="$SCRIPT_DIR/physx/source/compiler/cmakehip"
EXTRA_ARGS_FINGERPRINT="$BUILD_DIR/.px_hcu_cmake_extra_args"

case "$CMAKE_BUILD_TYPE" in
    Debug|Release|RelWithDebInfo|MinSizeRel) ;;
    *)
        echo "ERROR: unsupported CMAKE_BUILD_TYPE: $CMAKE_BUILD_TYPE" >&2
        echo "Use Debug, Release, RelWithDebInfo, or MinSizeRel." >&2
        exit 2
        ;;
esac

is_enabled() {
    case "${1:-}" in
        1|ON|on|TRUE|true|YES|yes) return 0 ;;
        *) return 1 ;;
    esac
}

echo "========================================"
echo " PhysX HCU Build"
echo " Architecture: $HCU_ARCH"
echo " Build type:   $CMAKE_BUILD_TYPE"
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
        if [[ "$CMAKE_EXTRA_ARGS" == *PX_DCU_* ]]; then
            echo "WARNING: PX_DCU_* CMake options are ignored after the HCU rename." >&2
            echo "         Replace them with the matching PX_HCU_* option." >&2
        fi
        read -r -a extra_args <<< "$CMAKE_EXTRA_ARGS"
    fi
    cmake "$CMAKE_DIR" \
        -DCMAKE_HIP_COMPILER="$HIP_CLANG" \
        -DCMAKE_CXX_COMPILER="$(which g++)" \
        -DCMAKE_BUILD_TYPE="$CMAKE_BUILD_TYPE" \
        -DHCU_ARCH="$HCU_ARCH" \
        -DPHYSX_ROOT_DIR="$SCRIPT_DIR/physx" \
        "${extra_args[@]}"
}

# Configure only when needed, and recover a cache created under another
# source/build path before cmake --build can trigger a failed auto-regenerate.
CACHE_FILE="$BUILD_DIR/CMakeCache.txt"
if [[ -d "$BUILD_DIR" ]]; then
    current_extra_args="${CMAKE_EXTRA_ARGS:-}"
    if [[ ! -f "$EXTRA_ARGS_FINGERPRINT" ]]; then
        echo "Discarding CMake cache without a CMAKE_EXTRA_ARGS fingerprint."
        reset_build_dir
    else
        cached_extra_args="$(<"$EXTRA_ARGS_FINGERPRINT")"
        if [[ "$cached_extra_args" != "$current_extra_args" ]]; then
            printf 'Discarding CMake cache after CMAKE_EXTRA_ARGS changed: old=%q new=%q\n' \
                "$cached_extra_args" "$current_extra_args"
            reset_build_dir
        fi
    fi
fi
if [[ -f "$CACHE_FILE" ]]; then
    cache_source="$(sed -n 's/^CMAKE_HOME_DIRECTORY:INTERNAL=//p' "$CACHE_FILE" | tail -1)"
    cache_build="$(sed -n 's/^CMAKE_CACHEFILE_DIR:INTERNAL=//p' "$CACHE_FILE" | tail -1)"
    cache_physx="$(sed -n 's/^PHYSX_ROOT_DIR:[^=]*=//p' "$CACHE_FILE" | tail -1)"
    cache_hip="$(sed -n 's/^CMAKE_HIP_COMPILER:[^=]*=//p' "$CACHE_FILE" | tail -1)"
    cache_build_type="$(sed -n 's/^CMAKE_BUILD_TYPE:[^=]*=//p' "$CACHE_FILE" | tail -1)"
    cache_arch="$(sed -n 's/^HCU_ARCH:[^=]*=//p' "$CACHE_FILE" | tail -1)"
    if [[ "$cache_source" != "$CMAKE_DIR" || "$cache_build" != "$BUILD_DIR" || \
          "$cache_physx" != "$SCRIPT_DIR/physx" || "$cache_hip" != "$HIP_CLANG" || \
          "$cache_build_type" != "$CMAKE_BUILD_TYPE" || "$cache_arch" != "$HCU_ARCH" || \
          ! -x "$cache_hip" ]]; then
        printf 'Discarding incompatible CMake cache: source=%s build=%s physx=%s hip=%s type=%s arch=%s\n' \
            "${cache_source:-missing}" "${cache_build:-missing}" "${cache_physx:-missing}" \
            "${cache_hip:-missing}" "${cache_build_type:-missing}" "${cache_arch:-missing}"
        reset_build_dir
    fi
fi

if [[ ! -f "$CACHE_FILE" ]]; then
    reset_build_dir
fi

# Always configure so CMAKE_BUILD_TYPE, HCU_ARCH, and CMAKE_EXTRA_ARGS changes
# take effect instead of being silently ignored by an existing cache.
echo "Configuring HCU build..."
configure_build
printf '%s\n' "${CMAKE_EXTRA_ARGS:-}" > "$EXTRA_ARGS_FINGERPRINT"

# Build changed files only
cmake --build . -j1 2>&1 | tee build.log

# Remove stale code objects before extracting the current build's fat binaries.
rm -rf "$BUILD_DIR/kernels"
python3 "$CMAKE_DIR/extract_hsaco.py" \
    "$BUILD_DIR/CMakeFiles/PhysXGpuHCU.dir" \
    "$BUILD_DIR/kernels"

KERNEL_COUNT="$(find "$BUILD_DIR/kernels" -maxdepth 1 -type f -name '*.hsaco' | wc -l)"
if [ "$KERNEL_COUNT" -eq 0 ]; then
    echo "ERROR: No HCU .hsaco modules were generated."
    exit 1
fi
echo "HCU kernel modules: $KERNEL_COUNT"

if is_enabled "${PX_HCU_SYNC_KERNELS:-}"; then
    echo "WARNING: PX_HCU_SYNC_KERNELS is enabled; benchmark results will be much slower." >&2
fi
if is_enabled "${PX_HCU_TRACE_KERNELS:-}"; then
    echo "WARNING: PX_HCU_TRACE_KERNELS is enabled; benchmark results will be much slower." >&2
fi

echo ""
echo "Build complete."
echo "Run: ./build_hcu/test_functional"
