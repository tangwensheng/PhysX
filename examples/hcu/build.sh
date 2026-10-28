#!/usr/bin/env bash
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

# Build PhysX HCU Examples
set -euo pipefail

# Keep the legacy marker value: external build verification helpers match it verbatim.
BUILD_SCRIPT_MARKER="PX_DCU_EXAMPLES_BUILD_R2_FOREIGN_CMAKE_CACHE_RECOVERY"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd -P)"
BUILD_DIR="$SCRIPT_DIR/build"
PHYSX_ROOT="$SCRIPT_DIR/../../physx"
PHYSX_BUILD_DIR="${PHYSX_BUILD_DIR:-$SCRIPT_DIR/../../build_cpu}"
CMAKE_BUILD_TYPE="${CMAKE_BUILD_TYPE:-Release}"

case "$CMAKE_BUILD_TYPE" in
    Debug|Release|RelWithDebInfo|MinSizeRel) ;;
    *)
        echo "ERROR: unsupported CMAKE_BUILD_TYPE: $CMAKE_BUILD_TYPE" >&2
        echo "Use Debug, Release, RelWithDebInfo, or MinSizeRel." >&2
        exit 2
        ;;
esac

reset_build_dir()
{
    if [[ -z "$BUILD_DIR" || "$BUILD_DIR" != "$SCRIPT_DIR/build" ]]; then
        printf 'ERROR: refusing to reset unexpected build directory: %s\n' "$BUILD_DIR" >&2
        exit 2
    fi
    rm -rf -- "$BUILD_DIR"
    mkdir -p "$BUILD_DIR"
}

# Find a usable HIP compiler (same search order as build_hcu.sh).
HIP_CLANG=""
if [[ -n "${HIPCXX:-}" ]]; then
    if [[ -x "$HIPCXX" ]]; then
        HIP_CLANG="$HIPCXX"
    else
        HIP_CLANG="$(command -v "$HIPCXX" 2>/dev/null || true)"
    fi
fi
if [[ -z "$HIP_CLANG" ]]; then
    for compiler in /opt/dtk/bin/clang++ /opt/dtk/dcc/bin/clang++ /opt/rocm/bin/clang++ /opt/rocm/llvm/bin/clang++; do
        [[ -x "$compiler" ]] && HIP_CLANG="$compiler" && break
    done
fi
if [[ -z "$HIP_CLANG" ]]; then
    HIP_CLANG="$(command -v clang++ 2>/dev/null || true)"
fi
if [[ -z "$HIP_CLANG" || ! -x "$HIP_CLANG" ]]; then
    echo "ERROR: Cannot find a usable HIP compiler" >&2
    exit 1
fi

echo "HIP Clang: $HIP_CLANG"
echo "Build type: $CMAKE_BUILD_TYPE"
echo "Building PhysX HCU Examples..."
printf 'build_script_marker=%s\n' "$BUILD_SCRIPT_MARKER"

# CMake caches absolute paths. A cache copied from Windows/WSL or another checkout cannot be
# reused on the remote node, so discard only this exact example build directory on mismatch.
CACHE_FILE="$BUILD_DIR/CMakeCache.txt"
if [[ -f "$CACHE_FILE" ]]; then
    cache_source="$(sed -n 's/^CMAKE_HOME_DIRECTORY:INTERNAL=//p' "$CACHE_FILE" | tail -1)"
    cache_build="$(sed -n 's/^CMAKE_CACHEFILE_DIR:INTERNAL=//p' "$CACHE_FILE" | tail -1)"
    cache_hip="$(sed -n 's/^CMAKE_HIP_COMPILER:[^=]*=//p' "$CACHE_FILE" | tail -1)"
    cache_physx_root="$(sed -n 's/^PHYSX_ROOT_DIR:[^=]*=//p' "$CACHE_FILE" | tail -1)"
    cache_physx_build="$(sed -n 's/^PHYSX_BUILD_DIR:[^=]*=//p' "$CACHE_FILE" | tail -1)"
    cache_build_type="$(sed -n 's/^CMAKE_BUILD_TYPE:[^=]*=//p' "$CACHE_FILE" | tail -1)"
    if [[ "$cache_source" != "$SCRIPT_DIR" || "$cache_build" != "$BUILD_DIR" || \
          "$cache_hip" != "$HIP_CLANG" || "$cache_physx_root" != "$PHYSX_ROOT" || \
          "$cache_physx_build" != "$PHYSX_BUILD_DIR" || \
          "$cache_build_type" != "$CMAKE_BUILD_TYPE" || ! -x "$cache_hip" ]]; then
        printf 'Discarding incompatible CMake cache: source=%s build=%s hip=%s physx=%s physx_build=%s type=%s\n' \
            "${cache_source:-missing}" "${cache_build:-missing}" \
            "${cache_hip:-missing}" "${cache_physx_root:-missing}" \
            "${cache_physx_build:-missing}" "${cache_build_type:-missing}"
        reset_build_dir
    fi
elif [[ -e "$BUILD_DIR" ]]; then
    printf 'Discarding incomplete example build directory without CMakeCache.txt: %s\n' "$BUILD_DIR"
    reset_build_dir
else
    mkdir -p "$BUILD_DIR"
fi

cd "$BUILD_DIR"

# Always configure so build type, source paths, and example rules stay current.
cmake "$SCRIPT_DIR" \
    -DCMAKE_CXX_COMPILER="$(command -v g++)" \
    -DCMAKE_HIP_COMPILER="$HIP_CLANG" \
    -DCMAKE_BUILD_TYPE="$CMAKE_BUILD_TYPE" \
    -DPHYSX_ROOT_DIR="$PHYSX_ROOT" \
    -DPHYSX_BUILD_DIR="$PHYSX_BUILD_DIR"

TARGET="${1:-bench_physx_scene}"
cmake --build . -j1 --target "$TARGET"

if [[ ! -x "$TARGET" ]]; then
    printf 'ERROR: target was not produced or is not executable: %s/%s\n' "$BUILD_DIR" "$TARGET" >&2
    exit 1
fi
printf 'Built %s/%s\n' "$BUILD_DIR" "$TARGET"
