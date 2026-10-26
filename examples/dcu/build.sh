#!/usr/bin/env bash
# Build PhysX DCU Examples
set -euo pipefail

BUILD_SCRIPT_MARKER="PX_DCU_EXAMPLES_BUILD_R2_FOREIGN_CMAKE_CACHE_RECOVERY"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd -P)"
BUILD_DIR="$SCRIPT_DIR/build"
PHYSX_ROOT="$SCRIPT_DIR/../../physx"
PHYSX_BUILD_DIR="${PHYSX_BUILD_DIR:-$SCRIPT_DIR/../../build_cpu}"

reset_build_dir()
{
    if [[ -z "$BUILD_DIR" || "$BUILD_DIR" != "$SCRIPT_DIR/build" ]]; then
        printf 'ERROR: refusing to reset unexpected build directory: %s\n' "$BUILD_DIR" >&2
        exit 2
    fi
    rm -rf -- "$BUILD_DIR"
    mkdir -p "$BUILD_DIR"
}

# Find a usable HIP compiler (same search order as build_dcu.sh).
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
if [[ -z "$HIP_CLANG" ]]; then
    echo "ERROR: Cannot find a usable HIP compiler" >&2
    exit 1
fi

echo "HIP Clang: $HIP_CLANG"
echo "Building PhysX DCU Examples..."
printf 'build_script_marker=%s\n' "$BUILD_SCRIPT_MARKER"

# CMake caches absolute paths. A cache copied from Windows/WSL or another checkout cannot be
# reused on the remote node, so discard only this exact example build directory on mismatch.
CACHE_FILE="$BUILD_DIR/CMakeCache.txt"
if [[ -f "$CACHE_FILE" ]]; then
    cache_source="$(sed -n 's/^CMAKE_HOME_DIRECTORY:INTERNAL=//p' "$CACHE_FILE" | tail -1)"
    cache_build="$(sed -n 's/^CMAKE_CACHEFILE_DIR:INTERNAL=//p' "$CACHE_FILE" | tail -1)"
    cache_hip="$(sed -n 's/^CMAKE_HIP_COMPILER:[^=]*=//p' "$CACHE_FILE" | tail -1)"
    if [[ "$cache_source" != "$SCRIPT_DIR" || "$cache_build" != "$BUILD_DIR" || \
          "$cache_hip" != "$HIP_CLANG" || ! -x "$cache_hip" ]]; then
        printf 'Discarding stale/foreign CMake cache: source=%s build=%s hip=%s expected_hip=%s\n' \
            "${cache_source:-missing}" "${cache_build:-missing}" \
            "${cache_hip:-missing}" "$HIP_CLANG"
        reset_build_dir
    fi
elif [[ -e "$BUILD_DIR" ]]; then
    printf 'Discarding incomplete example build directory without CMakeCache.txt: %s\n' "$BUILD_DIR"
    reset_build_dir
else
    mkdir -p "$BUILD_DIR"
fi

cd "$BUILD_DIR"

# Run cmake if needed, or when example build rules changed.
if [[ ! -f Makefile || "$SCRIPT_DIR/CMakeLists.txt" -nt CMakeCache.txt ]]; then
    cmake "$SCRIPT_DIR" \
        -DCMAKE_CXX_COMPILER="$(command -v g++)" \
        -DCMAKE_HIP_COMPILER="$HIP_CLANG" \
        -DPHYSX_ROOT_DIR="$PHYSX_ROOT" \
        -DPHYSX_BUILD_DIR="$PHYSX_BUILD_DIR"
fi

TARGET="${1:-bench_physx_scene}"
cmake --build . -j1 --target "$TARGET"

if [[ ! -x "$TARGET" ]]; then
    printf 'ERROR: target was not produced or is not executable: %s/%s\n' "$BUILD_DIR" "$TARGET" >&2
    exit 1
fi
printf 'Built %s/%s\n' "$BUILD_DIR" "$TARGET"
