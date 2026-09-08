#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
BUILD_DIR="${SCRIPT_DIR}/build"
ARCH="${DCU_ARCH:-gfx936}"

for candidate in /opt/dtk/bin/hipcc /opt/dtk/dcc/bin/hipcc /opt/rocm/bin/hipcc; do
    if [ -x "${candidate}" ]; then
        HIPCC="${candidate}"
        break
    fi
done

if [ -z "${HIPCC:-}" ]; then
    echo "ERROR: hipcc was not found under /opt/dtk or /opt/rocm" >&2
    exit 1
fi

mkdir -p "${BUILD_DIR}"

COMMON_FLAGS=(-O2 -std=c++17 "--offload-arch=${ARCH}")
"${HIPCC}" "${COMMON_FLAGS[@]}" \
    "${SCRIPT_DIR}/host_mapped_atomic_drop_repro.hip" \
    -o "${BUILD_DIR}/host_mapped_atomic_drop_repro"
"${HIPCC}" "${COMMON_FLAGS[@]}" \
    "${SCRIPT_DIR}/device_atomic_readback_fixed.hip" \
    -o "${BUILD_DIR}/device_atomic_readback_fixed"

echo "Built for ${ARCH}:"
echo "  ${BUILD_DIR}/device_atomic_readback_fixed"
echo "  ${BUILD_DIR}/host_mapped_atomic_drop_repro"
echo "The build script does not run either executable."
