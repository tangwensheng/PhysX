#!/usr/bin/env bash
set -u

# Low-risk DCU smoke matrix for PhysX complex triangle mesh.
# Runs cases in separate processes so a VMFault/segfault does not hide the last stable size.

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT_DIR"

TARGET=${TARGET:-bench_complex_trimesh_smoke}
BIN="./build/${TARGET}"
WAIT_SECONDS=${WAIT_SECONDS:-15}

if [[ ! -x "$BIN" ]]; then
    echo "Binary not found: $BIN"
    echo "Build it first: ./build.sh ${TARGET}"
    exit 1
fi

run_case() {
    local name="$1"
    shift
    echo "========================================"
    echo "CASE: ${name}"
    echo "CMD : $BIN $* --cleanexit-wait ${WAIT_SECONDS}"
    echo "========================================"
    "$BIN" "$@" --cleanexit-wait "${WAIT_SECONDS}"
    local rc=$?
    echo "CASE ${name} exit code: ${rc}"
    echo
    return ${rc}
}

# Cooking / scene-only style probes first. --bodies 0 still simulates the static mesh scene.
run_case "grid24_bodies0_steps1" 1 --grid 24 --bodies 0 --print-all-steps || exit $?
run_case "grid48_bodies0_steps1" 1 --grid 48 --bodies 0 --print-all-steps || exit $?
run_case "grid96_bodies0_steps1" 1 --grid 96 --bodies 0 --print-all-steps || exit $?

# Known-low body counts before scaling toward the previously-failing default.
run_case "grid48_bodies8_steps60" 60 --grid 48 --bodies 8 --print-all-steps || exit $?
run_case "grid48_bodies32_steps120" 120 --grid 48 --bodies 32 --print-all-steps || exit $?
run_case "grid48_bodies64_steps240" 240 --grid 48 --bodies 64 --print-all-steps || exit $?
run_case "grid48_bodies128_steps360" 360 --grid 48 --bodies 128 --print-all-steps || exit $?
run_case "grid48_bodies256_steps720" 720 --grid 48 --bodies 256 --print-all-steps || exit $?

echo "All complex triangle mesh smoke cases passed."
