#!/bin/bash
# Low-risk DCU smoke matrix for PhysX PBD particle cloth.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

TARGET=${TARGET:-bench_pbd_cloth_smoke}
WAIT_SECONDS="${CLEANEXIT_WAIT:-15}"
SLEEP_SECONDS="${BETWEEN_RUN_WAIT:-5}"

./build.sh "$TARGET"

run_case() {
    local steps="$1"
    local dim="$2"
    shift 2
    echo ""
    echo "=== PBD cloth smoke: steps=${steps} dim=${dim} $* ==="
    "./build/${TARGET}" "$steps" --dim "$dim" --cleanexit-wait "$WAIT_SECONDS" "$@"
    sleep "$SLEEP_SECONDS"
}

run_case 1 2 --print-all-steps
run_case 10 2 --print-all-steps
run_case 10 4 --print-all-steps
run_case 60 4
run_case 60 8

# Optional GPU broadphase probe; keep it last so a failure does not block default-BP evidence.
run_case 10 4 --force-gpu-bp --print-all-steps
