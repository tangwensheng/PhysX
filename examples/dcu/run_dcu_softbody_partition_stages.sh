#!/usr/bin/env bash
# Localize the gfx936 deformable-volume partition kernel with one pinned binary.

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="${REPO_ROOT:-$(cd "$SCRIPT_DIR/../.." && pwd)}"
EXECUTABLE="${EXECUTABLE:-$SCRIPT_DIR/build/bench_deformable_volume_smoke}"
HSACO="${HSACO:-$REPO_ROOT/build_dcu/kernels/softBodyGM.hsaco}"
GPU_LIBRARY="${GPU_LIBRARY:-$REPO_ROOT/build_dcu/libPhysXGpuDCU.so}"
DEVICE="${TARGET_HIP_DEVICE:-1}"
EXPECTED_HCU_COUNT="${EXPECTED_HCU_COUNT:-8}"
TARGET_CALL="${TARGET_CALL:-291}"
TARGET_PARTITION="${TARGET_PARTITION:-0}"
STEPS="${STEPS:-300}"
VOXELS="${VOXELS:-8}"
TIMEOUT_SECONDS="${TIMEOUT_SECONDS:-1800}"
LOG_ROOT="${LOG_ROOT:-$SCRIPT_DIR/logs}"

HOST_MARKER="PX_DCU_SOFTBODY_PARTITION_DIAG_V3_TARGETED_STAGE"
HOST_LAUNCH_MARKER="PX_DCU_SOFTBODY_PARTITION_LAUNCH_V5_SINGLE_BLOCK_GRID_STRIDE"
DEVICE_MARKER="PX_DCU_SOFTBODY_HEX_GS_V5_SINGLE_BLOCK_GRID_STRIDE"

check_health()
{
    local output count alerts
    output="$(timeout 15s hy-smi 2>&1 || true)"
    count="$(printf '%s\n' "$output" | awk '/^[[:space:]]*[0-9]+[[:space:]]/{count++} END{print count+0}')"
    alerts="$(printf '%s\n' "$output" | grep -Eic 'Open mkfd failed|No device available|initialization failed|[[:space:]]N/A[[:space:]]' || true)"
    printf 'health hcu_count=%s expected=%s alerts=%s healthy=%s\n' \
        "$count" "$EXPECTED_HCU_COUNT" "$alerts" \
        "$([[ "$count" -eq "$EXPECTED_HCU_COUNT" && "$alerts" -eq 0 ]] && printf 1 || printf 0)"
    [[ "$count" -eq "$EXPECTED_HCU_COUNT" && "$alerts" -eq 0 ]]
}

if [[ ! -x "$EXECUTABLE" ]]; then
    printf 'ERROR: executable missing or not executable: %s\n' "$EXECUTABLE" >&2
    exit 2
fi
if [[ ! -f "$HSACO" ]]; then
    printf 'ERROR: hsaco missing: %s\n' "$HSACO" >&2
    exit 2
fi
if ! grep -aFq "$HOST_MARKER" "$EXECUTABLE"; then
    printf 'ERROR: host marker missing from executable: %s\n' "$HOST_MARKER" >&2
    exit 3
fi
if ! grep -aFq "$HOST_LAUNCH_MARKER" "$EXECUTABLE"; then
    printf 'ERROR: host launch marker missing from executable: %s\n' "$HOST_LAUNCH_MARKER" >&2
    exit 3
fi
if ! grep -aFq "$DEVICE_MARKER" "$HSACO"; then
    printf 'ERROR: device marker missing from hsaco: %s\n' "$DEVICE_MARKER" >&2
    exit 3
fi

timestamp="$(date +%Y%m%d_%H%M%S)"
RUN_DIR="$LOG_ROOT/dcu_softbody_partition_stages_${timestamp}"
SUMMARY_LOG="$RUN_DIR/summary.log"
IDENTITY_LOG="$RUN_DIR/identity.log"
mkdir -p "$RUN_DIR"

{
    printf 'timestamp=%s\n' "$timestamp"
    printf 'target_hip_device=%s\n' "$DEVICE"
    printf 'target_call=%s\n' "$TARGET_CALL"
    printf 'target_partition=%s\n' "$TARGET_PARTITION"
    printf 'steps=%s\n' "$STEPS"
    printf 'voxels=%s\n' "$VOXELS"
    printf 'host_marker=%s\n' "$HOST_MARKER"
    printf 'host_launch_marker=%s\n' "$HOST_LAUNCH_MARKER"
    printf 'device_marker=%s\n' "$DEVICE_MARKER"
    md5sum "$EXECUTABLE" "$HSACO" "$GPU_LIBRARY" 2>/dev/null || true
    hy-smi || true
} > "$IDENTITY_LOG" 2>&1

printf 'run_dir=%s\n' "$RUN_DIR" | tee "$SUMMARY_LOG"

for stage in $(seq 1 9); do
    log="$RUN_DIR/stage${stage}.log"
    printf '\n=== stage %s ===\n' "$stage" | tee -a "$SUMMARY_LOG"

    if ! check_health | tee "$log"; then
        printf 'stage=%s result=DEVICE_UNHEALTHY\n' "$stage" | tee -a "$SUMMARY_LOG"
        exit 1
    fi

    timeout --signal=TERM --kill-after=10s "${TIMEOUT_SECONDS}s" \
        env HIP_VISIBLE_DEVICES="$DEVICE" \
        PX_DCU_SOFTBODY_PARTITION_DIAG=1 \
        PX_DCU_SOFTBODY_PARTITION_DIAG_CALL="$TARGET_CALL" \
        PX_DCU_SOFTBODY_PARTITION_DIAG_PARTITION="$TARGET_PARTITION" \
        PX_DCU_SOFTBODY_PARTITION_DIAG_STAGE="$stage" \
        stdbuf -oL -eL "$EXECUTABLE" "$STEPS" --voxels "$VOXELS" \
        2>&1 | tee -a "$log"
    run_exit=${PIPESTATUS[0]}

    post_health=0
    check_health | tee -a "$log" || post_health=1
    input_valid=0
    stage_complete=0
    grep -Eq 'phase=input_validation .*verdict=VALID$' "$log" && input_valid=1
    grep -Fq "phase=stage_complete partition=$TARGET_PARTITION stage=$stage sync=0 exit=96" "$log" && stage_complete=1

    if [[ "$run_exit" -eq 96 && "$post_health" -eq 0 && "$input_valid" -eq 1 && "$stage_complete" -eq 1 ]]; then
        printf 'stage=%s run_exit=%s input_valid=%s stage_complete=%s post_health=%s result=STABLE log=%s\n' \
            "$stage" "$run_exit" "$input_valid" "$stage_complete" "$post_health" "$log" | tee -a "$SUMMARY_LOG"
    else
        printf 'stage=%s run_exit=%s input_valid=%s stage_complete=%s post_health=%s result=FAIL log=%s\n' \
            "$stage" "$run_exit" "$input_valid" "$stage_complete" "$post_health" "$log" | tee -a "$SUMMARY_LOG"
        printf 'STOP: first non-STABLE stage=%s\n' "$stage" | tee -a "$SUMMARY_LOG"
        exit 1
    fi
done

printf '\nALL_STAGES_STABLE=1\n' | tee -a "$SUMMARY_LOG"
