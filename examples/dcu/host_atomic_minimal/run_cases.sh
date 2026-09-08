#!/usr/bin/env bash
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
BUILD_DIR="${SCRIPT_DIR}/build"
FIXED_EXE="${BUILD_DIR}/device_atomic_readback_fixed"
REPRO_EXE="${BUILD_DIR}/host_mapped_atomic_drop_repro"
TARGET_HIP_DEVICE="${TARGET_HIP_DEVICE:-1}"
EXPECTED_HCU_COUNT="${EXPECTED_HCU_COUNT:-8}"
HOLD_SECONDS="${HOLD_SECONDS:-20}"
POST_WAIT_SECONDS="${POST_WAIT_SECONDS:-5}"
STAMP="$(date +%Y%m%d_%H%M%S)"
RUN_LOG="${BUILD_DIR}/host_atomic_cases_${STAMP}.log"
DMESG_PATTERN='GCEA err detected|0x2320|atomic err|UR_ATOMIC_OPCODE|VMFault|page fault|GPU reset|ring.*timeout|device.*lost|PCIe error|pcie.*error|hycu.*err'

mkdir -p "${BUILD_DIR}"
: > "${RUN_LOG}"

log()
{
    printf '%s\n' "$*" | tee -a "${RUN_LOG}"
}

sample_health()
{
    local phase="$1"
    local output
    local command_exit
    local hcu_count
    local alert_count
    local target_line

    output=$(timeout 10s hy-smi 2>&1)
    command_exit=$?
    hcu_count=$(printf '%s\n' "${output}" | awk '/^[[:space:]]*[0-9]+[[:space:]]/{count++} END{print count+0}')
    alert_count=$(printf '%s\n' "${output}" | grep -Eic 'Open mkfd failed|No device available|initialization failed|[[:space:]]N/A[[:space:]]')
    target_line=$(printf '%s\n' "${output}" | awk -v target="${TARGET_HIP_DEVICE}" '$1 == target {print; exit}')

    if [ "${command_exit}" -eq 0 ] && \
        [ "${hcu_count}" -eq "${EXPECTED_HCU_COUNT}" ] && \
        [ "${alert_count}" -eq 0 ] && [ -n "${target_line}" ]; then
        LAST_HEALTHY=1
    else
        LAST_HEALTHY=0
    fi

    echo "health phase=${phase} hy_smi_exit=${command_exit} hcu_count=${hcu_count} expected_hcu_count=${EXPECTED_HCU_COUNT} target_index=${TARGET_HIP_DEVICE} target_present=$([ -n "${target_line}" ] && echo 1 || echo 0) alert_count=${alert_count} healthy=${LAST_HEALTHY}" >> "${RUN_LOG}"
    if [ "${LAST_HEALTHY}" -eq 0 ]; then
        printf '%s\n' "health_raw_begin phase=${phase}" "${output}" "health_raw_end phase=${phase}" >> "${RUN_LOG}"
    fi
}

collect_dmesg()
{
    local phase="$1"
    local since="$2"
    local output
    local alerts

    output=$(dmesg --since "${since}" 2>&1)
    LAST_DMESG_EXIT=$?
    alerts=$(printf '%s\n' "${output}" | grep -Ei "${DMESG_PATTERN}")
    LAST_ALERT_COUNT=$(printf '%s\n' "${alerts}" | sed '/^$/d' | wc -l | tr -d ' ')
    LAST_UR_COUNT=$(printf '%s\n' "${output}" | grep -Eic 'UR_ATOMIC_OPCODE|Found atomic err')
    LAST_GCEA_COUNT=$(printf '%s\n' "${output}" | grep -Eic 'GCEA err detected|0x2320')

    echo "dmesg phase=${phase} since=${since} exit=${LAST_DMESG_EXIT} alert_count=${LAST_ALERT_COUNT} ur_atomic_count=${LAST_UR_COUNT} gcea_count=${LAST_GCEA_COUNT}" >> "${RUN_LOG}"
    if [ "${LAST_DMESG_EXIT}" -ne 0 ]; then
        printf '%s\n' "${output}" | tail -n 10 | sed "s/^/dmesg_error phase=${phase} /" >> "${RUN_LOG}"
    elif [ "${LAST_ALERT_COUNT}" -gt 0 ]; then
        printf '%s\n' "${alerts}" | tail -n 30 | sed "s/^/dmesg_alert phase=${phase} /" >> "${RUN_LOG}"
    fi
}

log "experiment=standalone_host_atomic target_hip_device=${TARGET_HIP_DEVICE} hold_seconds=${HOLD_SECONDS}"
log "run_log=${RUN_LOG}"

repro_source_count=$(grep -Fc 'atomicMax(hostMappedCounter, *deviceCandidate)' "${SCRIPT_DIR}/host_mapped_atomic_drop_repro.hip")
fixed_source_count=$(grep -Fc 'atomicMax(deviceCounter, *deviceCandidate)' "${SCRIPT_DIR}/device_atomic_readback_fixed.hip")
fixed_copy_count=$(grep -Fc 'hipMemcpyDtoHAsync' "${SCRIPT_DIR}/device_atomic_readback_fixed.hip")
if [ "${repro_source_count}" -eq 1 ] && [ "${fixed_source_count}" -eq 1 ] && [ "${fixed_copy_count}" -eq 1 ]; then
    source_gate=1
else
    source_gate=0
fi
echo "source_gate=${source_gate} repro_atomic_count=${repro_source_count} fixed_atomic_count=${fixed_source_count} fixed_dtoh_count=${fixed_copy_count}" >> "${RUN_LOG}"

if [ "${source_gate}" -eq 1 ]; then
    bash "${SCRIPT_DIR}/build.sh" >> "${RUN_LOG}" 2>&1
    build_exit=$?
else
    build_exit=125
fi

if [ "${build_exit}" -eq 0 ] && [ -x "${FIXED_EXE}" ] && [ -x "${REPRO_EXE}" ] && \
    strings "${FIXED_EXE}" | grep -F 'atomic_target_gpu_device_buffer=' >/dev/null && \
    strings "${REPRO_EXE}" | grep -F 'atomic_target_cpu_host_backing=' >/dev/null; then
    build_gate=1
else
    build_gate=0
fi
log "build_gate=${build_gate} build_exit=${build_exit}"

sample_health preflight
preflight_health="${LAST_HEALTHY}"
log "preflight_gate=${preflight_health}"
if [ "${build_gate}" -ne 1 ] || [ "${preflight_health}" -ne 1 ]; then
    log "overall_verdict=NOT_TESTED_GATE_REJECTED"
    exit 125
fi

fixed_since=$(date '+%Y-%m-%d %H:%M:%S')
echo "phase=fixed begin=$(date --iso-8601=seconds)" >> "${RUN_LOG}"
env HIP_VISIBLE_DEVICES="${TARGET_HIP_DEVICE}" "${FIXED_EXE}" >> "${RUN_LOG}" 2>&1
fixed_exit=$?
sleep "${POST_WAIT_SECONDS}"
sample_health fixed-post
fixed_health="${LAST_HEALTHY}"
collect_dmesg fixed "${fixed_since}"
fixed_dmesg_exit="${LAST_DMESG_EXIT}"
fixed_alert_count="${LAST_ALERT_COUNT}"
fixed_pass_count=$(grep -c '^\[FIXED\].*verdict=PASS$' "${RUN_LOG}")

if [ "${fixed_exit}" -eq 0 ] && [ "${fixed_pass_count}" -eq 1 ] && \
    [ "${fixed_health}" -eq 1 ] && [ "${fixed_dmesg_exit}" -eq 0 ] && \
    [ "${fixed_alert_count}" -eq 0 ]; then
    fixed_verdict=FIXED_STABLE
else
    fixed_verdict=FIXED_FAILED_OR_INCONCLUSIVE
fi
log "fixed_exit=${fixed_exit} fixed_health=${fixed_health} fixed_dmesg_exit=${fixed_dmesg_exit} fixed_alert_count=${fixed_alert_count} fixed_verdict=${fixed_verdict}"

if [ "${fixed_verdict}" != "FIXED_STABLE" ]; then
    log "dangerous_skipped=fixed_gate"
    log "overall_verdict=${fixed_verdict}"
    exit 1
fi

if [ "${RUN_DANGEROUS:-}" != "I_UNDERSTAND" ]; then
    log "dangerous_skipped=confirmation_required set_RUN_DANGEROUS=I_UNDERSTAND"
    log "overall_verdict=FIXED_STABLE_DANGEROUS_NOT_RUN"
    exit 0
fi

sample_health dangerous-preflight
dangerous_preflight_health="${LAST_HEALTHY}"
if [ "${dangerous_preflight_health}" -ne 1 ]; then
    log "dangerous_skipped=device_not_healthy"
    log "overall_verdict=FIXED_STABLE_DANGEROUS_NOT_TESTED"
    exit 125
fi

dangerous_since=$(date '+%Y-%m-%d %H:%M:%S')
echo "phase=dangerous begin=$(date --iso-8601=seconds)" >> "${RUN_LOG}"
timeout --signal=TERM --kill-after=5s "$((HOLD_SECONDS + 15))s" \
    env DCU_ALLOW_HOST_ATOMIC_DROP=I_UNDERSTAND \
    HIP_VISIBLE_DEVICES="${TARGET_HIP_DEVICE}" \
    "${REPRO_EXE}" "${HOLD_SECONDS}" >> "${RUN_LOG}" 2>&1
dangerous_exit=$?

sleep "${POST_WAIT_SECONDS}"
sample_health dangerous-post
dangerous_health="${LAST_HEALTHY}"
collect_dmesg dangerous "${dangerous_since}"
dangerous_dmesg_exit="${LAST_DMESG_EXIT}"
dangerous_alert_count="${LAST_ALERT_COUNT}"
dangerous_ur_count="${LAST_UR_COUNT}"
dangerous_gcea_count="${LAST_GCEA_COUNT}"
rejected_count=$(grep -c '^\[REPRO\] verdict=ATOMIC_REJECTED_WITHOUT_DEVICE_DROP$' "${RUN_LOG}")
committed_count=$(grep -c '^\[REPRO\] verdict=ATOMIC_COMMITTED_NO_DEVICE_DROP$' "${RUN_LOG}")

if [ "${dangerous_dmesg_exit}" -ne 0 ]; then
    dangerous_verdict=INCONCLUSIVE_DMESG_UNAVAILABLE
elif [ "${dangerous_ur_count}" -gt 0 ] && \
    { [ "${dangerous_exit}" -eq 137 ] || [ "${dangerous_health}" -eq 0 ]; }; then
    dangerous_verdict=FULL_DEVICE_DROP_REPRO
elif [ "${dangerous_exit}" -eq 5 ] && [ "${rejected_count}" -eq 1 ] && \
    [ "${dangerous_health}" -eq 1 ] && [ "${dangerous_ur_count}" -eq 0 ]; then
    dangerous_verdict=ATOMIC_REJECTED_WITHOUT_DEVICE_DROP
elif [ "${dangerous_exit}" -eq 0 ] && [ "${committed_count}" -eq 1 ] && \
    [ "${dangerous_health}" -eq 1 ]; then
    dangerous_verdict=ATOMIC_COMMITTED_NO_DEVICE_DROP
else
    dangerous_verdict=INCONCLUSIVE
fi

log "dangerous_exit=${dangerous_exit} dangerous_health=${dangerous_health} dangerous_dmesg_exit=${dangerous_dmesg_exit} dangerous_alert_count=${dangerous_alert_count} ur_atomic_count=${dangerous_ur_count} gcea_count=${dangerous_gcea_count} dangerous_verdict=${dangerous_verdict}"
log "overall_verdict=fixed:${fixed_verdict},dangerous:${dangerous_verdict}"

if [ "${dangerous_verdict}" = "INCONCLUSIVE" ] || \
    [ "${dangerous_verdict}" = "INCONCLUSIVE_DMESG_UNAVAILABLE" ]; then
    exit 1
fi
exit 0
