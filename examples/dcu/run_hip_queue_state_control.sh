#!/bin/bash

ROOT=/public/home/tangwsh/PhysX
SOURCE=${ROOT}/examples/dcu/hip_queue_state_reproducer.hip
BUILD_DIR=${ROOT}/examples/dcu/build
EXE=${BUILD_DIR}/hip_queue_state_reproducer
HIPCC=/opt/dtk/bin/hipcc
BUILD_MARKER=HIP_QUEUE_STATE_CONTROL_V2_PAIR56_DIRECT_HIPCC
RUNNER_MARKER=HIP_QUEUE_STATE_CONTROL_RUNNER_V2_IDLEFIX
STAMP=$(date +%Y%m%d_%H%M%S)
BUILD_LOG=${BUILD_DIR}/hip_queue_state_control_${STAMP}_build.log
RUN_LOG=${BUILD_DIR}/hip_queue_state_control_${STAMP}_run.log
PCI_FUNCTIONS="0000:05:00.0 0000:56:00.0 0000:5d:00.0 0000:9f:00.0 0000:b1:00.0 0000:c1:00.0 0000:ca:00.0 0000:e8:00.0"
DMESG_PATTERN='Out of memory|Killed process|oom-kill|oom_reaper|Memory cgroup out of memory|VMFault|page fault|atomic err|UR_ATOMIC_OPCODE|GPU reset|ring.*timeout|device.*lost|PCIe error|pcie.*error|hycu.*error'

mkdir -p "${BUILD_DIR}"
: > "${BUILD_LOG}"
: > "${RUN_LOG}"

count_pci_missing()
{
    local missing=0
    local device
    for device in ${PCI_FUNCTIONS}; do
        if [ ! -e "/sys/bus/pci/devices/${device}" ]; then
            missing=$((missing + 1))
        fi
    done
    echo "${missing}"
}

sample_health()
{
    local phase=$1
    local sample=$2
    local output
    local command_exit
    local hcu_count
    local alert_count
    local target_line
    local target_vram
    local target_hcu
    local pci_missing
    local healthy

    output=$(timeout 8s hy-smi 2>&1)
    command_exit=$?
    hcu_count=$(printf '%s\n' "${output}" | awk '/^[[:space:]]*[0-9]+[[:space:]]/{count++} END{print count+0}')
    alert_count=$(printf '%s\n' "${output}" | grep -Eic 'Open mkfd failed|No device available|initialization failed|[[:space:]]N/A[[:space:]]')
    target_line=$(printf '%s\n' "${output}" | awk '$1 == 7 {print; exit}')
    target_vram=$(printf '%s\n' "${target_line}" | awk '{value=$6; gsub(/%/, "", value); print value}')
    target_hcu=$(printf '%s\n' "${target_line}" | awk '{value=$7; gsub(/%/, "", value); print value}')
    pci_missing=$(count_pci_missing)

    if [ ${command_exit} -eq 0 ] && [ "${hcu_count}" -eq 8 ] && [ "${alert_count}" -eq 0 ] && \
        [ -n "${target_vram}" ] && [ -n "${target_hcu}" ] && [ "${pci_missing}" -eq 0 ]; then
        healthy=1
    else
        healthy=0
    fi

    echo "health phase=${phase} sample=${sample} hy_smi_exit=${command_exit} hcu_count=${hcu_count} alert_count=${alert_count} target_index=7 target_vram=${target_vram:-unknown} target_hcu=${target_hcu:-unknown} pci_missing=${pci_missing} healthy=${healthy}" >> "${RUN_LOG}"
    if [ "${healthy}" -eq 0 ]; then
        printf '%s\n' "health_raw_begin phase=${phase} sample=${sample}" "${output}" "health_raw_end phase=${phase} sample=${sample}" >> "${RUN_LOG}"
    fi
}

{
    echo "experiment=hip_queue_state_control hypothesis=direct-hip-none-prefix-remains-stable-without-PhysX mode=none stack_mib=256 hold_seconds=40 runner=${RUNNER_MARKER}"
    echo "source=${SOURCE}"
    echo "compiler=${HIPCC} marker=${BUILD_MARKER}"
    date
    source_marker_count=$(grep -Fc "${BUILD_MARKER}" "${SOURCE}")
    memset_call_count=$(grep -Ec '^[[:space:]]*memsetResults\[[0-4]\][[:space:]]*=[[:space:]]*hipMemsetD32Async' "${SOURCE}")
    d2h_call_count=$(grep -Ec 'copyResults\[index\][[:space:]]*=[[:space:]]*hipMemcpyDtoH' "${SOURCE}")
    hip_free_null_count=$(grep -Fc 'hipFree(nullptr)' "${SOURCE}")
    if [ "${source_marker_count}" -eq 1 ] && [ "${memset_call_count}" -eq 5 ] && \
        [ "${d2h_call_count}" -eq 1 ] && [ "${hip_free_null_count}" -eq 0 ]; then
        source_check_exit=0
    else
        source_check_exit=1
    fi
    echo "source_marker_count=${source_marker_count} memset_call_count=${memset_call_count} d2h_loop_call_count=${d2h_call_count} hip_free_null_count=${hip_free_null_count} source_check_exit=${source_check_exit}"
    if [ -x "${HIPCC}" ] && [ ${source_check_exit} -eq 0 ]; then
        "${HIPCC}" -O2 -std=c++17 --offload-arch=gfx936 -pthread "${SOURCE}" -o "${EXE}"
        compile_exit=$?
    else
        compile_exit=125
    fi
    if [ ${compile_exit} -eq 0 ]; then
        strings "${EXE}" | grep -Fq "${BUILD_MARKER}"
        binary_marker_exit=$?
    else
        binary_marker_exit=125
    fi
    if [ ${source_check_exit} -eq 0 ] && [ ${compile_exit} -eq 0 ] && [ ${binary_marker_exit} -eq 0 ]; then
        build_gate=1
    else
        build_gate=0
    fi
    echo "compile_exit=${compile_exit} binary_marker_exit=${binary_marker_exit} build_gate=${build_gate}"
} >> "${BUILD_LOG}" 2>&1

echo "experiment=hip_queue_state_control mode=none stack_mib=256 hold_seconds=40 marker=${BUILD_MARKER} runner=${RUNNER_MARKER} build_gate=${build_gate}" >> "${RUN_LOG}"
sample_health preflight 1
sleep 3
sample_health preflight 2

preflight_healthy=$(awk '/^health phase=preflight / && /healthy=1/{count++} END{print count+0}' "${RUN_LOG}")
preflight_idle=$(awk '
    /^health phase=preflight / {
        vram=""; hcu="";
        for (field=1; field<=NF; ++field) {
            if (index($field, "target_vram=") == 1) {split($field, value, "="); vram=value[2]}
            if (index($field, "target_hcu=") == 1) {split($field, value, "="); hcu=value[2]}
        }
        if (vram != "unknown" && hcu != "unknown" && vram + 0 <= 10 && hcu + 0 <= 10) count++
    }
    END {print count+0}
' "${RUN_LOG}")

if [ "${preflight_healthy}" -eq 2 ] && [ "${preflight_idle}" -eq 2 ]; then
    preflight_gate=1
else
    preflight_gate=0
fi
echo "preflight_healthy_samples=${preflight_healthy} preflight_idle_samples=${preflight_idle} idle_limits_vram_hcu=10 preflight_gate=${preflight_gate}" >> "${RUN_LOG}"

run_exit=125
if [ "${build_gate}" -eq 1 ] && [ "${preflight_gate}" -eq 1 ]; then
    dmesg_since=$(date '+%Y-%m-%d %H:%M:%S')
    echo "dmesg_since=${dmesg_since}" >> "${RUN_LOG}"
    timeout --signal=TERM --kill-after=5s 65s env HIP_VISIBLE_DEVICES=7 \
        "${EXE}" --mode none --stack-mib 256 --hold-seconds 40 >> "${RUN_LOG}" 2>&1 &
    app_pid=$!
    echo "app_pid=${app_pid}" >> "${RUN_LOG}"

    sleep 5
    sample_health hold 1
    sleep 20
    sample_health hold 2

    wait "${app_pid}"
    run_exit=$?
    echo "run_exit=${run_exit}" >> "${RUN_LOG}"
    sample_health post-exit 1

    dmesg_output=$(dmesg --since "${dmesg_since}" 2>&1)
    dmesg_exit=$?
    dmesg_alerts=$(printf '%s\n' "${dmesg_output}" | grep -Ei "${DMESG_PATTERN}")
    dmesg_alert_count=$(printf '%s\n' "${dmesg_alerts}" | sed '/^$/d' | wc -l)
    echo "dmesg_summary exit=${dmesg_exit} alert_count=${dmesg_alert_count}" >> "${RUN_LOG}"
    if [ "${dmesg_exit}" -ne 0 ]; then
        printf '%s\n' "${dmesg_output}" | tail -n 5 | sed 's/^/dmesg_error /' >> "${RUN_LOG}"
    elif [ "${dmesg_alert_count}" -gt 0 ]; then
        printf '%s\n' "${dmesg_alerts}" | tail -n 20 | sed 's/^/dmesg_alert /' >> "${RUN_LOG}"
    fi
else
    echo "run_skipped=gate build_gate=${build_gate} preflight_gate=${preflight_gate}" >> "${RUN_LOG}"
    dmesg_exit=125
    dmesg_alert_count=0
fi

control_marker_count=$(grep -c '^VERDICT=CONTROL_STABLE$' "${RUN_LOG}")
prefix_marker_count=$(grep -Ec '^phase=prefix .*zeros=1$' "${RUN_LOG}")
health_failure_count=$(grep -Ec '^health phase=(hold|post-exit) .*healthy=0$' "${RUN_LOG}")

if [ "${build_gate}" -ne 1 ] || [ "${preflight_gate}" -ne 1 ]; then
    experiment_exit=125
    hypothesis_result=not-tested-gate-rejected
elif [ "${run_exit}" -eq 0 ] && [ "${control_marker_count}" -eq 1 ] && [ "${prefix_marker_count}" -eq 1 ] && \
    [ "${health_failure_count}" -eq 0 ] && [ "${dmesg_exit}" -eq 0 ] && [ "${dmesg_alert_count}" -eq 0 ]; then
    experiment_exit=0
    hypothesis_result=standalone-none-control-stable
elif [ "${health_failure_count}" -gt 0 ] || [ "${dmesg_alert_count}" -gt 0 ]; then
    experiment_exit=1
    hypothesis_result=standalone-none-observed-device-or-driver-failure
elif [ "${dmesg_exit}" -ne 0 ]; then
    experiment_exit=1
    hypothesis_result=standalone-none-dmesg-unavailable
else
    experiment_exit=1
    hypothesis_result=standalone-none-incomplete-or-process-failure
fi

echo "control_marker_count=${control_marker_count} prefix_marker_count=${prefix_marker_count} health_failure_count=${health_failure_count} dmesg_exit=${dmesg_exit} dmesg_alert_count=${dmesg_alert_count}" >> "${RUN_LOG}"
echo "hypothesis_result=${hypothesis_result} experiment_exit=${experiment_exit}" >> "${RUN_LOG}"
echo "build_log=${BUILD_LOG}"
echo "run_log=${RUN_LOG}"
echo "experiment_exit=${experiment_exit}"
exit "${experiment_exit}"
