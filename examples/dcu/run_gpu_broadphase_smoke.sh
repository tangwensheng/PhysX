#!/usr/bin/env bash
# Fingerprinted, fail-fast runner for bench_gpu_broadphase_smoke.

set -u

ROOT=/public/home/tangwsh/PhysX
LOG_DIR=${ROOT}/examples/dcu
EXE=${LOG_DIR}/build/bench_gpu_broadphase_smoke
RUNNER=${LOG_DIR}/run_gpu_broadphase_smoke.sh
BASELINE=${LOG_DIR}/gpu_broadphase_smoke_baseline.md5
GPU_LIB=${ROOT}/build_dcu/libPhysXGpuDCU.so
BP_HSACO=${ROOT}/build_dcu/kernels/broadphase.hsaco
AGGREGATE_HSACO=${ROOT}/build_dcu/kernels/aggregate.hsaco
RADIX_HSACO=${ROOT}/build_dcu/kernels/radixSortImpl.hsaco
BP_HEADER=${ROOT}/physx/source/gpubroadphase/include/PxgBroadPhaseKernelIndices.h
BP_SOURCE=${ROOT}/physx/source/gpubroadphase/src/CUDA/broadphase.cu
AGGREGATE_SOURCE=${ROOT}/physx/source/gpubroadphase/src/CUDA/aggregate.cu
RADIX_SOURCE=${ROOT}/physx/source/gpucommon/src/CUDA/radixSortImpl.cu
RADIX_HEADER=${ROOT}/physx/source/gpucommon/include/PxgRadixSortKernelIndices.h
KERNEL_NAMES_SOURCE=${ROOT}/physx/source/gpucommon/include/PxgKernelNames.h
BP_HOST_SOURCE=${ROOT}/physx/source/gpubroadphase/src/PxgCudaBroadPhaseSap.cpp
BP_MANAGER_SOURCE=${ROOT}/physx/source/gpubroadphase/src/PxgBroadPhase.cpp
BP_AABB_SOURCE=${ROOT}/physx/source/gpubroadphase/src/PxgAABBManager.cpp
BENCH_SOURCE=${LOG_DIR}/bench_gpu_broadphase_smoke.cpp
CMAKE_SOURCE=${LOG_DIR}/CMakeLists.txt

PIN=${1:-}
DEVICE=${TARGET_HIP_DEVICE:-1}
EXPECTED_HCU_COUNT=${EXPECTED_HCU_COUNT:-8}
REPEAT=${REPEAT:-1}
REPEAT_DELAY=${REPEAT_DELAY:-5}
POST_HEALTH_SAMPLES=${POST_HEALTH_SAMPLES:-5}
HEALTH_INTERVAL=${HEALTH_INTERVAL:-2}
TIMEOUT_SECONDS=${TIMEOUT_SECONDS:-180}
BODIES=${BODIES:-512}
CYCLES=${CYCLES:-8}
TRANSITION_STEPS=${TRANSITION_STEPS:-2}
SETTLE_STEPS=${SETTLE_STEPS:-60}
MARKER=PX_DCU_GPU_BROADPHASE_SMOKE_V1_PAIR_CHURN
REPORT_MARKER=PX_DCU_GPU_BROADPHASE_REPORT_V4_DEVICE_TO_HOST_READBACK
REPORT_STAGE_MARKER=PX_DCU_GPU_BROADPHASE_REPORT_V5_SERIAL_FULL_COMPACTION
RADIX_MARKER=PX_DCU_GPU_BROADPHASE_RADIX_V3_SERIAL_COUNTING_PASS
RADIX_KERNEL=radixSortBroadPhaseSerialCountingPassDcuV3

FINGERPRINT_SRC="
${BP_HEADER}
${BP_SOURCE}
${AGGREGATE_SOURCE}
${RADIX_SOURCE}
${RADIX_HEADER}
${KERNEL_NAMES_SOURCE}
${BP_HOST_SOURCE}
${BP_MANAGER_SOURCE}
${BP_AABB_SOURCE}
${BENCH_SOURCE}
${CMAKE_SOURCE}
${RUNNER}
"
FINGERPRINT_BIN="
${EXE}
${BP_HSACO}
${AGGREGATE_HSACO}
${RADIX_HSACO}
${GPU_LIB}
"

compute_fingerprint()
{
    local file
    for file in ${FINGERPRINT_SRC} ${FINGERPRINT_BIN}; do
        if [ -e "${file}" ]; then
            md5sum "${file}" 2>/dev/null
        else
            echo "MISSING ${file}"
        fi
    done
}

check_build_input()
{
    local source=$1
    local output=$2
    local label=$3
    if [ ! -e "${output}" ]; then
        echo "STALE: ${label} output is missing: ${output}"
        stale=$((stale + 1))
    elif [ -e "${source}" ] && [ "${source}" -nt "${output}" ]; then
        echo "STALE: ${source} is newer than ${label} output ${output}"
        stale=$((stale + 1))
    fi
}

check_device_health()
{
    local output hcu_count alerts
    output=$(timeout 15s hy-smi 2>&1)
    hcu_count=$(printf '%s\n' "${output}" | awk '/^[[:space:]]*[0-9]+[[:space:]]/{count++} END{print count+0}')
    alerts=$(printf '%s\n' "${output}" | grep -Eic 'Open mkfd failed|No device available|initialization failed|[[:space:]]N/A[[:space:]]')
    if [ "${hcu_count}" -eq "${EXPECTED_HCU_COUNT}" ] && [ "${alerts}" -eq 0 ]; then
        health_gate=1
    else
        health_gate=0
    fi
    echo "health hcu_count=${hcu_count} expected=${EXPECTED_HCU_COUNT} alerts=${alerts} healthy=${health_gate}"
}

if [ "${PIN}" = "--pin" ]; then
    candidate=${BASELINE}.candidate.$$
    verify=${BASELINE}.verify.$$
    compute_fingerprint > "${candidate}"
    if grep -q '^MISSING ' "${candidate}"; then
        echo "pin rejected: fingerprint input is missing" >&2
        cat "${candidate}" >&2
        rm -f "${candidate}"
        exit 3
    fi
    mv "${candidate}" "${BASELINE}"
    compute_fingerprint > "${verify}"
    if diff -q "${BASELINE}" "${verify}" >/dev/null 2>&1; then
        pin_verify=1
    else
        pin_verify=0
        echo "--- pin verification DRIFT ---" >&2
        diff "${BASELINE}" "${verify}" >&2
    fi
    rm -f "${verify}"
    echo "baseline pinned -> ${BASELINE}"
    cat "${BASELINE}"
    echo "pin_verify=${pin_verify}"
    if [ "${pin_verify}" -ne 1 ]; then
        exit 3
    fi
    exit 0
fi

if [ -n "${PIN}" ]; then
    echo "usage: $0 [--pin]" >&2
    exit 2
fi

if [ ! -f "${BASELINE}" ]; then
    echo "no baseline fingerprint; build, verify, then run $0 --pin" >&2
    exit 2
fi

run_once()
{
    local iteration=$1
    local stamp run_log now_md5
    stamp=$(date +%Y%m%d_%H%M%S)
    run_log=${LOG_DIR}/gpu_broadphase_smoke_${stamp}_run${iteration}.log
    now_md5=/tmp/gpu_broadphase_smoke_now_$$.md5

    {
        echo "experiment=gfx936_forced_gpu_broadphase iteration=${iteration} device=${DEVICE} bodies=${BODIES} cycles=${CYCLES} transition_steps=${TRANSITION_STEPS} settle_steps=${SETTLE_STEPS}"
        date

        compute_fingerprint > "${now_md5}"
        cat "${now_md5}"
        if diff -q "${BASELINE}" "${now_md5}" >/dev/null 2>&1; then
            fingerprint_gate=1
        else
            fingerprint_gate=0
            echo "--- fingerprint DRIFT (expected vs actual) ---"
            diff "${BASELINE}" "${now_md5}"
        fi
        rm -f "${now_md5}"
        echo "fingerprint_gate=${fingerprint_gate}"

        stale=0
        check_build_input "${BENCH_SOURCE}" "${EXE}" "executable"
        check_build_input "${CMAKE_SOURCE}" "${EXE}" "executable"
        check_build_input "${BP_HOST_SOURCE}" "${EXE}" "executable"
        check_build_input "${BP_MANAGER_SOURCE}" "${EXE}" "executable"
        check_build_input "${BP_AABB_SOURCE}" "${EXE}" "executable"
        check_build_input "${BP_HEADER}" "${EXE}" "executable"
        check_build_input "${RADIX_HEADER}" "${EXE}" "executable"
        check_build_input "${KERNEL_NAMES_SOURCE}" "${EXE}" "executable"
        check_build_input "${BP_HEADER}" "${BP_HSACO}" "broadphase hsaco"
        check_build_input "${BP_HEADER}" "${AGGREGATE_HSACO}" "aggregate hsaco"
        check_build_input "${BP_SOURCE}" "${BP_HSACO}" "broadphase hsaco"
        check_build_input "${AGGREGATE_SOURCE}" "${AGGREGATE_HSACO}" "aggregate hsaco"
        check_build_input "${RADIX_SOURCE}" "${RADIX_HSACO}" "radixSortImpl hsaco"
        check_build_input "${RADIX_HEADER}" "${RADIX_HSACO}" "radixSortImpl hsaco"
        if [ "${stale}" -eq 0 ]; then build_gate=1; else build_gate=0; fi
        echo "build_gate=${build_gate} stale_sources=${stale}"

        if strings "${EXE}" 2>/dev/null | grep -Fq "${MARKER}" \
            && strings "${EXE}" 2>/dev/null | grep -Fq "${REPORT_MARKER}" \
            && strings "${EXE}" 2>/dev/null | grep -Fq "${RADIX_KERNEL}"; then
            host_marker_gate=1
        else
            host_marker_gate=0
            echo "HOST MARKER MISSING: ${MARKER}, ${REPORT_MARKER}, or V3 DCU broadphase radix kernel"
        fi
        if grep -Eq 'BP_COMPUTE_ACTIVE_HISTOGRAM[[:space:]]*=[[:space:]]*256' "${BP_HEADER}" \
            && strings "${BP_HSACO}" 2>/dev/null | grep -Fq 'computeStartAndActiveRegionHistogram' \
            && strings "${BP_HSACO}" 2>/dev/null | grep -Fq "${REPORT_STAGE_MARKER}" \
            && strings "${RADIX_HSACO}" 2>/dev/null | grep -Fq "${RADIX_MARKER}" \
            && strings "${RADIX_HSACO}" 2>/dev/null | grep -Fq "${RADIX_KERNEL}"; then
            device_marker_gate=1
        else
            device_marker_gate=0
            echo "DEVICE MARKER MISSING: expected V5 report-stage marker, active histogram, and V3 DCU broadphase radix marker/kernel"
        fi
        if [ "${host_marker_gate}" -eq 1 ] && [ "${device_marker_gate}" -eq 1 ]; then
            marker_gate=1
        else
            marker_gate=0
        fi
        echo "host_marker_gate=${host_marker_gate} device_marker_gate=${device_marker_gate} marker_gate=${marker_gate}"

        check_device_health
        device_gate=${health_gate}
        echo "device_gate=${device_gate}"

        if [ "${fingerprint_gate}" -eq 1 ] && [ "${build_gate}" -eq 1 ] && \
           [ "${marker_gate}" -eq 1 ] && [ "${device_gate}" -eq 1 ]; then
            echo "--- run ---"
            timeout --signal=TERM --kill-after=5s "${TIMEOUT_SECONDS}s" \
                env HIP_VISIBLE_DEVICES="${DEVICE}" \
                "${EXE}" \
                --bodies "${BODIES}" \
                --cycles "${CYCLES}" \
                --transition-steps "${TRANSITION_STEPS}" \
                --settle-steps "${SETTLE_STEPS}"
            run_exit=$?
            echo "run_exit=${run_exit} expected=0"

            post_failures=0
            for sample in $(seq 1 "${POST_HEALTH_SAMPLES}"); do
                check_device_health
                echo "health phase=post sample=${sample} healthy=${health_gate}"
                if [ "${health_gate}" -ne 1 ]; then
                    post_failures=$((post_failures + 1))
                fi
                sleep "${HEALTH_INTERVAL}"
            done
            echo "post_health_failures=${post_failures}"

            if [ "${run_exit}" -eq 0 ] && [ "${post_failures}" -eq 0 ]; then
                verdict=STABLE
            elif [ "${post_failures}" -gt 0 ]; then
                verdict=DEVICE_DROPPED
            else
                verdict=FAILED
            fi
        else
            echo "run_skipped fingerprint_gate=${fingerprint_gate} build_gate=${build_gate} marker_gate=${marker_gate} device_gate=${device_gate}"
            verdict=NOT_TESTED
        fi

        echo "iteration=${iteration} verdict=${verdict}"
    } 2>&1 | tee "${run_log}"

    echo "log: ${run_log}"
    grep -q 'verdict=STABLE$' "${run_log}"
}

iteration=1
while [ "${iteration}" -le "${REPEAT}" ]; do
    echo "########## forced GPU broadphase run ${iteration}/${REPEAT} ##########"
    if ! run_once "${iteration}"; then
        echo "batch_stop iteration=${iteration} reason=non_stable"
        exit 1
    fi
    if [ "${iteration}" -lt "${REPEAT}" ]; then
        sleep "${REPEAT_DELAY}"
    fi
    iteration=$((iteration + 1))
done

exit 0
