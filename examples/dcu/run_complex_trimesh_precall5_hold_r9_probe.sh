#!/bin/bash

ROOT=/public/home/tangwsh/PhysX
LOG_DIR=/public/home/tangwsh/PhysX/examples/dcu/build
STAMP=$(date +%Y%m%d_%H%M%S)
EXPERIMENT=complex_trimesh_precall5_hold_r9_${STAMP}
BUILD_LOG=${LOG_DIR}/${EXPERIMENT}_build.log
RUN_LOG=${LOG_DIR}/${EXPERIMENT}_run.log
CPP=${ROOT}/physx/source/gpunarrowphase/src/PxgNarrowphaseCore.cpp
THREAD_CPP=${ROOT}/physx/source/foundation/unix/FdUnixThread.cpp
CU=${ROOT}/physx/source/gpunarrowphase/src/CUDA/convexMeshPostProcess.cu
HSACO=${ROOT}/build_dcu/kernels/convexMeshPostProcess.hsaco
EXE=${ROOT}/examples/dcu/build/bench_complex_trimesh_smoke
PCI_FUNCTIONS="0000:05:00.0 0000:56:00.0 0000:5d:00.0 0000:9f:00.0 0000:b1:00.0 0000:c1:00.0 0000:ca:00.0 0000:e8:00.0"
DMESG_PATTERN='Out of memory|Killed process|oom-kill|oom_reaper|Memory cgroup out of memory|VMFault|page fault|atomic err|UR_ATOMIC_OPCODE|GPU reset|ring.*timeout|device.*lost|PCIe error|pcie.*error|hycu.*error'

mkdir -p "${LOG_DIR}"
mkdir_exit=$?
: > "${BUILD_LOG}"
build_log_init_exit=$?
: > "${RUN_LOG}"
run_log_init_exit=$?

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
	local smi_output
	local smi_exit
	local hcu_count
	local smi_alert_count
	local pci_missing
	local healthy
	local app_alive_after

	smi_output=$(timeout 5s hy-smi 2>&1)
	smi_exit=$?
	hcu_count=$(printf '%s\n' "${smi_output}" | awk '/^[[:space:]]*[0-9]+[[:space:]]/{count++} END{print count+0}')
	smi_alert_count=$(printf '%s\n' "${smi_output}" | grep -Eic 'Open mkfd failed|No device available|initialization failed|[[:space:]]N/A[[:space:]]')
	pci_missing=$(count_pci_missing)
	if [ ${smi_exit} -eq 0 ] && [ ${hcu_count} -eq 8 ] && [ ${smi_alert_count} -eq 0 ] && [ ${pci_missing} -eq 0 ]; then
		healthy=1
	else
		healthy=0
	fi
	if [ "${phase}" = post-exit ]; then
		app_alive_after=0
	elif kill -0 "${app_pid}" 2>/dev/null; then
		app_alive_after=1
	else
		app_alive_after=0
	fi
	echo "health_sample phase=${phase} sample=${sample} smi_exit=${smi_exit} hcu_count=${hcu_count} smi_alert_count=${smi_alert_count} pci_missing=${pci_missing} healthy=${healthy} app_alive_after=${app_alive_after}" >> "${RUN_LOG}"
	if [ ${healthy} -eq 0 ] && ! grep -q '^first_health_failure=' "${RUN_LOG}"; then
		echo "first_health_failure=1 phase=${phase} sample=${sample}" >> "${RUN_LOG}"
		printf '%s\n' "${smi_output}" >> "${RUN_LOG}"
	fi
}

sample_dmesg()
{
	local phase=$1
	local output
	local dmesg_exit
	local alerts
	local alert_grep_exit
	local alert_count
	local line

	output=$(dmesg --since "${SINCE}" 2>&1)
	dmesg_exit=$?
	alerts=$(printf '%s\n' "${output}" | grep -Ei "${DMESG_PATTERN}")
	alert_grep_exit=$?
	alert_count=$(printf '%s\n' "${alerts}" | sed '/^$/d' | wc -l)
	echo "dmesg_sample phase=${phase} dmesg_exit=${dmesg_exit} alert_grep_exit=${alert_grep_exit} alert_count=${alert_count}" >> "${RUN_LOG}"
	if [ -n "${alerts}" ]; then
		while IFS= read -r line; do
			if ! grep -Fq -- "${line}" "${RUN_LOG}"; then
				echo "dmesg_alert phase=${phase} ${line}" >> "${RUN_LOG}"
			fi
		done <<< "${alerts}"
	fi
}

{
	echo "experiment=${EXPERIMENT} mkdir_exit=${mkdir_exit} build_log_init_exit=${build_log_init_exit} run_log_init_exit=${run_log_init_exit}"
	date
	grep -n "PX_DCU_NP_TRIMESH_PREMIDPHASE_PROBE\|PX_DCU_NP_TRIMESH_PREMIDPHASE_HOLD_MS\|DCU NP TRIMESH PREMIDPHASE HOLD" "${CPP}"
	host_probe_exit=$?
	grep -nF 'sleepTime.tv_sec = (ms - remainder) / 1000;' "${THREAD_CPP}"
	host_thread_fix_exit=$?
	cd "${ROOT}/build_cpu"
	cpu_cd_exit=$?
	if [ ${cpu_cd_exit} -eq 0 ]; then cmake --build . -j1; cpu_build_exit=$?; else cpu_build_exit=125; fi
	cd "${ROOT}/examples/dcu/build"
	example_cd_exit=$?
	if [ ${example_cd_exit} -eq 0 ]; then cmake --build . -j1 --target bench_complex_trimesh_smoke; example_build_exit=$?; else example_build_exit=125; fi
	stat -c "cu_mtime=%y path=%n" "${CU}"
	cu_stat_exit=$?
	stat -c "hsaco_mtime=%y path=%n" "${HSACO}"
	hsaco_stat_exit=$?
	if [ ${cu_stat_exit} -eq 0 ] && [ ${hsaco_stat_exit} -eq 0 ] && [ "${HSACO}" -nt "${CU}" ]; then hsaco_newer=1; else hsaco_newer=0; fi
	stat -c "cpp_mtime=%y path=%n" "${CPP}"
	cpp_stat_exit=$?
	stat -c "thread_cpp_mtime=%y path=%n" "${THREAD_CPP}"
	thread_cpp_stat_exit=$?
	stat -c "exe_mtime=%y path=%n" "${EXE}"
	exe_stat_exit=$?
	if [ ${cpp_stat_exit} -eq 0 ] && [ ${thread_cpp_stat_exit} -eq 0 ] && [ ${exe_stat_exit} -eq 0 ] && \
		[ "${EXE}" -nt "${CPP}" ] && [ "${EXE}" -nt "${THREAD_CPP}" ]; then executable_newer=1; else executable_newer=0; fi
	if [ ${host_probe_exit} -eq 0 ] && [ ${host_thread_fix_exit} -eq 0 ] && [ ${cpu_build_exit} -eq 0 ] && [ ${example_build_exit} -eq 0 ] && \
		[ ${hsaco_newer} -eq 1 ] && [ ${executable_newer} -eq 1 ]; then build_gate=1; else build_gate=0; fi
	echo "host_probe_exit=${host_probe_exit} host_thread_fix_exit=${host_thread_fix_exit} cpu_cd_exit=${cpu_cd_exit} cpu_build_exit=${cpu_build_exit} example_cd_exit=${example_cd_exit} example_build_exit=${example_build_exit}"
	echo "cu_stat_exit=${cu_stat_exit} hsaco_stat_exit=${hsaco_stat_exit} hsaco_newer=${hsaco_newer} cpp_stat_exit=${cpp_stat_exit} thread_cpp_stat_exit=${thread_cpp_stat_exit} exe_stat_exit=${exe_stat_exit} executable_newer=${executable_newer} build_gate=${build_gate}"
} >> "${BUILD_LOG}" 2>&1
build_block_exit=$?
echo "build_block_exit=${build_block_exit}" >> "${BUILD_LOG}"

echo "experiment=${EXPERIMENT} build_gate=${build_gate}" >> "${RUN_LOG}"
preflight_smi=$(timeout 15s hy-smi 2>&1)
hy_smi_pre_exit=$?
printf '%s\n' "${preflight_smi}" >> "${RUN_LOG}"
preflight_hcu_count=$(printf '%s\n' "${preflight_smi}" | awk '/^[[:space:]]*[0-9]+[[:space:]]/{count++} END{print count+0}')
preflight_smi_alert_count=$(printf '%s\n' "${preflight_smi}" | grep -Eic 'Open mkfd failed|No device available|initialization failed|[[:space:]]N/A[[:space:]]')
preflight_pci_missing=$(count_pci_missing)
if [ ${hy_smi_pre_exit} -eq 0 ] && [ ${preflight_hcu_count} -eq 8 ] && [ ${preflight_smi_alert_count} -eq 0 ] && [ ${preflight_pci_missing} -eq 0 ]; then preflight_gate=1; else preflight_gate=0; fi
echo "hy_smi_pre_exit=${hy_smi_pre_exit} preflight_hcu_count=${preflight_hcu_count} preflight_smi_alert_count=${preflight_smi_alert_count} preflight_pci_missing=${preflight_pci_missing} preflight_gate=${preflight_gate}" >> "${RUN_LOG}"

run_exit=125
monitor_wait_exit=125
watchdog_wait_exit=125
watchdog_fired=0
SINCE=$(date '+%Y-%m-%d %H:%M:%S')
echo "dmesg_since=${SINCE}" >> "${RUN_LOG}"

if [ ${build_gate} -eq 1 ] && [ ${preflight_gate} -eq 1 ]; then
	cd "${LOG_DIR}"
	run_cd_exit=$?
	if [ ${run_cd_exit} -eq 0 ]; then
		env \
			HIP_VISIBLE_DEVICES=${TARGET_HIP_DEVICE:-7} \
			PX_DCU_NP_TRIMESH_PREMIDPHASE_PROBE=5 \
			PX_DCU_NP_TRIMESH_PREMIDPHASE_HOLD_MS=90000 \
			"${EXE}" 45 --bodies 8 --grid 48 --start-height 3.0 >> "${RUN_LOG}" 2>&1 &
	else
		( exit 125 ) &
	fi
	app_pid=$!
	echo "run_cd_exit=${run_cd_exit} app_pid=${app_pid} app_start_exit=0" >> "${RUN_LOG}"
	(
		sleep 150
		if kill -0 "${app_pid}" 2>/dev/null; then
			echo "watchdog_fired=1 app_pid=${app_pid}" >> "${RUN_LOG}"
			kill -TERM "${app_pid}" 2>/dev/null
			sleep 5
			kill -KILL "${app_pid}" 2>/dev/null
		fi
	) &
	watchdog_pid=$!
	(
		sample=0
		while kill -0 "${app_pid}" 2>/dev/null; do
			sample=$((sample + 1))
			if grep -q '\[DCU NP TRIMESH PREMIDPHASE HOLD\] phase=pre-exit' "${RUN_LOG}"; then phase=pre-exit
			elif grep -q '\[DCU NP TRIMESH PREMIDPHASE HOLD\] phase=begin' "${RUN_LOG}"; then phase=hold
			else phase=startup; fi
			sample_health "${phase}" "${sample}"
			if [ $((sample % 5)) -eq 0 ]; then sample_dmesg "${phase}"; fi
			sleep 2
		done
	) &
	monitor_pid=$!
	wait "${app_pid}"
	run_exit=$?
	wait "${monitor_pid}"
	monitor_wait_exit=$?
	kill "${watchdog_pid}" 2>/dev/null
	wait "${watchdog_pid}" 2>/dev/null
	watchdog_wait_exit=$?
	if grep -q '^watchdog_fired=1 ' "${RUN_LOG}"; then watchdog_fired=1; fi
	echo "run_exit=${run_exit} expected_run_exit=96 monitor_wait_exit=${monitor_wait_exit} watchdog_wait_exit=${watchdog_wait_exit} watchdog_fired=${watchdog_fired}" >> "${RUN_LOG}"
	post_sample=0
	while [ ${post_sample} -lt 15 ]; do
		post_sample=$((post_sample + 1))
		sample_health post-exit "${post_sample}"
		if [ $((post_sample % 5)) -eq 0 ]; then sample_dmesg post-exit; fi
		sleep 2
	done
else
	echo "run_skipped=gate build_gate=${build_gate} preflight_gate=${preflight_gate}" >> "${RUN_LOG}"
fi

grep -Eq '\[DCU NP TRIMESH PREMIDPHASE HOLD\] phase=ready launch=5 .*zero=1' "${RUN_LOG}"
hold_ready_exit=$?
grep -q '\[DCU NP TRIMESH PREMIDPHASE HOLD\] phase=begin ms=90000' "${RUN_LOG}"
hold_begin_exit=$?
grep -Eq '\[DCU NP TRIMESH PREMIDPHASE HOLD\] phase=pre-exit ms=10000 actualHoldMs=[0-9]+' "${RUN_LOG}"
hold_pre_exit_marker_exit=$?
grep -Eq '\[DCU NP TRIMESH PREMIDPHASE HOLD\] phase=end requestedTotalMs=100000 actualTotalMs=[0-9]+' "${RUN_LOG}"
hold_end_exit=$?
actual_hold_ms=$(sed -n 's/.*phase=pre-exit ms=10000 actualHoldMs=\([0-9][0-9]*\).*/\1/p' "${RUN_LOG}" | tail -n 1)
actual_total_ms=$(sed -n 's/.*phase=end requestedTotalMs=100000 actualTotalMs=\([0-9][0-9]*\).*/\1/p' "${RUN_LOG}" | tail -n 1)
if [ -n "${actual_hold_ms}" ] && [ ${actual_hold_ms} -ge 90000 ] && [ ${actual_hold_ms} -le 95000 ]; then hold_timing_valid=1; else hold_timing_valid=0; fi
if [ -n "${actual_total_ms}" ] && [ ${actual_total_ms} -ge 100000 ] && [ ${actual_total_ms} -le 105000 ]; then total_timing_valid=1; else total_timing_valid=0; fi
if grep -q '\[DCU NP TRIMESH PRECORE HOLD\]\|\[DCU NP TRIMESH ORDERED PREFIX\]\|\[DCU NP TRIMESH TINYSORT PACKED\]' "${RUN_LOG}"; then later_marker_absent=0; else later_marker_absent=1; fi

alive_health_failures=$(awk '/^health_sample / && /phase=(startup|hold|pre-exit)/ && /healthy=0/ && /app_alive_after=1/{count++} END{print count+0}' "${RUN_LOG}")
hold_healthy_samples=$(awk '/^health_sample / && /phase=hold/ && /healthy=1/ && /app_alive_after=1/{count++} END{print count+0}' "${RUN_LOG}")
pre_exit_healthy_samples=$(awk '/^health_sample / && /phase=pre-exit/ && /healthy=1/ && /app_alive_after=1/{count++} END{print count+0}' "${RUN_LOG}")
post_health_failures=$(awk '/^health_sample / && /phase=post-exit/ && /healthy=0/{count++} END{print count+0}' "${RUN_LOG}")
dmesg_alert_count=$(grep -c '^dmesg_alert ' "${RUN_LOG}")

if [ ${run_exit} -eq 96 ] && [ ${hold_ready_exit} -eq 0 ] && [ ${hold_begin_exit} -eq 0 ] && \
	[ ${hold_pre_exit_marker_exit} -eq 0 ] && [ ${hold_end_exit} -eq 0 ] && [ ${later_marker_absent} -eq 1 ] && \
	[ ${hold_timing_valid} -eq 1 ] && [ ${total_timing_valid} -eq 1 ] && \
	[ ${alive_health_failures} -eq 0 ] && [ ${hold_healthy_samples} -gt 0 ] && [ ${pre_exit_healthy_samples} -gt 0 ] && \
	[ ${post_health_failures} -eq 0 ] && [ ${watchdog_fired} -eq 0 ]; then
	probe_exit=0
	hypothesis_result=first-four-trimesh-calls-stable-trigger-requires-call-5-through-8
elif [ ${alive_health_failures} -gt 0 -o ${post_health_failures} -gt 0 ]; then
	probe_exit=1
	hypothesis_result=trimesh-call-5-or-earlier-introduces-device-failure
elif [ ${run_exit} -eq 137 ] && [ ${alive_health_failures} -eq 0 ] && [ ${post_health_failures} -eq 0 ]; then
	probe_exit=1
	hypothesis_result=external-sigkill-without-device-failure
elif [ ${run_exit} -eq 97 ]; then
	probe_exit=1
	hypothesis_result=premidphase-counter-initialization-invalid
elif [ ${watchdog_fired} -eq 1 ]; then
	probe_exit=1
	hypothesis_result=probe-timeout
else
	probe_exit=1
	hypothesis_result=incomplete
fi

if [ ${probe_exit} -eq 0 ]; then experiment_exit=0; else experiment_exit=1; fi
echo "hold_ready_exit=${hold_ready_exit} hold_begin_exit=${hold_begin_exit} hold_pre_exit_marker_exit=${hold_pre_exit_marker_exit} hold_end_exit=${hold_end_exit} later_marker_absent=${later_marker_absent}" >> "${RUN_LOG}"
echo "actual_hold_ms=${actual_hold_ms:-missing} hold_timing_valid=${hold_timing_valid} actual_total_ms=${actual_total_ms:-missing} total_timing_valid=${total_timing_valid}" >> "${RUN_LOG}"
echo "alive_health_failures=${alive_health_failures} hold_healthy_samples=${hold_healthy_samples} pre_exit_healthy_samples=${pre_exit_healthy_samples} post_health_failures=${post_health_failures} dmesg_alert_count=${dmesg_alert_count}" >> "${RUN_LOG}"
echo "probe_exit=${probe_exit} hypothesis_result=${hypothesis_result} experiment_exit=${experiment_exit}" >> "${RUN_LOG}"
echo "build_log=${BUILD_LOG}"
echo "run_log=${RUN_LOG}"
echo "experiment_exit=${experiment_exit}"
exit ${experiment_exit}
