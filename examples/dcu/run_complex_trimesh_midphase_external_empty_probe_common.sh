#!/bin/bash

ROOT=/public/home/tangwsh/PhysX
LOG_DIR=/public/home/tangwsh/PhysX/examples/dcu/build
STAMP=$(date +%Y%m%d_%H%M%S)
: "${EXPERIMENT_BASE:?EXPERIMENT_BASE must be set by the experiment wrapper}"
: "${UNIT_BLOCK:?UNIT_BLOCK must be set by the experiment wrapper}"
: "${SKIP_DISPATCH:?SKIP_DISPATCH must be set by the experiment wrapper}"
: "${NONNULL_PARAMS:?NONNULL_PARAMS must be set by the experiment wrapper}"
: "${NONNULL_PARAM_ENTRY:?NONNULL_PARAM_ENTRY must be set by the experiment wrapper}"
: "${SCALAR_ARG_KERNEL:=0}"
: "${KNOWN_EXISTING_KERNEL:=0}"
: "${FRESH_STREAM:=0}"
: "${D2D_COMMAND:=0}"
: "${READY_PATTERN:?READY_PATTERN must be set by the experiment wrapper}"
: "${STABLE_RESULT:?STABLE_RESULT must be set by the experiment wrapper}"
: "${DEVICE_FAILURE_RESULT:?DEVICE_FAILURE_RESULT must be set by the experiment wrapper}"
: "${PROBE_ENTRY:?PROBE_ENTRY must be set by the experiment wrapper}"
EXPERIMENT=${EXPERIMENT_BASE}_${STAMP}
BUILD_LOG=${LOG_DIR}/${EXPERIMENT}_build.log
RUN_LOG=${LOG_DIR}/${EXPERIMENT}_run.log
CPP=${ROOT}/physx/source/gpunarrowphase/src/PxgNarrowphaseCore.cpp
NAMES=${ROOT}/physx/source/gpucommon/include/PxgKernelNames.h
CU=${ROOT}/physx/source/gpunarrowphase/src/CUDA/pairManagement.cu
HSACO=${ROOT}/build_dcu/kernels/pairManagement.hsaco
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
		if [ ! -e "/sys/bus/pci/devices/${device}" ]; then missing=$((missing + 1)); fi
	done
	echo "${missing}"
}

summarize_idle()
{
	awk '
		/^[[:space:]]*[0-9]+[[:space:]]/ {
			vram=$6; hcu=$7;
			gsub(/%/, "", vram); gsub(/%/, "", hcu);
			if (vram == "" || hcu == "") { bad++; next; }
			count++;
			if (vram + 0 > max_vram) max_vram=vram + 0;
			if (hcu + 0 > max_hcu) max_hcu=hcu + 0;
		}
		END {
			idle=(count == 8 && bad == 0 && max_vram <= 10 && max_hcu <= 10) ? 1 : 0;
			printf "parsed=%d bad=%d max_vram=%.1f max_hcu=%.1f idle=%d", count + 0, bad + 0, max_vram + 0, max_hcu + 0, idle;
		}
	'
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
	if [ ${smi_exit} -eq 0 ] && [ ${hcu_count} -eq 8 ] && [ ${smi_alert_count} -eq 0 ] && [ ${pci_missing} -eq 0 ]; then healthy=1; else healthy=0; fi
	if [ "${phase}" = post-exit ]; then app_alive_after=0
	elif kill -0 "${app_pid}" 2>/dev/null; then app_alive_after=1
	else app_alive_after=0; fi
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
			if ! grep -Fq -- "${line}" "${RUN_LOG}"; then echo "dmesg_alert phase=${phase} ${line}" >> "${RUN_LOG}"; fi
		done <<< "${alerts}"
	fi
}

{
	echo "probe_entry=${PROBE_ENTRY} experiment_base=${EXPERIMENT_BASE} unit_block=${UNIT_BLOCK} skip_dispatch=${SKIP_DISPATCH} nonnull_params=${NONNULL_PARAMS} nonnull_param_entry=${NONNULL_PARAM_ENTRY} scalar_arg_kernel=${SCALAR_ARG_KERNEL} known_existing_kernel=${KNOWN_EXISTING_KERNEL} fresh_stream=${FRESH_STREAM} d2d_command=${D2D_COMMAND}"
	echo "experiment=${EXPERIMENT} mkdir_exit=${mkdir_exit} build_log_init_exit=${build_log_init_exit} run_log_init_exit=${run_log_init_exit}"
	date
	grep -n "PX_DCU_NP_TRIMESH_MIDPHASE_EXTERNAL_EMPTY_REPLACE\|PX_DCU_NP_TRIMESH_MIDPHASE_EXTERNAL_EMPTY_UNIT_BLOCK\|externalEmptyReplace\|unitBlock\|PX_DCU_NP_TRIMESH_EXTERNAL_HSACO_PROOF" "${CPP}"
	host_probe_exit=$?
	if [ "${UNIT_BLOCK}" -eq 1 ]; then
		grep -Fq 'PX_DCU_NP_TRIMESH_MIDPHASE_EXTERNAL_EMPTY_UNIT_BLOCK' "${CPP}"
		unit_block_env_source_exit=$?
		grep -Fq 'midphaseExternalEmptyUnitBlock ? 1u : WARP_SIZE' "${CPP}"
		unit_block_launch_source_exit=$?
		grep -Fq 'unitBlock=%u grid=(%u,1,1) block=(%u,%u,1) originalBlock=(%u,%u,1)' "${CPP}"
		unit_block_ready_source_exit=$?
	else
		unit_block_env_source_exit=0
		unit_block_launch_source_exit=0
		unit_block_ready_source_exit=0
	fi
	if [ "${SKIP_DISPATCH}" -eq 1 ]; then
		grep -Fq 'PX_DCU_NP_TRIMESH_MIDPHASE_EXTERNAL_EMPTY_SKIP_DISPATCH' "${CPP}"
		skip_dispatch_env_source_exit=$?
		grep -Fq 'if (midphaseExternalEmptySkipDispatch)' "${CPP}"
		skip_dispatch_branch_source_exit=$?
		grep -Fq 'skipDispatch=%u submitted=%u' "${CPP}"
		skip_dispatch_ready_source_exit=$?
	else
		skip_dispatch_env_source_exit=0
		skip_dispatch_branch_source_exit=0
		skip_dispatch_ready_source_exit=0
	fi
	if [ "${NONNULL_PARAMS}" -eq 1 ]; then
		grep -Fq 'PX_DCU_NP_TRIMESH_MIDPHASE_EXTERNAL_EMPTY_NONNULL_PARAMS' "${CPP}"
		nonnull_params_env_source_exit=$?
		grep -Fq 'midphaseExternalEmptyNonNullParams ? nonNullEmptyKernelParams' "${CPP}"
		nonnull_params_launch_source_exit=$?
		grep -Fq 'launchResult=%d sync=%d copy=%d' "${CPP}"
		nonnull_params_launch_result_source_exit=$?
		grep -Fq 'nonNullParams=%u nonNullParamEntry=%u skipDispatch=%u submitted=%u' "${CPP}"
		nonnull_params_ready_source_exit=$?
	else
		nonnull_params_env_source_exit=0
		nonnull_params_launch_source_exit=0
		nonnull_params_launch_result_source_exit=0
		nonnull_params_ready_source_exit=0
	fi
	if [ "${NONNULL_PARAM_ENTRY}" -eq 1 ]; then
		grep -Fq 'PX_DCU_NP_TRIMESH_MIDPHASE_EXTERNAL_EMPTY_NONNULL_PARAM_ENTRY' "${CPP}"
		nonnull_param_entry_env_source_exit=$?
		grep -Fq 'midphaseExternalEmptyNonNullParamEntry ? &externalEmptyDiagnosticValue : NULL' "${CPP}"
		nonnull_param_entry_launch_source_exit=$?
		grep -Fq 'nonNullParamEntry=%u skipDispatch=%u submitted=%u' "${CPP}"
		nonnull_param_entry_ready_source_exit=$?
	else
		nonnull_param_entry_env_source_exit=0
		nonnull_param_entry_launch_source_exit=0
		nonnull_param_entry_ready_source_exit=0
	fi
	if [ "${SCALAR_ARG_KERNEL}" -eq 1 ]; then
		grep -Fq 'PX_DCU_NP_TRIMESH_MIDPHASE_EXTERNAL_SCALAR_ARG_KERNEL' "${CPP}"
		scalar_arg_env_source_exit=$?
		grep -Fq 'PxgKernelIds::CONVEX_TRIMESH_MIDPHASE_DIAGNOSTIC_EXTERNAL_SCALAR' "${CPP}"
		scalar_arg_launch_source_exit=$?
		grep -Fq 'scalarArgKernel=%u' "${CPP}"
		scalar_arg_ready_source_exit=$?
		grep -Fq 'void midphaseDiagnosticExternalScalar(physx::PxU32 diagnosticValue)' "${CU}"
		scalar_arg_kernel_source_exit=$?
		grep -Fq 'CONVEX_TRIMESH_MIDPHASE_DIAGNOSTIC_EXTERNAL_SCALAR, "midphaseDiagnosticExternalScalar"' "${NAMES}"
		scalar_arg_name_source_exit=$?
	else
		scalar_arg_env_source_exit=0
		scalar_arg_launch_source_exit=0
		scalar_arg_ready_source_exit=0
		scalar_arg_kernel_source_exit=0
		scalar_arg_name_source_exit=0
	fi
	if [ "${KNOWN_EXISTING_KERNEL}" -eq 1 ]; then
		grep -Fq 'PX_DCU_NP_TRIMESH_MIDPHASE_KNOWN_EXISTING_KERNEL' "${CPP}"
		known_existing_env_source_exit=$?
		grep -Fq 'midphaseKnownExistingKernel ? PxgKernelIds::INITIALIZE_MANIFOLDS' "${CPP}"
		known_existing_launch_source_exit=$?
		grep -Fq 'PxU32 knownKernelDataSize = sizeof(uint4);' "${CPP}"
		known_existing_data_size_source_exit=$?
		grep -Fq 'PxU32 knownKernelReplicateCount = 0;' "${CPP}"
		known_existing_count_source_exit=$?
		grep -Fq 'knownExistingKernel=%u knownDataSize=%u knownReplicateCount=%u' "${CPP}"
		known_existing_ready_source_exit=$?
	else
		known_existing_env_source_exit=0
		known_existing_launch_source_exit=0
		known_existing_data_size_source_exit=0
		known_existing_count_source_exit=0
		known_existing_ready_source_exit=0
	fi
	if [ "${FRESH_STREAM}" -eq 1 ]; then
		grep -Fq 'PX_DCU_NP_TRIMESH_MIDPHASE_FRESH_STREAM' "${CPP}"
		fresh_stream_env_source_exit=$?
		grep -Fq 'streamCreate(&midphaseLaunchStream, CU_STREAM_NON_BLOCKING)' "${CPP}"
		fresh_stream_create_source_exit=$?
		grep -Fq 'eventCreate(&midphaseDependencyEvent, CU_EVENT_DISABLE_TIMING)' "${CPP}"
		fresh_event_create_source_exit=$?
		grep -Fq 'eventRecord(midphaseDependencyEvent, mStream)' "${CPP}"
		fresh_event_record_source_exit=$?
		grep -Fq 'streamWaitEvent(midphaseLaunchStream, midphaseDependencyEvent, 0)' "${CPP}"
		fresh_stream_wait_source_exit=$?
		grep -Fq '0, midphaseLaunchStream, externalEmptyKernelParams' "${CPP}"
		fresh_stream_launch_source_exit=$?
		grep -Fq 'streamSynchronize(midphaseLaunchStream)' "${CPP}"
		fresh_stream_sync_source_exit=$?
		grep -Fq 'eventDestroy(midphaseDependencyEvent)' "${CPP}"
		fresh_event_destroy_source_exit=$?
		grep -Fq 'streamDestroy(midphaseLaunchStream)' "${CPP}"
		fresh_stream_destroy_source_exit=$?
		grep -Fq 'freshStream=%u streamCreate=%d eventCreate=%d eventRecord=%d streamWait=%d streamSync=%d baseStreamSync=%d eventDestroy=%d streamDestroy=%d' "${CPP}"
		fresh_stream_ready_source_exit=$?
	else
		fresh_stream_env_source_exit=0
		fresh_stream_create_source_exit=0
		fresh_event_create_source_exit=0
		fresh_event_record_source_exit=0
		fresh_stream_wait_source_exit=0
		fresh_stream_launch_source_exit=0
		fresh_stream_sync_source_exit=0
		fresh_event_destroy_source_exit=0
		fresh_stream_destroy_source_exit=0
		fresh_stream_ready_source_exit=0
	fi
	if [ "${D2D_COMMAND}" -eq 1 ]; then
		grep -Fq 'PX_DCU_NP_TRIMESH_MIDPHASE_D2D_COMMAND' "${CPP}"
		d2d_env_source_exit=$?
		grep -Fq 'midphaseD2DResult = mCudaContext->memcpyDtoDAsync(' "${CPP}"
		d2d_command_source_exit=$?
		grep -Fq 'gpuIntermStack + sizeof(uint4), gpuIntermStack, sizeof(uint4), mStream' "${CPP}"
		d2d_shape_source_exit=$?
		grep -Fq 'command=%s d2dResult=%d d2dBytes=%u' "${CPP}"
		d2d_ready_source_exit=$?
	else
		d2d_env_source_exit=0
		d2d_command_source_exit=0
		d2d_shape_source_exit=0
		d2d_ready_source_exit=0
	fi
	grep -n "void midphaseDiagnosticExternalEmpty" "${CU}"
	kernel_declaration_exit=$?
	grep -n "void initializeManifolds" "${CU}"
	known_path_kernel_exit=$?
	grep -n "getCuFunction(PxgKernelIds::INITIALIZE_MANIFOLDS)" "${CPP}"
	known_path_launch_exit=$?
	grep -n 'CONVEX_TRIMESH_MIDPHASE_DIAGNOSTIC_EXTERNAL_EMPTY, "midphaseDiagnosticExternalEmpty"' "${NAMES}"
	kernel_name_exit=$?

	cd "${ROOT}/build_dcu"
	dcu_cd_exit=$?
	if [ ${dcu_cd_exit} -eq 0 ]; then cmake --build . -j1; dcu_build_exit=$?; else dcu_build_exit=125; fi
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
	stat -c "names_mtime=%y path=%n" "${NAMES}"
	names_stat_exit=$?
	stat -c "exe_mtime=%y path=%n" "${EXE}"
	exe_stat_exit=$?
	if [ ${cpp_stat_exit} -eq 0 ] && [ ${names_stat_exit} -eq 0 ] && [ ${exe_stat_exit} -eq 0 ] && \
		[ "${EXE}" -nt "${CPP}" ] && [ "${EXE}" -nt "${NAMES}" ]; then executable_newer=1; else executable_newer=0; fi
	if [ "${UNIT_BLOCK}" -eq 1 ]; then
		strings "${EXE}" | grep -Fq 'unitBlock=%u grid=(%u,1,1) block=(%u,%u,1) originalBlock=(%u,%u,1)'
		unit_block_ready_binary_exit=$?
	else
		unit_block_ready_binary_exit=0
	fi
	if [ "${SKIP_DISPATCH}" -eq 1 ]; then
		strings "${EXE}" | grep -Fq 'skipDispatch=%u submitted=%u'
		skip_dispatch_ready_binary_exit=$?
	else
		skip_dispatch_ready_binary_exit=0
	fi
	if [ "${NONNULL_PARAMS}" -eq 1 ]; then
		strings "${EXE}" | grep -Fq 'launchResult=%d sync=%d copy=%d'
		nonnull_params_launch_result_binary_exit=$?
		strings "${EXE}" | grep -Fq 'nonNullParams=%u nonNullParamEntry=%u skipDispatch=%u submitted=%u'
		nonnull_params_ready_binary_exit=$?
	else
		nonnull_params_launch_result_binary_exit=0
		nonnull_params_ready_binary_exit=0
	fi
	if [ "${NONNULL_PARAM_ENTRY}" -eq 1 ]; then
		strings "${EXE}" | grep -Fq 'nonNullParamEntry=%u skipDispatch=%u submitted=%u'
		nonnull_param_entry_ready_binary_exit=$?
	else
		nonnull_param_entry_ready_binary_exit=0
	fi
	if [ "${SCALAR_ARG_KERNEL}" -eq 1 ]; then
		strings "${EXE}" | grep -Fq 'scalarArgKernel=%u'
		scalar_arg_ready_binary_exit=$?
		strings "${HSACO}" | grep -Fq 'midphaseDiagnosticExternalScalar'
		scalar_arg_symbol_binary_exit=$?
	else
		scalar_arg_ready_binary_exit=0
		scalar_arg_symbol_binary_exit=0
	fi
	if [ "${KNOWN_EXISTING_KERNEL}" -eq 1 ]; then
		strings "${EXE}" | grep -Fq 'knownExistingKernel=%u knownDataSize=%u knownReplicateCount=%u'
		known_existing_ready_binary_exit=$?
		strings "${HSACO}" | grep -Fq 'initializeManifolds'
		known_existing_symbol_binary_exit=$?
	else
		known_existing_ready_binary_exit=0
		known_existing_symbol_binary_exit=0
	fi
	if [ "${FRESH_STREAM}" -eq 1 ]; then
		strings "${EXE}" | grep -Fq 'freshStream=%u streamCreate=%d eventCreate=%d eventRecord=%d streamWait=%d streamSync=%d baseStreamSync=%d eventDestroy=%d streamDestroy=%d'
		fresh_stream_ready_binary_exit=$?
	else
		fresh_stream_ready_binary_exit=0
	fi
	if [ "${D2D_COMMAND}" -eq 1 ]; then
		strings "${EXE}" | grep -Fq 'command=%s d2dResult=%d d2dBytes=%u'
		d2d_ready_binary_exit=$?
	else
		d2d_ready_binary_exit=0
	fi

	readelf_available=0
	readelf_dump_exit=127
	readelf_grep_exit=127
	nm_available=0
	nm_dump_exit=127
	nm_grep_exit=127
	dccobjdump_available=0
	dccobjdump_dump_exit=127
	dccobjdump_grep_exit=127
	strings_available=0
	strings_dump_exit=127
	strings_grep_exit=127
	symbol_tool=none
	symbol_dump_exit=127
	symbol_grep_exit=127
	if command -v readelf >/dev/null 2>&1; then
		readelf_available=1
		readelf -Ws "${HSACO}" | grep -F midphaseDiagnosticExternalEmpty
		symbol_status=("${PIPESTATUS[@]}")
		readelf_dump_exit=${symbol_status[0]}
		readelf_grep_exit=${symbol_status[1]}
	fi
	if [ ${readelf_dump_exit} -eq 0 ] && [ ${readelf_grep_exit} -eq 0 ]; then
		symbol_tool=readelf
		symbol_dump_exit=${readelf_dump_exit}
		symbol_grep_exit=${readelf_grep_exit}
	elif command -v nm >/dev/null 2>&1; then
		nm_available=1
		nm -a "${HSACO}" | grep -F midphaseDiagnosticExternalEmpty
		symbol_status=("${PIPESTATUS[@]}")
		nm_dump_exit=${symbol_status[0]}
		nm_grep_exit=${symbol_status[1]}
		if [ ${nm_dump_exit} -eq 0 ] && [ ${nm_grep_exit} -eq 0 ]; then
			symbol_tool=nm
			symbol_dump_exit=${nm_dump_exit}
			symbol_grep_exit=${nm_grep_exit}
		fi
	fi
	if [ "${symbol_tool}" = none ] && command -v dccobjdump >/dev/null 2>&1; then
		dccobjdump_available=1
		dccobjdump -t "${HSACO}" | grep -F midphaseDiagnosticExternalEmpty
		symbol_status=("${PIPESTATUS[@]}")
		dccobjdump_dump_exit=${symbol_status[0]}
		dccobjdump_grep_exit=${symbol_status[1]}
		if [ ${dccobjdump_dump_exit} -eq 0 ] && [ ${dccobjdump_grep_exit} -eq 0 ]; then
			symbol_tool=dccobjdump
			symbol_dump_exit=${dccobjdump_dump_exit}
			symbol_grep_exit=${dccobjdump_grep_exit}
		fi
	fi
	if [ "${symbol_tool}" = none ] && command -v strings >/dev/null 2>&1; then
		strings_available=1
		strings "${HSACO}" | grep -F midphaseDiagnosticExternalEmpty
		symbol_status=("${PIPESTATUS[@]}")
		strings_dump_exit=${symbol_status[0]}
		strings_grep_exit=${symbol_status[1]}
		if [ ${strings_dump_exit} -eq 0 ] && [ ${strings_grep_exit} -eq 0 ]; then
			symbol_tool=strings
			symbol_dump_exit=${strings_dump_exit}
			symbol_grep_exit=${strings_grep_exit}
		fi
	fi
	if [ ${symbol_dump_exit} -eq 0 ] && [ ${symbol_grep_exit} -eq 0 ]; then symbol_present=1; else symbol_present=0; fi

	if [ ${host_probe_exit} -eq 0 ] && [ ${unit_block_env_source_exit} -eq 0 ] && \
		[ ${unit_block_launch_source_exit} -eq 0 ] && [ ${unit_block_ready_source_exit} -eq 0 ] && \
		[ ${unit_block_ready_binary_exit} -eq 0 ] && [ ${skip_dispatch_env_source_exit} -eq 0 ] && \
		[ ${skip_dispatch_branch_source_exit} -eq 0 ] && [ ${skip_dispatch_ready_source_exit} -eq 0 ] && \
		[ ${skip_dispatch_ready_binary_exit} -eq 0 ] && [ ${nonnull_params_env_source_exit} -eq 0 ] && \
		[ ${nonnull_params_launch_source_exit} -eq 0 ] && [ ${nonnull_params_launch_result_source_exit} -eq 0 ] && \
		[ ${nonnull_params_ready_source_exit} -eq 0 ] && [ ${nonnull_params_launch_result_binary_exit} -eq 0 ] && \
		[ ${nonnull_params_ready_binary_exit} -eq 0 ] && [ ${nonnull_param_entry_env_source_exit} -eq 0 ] && \
		[ ${nonnull_param_entry_launch_source_exit} -eq 0 ] && [ ${nonnull_param_entry_ready_source_exit} -eq 0 ] && \
		[ ${nonnull_param_entry_ready_binary_exit} -eq 0 ] && [ ${scalar_arg_env_source_exit} -eq 0 ] && \
		[ ${scalar_arg_launch_source_exit} -eq 0 ] && [ ${scalar_arg_ready_source_exit} -eq 0 ] && \
		[ ${scalar_arg_kernel_source_exit} -eq 0 ] && [ ${scalar_arg_name_source_exit} -eq 0 ] && \
		[ ${scalar_arg_ready_binary_exit} -eq 0 ] && [ ${scalar_arg_symbol_binary_exit} -eq 0 ] && \
		[ ${known_existing_env_source_exit} -eq 0 ] && [ ${known_existing_launch_source_exit} -eq 0 ] && \
		[ ${known_existing_data_size_source_exit} -eq 0 ] && [ ${known_existing_count_source_exit} -eq 0 ] && \
		[ ${known_existing_ready_source_exit} -eq 0 ] && [ ${known_existing_ready_binary_exit} -eq 0 ] && \
		[ ${known_existing_symbol_binary_exit} -eq 0 ] && [ ${fresh_stream_env_source_exit} -eq 0 ] && \
		[ ${fresh_stream_create_source_exit} -eq 0 ] && [ ${fresh_event_create_source_exit} -eq 0 ] && \
		[ ${fresh_event_record_source_exit} -eq 0 ] && [ ${fresh_stream_wait_source_exit} -eq 0 ] && \
		[ ${fresh_stream_launch_source_exit} -eq 0 ] && [ ${fresh_stream_sync_source_exit} -eq 0 ] && \
		[ ${fresh_event_destroy_source_exit} -eq 0 ] && [ ${fresh_stream_destroy_source_exit} -eq 0 ] && \
		[ ${fresh_stream_ready_source_exit} -eq 0 ] && [ ${fresh_stream_ready_binary_exit} -eq 0 ] && \
		[ ${d2d_env_source_exit} -eq 0 ] && [ ${d2d_command_source_exit} -eq 0 ] && \
		[ ${d2d_shape_source_exit} -eq 0 ] && [ ${d2d_ready_source_exit} -eq 0 ] && \
		[ ${d2d_ready_binary_exit} -eq 0 ] && \
		[ ${kernel_declaration_exit} -eq 0 ] && \
		[ ${known_path_kernel_exit} -eq 0 ] && [ ${known_path_launch_exit} -eq 0 ] && [ ${kernel_name_exit} -eq 0 ] && \
		[ ${dcu_build_exit} -eq 0 ] && [ ${cpu_build_exit} -eq 0 ] && [ ${example_build_exit} -eq 0 ] && \
		[ ${hsaco_newer} -eq 1 ] && [ ${executable_newer} -eq 1 ] && [ ${symbol_present} -eq 1 ]; then build_gate=1; else build_gate=0; fi
	echo "host_probe_exit=${host_probe_exit} kernel_declaration_exit=${kernel_declaration_exit} known_path_kernel_exit=${known_path_kernel_exit} known_path_launch_exit=${known_path_launch_exit} kernel_name_exit=${kernel_name_exit}"
	echo "unit_block_env_source_exit=${unit_block_env_source_exit} unit_block_launch_source_exit=${unit_block_launch_source_exit} unit_block_ready_source_exit=${unit_block_ready_source_exit} unit_block_ready_binary_exit=${unit_block_ready_binary_exit}"
	echo "skip_dispatch_env_source_exit=${skip_dispatch_env_source_exit} skip_dispatch_branch_source_exit=${skip_dispatch_branch_source_exit} skip_dispatch_ready_source_exit=${skip_dispatch_ready_source_exit} skip_dispatch_ready_binary_exit=${skip_dispatch_ready_binary_exit}"
	echo "nonnull_params_env_source_exit=${nonnull_params_env_source_exit} nonnull_params_launch_source_exit=${nonnull_params_launch_source_exit} nonnull_params_launch_result_source_exit=${nonnull_params_launch_result_source_exit} nonnull_params_ready_source_exit=${nonnull_params_ready_source_exit} nonnull_params_launch_result_binary_exit=${nonnull_params_launch_result_binary_exit} nonnull_params_ready_binary_exit=${nonnull_params_ready_binary_exit}"
	echo "nonnull_param_entry_env_source_exit=${nonnull_param_entry_env_source_exit} nonnull_param_entry_launch_source_exit=${nonnull_param_entry_launch_source_exit} nonnull_param_entry_ready_source_exit=${nonnull_param_entry_ready_source_exit} nonnull_param_entry_ready_binary_exit=${nonnull_param_entry_ready_binary_exit}"
	echo "scalar_arg_env_source_exit=${scalar_arg_env_source_exit} scalar_arg_launch_source_exit=${scalar_arg_launch_source_exit} scalar_arg_ready_source_exit=${scalar_arg_ready_source_exit} scalar_arg_kernel_source_exit=${scalar_arg_kernel_source_exit} scalar_arg_name_source_exit=${scalar_arg_name_source_exit} scalar_arg_ready_binary_exit=${scalar_arg_ready_binary_exit} scalar_arg_symbol_binary_exit=${scalar_arg_symbol_binary_exit}"
	echo "known_existing_env_source_exit=${known_existing_env_source_exit} known_existing_launch_source_exit=${known_existing_launch_source_exit} known_existing_data_size_source_exit=${known_existing_data_size_source_exit} known_existing_count_source_exit=${known_existing_count_source_exit} known_existing_ready_source_exit=${known_existing_ready_source_exit} known_existing_ready_binary_exit=${known_existing_ready_binary_exit} known_existing_symbol_binary_exit=${known_existing_symbol_binary_exit}"
	echo "fresh_stream_env_source_exit=${fresh_stream_env_source_exit} fresh_stream_create_source_exit=${fresh_stream_create_source_exit} fresh_event_create_source_exit=${fresh_event_create_source_exit} fresh_event_record_source_exit=${fresh_event_record_source_exit} fresh_stream_wait_source_exit=${fresh_stream_wait_source_exit} fresh_stream_launch_source_exit=${fresh_stream_launch_source_exit} fresh_stream_sync_source_exit=${fresh_stream_sync_source_exit} fresh_event_destroy_source_exit=${fresh_event_destroy_source_exit} fresh_stream_destroy_source_exit=${fresh_stream_destroy_source_exit} fresh_stream_ready_source_exit=${fresh_stream_ready_source_exit} fresh_stream_ready_binary_exit=${fresh_stream_ready_binary_exit}"
	echo "d2d_env_source_exit=${d2d_env_source_exit} d2d_command_source_exit=${d2d_command_source_exit} d2d_shape_source_exit=${d2d_shape_source_exit} d2d_ready_source_exit=${d2d_ready_source_exit} d2d_ready_binary_exit=${d2d_ready_binary_exit}"
	echo "dcu_cd_exit=${dcu_cd_exit} dcu_build_exit=${dcu_build_exit} cpu_cd_exit=${cpu_cd_exit} cpu_build_exit=${cpu_build_exit} example_cd_exit=${example_cd_exit} example_build_exit=${example_build_exit}"
	echo "cu_stat_exit=${cu_stat_exit} hsaco_stat_exit=${hsaco_stat_exit} hsaco_newer=${hsaco_newer} cpp_stat_exit=${cpp_stat_exit} names_stat_exit=${names_stat_exit} exe_stat_exit=${exe_stat_exit} executable_newer=${executable_newer}"
	echo "readelf_available=${readelf_available} readelf_dump_exit=${readelf_dump_exit} readelf_grep_exit=${readelf_grep_exit}"
	echo "nm_available=${nm_available} nm_dump_exit=${nm_dump_exit} nm_grep_exit=${nm_grep_exit}"
	echo "dccobjdump_available=${dccobjdump_available} dccobjdump_dump_exit=${dccobjdump_dump_exit} dccobjdump_grep_exit=${dccobjdump_grep_exit}"
	echo "strings_available=${strings_available} strings_dump_exit=${strings_dump_exit} strings_grep_exit=${strings_grep_exit}"
	echo "symbol_tool=${symbol_tool} symbol_dump_exit=${symbol_dump_exit} symbol_grep_exit=${symbol_grep_exit} symbol_present=${symbol_present} build_gate=${build_gate}"
} >> "${BUILD_LOG}" 2>&1
build_block_exit=$?
echo "build_block_exit=${build_block_exit}" >> "${BUILD_LOG}"

echo "probe_entry=${PROBE_ENTRY} experiment_base=${EXPERIMENT_BASE} unit_block=${UNIT_BLOCK} skip_dispatch=${SKIP_DISPATCH} nonnull_params=${NONNULL_PARAMS} nonnull_param_entry=${NONNULL_PARAM_ENTRY} scalar_arg_kernel=${SCALAR_ARG_KERNEL} known_existing_kernel=${KNOWN_EXISTING_KERNEL} fresh_stream=${FRESH_STREAM} d2d_command=${D2D_COMMAND}" >> "${RUN_LOG}"
echo "experiment=${EXPERIMENT} build_gate=${build_gate}" >> "${RUN_LOG}"
preflight_smi_1=$(timeout 15s hy-smi 2>&1)
hy_smi_pre_exit_1=$?
printf '%s\n' "preflight_sample=1" "${preflight_smi_1}" >> "${RUN_LOG}"
preflight_idle_1=$(printf '%s\n' "${preflight_smi_1}" | summarize_idle)
sleep 3
preflight_smi_2=$(timeout 15s hy-smi 2>&1)
hy_smi_pre_exit_2=$?
printf '%s\n' "preflight_sample=2" "${preflight_smi_2}" >> "${RUN_LOG}"
preflight_idle_2=$(printf '%s\n' "${preflight_smi_2}" | summarize_idle)
preflight_hcu_count_1=$(printf '%s\n' "${preflight_smi_1}" | awk '/^[[:space:]]*[0-9]+[[:space:]]/{count++} END{print count+0}')
preflight_hcu_count_2=$(printf '%s\n' "${preflight_smi_2}" | awk '/^[[:space:]]*[0-9]+[[:space:]]/{count++} END{print count+0}')
preflight_smi_alert_count_1=$(printf '%s\n' "${preflight_smi_1}" | grep -Eic 'Open mkfd failed|No device available|initialization failed|[[:space:]]N/A[[:space:]]')
preflight_smi_alert_count_2=$(printf '%s\n' "${preflight_smi_2}" | grep -Eic 'Open mkfd failed|No device available|initialization failed|[[:space:]]N/A[[:space:]]')
preflight_idle_gate_1=$(printf '%s\n' "${preflight_idle_1}" | sed -n 's/.* idle=\([01]\)$/\1/p')
preflight_idle_gate_2=$(printf '%s\n' "${preflight_idle_2}" | sed -n 's/.* idle=\([01]\)$/\1/p')
preflight_pci_missing=$(count_pci_missing)
if [ ${hy_smi_pre_exit_1} -eq 0 ] && [ ${hy_smi_pre_exit_2} -eq 0 ] && \
	[ ${preflight_hcu_count_1} -eq 8 ] && [ ${preflight_hcu_count_2} -eq 8 ] && \
	[ ${preflight_smi_alert_count_1} -eq 0 ] && [ ${preflight_smi_alert_count_2} -eq 0 ] && \
	[ "${preflight_idle_gate_1}" = 1 ] && [ "${preflight_idle_gate_2}" = 1 ] && \
	[ ${preflight_pci_missing} -eq 0 ]; then preflight_gate=1; else preflight_gate=0; fi
echo "preflight_sample=1 hy_smi_exit=${hy_smi_pre_exit_1} hcu_count=${preflight_hcu_count_1} smi_alert_count=${preflight_smi_alert_count_1} ${preflight_idle_1}" >> "${RUN_LOG}"
echo "preflight_sample=2 hy_smi_exit=${hy_smi_pre_exit_2} hcu_count=${preflight_hcu_count_2} smi_alert_count=${preflight_smi_alert_count_2} ${preflight_idle_2}" >> "${RUN_LOG}"
echo "preflight_limits max_vram=10.0 max_hcu=10.0 samples=2 interval_seconds=3 pci_missing=${preflight_pci_missing} preflight_gate=${preflight_gate}" >> "${RUN_LOG}"

run_exit=125
monitor_wait_exit=125
watchdog_wait_exit=125
watchdog_fired=0
run_skipped=0
SINCE=$(date '+%Y-%m-%d %H:%M:%S')
echo "dmesg_since=${SINCE}" >> "${RUN_LOG}"

if [ ${build_gate} -eq 1 ] && [ ${preflight_gate} -eq 1 ]; then
	cd "${LOG_DIR}"
	run_cd_exit=$?
	if [ ${run_cd_exit} -eq 0 ]; then
		run_env=( \
			HIP_VISIBLE_DEVICES=${TARGET_HIP_DEVICE:-7} \
			PX_DCU_NP_TRIMESH_PRECORE_PROBE=5 \
			PX_DCU_NP_TRIMESH_MIDPHASE_EXTERNAL_EMPTY_REPLACE=1 \
			PX_DCU_NP_TRIMESH_EXTERNAL_HSACO_PROOF=1 \
			PX_DCU_NP_TRIMESH_PRECORE_HOLD_MS=30000 )
		if [ "${UNIT_BLOCK}" -eq 1 ]; then
			run_env+=(PX_DCU_NP_TRIMESH_MIDPHASE_EXTERNAL_EMPTY_UNIT_BLOCK=1)
		fi
		if [ "${SKIP_DISPATCH}" -eq 1 ]; then
			run_env+=(PX_DCU_NP_TRIMESH_MIDPHASE_EXTERNAL_EMPTY_SKIP_DISPATCH=1)
		fi
		if [ "${NONNULL_PARAMS}" -eq 1 ]; then
			run_env+=(PX_DCU_NP_TRIMESH_MIDPHASE_EXTERNAL_EMPTY_NONNULL_PARAMS=1)
		fi
		if [ "${NONNULL_PARAM_ENTRY}" -eq 1 ]; then
			run_env+=(PX_DCU_NP_TRIMESH_MIDPHASE_EXTERNAL_EMPTY_NONNULL_PARAM_ENTRY=1)
		fi
		if [ "${SCALAR_ARG_KERNEL}" -eq 1 ]; then
			run_env+=(PX_DCU_NP_TRIMESH_MIDPHASE_EXTERNAL_SCALAR_ARG_KERNEL=1)
		fi
		if [ "${KNOWN_EXISTING_KERNEL}" -eq 1 ]; then
			run_env+=(PX_DCU_NP_TRIMESH_MIDPHASE_KNOWN_EXISTING_KERNEL=1)
		fi
		if [ "${FRESH_STREAM}" -eq 1 ]; then
			run_env+=(PX_DCU_NP_TRIMESH_MIDPHASE_FRESH_STREAM=1)
		fi
		if [ "${D2D_COMMAND}" -eq 1 ]; then
			run_env+=(PX_DCU_NP_TRIMESH_MIDPHASE_D2D_COMMAND=1)
		fi
		env "${run_env[@]}" "${EXE}" 45 --bodies 8 --grid 48 --start-height 3.0 >> "${RUN_LOG}" 2>&1 &
	else
		( exit 125 ) &
	fi
	app_pid=$!
	echo "run_cd_exit=${run_cd_exit} app_pid=${app_pid} app_start_exit=0" >> "${RUN_LOG}"
	(
		sleep 75
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
			if grep -q '\[DCU NP TRIMESH PRECORE HOLD\] phase=pre-exit' "${RUN_LOG}"; then phase=pre-exit
			elif grep -q '\[DCU NP TRIMESH PRECORE HOLD\] phase=begin' "${RUN_LOG}"; then phase=hold
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
	echo "run_exit=${run_exit} expected_run_exit=94 validation_failure_exit=95 monitor_wait_exit=${monitor_wait_exit} watchdog_wait_exit=${watchdog_wait_exit} watchdog_fired=${watchdog_fired}" >> "${RUN_LOG}"
	post_sample=0
	while [ ${post_sample} -lt 10 ]; do
		post_sample=$((post_sample + 1))
		sample_health post-exit "${post_sample}"
		if [ $((post_sample % 5)) -eq 0 ]; then sample_dmesg post-exit; fi
		sleep 2
	done
else
	run_skipped=1
	echo "run_skipped=gate build_gate=${build_gate} preflight_gate=${preflight_gate}" >> "${RUN_LOG}"
fi

grep -Eq "${READY_PATTERN}" "${RUN_LOG}"
hold_ready_exit=$?
grep -Eq '\[DCU NP TRIMESH EXTERNAL HSACO PROOF\] kernel=initializeManifolds module=pairManagement managers=[1-9][0-9]* launch=0' "${RUN_LOG}"
external_hsaco_proof_exit=$?
grep -q '\[DCU NP TRIMESH PRECORE HOLD\] phase=begin ms=30000' "${RUN_LOG}"
hold_begin_exit=$?
grep -q '\[DCU NP TRIMESH PRECORE HOLD\] phase=pre-exit ms=10000' "${RUN_LOG}"
hold_pre_exit_marker_exit=$?
grep -q '\[DCU NP TRIMESH PRECORE HOLD\] phase=end totalMs=40000' "${RUN_LOG}"
hold_end_exit=$?

alive_health_failures=$(awk '/^health_sample / && /phase=(startup|hold|pre-exit)/ && /healthy=0/ && /app_alive_after=1/{count++} END{print count+0}' "${RUN_LOG}")
hold_healthy_samples=$(awk '/^health_sample / && /phase=hold/ && /healthy=1/ && /app_alive_after=1/{count++} END{print count+0}' "${RUN_LOG}")
pre_exit_healthy_samples=$(awk '/^health_sample / && /phase=pre-exit/ && /healthy=1/ && /app_alive_after=1/{count++} END{print count+0}' "${RUN_LOG}")
post_health_failures=$(awk '/^health_sample / && /phase=post-exit/ && /healthy=0/{count++} END{print count+0}' "${RUN_LOG}")
dmesg_alert_count=$(grep -c '^dmesg_alert ' "${RUN_LOG}")

if [ ${run_skipped} -eq 1 ]; then
	probe_exit=125
	hypothesis_result=not-tested-gate-rejected
elif [ ${hold_ready_exit} -ne 0 ]; then
	probe_exit=1
	hypothesis_result=experiment-configuration-not-proven
elif [ ${run_exit} -eq 94 ] && [ ${hold_ready_exit} -eq 0 ] && [ ${external_hsaco_proof_exit} -eq 0 ] && [ ${hold_begin_exit} -eq 0 ] && \
	[ ${hold_pre_exit_marker_exit} -eq 0 ] && [ ${hold_end_exit} -eq 0 ] && \
	[ ${alive_health_failures} -eq 0 ] && [ ${hold_healthy_samples} -gt 0 ] && [ ${pre_exit_healthy_samples} -gt 0 ] && \
	[ ${post_health_failures} -eq 0 ] && [ ${watchdog_fired} -eq 0 ]; then
	probe_exit=0
	hypothesis_result=${STABLE_RESULT}
elif [ ${run_exit} -eq 95 ]; then
	probe_exit=1
	hypothesis_result=external-empty-counter-validation-failed
elif [ ${external_hsaco_proof_exit} -ne 0 ]; then
	probe_exit=1
	hypothesis_result=external-hsaco-prior-execution-not-proven
elif [ ${alive_health_failures} -gt 0 ] || [ ${post_health_failures} -gt 0 ]; then
	probe_exit=1
	hypothesis_result=${DEVICE_FAILURE_RESULT}
elif [ ${run_exit} -eq 137 ]; then
	probe_exit=1
	hypothesis_result=external-sigkill-without-observed-device-failure
elif [ ${watchdog_fired} -eq 1 ]; then
	probe_exit=1
	hypothesis_result=probe-timeout
else
	probe_exit=1
	hypothesis_result=incomplete
fi

if [ ${probe_exit} -eq 0 ]; then experiment_exit=0
elif [ ${probe_exit} -eq 125 ]; then experiment_exit=125
else experiment_exit=1; fi
echo "hold_ready_exit=${hold_ready_exit} external_hsaco_proof_exit=${external_hsaco_proof_exit} hold_begin_exit=${hold_begin_exit} hold_pre_exit_marker_exit=${hold_pre_exit_marker_exit} hold_end_exit=${hold_end_exit}" >> "${RUN_LOG}"
echo "alive_health_failures=${alive_health_failures} hold_healthy_samples=${hold_healthy_samples} pre_exit_healthy_samples=${pre_exit_healthy_samples} post_health_failures=${post_health_failures} dmesg_alert_count=${dmesg_alert_count}" >> "${RUN_LOG}"
echo "probe_exit=${probe_exit} hypothesis_result=${hypothesis_result} experiment_exit=${experiment_exit}" >> "${RUN_LOG}"
echo "build_log=${BUILD_LOG}"
echo "run_log=${RUN_LOG}"
echo "experiment_exit=${experiment_exit}"
exit ${experiment_exit}
