#!/bin/bash
#
# DCU device-drop probe runner (2026-08-19 rewrite).
#
# Replaces run_complex_trimesh_*_probe.sh. Those scripts only checked that the
# executable was newer than its sources, never that the sources still matched
# the experiment being replayed -- so the r9/r21 reruns silently tested code
# that was six days newer than the original runs, and their results were void.
#
# This script records an md5 of every file that can change behaviour, and
# refuses to run if the fingerprint moved since the baseline was pinned.
# Every log therefore carries proof of exactly what was tested.
#
# Usage:
#   ./run_dcu_probe.sh <site|full> [--pin]
#
# Sites, in the order the kernels are issued inside one call of
# testSDKConvexTriMeshSATGpu:
#
#   pre-midphase          before any kernel of this call is queued
#   post-midphase         after CONVEX_TRIMESH_MIDPHASE
#   post-core             after CONVEX_TRIMESH_CORE
#   post-sorttriangles    after CONVEX_TRIMESH_SORT_TRIANGLES
#   post-postprocess      after CONVEX_TRIMESH_POST_PROCESS
#   post-correlate        after CONVEX_TRIMESH_CORRELATE
#   post-finishcontacts   after CONVEX_TRIMESH_FINISHCONTACTS
#   post-compact          after both compactLostFoundPairs kernels
#   post-convextrimesh-return after the convex-trimesh callee returns
#   post-update-friction  after updateFrictionPatches returns
#   full                  run the complete workload; expect exit 0/PASS
#
# Examples:
#   ./run_dcu_probe.sh pre-midphase --pin
#   PROBE_CALL=2 ./run_dcu_probe.sh post-midphase
#   REPEAT=3 HOLD_MS=20000 ./run_dcu_probe.sh post-core
#   REPEAT=5 STEPS=720 BODIES=256 ./run_dcu_probe.sh full
#   PX_DCU_CORE_BLOCKS=fit ./run_dcu_probe.sh post-core
#   PX_DCU_CORE_BLOCKS=fit PX_DCU_CORE_TEMP_INDEX=1 ./run_dcu_probe.sh full
#   PROBE_CALL=10 PX_DCU_CORE_STAGE=1 ./run_dcu_probe.sh post-core
#   PROBE_CALL=10 PX_DCU_CORE_STAGES="9 15 16 17 18 19 20 21 10" \
#       ./run_dcu_probe.sh post-core
#   PROBE_CALL=10 REPEAT=2 REPEAT_DELAY=5 \
#       PX_DCU_CORE_STAGES="38 39 40 41 42 43 32 33 34 35 36 37" \
#       ./run_dcu_probe.sh post-core
#   PROBE_CALL=10 REPEAT=2 REPEAT_DELAY=3 \
#       PX_DCU_CORE_STAGES="44 45 46 47 48 49 50 51 52 53" \
#       ./run_dcu_probe.sh post-core
#   PX_DCU_MIDPHASE_DIAG=1 PX_DCU_MIDPHASE_MAX_TRIANGLES=4608 \
#       ./run_dcu_probe.sh post-midphase
#   STEPS=720 BODIES=256 GRID=48 START_HEIGHT=3.0 \
#       ./run_dcu_probe.sh post-core
#
set -u

SITE=${1:?usage: $0 <site|full> [--pin]   (see header for the list of sites)}
PIN=${2:-}

ROOT=/public/home/tangwsh/PhysX
LOG_DIR=${ROOT}/examples/dcu
EXE=${ROOT}/examples/dcu/build/bench_complex_trimesh_smoke
RUNNER=${ROOT}/examples/dcu/run_dcu_probe.sh
GPU_LIB=${ROOT}/build_dcu/libPhysXGpuDCU.so
FINGERPRINT_FILE=${LOG_DIR}/dcu_probe_baseline.md5

HOLD_MS=${HOLD_MS:-90000}
PROBE_CALL=${PROBE_CALL:-5}
DEVICE=${TARGET_HIP_DEVICE:-7}
REPEAT=${REPEAT:-1}
REPEAT_DELAY=${REPEAT_DELAY:-60}
NOSYNC=${NOSYNC:-}
CORE_BLOCKS=${PX_DCU_CORE_BLOCKS:-}
CORE_TEMP_INDEX=${PX_DCU_CORE_TEMP_INDEX:-}
CORE_STAGE=${PX_DCU_CORE_STAGE:-}
CORE_STAGES=${PX_DCU_CORE_STAGES:-}
SOLVER_SYNC_DIAG=${PX_DCU_SOLVER_SYNC_DIAG:-}
SOLVER_FLOW_DIAG=${PX_DCU_SOLVER_FLOW_DIAG:-}
SOLVER_PREPREP_DIAG=${PX_DCU_SOLVER_PREPREP_DIAG:-}
SOLVER_PREPREP_DIAG_CALL=${PX_DCU_SOLVER_PREPREP_DIAG_CALL:-}
SOLVER_PREPREP_DIAG_FIRST_ACTIVE=${PX_DCU_SOLVER_PREPREP_DIAG_FIRST_ACTIVE:-}
CONTACT_GEOM_DIAG=${PX_DCU_CONTACT_GEOM_DIAG:-}
FINISH_STREAM_DIAG=${PX_DCU_FINISH_STREAM_DIAG:-}
MIDPHASE_DIAG=${PX_DCU_MIDPHASE_DIAG:-}
MIDPHASE_DIAG_CALL=${PX_DCU_MIDPHASE_DIAG_CALL:-}
MIDPHASE_MAX_TRIANGLES=${PX_DCU_MIDPHASE_MAX_TRIANGLES:-}
MIDPHASE_DIAG_ABORT=${PX_DCU_MIDPHASE_DIAG_ABORT:-}
STEPS=${STEPS:-45}
BODIES=${BODIES:-8}
GRID=${GRID:-48}
START_HEIGHT=${START_HEIGHT:-3.0}
HEALTH_INTERVAL=${HEALTH_INTERVAL:-2}
POST_HEALTH_SAMPLES=${POST_HEALTH_SAMPLES:-10}
CORE_STAGE_HSACO=${ROOT}/build_dcu/kernels/convexMesh.hsaco
CORRELATE_STAGE_HSACO=${ROOT}/build_dcu/kernels/convexMeshCorrelate.hsaco
FINISH_CONTACTS_STAGE_HSACO=${ROOT}/build_dcu/kernels/convexMeshOutput.hsaco
GJKEPA_STAGE_HSACO=${ROOT}/build_dcu/kernels/cudaGJKEPA.hsaco
BOXBOX_STAGE_HSACO=${ROOT}/build_dcu/kernels/cudaBox.hsaco
PAIR_MANAGEMENT_STAGE_HSACO=${ROOT}/build_dcu/kernels/pairManagement.hsaco
SOLVER_PREPREP_STAGE_HSACO=${ROOT}/build_dcu/kernels/constraintBlockPrePrep.hsaco
SOLVER_CONTACT_PREP_STAGE_HSACO=${ROOT}/build_dcu/kernels/constraintBlockPrep.hsaco
SOLVER_CONTACT_PREP_TGS_HSACO=${ROOT}/build_dcu/kernels/constraintBlockPrepTGS.hsaco
SOLVER_ARTI_PREP_STAGE_HSACO=${ROOT}/build_dcu/kernels/artiConstraintPrep2.hsaco
SOLVER_STATIC_STAGE_HSACO=${ROOT}/build_dcu/kernels/solverMultiBlock.hsaco

# Everything that can change what the device sees.
FINGERPRINT_SRC="
${ROOT}/physx/source/gpunarrowphase/src/PxgNarrowphaseCore.cpp
${ROOT}/physx/source/gpunarrowphase/src/PxgNphaseImplementationContext.cpp
${ROOT}/physx/source/gpunarrowphase/src/CUDA/convexMeshMidphase.cu
${ROOT}/physx/source/gpunarrowphase/src/CUDA/convexMesh.cu
${ROOT}/physx/source/gpunarrowphase/src/CUDA/convexMeshCorrelate.cu
${ROOT}/physx/source/gpunarrowphase/src/CUDA/convexMeshOutput.cu
${ROOT}/physx/source/gpunarrowphase/src/CUDA/convexHeightfield.cu
${ROOT}/physx/source/gpunarrowphase/src/CUDA/convexTriangle.cuh
${ROOT}/physx/source/gpunarrowphase/src/CUDA/cudaGJKEPA.cu
${ROOT}/physx/source/gpunarrowphase/src/CUDA/cudaBox.cu
${ROOT}/physx/source/gpunarrowphase/src/CUDA/pairManagement.cu
${ROOT}/physx/source/gpusolver/src/PxgSolverCore.cpp
${ROOT}/physx/source/gpusolver/src/PxgCudaSolverCore.cpp
${ROOT}/physx/source/gpusolver/src/PxgTGSCudaSolverCore.cpp
${ROOT}/physx/source/gpusolver/include/PxgSolverCoreDesc.h
${ROOT}/physx/source/gpusolver/src/CUDA/constraintBlockPrePrep.cu
${ROOT}/physx/source/gpusolver/src/CUDA/constraintBlockPrep.cu
${ROOT}/physx/source/gpusolver/src/CUDA/contactConstraintBlockPrep.cuh
${ROOT}/physx/source/gpusolver/src/CUDA/solverMultiBlock.cu
${ROOT}/physx/source/gpusolver/src/CUDA/solverBlock.cuh
${ROOT}/physx/source/foundation/unix/FdUnixThread.cpp
${ROOT}/examples/dcu/bench_complex_trimesh_smoke.cpp
${RUNNER}
"
FINGERPRINT_BIN="
${EXE}
${ROOT}/build_dcu/kernels/convexMeshMidphase.hsaco
${ROOT}/build_dcu/kernels/convexMesh.hsaco
${ROOT}/build_dcu/kernels/convexMeshCorrelate.hsaco
${FINISH_CONTACTS_STAGE_HSACO}
${ROOT}/build_dcu/kernels/convexHeightfield.hsaco
${GJKEPA_STAGE_HSACO}
${BOXBOX_STAGE_HSACO}
${PAIR_MANAGEMENT_STAGE_HSACO}
${ROOT}/build_dcu/kernels/constraintBlockPrePrep.hsaco
${SOLVER_CONTACT_PREP_STAGE_HSACO}
${SOLVER_CONTACT_PREP_TGS_HSACO}
${SOLVER_ARTI_PREP_STAGE_HSACO}
${ROOT}/build_dcu/kernels/solverMultiBlock.hsaco
${GPU_LIB}
"

check_build_input()
{
	local src=$1
	local output=$2
	local label=$3

	if [ ! -e "${output}" ]; then
		echo "STALE: ${label} output is missing: ${output}"
		stale=$((stale + 1))
	elif [ -e "${src}" ] && [ "${src}" -nt "${output}" ]; then
		echo "STALE: ${src} is newer than ${label} output ${output}"
		stale=$((stale + 1))
	fi
}

compute_fingerprint()
{
	local f
	for f in ${FINGERPRINT_SRC} ${FINGERPRINT_BIN}; do
		if [ -e "${f}" ]; then
			md5sum "${f}" 2>/dev/null
		else
			echo "MISSING ${f}"
		fi
	done
}

case "${SITE}" in
	pre-midphase|post-midphase|post-core|post-sorttriangles|\
	post-postprocess|post-correlate|post-finishcontacts|post-compact|\
	post-convextrimesh-return|post-update-friction|full) ;;
	*)
		echo "unknown site: ${SITE}" >&2
		echo "valid: pre-midphase post-midphase post-core post-sorttriangles \
post-postprocess post-correlate post-finishcontacts post-compact \
post-convextrimesh-return post-update-friction full" >&2
		exit 2 ;;
esac

if [ "${PIN}" = "--pin" ]; then
	pin_candidate=${FINGERPRINT_FILE}.candidate.$$
	compute_fingerprint > "${pin_candidate}"
	if grep -q '^MISSING ' "${pin_candidate}"; then
		echo "pin rejected: fingerprint input is missing" >&2
		cat "${pin_candidate}" >&2
		rm -f "${pin_candidate}"
		exit 3
	fi
	mv "${pin_candidate}" "${FINGERPRINT_FILE}"
	echo "baseline pinned -> ${FINGERPRINT_FILE}"
	cat "${FINGERPRINT_FILE}"
	pin_verify_file=${FINGERPRINT_FILE}.verify.$$
	compute_fingerprint > "${pin_verify_file}"
	if diff -q "${FINGERPRINT_FILE}" "${pin_verify_file}" >/dev/null 2>&1; then
		pin_verify=1
	else
		pin_verify=0
		echo "--- pin verification DRIFT ---" >&2
		diff "${FINGERPRINT_FILE}" "${pin_verify_file}" >&2
	fi
	rm -f "${pin_verify_file}"
	echo "pin_verify=${pin_verify}"
	if [ ${pin_verify} -ne 1 ]; then exit 3; fi
	exit 0
fi

if [ ! -f "${FINGERPRINT_FILE}" ]; then
	echo "no baseline fingerprint; run once with --pin after building" >&2
	exit 2
fi

run_once()
{
	local iteration=$1
	local stamp
	stamp=$(date +%Y%m%d_%H%M%S)
	local stage_suffix=${CORE_STAGE:+_stage${CORE_STAGE}}
	local run_log=${LOG_DIR}/dcu_probe_${SITE}_call${PROBE_CALL}${stage_suffix}${NOSYNC:+_nosync}_${stamp}_run.log
	local expected_exit=96
	if [ "${SITE}" = "full" ]; then expected_exit=0; fi

	{
		echo "site=${SITE} iteration=${iteration} probe_call=${PROBE_CALL} hold_ms=${HOLD_MS} device=${DEVICE} nosync=${NOSYNC:-0} core_blocks=${CORE_BLOCKS:-default} core_temp_index=${CORE_TEMP_INDEX:-0} core_stage=${CORE_STAGE:-0} solver_sync_diag=${SOLVER_SYNC_DIAG:-0} solver_flow_diag=${SOLVER_FLOW_DIAG:-0} solver_preprep_diag=${SOLVER_PREPREP_DIAG:-0} solver_preprep_diag_call=${SOLVER_PREPREP_DIAG_CALL:-all} solver_preprep_first_active=${SOLVER_PREPREP_DIAG_FIRST_ACTIVE:-0} contact_geom_diag=${CONTACT_GEOM_DIAG:-0} finish_stream_diag=${FINISH_STREAM_DIAG:-0} middiag=${MIDPHASE_DIAG:-0} middiag_call=${MIDPHASE_DIAG_CALL:-all} max_triangles=${MIDPHASE_MAX_TRIANGLES:-unchecked} steps=${STEPS} bodies=${BODIES} grid=${GRID} start_height=${START_HEIGHT}"
		date

		# --- version gate: refuse to test code that is not the pinned baseline ---
		echo "--- fingerprint ---"
		compute_fingerprint | tee /tmp/dcu_probe_now.md5
		if diff -q "${FINGERPRINT_FILE}" /tmp/dcu_probe_now.md5 >/dev/null 2>&1; then
			fingerprint_gate=1
		else
			fingerprint_gate=0
			echo "--- fingerprint DRIFT (expected vs actual) ---"
			diff "${FINGERPRINT_FILE}" /tmp/dcu_probe_now.md5
		fi
		echo "fingerprint_gate=${fingerprint_gate}"

		# --- staleness gate: compare each source with the artifact that actually
		# consumes it. Host-side sources link into the example executable; device
		# sources compile into their corresponding hsaco modules.
		stale=0
		check_build_input "${ROOT}/physx/source/gpunarrowphase/src/PxgNarrowphaseCore.cpp" "${EXE}" "executable"
		check_build_input "${ROOT}/physx/source/gpunarrowphase/src/PxgNphaseImplementationContext.cpp" "${EXE}" "executable"
		check_build_input "${ROOT}/physx/source/foundation/unix/FdUnixThread.cpp" "${EXE}" "executable"
		check_build_input "${ROOT}/physx/source/gpusolver/src/PxgSolverCore.cpp" "${EXE}" "executable"
		check_build_input "${ROOT}/physx/source/gpusolver/src/PxgCudaSolverCore.cpp" "${EXE}" "executable"
		check_build_input "${ROOT}/physx/source/gpusolver/src/PxgTGSCudaSolverCore.cpp" "${EXE}" "executable"
		check_build_input "${ROOT}/physx/source/gpusolver/include/PxgSolverCoreDesc.h" "${EXE}" "executable"
		check_build_input "${ROOT}/physx/source/gpusolver/include/PxgSolverCoreDesc.h" "${SOLVER_CONTACT_PREP_STAGE_HSACO}" "constraintBlockPrep hsaco"
		check_build_input "${ROOT}/physx/source/gpusolver/include/PxgSolverCoreDesc.h" "${SOLVER_CONTACT_PREP_TGS_HSACO}" "constraintBlockPrepTGS hsaco"
		check_build_input "${ROOT}/physx/source/gpusolver/include/PxgSolverCoreDesc.h" "${SOLVER_PREPREP_STAGE_HSACO}" "constraintBlockPrePrep hsaco"
		check_build_input "${ROOT}/physx/source/gpusolver/include/PxgSolverCoreDesc.h" "${SOLVER_ARTI_PREP_STAGE_HSACO}" "artiConstraintPrep2 hsaco"
		check_build_input "${ROOT}/examples/dcu/bench_complex_trimesh_smoke.cpp" "${EXE}" "executable"
		check_build_input "${ROOT}/physx/source/gpunarrowphase/src/CUDA/convexMeshMidphase.cu" "${ROOT}/build_dcu/kernels/convexMeshMidphase.hsaco" "convexMeshMidphase hsaco"
		check_build_input "${ROOT}/physx/source/gpunarrowphase/src/CUDA/convexMesh.cu" "${ROOT}/build_dcu/kernels/convexMesh.hsaco" "convexMesh hsaco"
		check_build_input "${ROOT}/physx/source/gpunarrowphase/src/CUDA/convexMeshCorrelate.cu" "${CORRELATE_STAGE_HSACO}" "convexMeshCorrelate hsaco"
		check_build_input "${ROOT}/physx/source/gpunarrowphase/src/CUDA/convexMeshOutput.cu" "${FINISH_CONTACTS_STAGE_HSACO}" "convexMeshOutput hsaco"
		check_build_input "${ROOT}/physx/source/gpunarrowphase/src/CUDA/convexTriangle.cuh" "${ROOT}/build_dcu/kernels/convexMesh.hsaco" "convexMesh hsaco"
		check_build_input "${ROOT}/physx/source/gpunarrowphase/src/CUDA/convexHeightfield.cu" "${ROOT}/build_dcu/kernels/convexHeightfield.hsaco" "convexHeightfield hsaco"
		check_build_input "${ROOT}/physx/source/gpunarrowphase/src/CUDA/cudaGJKEPA.cu" "${GJKEPA_STAGE_HSACO}" "cudaGJKEPA hsaco"
		check_build_input "${ROOT}/physx/source/gpunarrowphase/src/CUDA/cudaBox.cu" "${BOXBOX_STAGE_HSACO}" "cudaBox hsaco"
		check_build_input "${ROOT}/physx/source/gpunarrowphase/src/CUDA/pairManagement.cu" "${PAIR_MANAGEMENT_STAGE_HSACO}" "pairManagement hsaco"
		check_build_input "${ROOT}/physx/source/gpusolver/src/CUDA/constraintBlockPrePrep.cu" "${SOLVER_PREPREP_STAGE_HSACO}" "constraintBlockPrePrep hsaco"
		check_build_input "${ROOT}/physx/source/gpusolver/src/CUDA/constraintBlockPrep.cu" "${SOLVER_CONTACT_PREP_STAGE_HSACO}" "constraintBlockPrep hsaco"
		check_build_input "${ROOT}/physx/source/gpusolver/src/CUDA/contactConstraintBlockPrep.cuh" "${SOLVER_CONTACT_PREP_STAGE_HSACO}" "constraintBlockPrep hsaco"
		check_build_input "${ROOT}/physx/source/gpusolver/src/CUDA/contactConstraintBlockPrep.cuh" "${SOLVER_CONTACT_PREP_TGS_HSACO}" "constraintBlockPrepTGS hsaco"
		check_build_input "${ROOT}/physx/source/gpusolver/src/CUDA/contactConstraintBlockPrep.cuh" "${SOLVER_ARTI_PREP_STAGE_HSACO}" "artiConstraintPrep2 hsaco"
		check_build_input "${ROOT}/physx/source/gpusolver/src/CUDA/solverMultiBlock.cu" "${SOLVER_STATIC_STAGE_HSACO}" "solverMultiBlock hsaco"
		check_build_input "${ROOT}/physx/source/gpusolver/src/CUDA/solverBlock.cuh" "${SOLVER_STATIC_STAGE_HSACO}" "solverMultiBlock hsaco"
		if [ ${stale} -eq 0 ]; then build_gate=1; else build_gate=0; fi
		echo "build_gate=${build_gate} stale_sources=${stale}"

		# --- probe symbol gates: verify both host and device halves were rebuilt ---
		if strings "${EXE}" 2>/dev/null | grep -Fq '[DCU CORE TEMP INDEX]' \
			&& strings "${EXE}" 2>/dev/null | grep -Fq 'PX_DCU_SOLVER_STAGE_V27_HOST_SYNC_BOUNDARY' \
			&& strings "${EXE}" 2>/dev/null | grep -Fq 'PX_DCU_SOLVER_STAGE_V40_VALIDATE_FRICTION_HISTORY_BOUNDS' \
			&& strings "${EXE}" 2>/dev/null | grep -Fq 'PX_DCU_SOLVER_STAGE_V39_CLEAR_CURRENT_FRICTION_COUNTS' \
			&& strings "${EXE}" 2>/dev/null | grep -Fq 'PX_DCU_SOLVER_STAGE_V46_FIRST_ACTIVE_PREPREP_TARGET' \
			&& strings "${EXE}" 2>/dev/null | grep -Fq 'PX_DCU_SOLVER_STAGE_V49_INTEGRATION_FLOW_READBACK' \
			&& strings "${EXE}" 2>/dev/null | grep -Fq 'PX_DCU_COMPLEX_TRIMESH_STAGE_V50_SPECULATIVE_CCD' \
			&& strings "${EXE}" 2>/dev/null | grep -Fq 'PX_DCU_NARROWPHASE_STAGE_V51_POST_COMPACT_BOUNDARY' \
			&& strings "${EXE}" 2>/dev/null | grep -Fq 'PX_DCU_NARROWPHASE_STAGE_V52_CALLER_RETURN_BOUNDARY' \
			&& strings "${EXE}" 2>/dev/null | grep -Fq 'PX_DCU_NARROWPHASE_STAGE_V53_POST_UPDATE_FRICTION_BOUNDARY' \
			&& strings "${EXE}" 2>/dev/null | grep -Fq 'PX_DCU_NARROWPHASE_STAGE_V44_NOINLINE_FINISH_STREAM_AGGREGATE'; then
			host_probe_gate=1
		else
			host_probe_gate=0
			echo "HOST PROBE MISSING: executable needs core temp, V27 solver sync, V39 friction-count clear, V40 friction-history bound, V44 noinline finish-stream, V46 first-active pre-prep, V49 integration-flow, V50 speculative-CCD, V51 post-compact, V52 caller-return, and V53 post-update-friction markers"
		fi
		if strings "${CORE_STAGE_HSACO}" 2>/dev/null | grep -Fq 'PX_DCU_CORE_STAGE_V20_SINGLE_LDS_BASE' \
			&& strings "${CORRELATE_STAGE_HSACO}" 2>/dev/null | grep -Fq 'PX_DCU_CORE_STAGE_V25_FINITE_SERIAL_CORRELATE' \
			&& strings "${FINISH_CONTACTS_STAGE_HSACO}" 2>/dev/null | grep -Fq 'PX_DCU_NARROWPHASE_STAGE_V41_SERIAL_FINISH_CONTACTS' \
			&& strings "${GJKEPA_STAGE_HSACO}" 2>/dev/null | grep -Fq 'PX_DCU_NARROWPHASE_STAGE_V54_SERIAL_LOST_FOUND_COMPACT' \
			&& strings "${BOXBOX_STAGE_HSACO}" 2>/dev/null | grep -Fq 'PX_DCU_NARROWPHASE_STAGE_V34_SERIAL_BOXBOX_GROUP' \
			&& strings "${PAIR_MANAGEMENT_STAGE_HSACO}" 2>/dev/null | grep -Fq 'PX_DCU_PAIR_REMOVE_STAGE_V55_SERIAL_COMPACT' \
			&& strings "${SOLVER_PREPREP_STAGE_HSACO}" 2>/dev/null | grep -Fq 'PX_DCU_SOLVER_STAGE_V45_TARGETED_STATIC_BATCH_AGGREGATE' \
			&& strings "${SOLVER_CONTACT_PREP_STAGE_HSACO}" 2>/dev/null | grep -Fq 'PX_DCU_SOLVER_STAGE_V40_VALIDATE_FRICTION_HISTORY_BOUNDS' \
			&& strings "${SOLVER_STATIC_STAGE_HSACO}" 2>/dev/null | grep -Fq 'PX_DCU_SOLVER_STAGE_V32_LEAN_STATIC_SOLVER_ABI'; then
			device_probe_gate=1
		else
			device_probe_gate=0
			echo "DEVICE PROBE MISSING: expected V20 core, V25 correlate, V41 finish contacts, V54 lost/found compact, V34 box-box, V55 pair remove, V45 solver pre-prep, V40 contact prep, and V32 static solver markers"
		fi
		if [ ${host_probe_gate} -eq 1 ] && [ ${device_probe_gate} -eq 1 ]; then probe_gate=1; else probe_gate=0; fi
		echo "host_probe_gate=${host_probe_gate} device_probe_gate=${device_probe_gate} probe_gate=${probe_gate}"

		# --- device gate ---
		smi=$(timeout 15s hy-smi 2>&1)
		hcu_count=$(printf '%s\n' "${smi}" | awk '/^[[:space:]]*[0-9]+[[:space:]]/{count++} END{print count+0}')
		smi_alerts=$(printf '%s\n' "${smi}" | grep -Eic 'Open mkfd failed|No device available|initialization failed|[[:space:]]N/A[[:space:]]')
		if [ "${hcu_count}" -eq 8 ] && [ "${smi_alerts}" -eq 0 ]; then device_gate=1; else device_gate=0; fi
		echo "preflight hcu_count=${hcu_count} smi_alerts=${smi_alerts} device_gate=${device_gate}"

		if [ ${fingerprint_gate} -eq 1 ] && [ ${device_gate} -eq 1 ] && [ ${build_gate} -eq 1 ] && [ ${probe_gate} -eq 1 ]; then
			echo "--- run ---"
			env HIP_VISIBLE_DEVICES=${DEVICE} \
				PX_DCU_PROBE=${SITE} \
				PX_DCU_PROBE_CALL=${PROBE_CALL} \
				PX_DCU_PROBE_HOLD_MS=${HOLD_MS} \
				${NOSYNC:+PX_DCU_PROBE_NOSYNC=1} \
				${CORE_BLOCKS:+PX_DCU_CORE_BLOCKS=${CORE_BLOCKS}} \
				${CORE_TEMP_INDEX:+PX_DCU_CORE_TEMP_INDEX=${CORE_TEMP_INDEX}} \
				${CORE_STAGE:+PX_DCU_CORE_STAGE=${CORE_STAGE}} \
				${SOLVER_SYNC_DIAG:+PX_DCU_SOLVER_SYNC_DIAG=${SOLVER_SYNC_DIAG}} \
				${SOLVER_FLOW_DIAG:+PX_DCU_SOLVER_FLOW_DIAG=${SOLVER_FLOW_DIAG}} \
				${SOLVER_PREPREP_DIAG:+PX_DCU_SOLVER_PREPREP_DIAG=${SOLVER_PREPREP_DIAG}} \
				${SOLVER_PREPREP_DIAG_CALL:+PX_DCU_SOLVER_PREPREP_DIAG_CALL=${SOLVER_PREPREP_DIAG_CALL}} \
				${SOLVER_PREPREP_DIAG_FIRST_ACTIVE:+PX_DCU_SOLVER_PREPREP_DIAG_FIRST_ACTIVE=${SOLVER_PREPREP_DIAG_FIRST_ACTIVE}} \
				${CONTACT_GEOM_DIAG:+PX_DCU_CONTACT_GEOM_DIAG=${CONTACT_GEOM_DIAG}} \
				${FINISH_STREAM_DIAG:+PX_DCU_FINISH_STREAM_DIAG=${FINISH_STREAM_DIAG}} \
				${MIDPHASE_DIAG:+PX_DCU_MIDPHASE_DIAG=${MIDPHASE_DIAG}} \
				${MIDPHASE_DIAG_CALL:+PX_DCU_MIDPHASE_DIAG_CALL=${MIDPHASE_DIAG_CALL}} \
				${MIDPHASE_MAX_TRIANGLES:+PX_DCU_MIDPHASE_MAX_TRIANGLES=${MIDPHASE_MAX_TRIANGLES}} \
				${MIDPHASE_DIAG_ABORT:+PX_DCU_MIDPHASE_DIAG_ABORT=${MIDPHASE_DIAG_ABORT}} \
				"${EXE}" "${STEPS}" --bodies "${BODIES}" --grid "${GRID}" --start-height "${START_HEIGHT}" 2>&1 &
			app_pid=$!

			# Sample device health while the app holds.
			(
				sample=0
				while kill -0 "${app_pid}" 2>/dev/null; do
					sample=$((sample + 1))
					s=$(timeout 5s hy-smi 2>&1)
					c=$(printf '%s\n' "${s}" | awk '/^[[:space:]]*[0-9]+[[:space:]]/{count++} END{print count+0}')
					a=$(printf '%s\n' "${s}" | grep -Eic 'Open mkfd failed|No device available|initialization failed|[[:space:]]N/A[[:space:]]')
					if [ "${c}" -eq 8 ] && [ "${a}" -eq 0 ]; then h=1; else h=0; fi
					echo "health phase=hold sample=${sample} hcu=${c} alerts=${a} healthy=${h}"
					sleep "${HEALTH_INTERVAL}"
				done
			) &
			monitor_pid=$!

			wait "${app_pid}"; run_exit=$?
			wait "${monitor_pid}" 2>/dev/null
			echo "run_exit=${run_exit} expected=${expected_exit}"

			post_fail=0
			for i in $(seq 1 "${POST_HEALTH_SAMPLES}"); do
				s=$(timeout 5s hy-smi 2>&1)
				c=$(printf '%s\n' "${s}" | awk '/^[[:space:]]*[0-9]+[[:space:]]/{count++} END{print count+0}')
				a=$(printf '%s\n' "${s}" | grep -Eic 'Open mkfd failed|No device available|initialization failed|[[:space:]]N/A[[:space:]]')
				if [ "${c}" -eq 8 ] && [ "${a}" -eq 0 ]; then h=1; else h=0; post_fail=$((post_fail + 1)); fi
				echo "health phase=post sample=${i} hcu=${c} alerts=${a} healthy=${h}"
				sleep "${HEALTH_INTERVAL}"
			done

			echo "post_health_failures=${post_fail}"

			if [ ${post_fail} -eq 0 ] && [ ${run_exit} -eq ${expected_exit} ]; then
				verdict=STABLE
			elif [ ${post_fail} -gt 0 ]; then
				verdict=DEVICE_DROPPED
			else
				verdict=INCONCLUSIVE
			fi
		else
			echo "run_skipped fingerprint_gate=${fingerprint_gate} device_gate=${device_gate} build_gate=${build_gate} probe_gate=${probe_gate}"
			verdict=NOT_TESTED
		fi

		echo "site=${SITE} call=${PROBE_CALL} nosync=${NOSYNC:-0} iteration=${iteration} verdict=${verdict}"
	} 2>&1 | tee "${run_log}"

	echo "log: ${run_log}"
	grep -q 'verdict=STABLE$' "${run_log}"
}

run_stage()
{
	local i=1
	while [ ${i} -le ${REPEAT} ]; do
		echo "########## ${SITE} stage ${CORE_STAGE:-0} run ${i}/${REPEAT} ##########"
		if ! run_once "${i}"; then
			echo "batch_stop stage=${CORE_STAGE:-0} iteration=${i} reason=non_stable"
			return 1
		fi
		if [ ${i} -lt ${REPEAT} ]; then sleep "${REPEAT_DELAY}"; fi
		i=$((i + 1))
	done
}

if [ -n "${CORE_STAGES}" ]; then
	for stage in ${CORE_STAGES//,/ }; do
		CORE_STAGE=${stage}
		if ! run_stage; then
			exit 1
		fi
	done
else
	run_stage
fi
