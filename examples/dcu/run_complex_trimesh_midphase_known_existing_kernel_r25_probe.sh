#!/bin/bash

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)

export EXPERIMENT_BASE=complex_trimesh_midphase_known_existing_kernel_r25
export UNIT_BLOCK=1
export SKIP_DISPATCH=0
export NONNULL_PARAMS=1
export NONNULL_PARAM_ENTRY=1
export SCALAR_ARG_KERNEL=0
export KNOWN_EXISTING_KERNEL=1
export READY_PATTERN='\[DCU NP TRIMESH PRECORE HOLD\] phase=ready launch=5 entryReturn=0 emptyReplace=0 launchResult=0 sync=0 copy=0 tests=3 midphase=0 padded=0 .*valid=1 externalEmptyReplace=1 unitBlock=1 grid=\(2,1,1\) block=\(1,1,1\) originalBlock=\(32,2,1\) nonNullParams=1 nonNullParamEntry=1 skipDispatch=0 submitted=1 scalarArgKernel=0 knownExistingKernel=1 knownDataSize=16 knownReplicateCount=0'
export STABLE_RESULT=known-existing-kernel-stable-new-diagnostic-function-entry-or-metadata-implicated-under-tested-state
export DEVICE_FAILURE_RESULT=known-existing-kernel-still-drops-new-diagnostic-function-entry-not-necessary
export PROBE_ENTRY=r25

printf 'probe_entry=r25 experiment_base=%s unit_block=%s skip_dispatch=%s known_existing_kernel=%s\n' "${EXPERIMENT_BASE}" "${UNIT_BLOCK}" "${SKIP_DISPATCH}" "${KNOWN_EXISTING_KERNEL}"
exec bash "${SCRIPT_DIR}/run_complex_trimesh_midphase_external_empty_probe_common.sh"
