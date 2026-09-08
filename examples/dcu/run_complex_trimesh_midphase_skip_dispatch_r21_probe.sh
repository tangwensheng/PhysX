#!/bin/bash

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)

export EXPERIMENT_BASE=complex_trimesh_midphase_skip_dispatch_r21
export UNIT_BLOCK=1
export SKIP_DISPATCH=1
export NONNULL_PARAMS=0
export NONNULL_PARAM_ENTRY=0
export READY_PATTERN='\[DCU NP TRIMESH PRECORE HOLD\] phase=ready launch=5 entryReturn=0 emptyReplace=0 launchResult=0 sync=0 copy=0 tests=3 midphase=0 padded=0 .*valid=1 externalEmptyReplace=1 unitBlock=1 grid=\(2,1,1\) block=\(1,1,1\) originalBlock=\(32,2,1\) nonNullParams=0 nonNullParamEntry=0 skipDispatch=1 submitted=0'
export STABLE_RESULT=skip-dispatch-stable-target-dispatch-necessary-trigger-under-tested-state
export DEVICE_FAILURE_RESULT=skip-dispatch-still-drops-target-dispatch-not-necessary-error-preexists
export PROBE_ENTRY=r21

printf 'probe_entry=r21 experiment_base=%s unit_block=%s skip_dispatch=%s\n' "${EXPERIMENT_BASE}" "${UNIT_BLOCK}" "${SKIP_DISPATCH}"
exec bash "${SCRIPT_DIR}/run_complex_trimesh_midphase_external_empty_probe_common.sh"
