#!/bin/bash

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)

export EXPERIMENT_BASE=complex_trimesh_midphase_external_empty_r19
export UNIT_BLOCK=0
export SKIP_DISPATCH=0
export NONNULL_PARAMS=0
export NONNULL_PARAM_ENTRY=0
export READY_PATTERN='\[DCU NP TRIMESH PRECORE HOLD\] phase=ready launch=5 entryReturn=0 emptyReplace=0 launchResult=0 sync=0 copy=0 tests=[0-9]+ midphase=0 padded=0 .*valid=1 externalEmptyReplace=1 unitBlock=0 grid=\(2,1,1\) block=\(32,2,1\) originalBlock=\(32,2,1\) nonNullParams=0 nonNullParamEntry=0 skipDispatch=0 submitted=1'
export STABLE_RESULT=external-module-empty-same-geometry-insufficient-convex-midphase-code-object-specific
export DEVICE_FAILURE_RESULT=external-module-empty-same-geometry-still-sufficient-generic-dispatch-or-geometry-implicated
export PROBE_ENTRY=r19

printf 'probe_entry=r19 experiment_base=%s unit_block=%s\n' "${EXPERIMENT_BASE}" "${UNIT_BLOCK}"
exec bash "${SCRIPT_DIR}/run_complex_trimesh_midphase_external_empty_probe_common.sh"
