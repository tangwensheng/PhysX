#!/bin/bash

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)

export EXPERIMENT_BASE=complex_trimesh_midphase_unit_block_r20
export UNIT_BLOCK=1
export SKIP_DISPATCH=0
export NONNULL_PARAMS=0
export NONNULL_PARAM_ENTRY=0
export READY_PATTERN='\[DCU NP TRIMESH PRECORE HOLD\] phase=ready launch=5 entryReturn=0 emptyReplace=0 launchResult=0 sync=0 copy=0 tests=3 midphase=0 padded=0 .*valid=1 externalEmptyReplace=1 unitBlock=1 grid=\(2,1,1\) block=\(1,1,1\) originalBlock=\(32,2,1\) nonNullParams=0 nonNullParamEntry=0 skipDispatch=0 submitted=1'
export STABLE_RESULT=unit-block-stable-original-block-geometry-implicated
export DEVICE_FAILURE_RESULT=unit-block-still-drops-original-block-geometry-not-necessary
export PROBE_ENTRY=r20

printf 'probe_entry=r20 experiment_base=%s unit_block=%s\n' "${EXPERIMENT_BASE}" "${UNIT_BLOCK}"
exec bash "${SCRIPT_DIR}/run_complex_trimesh_midphase_external_empty_probe_common.sh"
