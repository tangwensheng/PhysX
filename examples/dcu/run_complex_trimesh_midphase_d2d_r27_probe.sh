#!/bin/bash

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)

export EXPERIMENT_BASE=complex_trimesh_midphase_d2d_r27
export UNIT_BLOCK=1
export SKIP_DISPATCH=0
export NONNULL_PARAMS=1
export NONNULL_PARAM_ENTRY=1
export SCALAR_ARG_KERNEL=0
export KNOWN_EXISTING_KERNEL=1
export FRESH_STREAM=0
export D2D_COMMAND=1
export READY_PATTERN='\[DCU NP TRIMESH PRECORE HOLD\] phase=ready launch=5 entryReturn=0 emptyReplace=0 launchResult=0 sync=0 copy=0 tests=3 midphase=0 padded=0 .*valid=1 externalEmptyReplace=1 unitBlock=1 grid=\(2,1,1\) block=\(1,1,1\) originalBlock=\(32,2,1\) nonNullParams=1 nonNullParamEntry=1 skipDispatch=0 submitted=1 scalarArgKernel=0 knownExistingKernel=1 knownDataSize=16 knownReplicateCount=0 freshStream=0 .*command=d2d d2dResult=0 d2dBytes=16'
export STABLE_RESULT=d2d-command-stable-kernel-dispatch-path-necessary-under-tested-state
export DEVICE_FAILURE_RESULT=d2d-command-also-drops-any-gpu-command-or-broader-queue-context-device-state-sufficient
export PROBE_ENTRY=r27

printf 'probe_entry=r27 experiment_base=%s d2d_command=%s\n' "${EXPERIMENT_BASE}" "${D2D_COMMAND}"
exec bash "${SCRIPT_DIR}/run_complex_trimesh_midphase_external_empty_probe_common.sh"
