#!/bin/bash

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)

export EXPERIMENT_BASE=complex_trimesh_midphase_fresh_stream_r26
export UNIT_BLOCK=1
export SKIP_DISPATCH=0
export NONNULL_PARAMS=1
export NONNULL_PARAM_ENTRY=1
export SCALAR_ARG_KERNEL=0
export KNOWN_EXISTING_KERNEL=1
export FRESH_STREAM=1
export READY_PATTERN='\[DCU NP TRIMESH PRECORE HOLD\] phase=ready launch=5 entryReturn=0 emptyReplace=0 launchResult=0 sync=0 copy=0 tests=3 midphase=0 padded=0 .*valid=1 externalEmptyReplace=1 unitBlock=1 grid=\(2,1,1\) block=\(1,1,1\) originalBlock=\(32,2,1\) nonNullParams=1 nonNullParamEntry=1 skipDispatch=0 submitted=1 scalarArgKernel=0 knownExistingKernel=1 knownDataSize=16 knownReplicateCount=0 freshStream=1 streamCreate=0 eventCreate=0 eventRecord=0 streamWait=0 streamSync=0 baseStreamSync=0 eventDestroy=0 streamDestroy=0'
export STABLE_RESULT=fresh-stream-stable-existing-midphase-stream-or-queue-state-implicated-under-tested-state
export DEVICE_FAILURE_RESULT=fresh-stream-still-drops-existing-midphase-stream-or-queue-state-not-necessary
export PROBE_ENTRY=r26

printf 'probe_entry=r26 experiment_base=%s known_existing_kernel=%s fresh_stream=%s\n' "${EXPERIMENT_BASE}" "${KNOWN_EXISTING_KERNEL}" "${FRESH_STREAM}"
exec bash "${SCRIPT_DIR}/run_complex_trimesh_midphase_external_empty_probe_common.sh"
