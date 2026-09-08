#!/bin/bash

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)

export EXPERIMENT_BASE=complex_trimesh_midphase_scalar_arg_kernel_r24
export UNIT_BLOCK=1
export SKIP_DISPATCH=0
export NONNULL_PARAMS=1
export NONNULL_PARAM_ENTRY=1
export SCALAR_ARG_KERNEL=1
export READY_PATTERN='\[DCU NP TRIMESH PRECORE HOLD\] phase=ready launch=5 entryReturn=0 emptyReplace=0 launchResult=0 sync=0 copy=0 tests=3 midphase=0 padded=0 .*valid=1 externalEmptyReplace=1 unitBlock=1 grid=\(2,1,1\) block=\(1,1,1\) originalBlock=\(32,2,1\) nonNullParams=1 nonNullParamEntry=1 skipDispatch=0 submitted=1 scalarArgKernel=1'
export STABLE_RESULT=scalar-arg-kernel-stable-zero-argument-kernel-abi-necessary-trigger-under-tested-state
export DEVICE_FAILURE_RESULT=scalar-arg-kernel-still-drops-zero-argument-kernel-abi-not-necessary
export PROBE_ENTRY=r24

printf 'probe_entry=r24 experiment_base=%s unit_block=%s skip_dispatch=%s nonnull_params=%s nonnull_param_entry=%s scalar_arg_kernel=%s\n' "${EXPERIMENT_BASE}" "${UNIT_BLOCK}" "${SKIP_DISPATCH}" "${NONNULL_PARAMS}" "${NONNULL_PARAM_ENTRY}" "${SCALAR_ARG_KERNEL}"
exec bash "${SCRIPT_DIR}/run_complex_trimesh_midphase_external_empty_probe_common.sh"
