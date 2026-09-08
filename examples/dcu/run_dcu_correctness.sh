#!/usr/bin/env bash
# Table-driven, fingerprinted correctness runner for gfx936 DCU benchmarks.

set -u

RUNNER_MARKER="PX_DCU_CORRECTNESS_RUNNER_R5_PBD_STABLE_SORT_DIAG"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="${REPO_ROOT:-$(cd "$SCRIPT_DIR/../.." && pwd)}"
CASE_FILE="${CASE_FILE:-$SCRIPT_DIR/dcu_correctness_cases.tsv}"
MODE="run"
if [[ "${1:-}" == "--pin" ]]; then
    MODE="pin"
    SELECTOR="${2:-all}"
    if (( $# > 2 )); then
        printf 'ERROR: --pin accepts at most one selector\n' >&2
        exit 2
    fi
else
    SELECTOR="${1:-all}"
    if (( $# > 1 )); then
        printf 'ERROR: expected one selector\n' >&2
        exit 2
    fi
fi
BIN_DIR="${BIN_DIR:-$SCRIPT_DIR/build}"
GPU_LIBRARY="${GPU_LIBRARY:-$REPO_ROOT/build_dcu/libPhysXGpuDCU.so}"
HSACO_DIR="${HSACO_DIR:-$REPO_ROOT/build_dcu/kernels}"
CPU_BUILD_DIR="${CPU_BUILD_DIR:-$REPO_ROOT/build_cpu}"
DEVICE="${TARGET_HIP_DEVICE:-1}"
EXPECTED_HCU_COUNT="${EXPECTED_HCU_COUNT:-8}"
MAX_PREFLIGHT_VRAM_PERCENT="${MAX_PREFLIGHT_VRAM_PERCENT:-10}"
MAX_PREFLIGHT_HCU_PERCENT="${MAX_PREFLIGHT_HCU_PERCENT:-10}"
REPEAT="${REPEAT:-1}"
REPEAT_DELAY="${REPEAT_DELAY:-5}"
TIMEOUT_SECONDS="${TIMEOUT_SECONDS:-1800}"
POST_HEALTH_SAMPLES="${POST_HEALTH_SAMPLES:-4}"
HEALTH_INTERVAL="${HEALTH_INTERVAL:-2}"
BROADPHASE_DIAG="${PX_DCU_BROADPHASE_DIAG:-0}"
PBD_PRIMITIVE_RADIX_CHUNKED="${PX_DCU_PBD_PRIMITIVE_RADIX_CHUNKED:-0}"
PBD_PRIMITIVE_RANK_DIAG="${PX_DCU_PBD_PRIMITIVE_RANK_DIAG:-0}"
LOG_ROOT="${LOG_ROOT:-$SCRIPT_DIR/logs}"

case_tag="${CASE_FILE##*/}"
case_tag="${case_tag%.tsv}"
case_tag="$(printf '%s' "$case_tag" | tr -c 'A-Za-z0-9_.-' '_')"
BASELINE_OVERRIDE="${DCU_CORRECTNESS_BASELINE:-}"

usage()
{
    cat <<'EOF'
Usage: ./run_dcu_correctness.sh [--list|all|BATCH|CASE]
       ./run_dcu_correctness.sh --pin [all|BATCH|CASE]

Examples:
  ./run_dcu_correctness.sh --list
  ./run_dcu_correctness.sh --pin
  ./run_dcu_correctness.sh --pin trimesh_complex
  ./run_dcu_correctness.sh all
  ./run_dcu_correctness.sh batch2
  ./run_dcu_correctness.sh trimesh_complex

Environment overrides:
  REPEAT=N                    repeat each selected case N times
  TARGET_HIP_DEVICE=N         physical DCU exposed as HIP device 0
  EXPECTED_HCU_COUNT=N        expected healthy-card count from hy-smi
  MAX_PREFLIGHT_VRAM_PERCENT=N  maximum target VRAM use allowed before launch
  MAX_PREFLIGHT_HCU_PERCENT=N   maximum target HCU use allowed before launch
  TIMEOUT_SECONDS=N           timeout for each benchmark process
  POST_HEALTH_SAMPLES=N       health samples after each benchmark
  HEALTH_INTERVAL=N           seconds between post-run health samples
  PX_DCU_BROADPHASE_DIAG=0|1  enable expensive broadphase diagnostics (default 0)
  PX_DCU_PBD_PRIMITIVE_RADIX_CHUNKED=0|1  use opt-in chunked PBD primitive radix (default 0)
  PX_DCU_PBD_PRIMITIVE_RANK_DIAG=0|1  compare low-32 key/rank output with host stable_sort (default 0)
  CASE_FILE=PATH              alternate compatible case table
  BIN_DIR=PATH                benchmark executable directory
  LOG_ROOT=PATH               parent directory for timestamped logs
  DCU_CORRECTNESS_BASELINE=PATH  fingerprint baseline path
EOF
}

is_positive_integer()
{
    [[ "$1" =~ ^[1-9][0-9]*$ ]]
}

is_percentage()
{
    [[ "$1" =~ ^([0-9]+([.][0-9]+)?|[.][0-9]+)$ ]] &&
        awk -v value="$1" 'BEGIN { exit !(value >= 0 && value <= 100) }'
}

for value_name in REPEAT TIMEOUT_SECONDS POST_HEALTH_SAMPLES HEALTH_INTERVAL EXPECTED_HCU_COUNT; do
    value="${!value_name}"
    if ! is_positive_integer "$value"; then
        printf 'ERROR: %s must be a positive integer, got %s\n' "$value_name" "$value" >&2
        exit 2
    fi
done
if [[ ! "$DEVICE" =~ ^[0-9]+$ ]]; then
    printf 'ERROR: TARGET_HIP_DEVICE must be a non-negative integer, got %s\n' "$DEVICE" >&2
    exit 2
fi
if [[ ! "$BROADPHASE_DIAG" =~ ^[01]$ ]]; then
    printf 'ERROR: PX_DCU_BROADPHASE_DIAG must be 0 or 1, got %s\n' "$BROADPHASE_DIAG" >&2
    exit 2
fi
if [[ ! "$PBD_PRIMITIVE_RADIX_CHUNKED" =~ ^[01]$ ]]; then
    printf 'ERROR: PX_DCU_PBD_PRIMITIVE_RADIX_CHUNKED must be 0 or 1, got %s\n' "$PBD_PRIMITIVE_RADIX_CHUNKED" >&2
    exit 2
fi
if [[ ! "$PBD_PRIMITIVE_RANK_DIAG" =~ ^[01]$ ]]; then
    printf 'ERROR: PX_DCU_PBD_PRIMITIVE_RANK_DIAG must be 0 or 1, got %s\n' "$PBD_PRIMITIVE_RANK_DIAG" >&2
    exit 2
fi
for value_name in MAX_PREFLIGHT_VRAM_PERCENT MAX_PREFLIGHT_HCU_PERCENT; do
    value="${!value_name}"
    if ! is_percentage "$value"; then
        printf 'ERROR: %s must be a percentage from 0 through 100, got %s\n' "$value_name" "$value" >&2
        exit 2
    fi
done
if [[ ! -r "$CASE_FILE" ]]; then
    printf 'ERROR: case table not readable: %s\n' "$CASE_FILE" >&2
    exit 2
fi

declare -a CASE_ORDER=()
declare -a TARGET_ORDER=()
declare -A CASE_BATCH=()
declare -A CASE_TARGET=()
declare -A CASE_ARGS=()
declare -A CASE_NORMAL_EXIT=()
declare -A CASE_DESCRIPTION=()
declare -A CASE_ENVIRONMENT=()
declare -A TARGET_SEEN=()

while IFS=$'\t' read -r case_name batch target arguments normal_exit description environment; do
    [[ -z "$case_name" || "$case_name" == \#* ]] && continue
    description="${description%$'\r'}"
    environment="${environment%$'\r'}"
    CASE_ORDER+=("$case_name")
    CASE_BATCH["$case_name"]="$batch"
    CASE_TARGET["$case_name"]="$target"
    CASE_ARGS["$case_name"]="$arguments"
    CASE_NORMAL_EXIT["$case_name"]="$normal_exit"
    CASE_DESCRIPTION["$case_name"]="$description"
    CASE_ENVIRONMENT["$case_name"]="$environment"
    if [[ -z "${TARGET_SEEN[$target]+x}" ]]; then
        TARGET_SEEN["$target"]=1
        TARGET_ORDER+=("$target")
    fi
done < "$CASE_FILE"

if (( ${#CASE_ORDER[@]} == 0 )); then
    printf 'ERROR: no cases found in %s\n' "$CASE_FILE" >&2
    exit 2
fi

list_cases()
{
    printf '%-30s %-12s %-42s %s\n' CASE BATCH TARGET ARGUMENTS
    local case_name
    for case_name in "${CASE_ORDER[@]}"; do
        printf '%-30s %-12s %-42s %s\n' \
            "$case_name" "${CASE_BATCH[$case_name]}" "${CASE_TARGET[$case_name]}" "${CASE_ARGS[$case_name]}"
    done
}

case "$SELECTOR" in
    -h|--help|help)
        usage
        exit 0
        ;;
    --list|list)
        list_cases
        exit 0
        ;;
esac

declare -a SELECTED_CASES=()
if [[ "$SELECTOR" == "all" ]]; then
    SELECTED_CASES=("${CASE_ORDER[@]}")
elif [[ -n "${CASE_TARGET[$SELECTOR]+x}" ]]; then
    SELECTED_CASES=("$SELECTOR")
else
    for case_name in "${CASE_ORDER[@]}"; do
        [[ "${CASE_BATCH[$case_name]}" == "$SELECTOR" ]] && SELECTED_CASES+=("$case_name")
    done
fi

selector_tag="$(printf '%s' "$SELECTOR" | tr -c 'A-Za-z0-9_.-' '_')"
if [[ -n "$BASELINE_OVERRIDE" ]]; then
    BASELINE="$BASELINE_OVERRIDE"
elif [[ "$SELECTOR" == "all" ]]; then
    BASELINE="$SCRIPT_DIR/${case_tag}_baseline.md5"
else
    BASELINE="$SCRIPT_DIR/${case_tag}_${selector_tag}_baseline.md5"
fi

if (( ${#SELECTED_CASES[@]} == 0 )); then
    printf 'ERROR: unknown or empty selector: %s\n\n' "$SELECTOR" >&2
    usage >&2
    exit 2
fi

linked_cpu_archives()
{
    local archive
    for archive in \
        libFoundation.a libCommon.a libPhysXSDK.a libPhysXExtensions.a \
        libPhysXTask.a libPhysXCooking.a libLowLevel.a libLowLevelAABB.a \
        libLowLevelDynamics.a libSceneQuery.a libSimulationController.a \
        libGeomUtils.a libHipContext.a; do
        printf '%s/%s\n' "$CPU_BUILD_DIR" "$archive"
    done
}

fingerprint_paths()
{
    printf '%s\n' \
        "$CASE_FILE" \
        "$SCRIPT_DIR/run_dcu_correctness.sh" \
        "$SCRIPT_DIR/build.sh" \
        "$SCRIPT_DIR/CMakeLists.txt" \
        "$REPO_ROOT/build_dcu.sh" \
        "$GPU_LIBRARY"

    local case_name target source_dir
    declare -A emitted_targets=()
    for case_name in "${SELECTED_CASES[@]}"; do
        target="${CASE_TARGET[$case_name]}"
        [[ -n "${emitted_targets[$target]+x}" ]] && continue
        emitted_targets["$target"]=1
        printf '%s\n' "$SCRIPT_DIR/${target}.cpp" "$BIN_DIR/$target"
    done
    linked_cpu_archives

    if [[ -d "$HSACO_DIR" ]]; then
        find "$HSACO_DIR" -maxdepth 1 -type f -name '*.hsaco' -print
    fi
    for source_dir in "$REPO_ROOT"/physx/source/gpu* "$REPO_ROOT/physx/source/cudamanager "$REPO_ROOT/physx/include/cudamanager; do
        if [[ -d "$source_dir" ]]; then
            find "$source_dir" -type f \( \
                -name '*.cpp' -o -name '*.h' -o -name '*.hpp' -o \
                -name '*.cu' -o -name '*.cuh' \) -print
        fi
    done
}

compute_fingerprint()
{
    local file
    fingerprint_paths | LC_ALL=C sort -u | while IFS= read -r file; do
        if [[ -e "$file" ]]; then
            md5sum "$file" 2>/dev/null
        else
            printf 'MISSING %s\n' "$file"
        fi
    done
}

verify_build_state()
{
    local errors=0 case_name target executable source input archive hsaco_count
    declare -A checked_targets=()

    if [[ ! -f "$GPU_LIBRARY" ]]; then
        printf 'BUILD ERROR: GPU library missing: %s\n' "$GPU_LIBRARY" >&2
        errors=$((errors + 1))
    fi
    hsaco_count=0
    if [[ -d "$HSACO_DIR" ]]; then
        hsaco_count="$(find "$HSACO_DIR" -maxdepth 1 -type f -name '*.hsaco' | wc -l)"
    fi
    if (( hsaco_count == 0 )); then
        printf 'BUILD ERROR: no hsaco files found in %s\n' "$HSACO_DIR" >&2
        errors=$((errors + 1))
    fi

    while IFS= read -r archive; do
        if [[ ! -f "$archive" ]]; then
            printf 'BUILD ERROR: linked CPU archive missing: %s\n' "$archive" >&2
            errors=$((errors + 1))
        fi
    done < <(linked_cpu_archives)

    for case_name in "${SELECTED_CASES[@]}"; do
        target="${CASE_TARGET[$case_name]}"
        [[ -n "${checked_targets[$target]+x}" ]] && continue
        checked_targets["$target"]=1
        executable="$BIN_DIR/$target"
        source="$SCRIPT_DIR/${target}.cpp"
        if [[ ! -x "$executable" ]]; then
            printf 'BUILD ERROR: executable missing or not executable: %s\n' "$executable" >&2
            errors=$((errors + 1))
            continue
        fi
        for input in "$source" "$SCRIPT_DIR/CMakeLists.txt"; do
            if [[ ! -f "$input" ]]; then
                printf 'BUILD ERROR: input missing: %s\n' "$input" >&2
                errors=$((errors + 1))
            elif [[ "$input" -nt "$executable" ]]; then
                printf 'STALE: %s is newer than executable %s\n' "$input" "$executable" >&2
                errors=$((errors + 1))
            fi
        done
        while IFS= read -r archive; do
            if [[ -f "$archive" && "$archive" -nt "$executable" ]]; then
                printf 'STALE: linked archive %s is newer than executable %s\n' "$archive" "$executable" >&2
                errors=$((errors + 1))
            fi
        done < <(linked_cpu_archives)
    done

    if (( errors == 0 )); then
        printf 'build_gate=1 stale_or_missing=0\n'
        return 0
    fi
    printf 'build_gate=0 stale_or_missing=%d\n' "$errors" >&2
    return 1
}

check_device_health()
{
    local phase="${1:-post}"
    local output hcu_count alerts target_metrics target_vram target_hcu
    local target_present=0 utilization_ok=1
    output="$(timeout 15s hy-smi 2>&1 || true)"
    hcu_count="$(printf '%s\n' "$output" | awk '/^[[:space:]]*[0-9]+[[:space:]]/{count++} END{print count+0}')"
    alerts="$(printf '%s\n' "$output" | grep -Eic 'Open mkfd failed|No device available|initialization failed|[[:space:]]N/A[[:space:]]' || true)"
    target_metrics="$(printf '%s\n' "$output" | awk -v target="$DEVICE" \
        '$1 ~ /^[0-9]+$/ && $1 == target {print $6, $7; exit}')"
    read -r target_vram target_hcu <<< "$target_metrics"
    target_vram="${target_vram:-}"
    target_hcu="${target_hcu:-}"
    target_vram="${target_vram%\%}"
    target_hcu="${target_hcu%\%}"
    if is_percentage "${target_vram:-invalid}" && is_percentage "${target_hcu:-invalid}"; then
        target_present=1
    else
        target_vram=unknown
        target_hcu=unknown
    fi

    if [[ "$phase" == "pre" && "$target_present" -eq 1 ]]; then
        if ! awk -v vram="$target_vram" -v hcu="$target_hcu" \
            -v max_vram="$MAX_PREFLIGHT_VRAM_PERCENT" -v max_hcu="$MAX_PREFLIGHT_HCU_PERCENT" \
            'BEGIN { exit !(vram <= max_vram && hcu <= max_hcu) }'; then
            utilization_ok=0
        fi
    fi

    if [[ "$hcu_count" -eq "$EXPECTED_HCU_COUNT" && "$alerts" -eq 0 && \
        "$target_present" -eq 1 && "$utilization_ok" -eq 1 ]]; then
        health_gate=1
    else
        health_gate=0
    fi
    health_line="$(printf 'health phase=%s hcu_count=%s expected=%s alerts=%s target=%s target_present=%s target_vram_percent=%s target_hcu_percent=%s max_preflight_vram_percent=%s max_preflight_hcu_percent=%s utilization_ok=%s healthy=%s' \
        "$phase" "$hcu_count" "$EXPECTED_HCU_COUNT" "$alerts" "$DEVICE" "$target_present" \
        "$target_vram" "$target_hcu" "$MAX_PREFLIGHT_VRAM_PERCENT" "$MAX_PREFLIGHT_HCU_PERCENT" \
        "$utilization_ok" "$health_gate"
    )"
}

if [[ "$MODE" == "pin" ]]; then
    if ! verify_build_state; then
        printf 'pin rejected: build state is stale or incomplete\n' >&2
        exit 3
    fi
    candidate="${BASELINE}.candidate.$$"
    verify="${BASELINE}.verify.$$"
    compute_fingerprint > "$candidate"
    if grep -q '^MISSING ' "$candidate"; then
        printf 'pin rejected: fingerprint input is missing\n' >&2
        grep '^MISSING ' "$candidate" >&2
        rm -f "$candidate"
        exit 3
    fi
    mv "$candidate" "$BASELINE"
    compute_fingerprint > "$verify"
    if diff -q "$BASELINE" "$verify" >/dev/null 2>&1; then
        pin_verify=1
    else
        pin_verify=0
        printf '%s\n' '--- pin verification DRIFT ---' >&2
        diff "$BASELINE" "$verify" >&2 || true
    fi
    rm -f "$verify"
    printf 'baseline pinned -> %s\n' "$BASELINE"
    printf 'pin_verify=%s\n' "$pin_verify"
    [[ "$pin_verify" -eq 1 ]]
    exit $?
fi

if [[ ! -f "$BASELINE" ]]; then
    printf 'ERROR: no fingerprint baseline: %s\n' "$BASELINE" >&2
    printf 'Complete both builds, relink all benchmark executables, then run %s --pin\n' "$0" >&2
    exit 3
fi

if ! verify_build_state; then
    printf 'ERROR: build gate failed; no benchmark was run\n' >&2
    exit 3
fi

current_fingerprint="/tmp/dcu_correctness_now_$$.md5"
compute_fingerprint > "$current_fingerprint"
if diff -q "$BASELINE" "$current_fingerprint" >/dev/null 2>&1; then
    fingerprint_gate=1
else
    fingerprint_gate=0
    printf '%s\n' '--- fingerprint DRIFT (expected vs actual) ---' >&2
    diff "$BASELINE" "$current_fingerprint" >&2 || true
fi
rm -f "$current_fingerprint"
printf 'fingerprint_gate=%s\n' "$fingerprint_gate"
if [[ "$fingerprint_gate" -ne 1 ]]; then
    printf 'ERROR: fingerprint gate failed; no benchmark was run\n' >&2
    exit 3
fi

timestamp="$(date +%Y%m%d_%H%M%S)"
RUN_DIR="$LOG_ROOT/dcu_correctness_${SELECTOR}_${timestamp}"
SUMMARY_LOG="$RUN_DIR/summary.log"
ENVIRONMENT_LOG="$RUN_DIR/environment.log"
mkdir -p "$RUN_DIR"

write_environment()
{
    printf 'timestamp=%s\n' "$timestamp"
    printf 'runner_marker=%s\n' "$RUNNER_MARKER"
    printf 'selector=%s\n' "$SELECTOR"
    printf 'repeat=%s\n' "$REPEAT"
    printf 'repo_root=%s\n' "$REPO_ROOT"
    printf 'case_file=%s\n' "$CASE_FILE"
    printf 'baseline=%s\n' "$BASELINE"
    printf 'bin_dir=%s\n' "$BIN_DIR"
    printf 'gpu_library=%s\n' "$GPU_LIBRARY"
    printf 'hsaco_dir=%s\n' "$HSACO_DIR"
    printf 'target_hip_device=%s\n' "$DEVICE"
    printf 'expected_hcu_count=%s\n' "$EXPECTED_HCU_COUNT"
    printf 'max_preflight_vram_percent=%s\n' "$MAX_PREFLIGHT_VRAM_PERCENT"
    printf 'max_preflight_hcu_percent=%s\n' "$MAX_PREFLIGHT_HCU_PERCENT"
    printf 'timeout_seconds=%s\n' "$TIMEOUT_SECONDS"
    printf 'post_health_samples=%s\n' "$POST_HEALTH_SAMPLES"
    printf 'health_interval=%s\n' "$HEALTH_INTERVAL"
    printf 'px_dcu_broadphase_diag=%s\n' "$BROADPHASE_DIAG"
    printf 'px_dcu_pbd_primitive_radix_chunked=%s\n' "$PBD_PRIMITIVE_RADIX_CHUNKED"
    printf 'px_dcu_pbd_primitive_rank_diag=%s\n' "$PBD_PRIMITIVE_RANK_DIAG"
    printf 'fingerprint_gate=%s\n' "$fingerprint_gate"
    printf 'baseline_md5='; md5sum "$BASELINE" | awk '{print $1}'
    printf '\n--- hy-smi ---\n'
    timeout 15s hy-smi || true
    printf '\n--- compiler/runtime versions ---\n'
    hipcc --version 2>/dev/null || true
    g++ --version || true
    printf '\n--- git ---\n'
    git -C "$REPO_ROOT" rev-parse HEAD 2>/dev/null || true
    git -C "$REPO_ROOT" branch --show-current 2>/dev/null || true
    printf 'dirty_paths=' 
    git -C "$REPO_ROOT" status --short --untracked-files=all 2>/dev/null | wc -l || true
    printf '\n--- selected executable identity ---\n'
    local case_name target executable
    declare -A emitted=()
    for case_name in "${SELECTED_CASES[@]}"; do
        target="${CASE_TARGET[$case_name]}"
        [[ -n "${emitted[$target]+x}" ]] && continue
        emitted["$target"]=1
        executable="$BIN_DIR/$target"
        md5sum "$SCRIPT_DIR/${target}.cpp" "$executable"
        ldd "$executable" 2>/dev/null || true
    done
    printf '\n--- GPU product identity ---\n'
    md5sum "$GPU_LIBRARY"
    find "$HSACO_DIR" -maxdepth 1 -type f -name '*.hsaco' -print | LC_ALL=C sort | xargs -r md5sum
}

write_environment 2>&1 | tee "$ENVIRONMENT_LOG"
printf 'run_dir=%s\n' "$RUN_DIR" | tee "$SUMMARY_LOG"

run_case()
{
    local case_name="$1" iteration="$2"
    local target="${CASE_TARGET[$case_name]}"
    local executable="$BIN_DIR/$target"
    local arguments_text="${CASE_ARGS[$case_name]}"
    local normal_exit="${CASE_NORMAL_EXIT[$case_name]}"
    local description="${CASE_DESCRIPTION[$case_name]}"
    local environment_text="${CASE_ENVIRONMENT[$case_name]}"
    local -a arguments=()
    local -a case_environment=()
    local log_file="$RUN_DIR/${case_name}_repeat${iteration}.log"
    local run_exit verdict_count pass_count fatal_count capacity_count perf_required perf_count perf_line
    local post_failures sample result assignment

    [[ -n "$arguments_text" ]] && read -r -a arguments <<< "$arguments_text"
    if strings "$executable" 2>/dev/null | grep -q 'PX_DCU_BENCH_SIMULATION_TIMING_V1'; then
        perf_required=1
    else
        perf_required=0
    fi
    if [[ -n "$environment_text" ]]; then
        read -r -a case_environment <<< "$environment_text"
        for assignment in "${case_environment[@]}"; do
            if [[ ! "$assignment" =~ ^[A-Za-z_][A-Za-z0-9_]*=.*$ ]]; then
                printf 'ERROR: invalid environment assignment for case %s: %s\n' "$case_name" "$assignment" >&2
                return 2
            fi
        done
    fi
    check_device_health pre
    device_gate="$health_gate"

    {
        printf 'case=%s\n' "$case_name"
        printf 'runner_marker=%s\n' "$RUNNER_MARKER"
        printf 'batch=%s\n' "${CASE_BATCH[$case_name]}"
        printf 'target=%s\n' "$target"
        printf 'description=%s\n' "$description"
        printf 'case_environment=%s\n' "$environment_text"
        printf 'px_dcu_broadphase_diag=%s\n' "$BROADPHASE_DIAG"
        printf 'px_dcu_pbd_primitive_radix_chunked=%s\n' "$PBD_PRIMITIVE_RADIX_CHUNKED"
        printf 'px_dcu_pbd_primitive_rank_diag=%s\n' "$PBD_PRIMITIVE_RANK_DIAG"
        printf 'performance_required=%s\n' "$perf_required"
        printf 'normal_exit_case=%s\n' "$normal_exit"
        printf 'repeat_index=%s/%s\n' "$iteration" "$REPEAT"
        printf 'target_hip_device=%s\n' "$DEVICE"
        printf 'source_md5='; md5sum "$SCRIPT_DIR/${target}.cpp" | awk '{print $1}'
        printf 'executable_md5='; md5sum "$executable" | awk '{print $1}'
        printf 'physx_gpu_md5='; md5sum "$GPU_LIBRARY" | awk '{print $1}'
        printf 'command='
        printf '%q ' env "HIP_VISIBLE_DEVICES=$DEVICE" "${case_environment[@]}" \
            "PX_DCU_BROADPHASE_DIAG=$BROADPHASE_DIAG" \
            "PX_DCU_PBD_PRIMITIVE_RADIX_CHUNKED=$PBD_PRIMITIVE_RADIX_CHUNKED" \
            "PX_DCU_PBD_PRIMITIVE_RANK_DIAG=$PBD_PRIMITIVE_RANK_DIAG" \
            "$executable" "${arguments[@]}"
        printf '\n\n--- pre-run health ---\n'
        printf '%s\n' "$health_line"
        printf 'device_gate=%s\n' "$device_gate"
    } | tee "$log_file"

    if [[ "$device_gate" -ne 1 ]]; then
        result=DEVICE_UNHEALTHY
        run_exit=125
        verdict_count=0
        pass_count=0
        fatal_count=0
        capacity_count=0
        perf_count=0
        perf_line=
        post_failures=0
    else
        printf '%s\n' '--- benchmark output ---' | tee -a "$log_file"
        timeout --signal=TERM --kill-after=10s "${TIMEOUT_SECONDS}s" \
            env "HIP_VISIBLE_DEVICES=$DEVICE" "${case_environment[@]}" \
            "PX_DCU_BROADPHASE_DIAG=$BROADPHASE_DIAG" \
            "PX_DCU_PBD_PRIMITIVE_RADIX_CHUNKED=$PBD_PRIMITIVE_RADIX_CHUNKED" \
            "PX_DCU_PBD_PRIMITIVE_RANK_DIAG=$PBD_PRIMITIVE_RANK_DIAG" \
            stdbuf -oL -eL "$executable" "${arguments[@]}" 2>&1 | tee -a "$log_file"
        run_exit=${PIPESTATUS[0]}

        verdict_count="$(grep -Ec '^[[:space:]]*VERDICT[[:space:]]*[:=]' "$log_file" || true)"
        pass_count="$(grep -Ec '^[[:space:]]*VERDICT[[:space:]]*[:=][[:space:]]*PASS([[:space:]]|$|\()' "$log_file" || true)"
        fatal_count="$(grep -Eic \
            'Could not find GPU function|VMFault|HSA_STATUS_ERROR|memory aperture|invalid address|segmentation fault|core dumped|dumped core|larger than launch bounds|^[[:space:]]*Killed[[:space:]]*$' \
            "$log_file" || true)"
        capacity_count="$(grep -Eic \
            'increase PxGpuDynamicsMemoryConfig|simulation will miss interactions|capacity[^[:alnum:]]*(is )?(exceeded|overflow)|buffer[^[:alnum:]]*(is )?(exceeded|overflow)' \
            "$log_file" || true)"
        perf_count="$(grep -Ec '^PERF marker=PX_DCU_BENCH_SIMULATION_TIMING_V1 ' "$log_file" || true)"
        if (( perf_count == 1 )); then
            perf_line="$(grep -E '^PERF marker=PX_DCU_BENCH_SIMULATION_TIMING_V1 ' "$log_file")"
        else
            perf_line=
        fi

        post_failures=0
        printf '%s\n' '--- post-run health ---' | tee -a "$log_file"
        for sample in $(seq 1 "$POST_HEALTH_SAMPLES"); do
            check_device_health post
            printf '%s\n' "$health_line" | tee -a "$log_file"
            printf 'health phase=post sample=%s healthy=%s\n' "$sample" "$health_gate" | tee -a "$log_file"
            if [[ "$health_gate" -ne 1 ]]; then
                post_failures=$((post_failures + 1))
            fi
            [[ "$sample" -lt "$POST_HEALTH_SAMPLES" ]] && sleep "$HEALTH_INTERVAL"
        done

        if (( post_failures > 0 )); then
            result=DEVICE_DROPPED
        elif (( capacity_count > 0 )); then
            result=CAPACITY_LIMIT
        elif (( run_exit == 124 )); then
            result=TIMEOUT
        elif (( run_exit != 0 || verdict_count != 1 || pass_count != 1 || fatal_count != 0 || (perf_required == 1 && perf_count != 1) || (perf_required == 0 && perf_count != 0) )); then
            result=FAIL
        else
            result=PASS
        fi
    fi

    {
        printf '\n--- harness result ---\n'
        printf 'run_exit=%s\n' "$run_exit"
        printf 'verdict_count=%s\n' "$verdict_count"
        printf 'pass_verdict_count=%s\n' "$pass_count"
        printf 'fatal_pattern_count=%s\n' "$fatal_count"
        printf 'capacity_pattern_count=%s\n' "$capacity_count"
        printf 'performance_required=%s\n' "$perf_required"
        printf 'performance_line_count=%s\n' "$perf_count"
        printf 'post_health_failures=%s\n' "$post_failures"
        printf 'harness_result=%s\n' "$result"
    } | tee -a "$log_file"

    if [[ -n "$perf_line" ]]; then
        printf 'performance case=%s repeat=%s %s\n' "$case_name" "$iteration" "$perf_line" | tee -a "$SUMMARY_LOG"
    fi
    printf 'case=%s repeat=%s run_exit=%s verdict_count=%s pass_count=%s fatal_count=%s capacity_count=%s perf_required=%s perf_count=%s post_health_failures=%s result=%s log=%s\n' \
        "$case_name" "$iteration" "$run_exit" "$verdict_count" "$pass_count" "$fatal_count" \
        "$capacity_count" "$perf_required" "$perf_count" "$post_failures" "$result" "$log_file" | tee -a "$SUMMARY_LOG"

    [[ "$result" == PASS ]]
}

summarize_case_performance()
{
    local case_name="$1" metric count median
    local -a values=()
    local -a summary_fields=()
    for metric in simulation_ms ms_per_step steps_per_second; do
        mapfile -t values < <(
            awk -v wanted_case="$case_name" -v wanted_metric="$metric" '
                $1 == "performance" {
                    matched = 0
                    value = ""
                    for (i = 2; i <= NF; ++i) {
                        split($i, field, "=")
                        if (field[1] == "case" && field[2] == wanted_case)
                            matched = 1
                        if (field[1] == wanted_metric)
                            value = field[2]
                    }
                    if (matched && value != "")
                        print value
                }
            ' "$SUMMARY_LOG" | LC_ALL=C sort -n
        )
        count="${#values[@]}"
        if (( count == 0 )); then
            continue
        fi
        if (( count % 2 == 1 )); then
            median="${values[count / 2]}"
        else
            median="$(awk -v a="${values[count / 2 - 1]}" -v b="${values[count / 2]}" 'BEGIN { printf "%.6f", (a + b) / 2.0 }')"
        fi
        summary_fields+=("${metric}_min=${values[0]}" "${metric}_median=$median" "${metric}_max=${values[count - 1]}")
    done
    if (( ${#summary_fields[@]} > 0 )); then
        printf 'performance_summary case=%s samples=%s %s\n' \
            "$case_name" "$count" "${summary_fields[*]}" | tee -a "$SUMMARY_LOG"
    fi
}

for case_name in "${SELECTED_CASES[@]}"; do
    for ((iteration = 1; iteration <= REPEAT; ++iteration)); do
        printf '\n=== %s repeat %d/%d ===\n' "$case_name" "$iteration" "$REPEAT" | tee -a "$SUMMARY_LOG"
        if ! run_case "$case_name" "$iteration"; then
            printf 'STOP: first non-PASS result at case=%s repeat=%d\n' "$case_name" "$iteration" | tee -a "$SUMMARY_LOG"
            printf 'Logs: %s\n' "$RUN_DIR"
            exit 1
        fi
        if [[ "$iteration" -lt "$REPEAT" ]]; then
            sleep "$REPEAT_DELAY"
        fi
    done
    summarize_case_performance "$case_name"
done

printf '\nALL_SELECTED_CASES_PASS=1\n' | tee -a "$SUMMARY_LOG"
printf 'Logs: %s\n' "$RUN_DIR"
exit 0
