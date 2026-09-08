#!/usr/bin/env bash
# Run GPU-enabled PhysX convex benches side-by-side and print a comparable summary.
# Usage:
#   cd /public/home/tangwsh/PhysX/examples/dcu
#   bash run_convex_gpu_equiv.sh [repeat]
# Env:
#   OUTDIR=./logs_convex_gpu   # output directory
#   ALLOW_CPU=1                # do not fail if bench reports CPU only

set -u

REPEAT="${1:-1}"
OUTDIR="${OUTDIR:-./logs_convex_gpu}"
ALLOW_CPU="${ALLOW_CPU:-0}"
mkdir -p "${OUTDIR}"

if ! [[ "${REPEAT}" =~ ^[0-9]+$ ]] || [ "${REPEAT}" -lt 1 ]; then
  echo "ERROR: repeat must be a positive integer, got '${REPEAT}'" >&2
  exit 2
fi

BENCHES=(
  "bench_convex"
  "bench_convexmesh"
)

for b in "${BENCHES[@]}"; do
  if [ ! -x "./build/${b}" ]; then
    echo "ERROR: missing executable ./build/${b}" >&2
    echo "Build first, e.g.: cmake --build ./build -j1 --target ${b}" >&2
    exit 3
  fi
done

CSV="${OUTDIR}/convex_gpu_equiv_summary.csv"
TXT="${OUTDIR}/convex_gpu_equiv_summary.txt"
: > "${TXT}"
echo "bench,run,status,gpu_line,avg_ms,fps,moving,total,min_y,max_y,log" > "${CSV}"

run_one() {
  local bench="$1"
  local idx="$2"
  local log="${OUTDIR}/${bench}_run${idx}.log"

  echo "============================================================" | tee -a "${TXT}"
  echo "RUN ${idx}: ${bench}" | tee -a "${TXT}"
  echo "LOG: ${log}" | tee -a "${TXT}"
  echo "============================================================" | tee -a "${TXT}"

  set +e
  "./build/${bench}" > "${log}" 2>&1
  local rc=$?
  set -e

  local gpu_line avg_line moving_line height_line pass_line
  gpu_line=$(grep -m1 '^GPU:' "${log}" || true)
  avg_line=$(grep -m1 'Avg frame time:' "${log}" || true)
  moving_line=$(grep -m1 'Moving bodies:' "${log}" || true)
  height_line=$(grep -m1 'Height range:' "${log}" || true)
  pass_line=$(grep -m1 'PASSED' "${log}" || true)

  local avg_ms fps moving total min_y max_y status
  avg_ms=$(echo "${avg_line}" | sed -nE 's/.*Avg frame time:[[:space:]]*([0-9.]+)[[:space:]]*ms.*/\1/p')
  fps=$(echo "${avg_line}" | sed -nE 's/.*\(([0-9.]+)[[:space:]]*FPS\).*/\1/p')
  moving=$(echo "${moving_line}" | sed -nE 's/.*Moving bodies:[[:space:]]*([0-9]+)[[:space:]]*\/[[:space:]]*([0-9]+).*/\1/p')
  total=$(echo "${moving_line}" | sed -nE 's/.*Moving bodies:[[:space:]]*([0-9]+)[[:space:]]*\/[[:space:]]*([0-9]+).*/\2/p')
  min_y=$(echo "${height_line}" | sed -nE 's/.*Height range:[[:space:]]*\[([-0-9.]+),[[:space:]]*([-0-9.]+)\].*/\1/p')
  max_y=$(echo "${height_line}" | sed -nE 's/.*Height range:[[:space:]]*\[([-0-9.]+),[[:space:]]*([-0-9.]+)\].*/\2/p')

  status="PASS"
  if [ "${rc}" -ne 0 ]; then
    status="FAIL_RC_${rc}"
  elif [ -z "${pass_line}" ]; then
    status="FAIL_NO_PASS_LINE"
  elif echo "${gpu_line}" | grep -qi 'CPU only\|NONE'; then
    if [ "${ALLOW_CPU}" = "1" ]; then
      status="PASS_CPU_ONLY_ALLOWED"
    else
      status="FAIL_GPU_DISABLED"
    fi
  fi

  {
    echo "status: ${status}"
    echo "${gpu_line:-GPU: <missing>}"
    echo "${avg_line:-Avg frame time: <missing>}"
    echo "${moving_line:-Moving bodies: <missing>}"
    echo "${height_line:-Height range: <missing>}"
    echo
  } | tee -a "${TXT}"

  # CSV quote gpu_line/log because GPU name may contain spaces or comma-like chars.
  printf '%s,%s,%s,"%s",%s,%s,%s,%s,%s,%s,"%s"\n' \
    "${bench}" "${idx}" "${status}" "${gpu_line//"/""}" \
    "${avg_ms}" "${fps}" "${moving}" "${total}" "${min_y}" "${max_y}" "${log}" >> "${CSV}"

  if [[ "${status}" == FAIL* ]]; then
    echo "ERROR: ${bench} run ${idx} failed: ${status}" >&2
    echo "Last 40 log lines:" >&2
    tail -40 "${log}" >&2 || true
    return 1
  fi
  return 0
}

FAILED=0
for i in $(seq 1 "${REPEAT}"); do
  for b in "${BENCHES[@]}"; do
    if ! run_one "${b}" "${i}"; then
      FAILED=1
      break
    fi
  done
  if [ "${FAILED}" -ne 0 ]; then
    break
  fi
done

echo "Summary written:" | tee -a "${TXT}"
echo "  ${TXT}" | tee -a "${TXT}"
echo "  ${CSV}" | tee -a "${TXT}"

# Optional compact table if column exists.
if command -v column >/dev/null 2>&1; then
  echo
  column -s, -t "${CSV}" | tee -a "${TXT}"
else
  echo
  cat "${CSV}" | tee -a "${TXT}"
fi

exit "${FAILED}"
