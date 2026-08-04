#!/bin/bash
# run_all_benchmarks.sh - 运行所有 benchmark 可执行文件并记录日志

set -e

# 可执行文件目录
BIN_DIR="./build"
# 日志目录
LOG_DIR="./logs"

# 创建日志目录
mkdir -p "$LOG_DIR"

# 获取当前时间戳
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
# 主日志文件
MAIN_LOG="${LOG_DIR}/all_benchmarks.log"

echo "=========================================="
echo "Running All Benchmarks"
echo "=========================================="
echo "Log directory: ${LOG_DIR}"
echo "Main log: ${MAIN_LOG}"
echo ""

# 写入日志头
{
    echo "=========================================="
    echo "Benchmark Run: $(date)"
    echo "=========================================="
    echo ""
} > "$MAIN_LOG"

# 计数器
TOTAL=0
SUCCESS=0
FAILED=0

# 查找所有可执行文件（递归查找）
EXECUTABLES=$(find "$BIN_DIR" -type f -executable ! -name "*.so" ! -name "*.a" 2>/dev/null | sort)

if [ -z "$EXECUTABLES" ]; then
    echo "❌ No executable files found in ${BIN_DIR}"
    exit 1
fi

echo "Found executables:"
echo "$EXECUTABLES" | sed 's/^/  /'
echo ""

# 遍历执行
for exe in $EXECUTABLES; do
    
    # 获取文件名（不含路径）
    exe_name=$(basename "$exe")
    # 单独日志文件
    single_log="${LOG_DIR}/${exe_name}_${TIMESTAMP}.log"
    
    echo ">>> [${TOTAL}] Running ${exe_name}..."
    echo "    Log: ${single_log}"
    
    # 写入主日志
    {
        echo "=========================================="
        echo ">>> Running: ${exe_name}"
        echo "    Time: $(date)"
        echo "    Command: ${exe}"
        echo "=========================================="
    } >> "$MAIN_LOG"
    
    # 执行并记录输出
    {
        echo "=========================================="
        echo "Benchmark: ${exe_name}"
        echo "Started at: $(date)"
        echo "=========================================="
        echo ""
        
        # 运行可执行文件，捕获 stdout 和 stderr
        if "$exe" 2>&1; then
            EXIT_CODE=$?
            echo ""
            echo "=========================================="
            echo "Finished at: $(date)"
            echo "Exit code: ${EXIT_CODE} ✅ SUCCESS"
            echo "=========================================="
        else
            EXIT_CODE=$?
            echo ""
            echo "=========================================="
            echo "Finished at: $(date)"
            echo "Exit code: ${EXIT_CODE} ❌ FAILED"
            echo "=========================================="
        fi
    } 2>&1 | tee "$single_log"
    
    # 追加到主日志
    {
        echo ""
        cat "$single_log"
        echo ""
        echo "------------------------------------------"
        echo ""
    } >> "$MAIN_LOG"
    
    echo ""
done

# 写入汇总信息
{
    echo "=========================================="
    echo "SUMMARY"
    echo "=========================================="
    echo "Total:   ${TOTAL}"
    echo "Success: ${SUCCESS}"
    echo "Failed:  ${FAILED}"
    echo ""
    echo "All logs saved to: ${LOG_DIR}"
    echo "Main log: ${MAIN_LOG}"
    echo "=========================================="
} | tee -a "$MAIN_LOG"

echo ""
echo "=========================================="
echo "✅ All benchmarks completed!"
echo "  Total:   ${TOTAL}"
echo "  Success: ${SUCCESS}"
echo "  Failed:  ${FAILED}"
echo "=========================================="
echo "Logs saved to: ${LOG_DIR}/"
echo "Main log: ${MAIN_LOG}"
