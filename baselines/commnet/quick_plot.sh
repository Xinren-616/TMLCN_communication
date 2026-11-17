#!/bin/bash

# 快速生成核心图表
# 用于初步查看实验结果

echo "=========================================="
echo "快速生成核心图表"
echo "=========================================="
echo ""

# 默认日期
DATE=${1:-20260122}

echo "使用日期: $DATE"
echo ""

# 检查日志文件是否存在
echo "检查日志文件..."
for model in commnet ic3net tarcomm magic gacomm; do
    log_file="train_tj_${model}_prune_online_shapley_${DATE}.log"
    if [ -f "$log_file" ]; then
        echo "  ✓ $log_file"
    else
        echo "  ✗ $log_file (不存在)"
    fi
done
echo ""

# 生成核心图表
echo "=========================================="
echo "[1/2] 生成训练性能对比图..."
echo "=========================================="
python plot_training_comparison.py --date $DATE --format both
echo ""

echo "=========================================="
echo "[2/2] 生成TEI指标分析图..."
echo "=========================================="
python plot_tei_metrics.py --date $DATE --format both
echo ""

echo "=========================================="
echo "完成！"
echo "=========================================="
echo ""
echo "查看结果:"
echo "  cd figures/"
echo "  ls -lh training_comparison/"
echo "  ls -lh tei_metrics/"
echo ""






