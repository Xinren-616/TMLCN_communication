#!/bin/bash

echo "======================================================================"
echo "可用的训练脚本"
echo "======================================================================"
echo ""

echo "【原有脚本 - 不带剪枝】"
echo "----------------------------------------------------------------------"
for model in commnet ic3net gacomm tarcomm magic; do
    script="train_tj_${model}_online_shapley.sh"
    if [ -f "$script" ]; then
        echo "  ✓ bash $script"
    fi
done
echo ""

echo "【新脚本 - 带基础剪枝】"
echo "----------------------------------------------------------------------"
for model in commnet ic3net gacomm tarcomm magic; do
    script="train_tj_${model}_basic_prune_online_shapley.sh"
    if [ -f "$script" ]; then
        echo "  ✓ bash $script"
    fi
done
echo ""

echo "======================================================================"
echo "功能对比"
echo "======================================================================"
echo ""

cat << 'EOF'
┌────────────────────────────────┬──────────────┬──────────────────┐
│ 功能                           │ 原有脚本     │ 新剪枝脚本       │
├────────────────────────────────┼──────────────┼──────────────────┤
│ 在线Shapley值计算              │ ✓            │ ✓                │
│ 基于Shapley值的拓扑剪枝        │ ✗            │ ✓                │
│ 通信统计（总次数、密度）       │ ✗            │ ✓                │
│ TEI指标计算                    │ ✗            │ ✓                │
│ 剪枝历史记录                   │ ✗            │ ✓                │
│ 向后兼容性                     │ -            │ ✓ 完全兼容       │
└────────────────────────────────┴──────────────┴──────────────────┘

配置参数差异：

原有脚本：
  --enable_shapley              # 启用Shapley值计算
  --shapley_interval 100        # 每100 epoch计算一次

新剪枝脚本（额外参数）：
  --enable_pruning              # 启用拓扑剪枝
  --pruning_threshold -0.3      # Shapley值阈值
  --pruning_start_epoch 500     # Warm-up期
  --pruning_interval 100        # 更新频率
  --enable_comm_stats           # 启用通信统计
EOF

echo ""
echo "======================================================================"
echo "使用建议"
echo "======================================================================"
echo ""
echo "1. 验证原功能：先运行原有脚本，确保基础功能正常"
echo "2. 测试剪枝：运行新剪枝脚本，观察剪枝效果"
echo "3. 参数调优：根据实验结果调整threshold和warm-up期"
echo ""
echo "更多信息请查看："
echo "  - PRUNING_README.md         # 详细使用文档"
echo "  - IMPLEMENTATION_SUMMARY.md # 实现总结"
echo ""








