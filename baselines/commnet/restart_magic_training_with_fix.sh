#!/bin/bash

# 重新开始MAGIC训练（应用Epoch 1160崩溃修复）
# 修复日期: 2026-01-15
# 修复内容: LSTM隐藏状态的计算图累积问题

echo "=========================================="
echo "重新开始MAGIC训练（应用修复）"
echo "=========================================="
echo ""
echo "修复内容:"
echo "  1. 动态设置LSTM隐藏状态的requires_grad"
echo "  2. 显式清理隐藏状态，防止计算图累积"
echo "  3. 强制清理CUDA缓存和Python垃圾"
echo ""
echo "预期效果:"
echo "  - Epoch 1160不再崩溃"
echo "  - Success rate保持稳定（0.95+）"
echo "  - 训练顺利进行到Epoch 2000"
echo ""
echo "=========================================="
echo ""

# 生成新的日志文件名（带日期和修复标记）
DATE=$(date +%Y%m%d)
LOG_FILE="train_tj_magic_one_round_online_shapley_${DATE}_fixed.log"

echo "日志文件: $LOG_FILE"
echo ""

# 确认是否继续
read -p "是否开始训练？(y/n) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]
then
    echo "训练已取消"
    exit 1
fi

echo ""
echo "开始训练（日志将同时保存到文件和显示在屏幕上）..."
echo "日志文件: $LOG_FILE"
echo ""
echo "关键检查点:"
echo "  - Epoch 1100: Shapley计算"
echo "  - Epoch 1160: 之前的崩溃点（现在应该正常）"
echo "  - Epoch 1200: Shapley计算"
echo ""
echo "=========================================="
echo ""

# 运行训练（前台运行，使用tee同时输出到屏幕和文件）
python -u baselines/train_with_online_shapley.py \
  --env_name traffic_junction \
  --nagents 5 \
  --model_type magic \
  --dim 6 \
  --max_steps 20 \
  --add_rate_min 0.3 \
  --add_rate_max 0.3 \
  --difficulty easy \
  --vision 1 \
  --nprocesses 16 \
  --num_epochs 2000 \
  --epoch_size 10 \
  --hid_size 128 \
  --lrate 0.001 \
  --value_coeff 0.01 \
  --gamma 1.0 \
  --normalize_rewards \
  --seed 0 \
  --save_dir ./saved/tj_magic_one_round_online_shapley_${DATE}_fixed \
  --save_every 100 \
  --shapley_interval 100 \
  --shapley_episodes 10 \
  --topology_threshold 0.1 \
  --enable_shapley \
  2>&1 | tee "$LOG_FILE"

echo ""
echo "=========================================="
echo "训练完成或中断"
echo "日志已保存到: $LOG_FILE"
echo "=========================================="

