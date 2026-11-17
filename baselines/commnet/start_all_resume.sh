#!/bin/bash

# 启动所有模型的resume训练脚本
# 这个脚本会启动ic3net和tarcomm的继续训练

echo "=================================================="
echo "开始启动所有模型的resume训练"
echo "=================================================="

# 进入工作目录
cd /home/zxr/Shapley_Value_Online/MAGIC-main_one_round/MAGIC-main_download

# 1. 启动IC3Net resume训练 (从epoch 1100继续)
echo ""
echo "正在启动 IC3Net resume训练..."
bash resume_tj_ic3net_online_shapley.sh &
IC3NET_PID=$!
echo "IC3Net 训练已启动 (PID: $IC3NET_PID)"

# 等待几秒避免同时启动太多进程
sleep 5

# 2. 启动TarMAC resume训练 (从epoch 700继续)
echo ""
echo "正在启动 TarMAC resume训练..."
bash resume_tj_tarcomm_online_shapley.sh &
TARCOMM_PID=$!
echo "TarMAC 训练已启动 (PID: $TARCOMM_PID)"

echo ""
echo "=================================================="
echo "所有resume训练已启动！"
echo "=================================================="
echo "IC3Net PID: $IC3NET_PID"
echo "TarMAC PID: $TARCOMM_PID"
echo ""
echo "查看训练日志："
echo "  IC3Net:  tail -f train_tj_ic3net_one_round_online_shapley_20260109_resume.log"
echo "  TarMAC:  tail -f train_tj_tarcomm_one_round_online_shapley_20260109_resume.log"
echo ""
echo "提示: 训练进程在后台运行。"
echo "=================================================="




