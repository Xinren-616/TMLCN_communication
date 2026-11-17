#!/bin/bash

# Traffic Junction Easy - 单进程模式（启用 Epoch 级别 CI 计算）
# 用于详细分析和理论研究

python run_baselines.py \
  --env_name traffic_junction \
  --nagents 5 \
  --nprocesses 1 \
  --num_epochs 10 \
  --epoch_size 10 \
  --batch_size 500 \
  --hid_size 128 \
  --recurrent \
  --rnn_type LSTM \
  --detach_gap 10 \
  --comm_mode avg \
  --comm_passes 1 \
  --commnet \
  --vision 1 \
  --dim 14 \
  --max_steps 40 \
  --lrate 0.001 \
  --seed 0 \
  2>&1 | tee train_tj_easy_epoch_ci_$(date +%m%d).log

# 说明：
# - nprocesses 1: 单进程模式，启用 Epoch CI 计算
# - num_epochs 10: 少量 epochs 用于测试（正式训练用 500）
# - 会同时输出瞬时 CI 和 Epoch 级别 CI (KL Divergence)



