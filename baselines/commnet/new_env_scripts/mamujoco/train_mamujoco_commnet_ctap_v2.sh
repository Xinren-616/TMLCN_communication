#!/bin/bash
# CommNet + CTAP v2 on MAMuJoCo (Ant-v2, 2x4 partition => 2 agents)
# MAMuJoCo 使用连续动作空间，nprocesses=1
export OMP_NUM_THREADS=1

DATE=$(date +%Y%m%d)
SAVE_DIR=./saved/mamujoco_commnet_ctap_v2_seed1_${DATE}
LOG_FILE=./new_env_scripts/mamujoco/train_mamujoco_commnet_ctap_v2_seed1_${DATE}.log

python -u baselines/train_with_online_shapley.py \
  --env_name mamujoco \
  --mamujoco_scenario Ant-v2 \
  --mamujoco_agent_conf 2x4 \
  --nagents 2 \
  --max_steps 1000 \
  --model_type commnet \
  --nprocesses 1 \
  --num_epochs 2000 \
  --epoch_size 10 \
  --hid_size 128 \
  --lrate 0.0003 \
  --value_coeff 0.5 \
  --gamma 0.99 \
  --normalize_rewards \
  --seed 1 \
  --save_dir ${SAVE_DIR} \
  --save_every 100 \
  --enable_shapley \
  --shapley_interval 100 \
  --shapley_episodes 10 \
  --topology_threshold 0.1 \
  --enable_ctap \
  --ctap_mode full \
  --pruning_start_epoch 300 \
  --pruning_interval 100 \
  --confidence_window 3 \
  --confidence_prob 0.6 \
  --initial_threshold 0.0 \
  --min_threshold -0.5 \
  --max_threshold 0.3 \
  --tei_learning_rate 0.05 \
  --tei_patience 2 \
  --enable_comm_stats \
  --cuda \
  2>&1 | tee ${LOG_FILE}
