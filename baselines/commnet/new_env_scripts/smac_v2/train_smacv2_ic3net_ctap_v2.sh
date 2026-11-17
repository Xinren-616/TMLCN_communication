#!/bin/bash
# IC3Net + CTAP v2 on SMAC v2 (10gen_terran, 10 agents)
export OMP_NUM_THREADS=1

DATE=$(date +%Y%m%d)
SAVE_DIR=./saved/smacv2_ic3net_ctap_v2_seed1_${DATE}
LOG_FILE=./new_env_scripts/smac_v2/train_smacv2_ic3net_ctap_v2_seed1_${DATE}.log

python -u baselines/train_with_online_shapley.py \
  --env_name smac_v2 \
  --smac_map 10gen_terran \
  --smac_difficulty 7 \
  --nagents 10 \
  --model_type ic3net \
  --nprocesses 1 \
  --num_epochs 100000 \
  --epoch_size 10 \
  --hid_size 128 \
  --lrate 0.0005 \
  --value_coeff 0.01 \
  --gamma 0.99 \
  --normalize_rewards \
  --hard_attn \
  --seed 1 \
  --save_dir ${SAVE_DIR} \
  --save_every 100 \
  --enable_shapley \
  --shapley_interval 100 \
  --shapley_episodes 20 \
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
