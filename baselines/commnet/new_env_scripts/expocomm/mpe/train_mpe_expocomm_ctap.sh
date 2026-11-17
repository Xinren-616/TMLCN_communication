#!/bin/bash
# ExpoComm + CTAP on MPE simple_spread (3 agents)
export OMP_NUM_THREADS=1

DATE=$(date +%Y%m%d)
SAVE_DIR=./saved/mpe_expocomm_ctap_seed1_${DATE}
LOG_FILE=./new_env_scripts/expocomm/mpe/train_mpe_expocomm_ctap_seed1_${DATE}.log

python -u baselines/train_with_online_shapley.py \
  --env_name mpe \
  --mpe_scenario simple_spread \
  --nagents 3 \
  --max_steps 25 \
  --model_type expocomm \
  --nprocesses 1 \
  --num_epochs 5000 \
  --epoch_size 10 \
  --hid_size 128 \
  --lrate 0.0005 \
  --value_coeff 0.01 \
  --gamma 0.95 \
  --normalize_rewards \
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
