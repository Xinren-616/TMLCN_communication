#!/bin/bash
# CommNet + Shapley Only (no pruning) on MPE simple_spread (3 agents)
export OMP_NUM_THREADS=1

DATE=$(date +%Y%m%d)
SAVE_DIR=./saved/mpe_commnet_shapley_only_seed1_${DATE}
LOG_FILE=./new_env_scripts/shapley_only/mpe/train_mpe_commnet_shapley_only_seed1_${DATE}.log

python -u baselines/train_with_shapley_only.py \
  --env_name mpe \
  --mpe_scenario simple_spread \
  --nagents 3 \
  --max_steps 25 \
  --model_type commnet \
  --nprocesses 1 \
  --num_epochs 100000 \
  --epoch_size 10 \
  --hid_size 128 \
  --lrate 0.0005 \
  --value_coeff 0.01 \
  --gamma 0.95 \
  --normalize_rewards \
  --seed 1 \
  --save_dir ${SAVE_DIR} \
  --save_every 100 \
  --shapley_interval 100 \
  --shapley_episodes 20 \
  --topology_threshold 0.1 \
  --cuda \
  2>&1 | tee ${LOG_FILE}
