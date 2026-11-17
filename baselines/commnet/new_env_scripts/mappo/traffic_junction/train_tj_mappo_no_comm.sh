#!/bin/bash
# MAPPO (no communication) on Traffic Junction easy (5 agents)
export OMP_NUM_THREADS=1

DATE=$(date +%Y%m%d)
SAVE_DIR=./saved/tj_mappo_no_comm_seed1_${DATE}
LOG_FILE=./new_env_scripts/mappo/traffic_junction/train_tj_mappo_no_comm_seed1_${DATE}.log

python -u baselines/train_with_shapley_only.py \
  --env_name traffic_junction \
  --nagents 5 \
  --max_steps 20 \
  --dim 6 \
  --add_rate_min 0.3 \
  --add_rate_max 0.3 \
  --difficulty easy \
  --vision 1 \
  --model_type mappo \
  --comm_mask_zero \
  --nprocesses 16 \
  --num_epochs 2000 \
  --epoch_size 10 \
  --hid_size 128 \
  --lrate 0.001 \
  --value_coeff 0.01 \
  --gamma 1.0 \
  --normalize_rewards \
  --seed 1 \
  --save_dir ${SAVE_DIR} \
  --save_every 100 \
  --shapley_interval 100 \
  --shapley_episodes 20 \
  --topology_threshold 0.1 \
  2>&1 | tee ${LOG_FILE}
