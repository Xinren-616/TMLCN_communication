#!/bin/bash
# MAPPO (no communication) on SMACv2 10gen_terran 5v5
export OMP_NUM_THREADS=1

DATE=$(date +%Y%m%d)
SAVE_DIR=./saved/smacv2_mappo_no_comm_5v5_seed1_${DATE}
LOG_FILE=./new_env_scripts/mappo/smac_v2/train_smacv2_mappo_no_comm_5v5_seed1_${DATE}.log

python -u baselines/train_with_shapley_only.py \
  --env_name smac_v2 \
  --smac_map 10gen_terran \
  --nagents 5 \
  --max_steps 150 \
  --model_type mappo \
  --comm_mask_zero \
  --nprocesses 1 \
  --num_epochs 10000 \
  --epoch_size 10 \
  --hid_size 128 \
  --lrate 0.0005 \
  --value_coeff 0.01 \
  --gamma 0.99 \
  --normalize_rewards \
  --seed 1 \
  --save_dir ${SAVE_DIR} \
  --save_every 100 \
  --shapley_interval 100 \
  --shapley_episodes 20 \
  --topology_threshold 0.1 \
  --cuda \
  2>&1 | tee ${LOG_FILE}
