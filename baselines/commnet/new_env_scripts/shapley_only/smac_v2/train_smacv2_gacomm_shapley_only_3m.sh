#!/bin/bash
# gacomm + Shapley Only on SMAC v2 (3m: 3 marines vs 3 marines, easy)
export OMP_NUM_THREADS=1

DATE=$(date +%Y%m%d)
SAVE_DIR=./saved/smacv2_gacomm_shapley_only_3m_seed1_${DATE}
LOG_FILE=./new_env_scripts/shapley_only/smac_v2/train_smacv2_gacomm_shapley_only_3m_seed1_${DATE}.log

python -u baselines/train_with_shapley_only.py \
  --env_name smac_v2 \
  --smac_map 3m \
  --nagents 3 \
  --model_type gacomm \
  --nprocesses 1 \
  --num_epochs 2000 \
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
