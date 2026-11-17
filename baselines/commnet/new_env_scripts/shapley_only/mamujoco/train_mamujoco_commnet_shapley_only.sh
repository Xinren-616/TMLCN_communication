#!/bin/bash
# commnet + Shapley Only (no pruning) on MAMuJoCo (Ant-v2, 2x4 => 2 agents)
export OMP_NUM_THREADS=1

DATE=$(date +%Y%m%d)
SAVE_DIR=./saved/mamujoco_commnet_shapley_only_seed1_${DATE}
LOG_FILE=./new_env_scripts/shapley_only/mamujoco/train_mamujoco_commnet_shapley_only_seed1_${DATE}.log

python -u baselines/train_with_shapley_only.py \
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
  --shapley_interval 100 \
  --shapley_episodes 10 \
  --topology_threshold 0.1 \
  2>&1 | tee ${LOG_FILE}
