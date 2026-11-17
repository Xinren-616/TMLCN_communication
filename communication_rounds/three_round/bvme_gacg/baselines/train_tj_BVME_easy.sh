#!/bin/bash
export OMP_NUM_THREADS=1

python -u run_baselines.py \
  --env_name traffic_junction \
  --nagents 5 \
  --dim 6 \
  --max_steps 20 \
  --add_rate_min 0.3 \
  --add_rate_max 0.3 \
  --difficulty easy \
  --vision 1 \
  --nprocesses 16 \
  --num_epochs 2000 \
  --epoch_size 10 \
  --hid_size 128 \
  --comm_passes 3 \
  --detach_gap 10 \
  --lrate 0.001 \
  --value_coeff 0.01 \
  --bvme \
  --ic3net \
  --recurrent \
  --bvme_kl_weight 0.1 \
  --bvme_sigma0 0.1 \
  --bvme_logvar_min -5 \
  --bvme_logvar_max 3 \
  --curr_start 0 \
  --curr_end 0 \
  --save \
  --seed 2 \
  | tee train_tj_three_round_BVME_easy3.log
