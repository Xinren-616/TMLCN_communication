#!/bin/bash

# Edge-Level Communication Shapley Value Analysis for Baseline Models
# 边级别通信Shapley值分析示例脚本 - Baseline模型版本

# ============================================================================
# 示例1: 分析 CommNet 模型 (Traffic Junction)
# ============================================================================
echo "示例1: CommNet - Traffic Junction Easy"
python edge_level_cshav_baselines.py \
  --checkpoint_dir ./saved/traffic_junction/commnet/run1 \
  --output_dir ./edge_cshav_results/tj_commnet \
  --model_type commnet \
  --env_name traffic_junction \
  --nagents 5 \
  --difficulty easy \
  --vision 0 \
  --add_rate_min 0.3 \
  --add_rate_max 0.3 \
  --hid_size 128 \
  --max_steps 20 \
  --n_episodes 20 \
  --topology_threshold 0.1 \
  --checkpoint_indices 0 4 9  # 分析epoch 100, 500, 1000

# ============================================================================
# 示例2: 分析 IC3Net 模型 (Traffic Junction)
# ============================================================================
echo ""
echo "示例2: IC3Net - Traffic Junction Easy"
python edge_level_cshav_baselines.py \
  --checkpoint_dir ./saved/traffic_junction/ic3net/run1 \
  --output_dir ./edge_cshav_results/tj_ic3net \
  --model_type ic3net \
  --env_name traffic_junction \
  --nagents 5 \
  --difficulty easy \
  --vision 0 \
  --add_rate_min 0.3 \
  --add_rate_max 0.3 \
  --hid_size 128 \
  --max_steps 20 \
  --n_episodes 20 \
  --topology_threshold 0.1 \
  --hard_attn

# ============================================================================
# 示例3: 分析 TarMAC 模型 (Traffic Junction)
# ============================================================================
echo ""
echo "示例3: TarMAC - Traffic Junction Easy"
python edge_level_cshav_baselines.py \
  --checkpoint_dir ./saved/traffic_junction/tar_commnet/run1 \
  --output_dir ./edge_cshav_results/tj_tarmac \
  --model_type tarmac \
  --env_name traffic_junction \
  --nagents 5 \
  --difficulty easy \
  --vision 0 \
  --add_rate_min 0.3 \
  --add_rate_max 0.3 \
  --hid_size 128 \
  --qk_hid_size 16 \
  --value_hid_size 32 \
  --max_steps 20 \
  --n_episodes 20 \
  --topology_threshold 0.1

# ============================================================================
# 示例4: 分析 GAComm 模型 (Predator-Prey)
# ============================================================================
echo ""
echo "示例4: GAComm - Predator-Prey Medium"
python edge_level_cshav_baselines.py \
  --checkpoint_dir ./saved/predator_prey/gacomm/run1 \
  --output_dir ./edge_cshav_results/pp_gacomm \
  --model_type gacomm \
  --env_name predator_prey \
  --nagents 5 \
  --nfriendly 5 \
  --dim 10 \
  --max_steps 40 \
  --vision 1 \
  --hid_size 128 \
  --n_episodes 20 \
  --topology_threshold 0.1 \
  --checkpoint_indices 0 2 4  # 分析epoch 100, 300, 500

# ============================================================================
# 高精度分析示例（更多episodes，更低阈值）
# ============================================================================
echo ""
echo "高精度分析示例: IC3Net - Traffic Junction"
python edge_level_cshav_baselines.py \
  --checkpoint_dir ./saved/traffic_junction/ic3net/run1 \
  --output_dir ./edge_cshav_results/tj_ic3net_highprecision \
  --model_type ic3net \
  --env_name traffic_junction \
  --nagents 5 \
  --difficulty easy \
  --vision 0 \
  --add_rate_min 0.3 \
  --add_rate_max 0.3 \
  --hid_size 128 \
  --max_steps 20 \
  --n_episodes 30 \
  --topology_threshold 0.05 \
  --hard_attn

echo ""
echo "所有分析完成！"
echo "结果保存在 ./edge_cshav_results/ 目录下"

