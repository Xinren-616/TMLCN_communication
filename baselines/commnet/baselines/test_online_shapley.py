"""
测试脚本 - 验证在线Shapley值计算功能

这个脚本用于快速测试在线Shapley值计算模块是否正常工作。
运行一个小规模的训练任务，在第100个epoch计算Shapley值。

使用方法：
python test_online_shapley.py
"""

import sys
import argparse
from pathlib import Path
import numpy as np
import torch
import os

# 添加父目录到path（支持从任何目录运行）
current_dir = Path(__file__).resolve().parent
parent_dir = current_dir.parent
sys.path.insert(0, str(parent_dir))

# 导入baseline模型
from comm import CommNetMLP

# 导入训练相关模块
import data
from utils import *
from action_utils import parse_action_args
from trainer import Trainer

# 导入在线Shapley值分析器
from online_edge_shapley import OnlineEdgeShapleyAnalyzer, integrate_with_training


def create_test_args():
    """创建测试用的参数"""
    parser = argparse.ArgumentParser()
    
    # 基本参数
    parser.add_argument('--env_name', type=str, default='traffic_junction')
    parser.add_argument('--nagents', type=int, default=5)
    parser.add_argument('--num_epochs', type=int, default=150)
    parser.add_argument('--epoch_size', type=int, default=5)
    parser.add_argument('--batch_size', type=int, default=500)
    parser.add_argument('--nprocesses', type=int, default=1)
    
    # 模型参数
    parser.add_argument('--hid_size', type=int, default=64)
    parser.add_argument('--recurrent', action='store_true', default=False)
    parser.add_argument('--rnn_type', default='MLP', type=str)
    
    # CommNet参数
    parser.add_argument('--commnet', action='store_true', default=True)
    parser.add_argument('--ic3net', action='store_true', default=False)
    parser.add_argument('--hard_attn', default=False, action='store_true')
    parser.add_argument('--comm_passes', type=int, default=1)
    parser.add_argument('--comm_mask_zero', action='store_true', default=False)
    parser.add_argument('--comm_mode', type=str, default='avg')
    parser.add_argument('--comm_init', default='uniform', type=str)
    parser.add_argument('--share_weights', default=False, action='store_true')
    
    # 优化参数
    parser.add_argument('--lrate', type=float, default=0.001)
    parser.add_argument('--gamma', type=float, default=1.0)
    parser.add_argument('--value_coeff', type=float, default=0.01)
    parser.add_argument('--entr', type=float, default=0)
    parser.add_argument('--normalize_rewards', action='store_true', default=False)
    
    # 环境参数
    parser.add_argument('--max_steps', type=int, default=40)
    parser.add_argument('--nactions', default='1', type=str)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--mean_ratio', default=1.0, type=float)
    parser.add_argument('--detach_gap', default=10000, type=int)
    
    # 从utils导入环境特定参数
    from utils import init_args_for_env
    init_args_for_env(parser)
    
    args = parser.parse_args([])  # 使用默认参数
    return args


def test_online_shapley():
    """测试在线Shapley值计算"""
    print("="*80)
    print("在线Shapley值计算 - Phase 1 测试")
    print("="*80)
    print("这个测试将:")
    print("  1. 训练一个小型CommNet模型")
    print("  2. 在第100个epoch计算Shapley值")
    print("  3. 验证所有功能是否正常工作")
    print("="*80 + "\n")
    
    # 创建参数
    args = create_test_args()
    
    # 设置随机种子
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    # 初始化环境
    print(f"[1/6] 初始化环境: {args.env_name}")
    env = data.init(args.env_name, args, False)
    
    # 设置环境参数
    args.obs_size = env.observation_dim
    args.num_actions = env.num_actions
    if not isinstance(args.num_actions, (list, tuple)):
        args.num_actions = [args.num_actions]
    args.dim_actions = env.dim_actions
    args.nfriendly = args.nagents
    args.num_inputs = env.observation_dim
    
    parse_action_args(args)
    
    # 创建模型
    print(f"[2/6] 创建CommNet模型")
    policy_net = CommNetMLP(args, env.observation_dim)
    policy_net = policy_net.double()
    
    # 创建trainer
    print(f"[3/6] 创建Trainer")
    trainer = Trainer(args, policy_net, env)
    
    # 创建在线Shapley值分析器
    print(f"[4/6] 创建在线Shapley值分析器")
    shapley_analyzer = OnlineEdgeShapleyAnalyzer(
        args=args,
        env=env,
        n_eval_episodes=5,  # 使用较少的episodes以加快测试
        topology_threshold=0.05  # 降低阈值以确保能分析到边
    )
    
    # 训练
    print(f"[5/6] 开始训练 (共{args.num_epochs}个epochs)")
    print("      注意: 在第100个epoch会触发Shapley值计算\n")
    
    for epoch in range(1, args.num_epochs + 1):
        # 训练一个epoch
        epoch_stats = {'reward': 0, 'success': 0, 'num_episodes': 0}
        
        for _ in range(args.epoch_size):
            stat, batch_comm_stats = trainer.train_batch(epoch)
            epoch_stats['reward'] += stat.get('reward', 0)
            epoch_stats['success'] += stat.get('success', 0)
            epoch_stats['num_episodes'] += stat.get('num_episodes', 0)
        
        # 计算平均值
        if epoch_stats['num_episodes'] > 0:
            epoch_stats['reward'] /= epoch_stats['num_episodes']
            epoch_stats['success'] /= epoch_stats['num_episodes']
        
        # 打印训练信息（每10个epoch）
        if epoch % 10 == 0:
            print(f"Epoch {epoch:3d}: Reward={epoch_stats['reward']:.4f}, "
                  f"Success={epoch_stats['success']:.4f}")
        
        # 在线Shapley值计算
        shapley_results = integrate_with_training(
            trainer=trainer,
            shapley_analyzer=shapley_analyzer,
            epoch=epoch,
            shapley_interval=100
        )
        
        # 如果计算了Shapley值，打印矩阵
        if shapley_results is not None:
            print("\n" + "="*80)
            print("Shapley值计算完成！")
            print("="*80)
            shapley_analyzer.print_shapley_matrix(epoch)
            
            # 验证结果
            print("\n验证结果:")
            edge_shapley = shapley_results['edge_shapley']
            print(f"  ✓ 计算了 {len(edge_shapley)} 条边的Shapley值")
            print(f"  ✓ 计算耗时: {shapley_results['computation_time']:.2f}秒")
            
            # 检查Shapley值的合理性
            shapley_values = [info['shapley_value'] for info in edge_shapley.values()]
            print(f"  ✓ Shapley值范围: [{min(shapley_values):.4f}, {max(shapley_values):.4f}]")
            print(f"  ✓ 平均Shapley值: {np.mean(shapley_values):.4f}")
    
    # 保存结果
    print(f"\n[6/6] 保存结果")
    save_dir = Path('./test_shapley_results')
    shapley_analyzer.save_results(save_dir)
    
    # 验证保存的文件
    print("\n验证保存的文件:")
    expected_files = [
        'online_shapley_values.json',
        'online_topologies.json',
        'online_shapley_stats.json'
    ]
    
    all_exist = True
    for filename in expected_files:
        filepath = save_dir / filename
        if filepath.exists():
            print(f"  ✓ {filename} 已保存")
        else:
            print(f"  ✗ {filename} 未找到")
            all_exist = False
    
    # 测试结果总结
    print("\n" + "="*80)
    print("测试完成！")
    print("="*80)
    
    if all_exist and len(shapley_analyzer.shapley_history) > 0:
        print("✓ 所有功能正常工作")
        print(f"✓ 共计算了 {len(shapley_analyzer.shapley_history)} 次Shapley值")
        print(f"✓ 结果已保存到: {save_dir}")
        print("\n下一步:")
        print("  1. 查看保存的JSON文件了解数据格式")
        print("  2. 尝试在实际训练中使用 train_with_online_shapley.py")
        print("  3. 根据需要调整 n_eval_episodes 和 topology_threshold 参数")
        return True
    else:
        print("✗ 测试失败，请检查错误信息")
        return False


if __name__ == '__main__':
    try:
        success = test_online_shapley()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"\n错误: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

