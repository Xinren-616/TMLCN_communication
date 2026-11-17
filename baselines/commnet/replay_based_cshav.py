"""
Replay-Based Communication Shapley Value (CShaV) Analysis

这个脚本基于训练过程中保存的checkpoint，分析不同训练阶段通信对性能的依赖程度。
通过在不同checkpoint上评估有/无特定智能体通信的性能，来估计训练时的通信贡献。
"""

import os
import sys
import argparse
from pathlib import Path
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
import json

# 导入训练相关模块
from magic import MAGIC
import data
from utils import *
from action_utils import parse_action_args


class ReplayBasedCShaVAnalyzer:
    """基于Replay的通信Shapley值分析器"""
    
    def __init__(self, args, checkpoint_dir, env):
        """
        初始化分析器
        
        Args:
            args: 训练参数
            checkpoint_dir: checkpoint目录路径
            env: 环境实例
        """
        self.args = args
        self.checkpoint_dir = Path(checkpoint_dir)
        self.env = env
        self.n_agents = args.nagents
        
        # 加载所有checkpoint
        self.checkpoints = self._load_checkpoints()
        print(f"找到 {len(self.checkpoints)} 个checkpoint")
        
    def _load_checkpoints(self):
        """加载所有可用的checkpoint"""
        checkpoints = []
        
        # 查找所有checkpoint文件 (model_ep*.pt)
        for checkpoint_file in sorted(self.checkpoint_dir.glob('model_ep*.pt')):
            epoch = int(checkpoint_file.stem.split('ep')[1])
            checkpoints.append({
                'epoch': epoch,
                'path': checkpoint_file
            })
        
        # 按epoch排序
        checkpoints.sort(key=lambda x: x['epoch'])
        
        return checkpoints
    
    def _load_model(self, checkpoint_path):
        """加载指定checkpoint的模型"""
        from magic import MAGIC
        
        # 创建新的模型实例
        policy_net = MAGIC(self.args)
        
        # 加载checkpoint
        checkpoint = torch.load(checkpoint_path)
        policy_net.load_state_dict(checkpoint['policy_net'])
        policy_net.eval()
        
        return policy_net
    
    def _evaluate_with_comm_mask(self, model, comm_mask, n_episodes=20):
        """
        在指定通信mask下评估模型性能
        
        Args:
            model: 策略网络
            comm_mask: [n, n] 通信mask矩阵
            n_episodes: 评估episode数量
            
        Returns:
            performance_dict: 包含成功率、平均奖励等指标的字典
        """
        total_rewards = []
        total_steps = []
        successes = []
        
        for ep in range(n_episodes):
            # 重置环境
            state = self.env.reset()
            episode_reward = 0
            episode_steps = 0
            
            # 初始化隐藏状态
            prev_hid = model.init_hidden(batch_size=state.shape[0])
            
            for t in range(self.args.max_steps):
                # 准备info字典，包含通信mask
                info = {
                    'comm_mask': comm_mask,
                    'epoch': -1  # 评估模式
                }
                
                # 前向传播
                with torch.no_grad():
                    x = [state, prev_hid]
                    action_out, value, prev_hid = model(x, info)
                
                # 选择动作
                from action_utils import select_action, translate_action
                action = select_action(self.args, action_out)
                action, actual = translate_action(self.args, self.env, action)
                
                # 执行动作
                next_state, reward, done, info = self.env.step(actual)
                
                episode_reward += reward[:self.args.nfriendly].sum()
                episode_steps += 1
                
                state = next_state
                
                if done or t == self.args.max_steps - 1:
                    break
            
            # 记录结果
            total_rewards.append(episode_reward)
            total_steps.append(episode_steps)
            
            # 检查是否成功（如果环境提供了成功信息）
            if hasattr(self.env, 'get_stat'):
                stat = self.env.get_stat()
                if 'success' in stat:
                    successes.append(stat['success'])
                else:
                    successes.append(1.0 if episode_reward > 0 else 0.0)
            else:
                successes.append(1.0 if episode_reward > 0 else 0.0)
        
        return {
            'success_rate': np.mean(successes),
            'avg_reward': np.mean(total_rewards),
            'avg_steps': np.mean(total_steps),
            'std_reward': np.std(total_rewards),
            'std_success': np.std(successes)
        }
    
    def compute_agent_contribution(self, checkpoint_info, agent_i, n_episodes=20):
        """
        计算特定agent在特定checkpoint的通信贡献
        
        Args:
            checkpoint_info: checkpoint信息字典
            agent_i: 目标agent的索引
            n_episodes: 评估episode数量
            
        Returns:
            contribution_dict: 包含边际贡献等信息的字典
        """
        print(f"  评估 Agent {agent_i} 在 Epoch {checkpoint_info['epoch']} 的贡献...")
        
        # 加载模型
        model = self._load_model(checkpoint_info['path'])
        
        # 1. 完整通信的性能
        comm_mask_full = torch.ones(self.n_agents, self.n_agents)
        comm_mask_full.fill_diagonal_(0)
        perf_full = self._evaluate_with_comm_mask(model, comm_mask_full, n_episodes)
        
        # 2. 移除agent i通信的性能
        comm_mask_no_i = comm_mask_full.clone()
        comm_mask_no_i[agent_i, :] = 0  # agent i不能发送消息
        comm_mask_no_i[:, agent_i] = 0  # agent i不能接收消息
        perf_no_i = self._evaluate_with_comm_mask(model, comm_mask_no_i, n_episodes)
        
        # 计算边际贡献
        marginal_success = perf_full['success_rate'] - perf_no_i['success_rate']
        marginal_reward = perf_full['avg_reward'] - perf_no_i['avg_reward']
        
        return {
            'epoch': checkpoint_info['epoch'],
            'agent_i': agent_i,
            'perf_full': perf_full,
            'perf_no_i': perf_no_i,
            'marginal_success': marginal_success,
            'marginal_reward': marginal_reward,
        }
    
    def analyze_all_agents_across_checkpoints(self, n_episodes=20):
        """
        分析所有agent在所有checkpoint的通信贡献
        
        Args:
            n_episodes: 每个评估的episode数量
            
        Returns:
            results: 嵌套字典，结构为 {agent_i: {epoch: contribution_dict}}
        """
        results = defaultdict(dict)
        
        print(f"\n开始分析 {self.n_agents} 个agent在 {len(self.checkpoints)} 个checkpoint的通信贡献...")
        
        for checkpoint_info in self.checkpoints:
            print(f"\n处理 Epoch {checkpoint_info['epoch']}...")
            
            for agent_i in range(self.n_agents):
                contribution = self.compute_agent_contribution(
                    checkpoint_info, 
                    agent_i, 
                    n_episodes
                )
                results[agent_i][checkpoint_info['epoch']] = contribution
        
        return dict(results)
    
    def compute_shapley_values(self, checkpoint_info, n_samples=50, n_episodes=10):
        """
        计算真正的Shapley值（通过采样联盟）
        
        这是一个更精确但计算更昂贵的方法
        
        Args:
            checkpoint_info: checkpoint信息
            n_samples: 采样的联盟数量
            n_episodes: 每次评估的episode数量
            
        Returns:
            shapley_values: 每个agent的Shapley值
        """
        print(f"  计算 Epoch {checkpoint_info['epoch']} 的Shapley值 (采样 {n_samples} 个联盟)...")
        
        # 加载模型
        model = self._load_model(checkpoint_info['path'])
        
        # 初始化Shapley值
        shapley_values = np.zeros(self.n_agents)
        
        for agent_i in range(self.n_agents):
            contributions = []
            
            for sample_idx in range(n_samples):
                # 随机采样联盟大小
                coalition_size = np.random.randint(0, self.n_agents)
                
                # 随机选择其他agent组成联盟
                other_agents = [j for j in range(self.n_agents) if j != agent_i]
                if len(other_agents) > 0 and coalition_size > 0:
                    coalition = np.random.choice(
                        other_agents, 
                        size=min(coalition_size, len(other_agents)), 
                        replace=False
                    )
                else:
                    coalition = []
                
                # 创建通信mask：只有联盟成员可以通信
                comm_mask_without = torch.zeros(self.n_agents, self.n_agents)
                for i in coalition:
                    for j in coalition:
                        if i != j:
                            comm_mask_without[i, j] = 1
                
                # 评估没有agent i的性能
                perf_without = self._evaluate_with_comm_mask(
                    model, comm_mask_without, n_episodes
                )
                
                # 创建包含agent i的通信mask
                comm_mask_with = comm_mask_without.clone()
                for j in range(self.n_agents):
                    if j != agent_i:
                        comm_mask_with[agent_i, j] = 1
                        comm_mask_with[j, agent_i] = 1
                
                # 评估包含agent i的性能
                perf_with = self._evaluate_with_comm_mask(
                    model, comm_mask_with, n_episodes
                )
                
                # 边际贡献
                marginal = perf_with['success_rate'] - perf_without['success_rate']
                contributions.append(marginal)
            
            # Agent i的Shapley值是所有边际贡献的平均
            shapley_values[agent_i] = np.mean(contributions)
            
            print(f"    Agent {agent_i}: {shapley_values[agent_i]:.4f}")
        
        return shapley_values
    
    def save_results(self, results, output_dir):
        """保存分析结果"""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 保存为JSON
        results_json = {}
        for agent_i, agent_results in results.items():
            results_json[f'agent_{agent_i}'] = {}
            for epoch, contribution in agent_results.items():
                results_json[f'agent_{agent_i}'][str(epoch)] = {
                    'marginal_success': float(contribution['marginal_success']),
                    'marginal_reward': float(contribution['marginal_reward']),
                    'success_full': float(contribution['perf_full']['success_rate']),
                    'success_no_i': float(contribution['perf_no_i']['success_rate']),
                }
        
        with open(output_dir / 'replay_cshav_results.json', 'w') as f:
            json.dump(results_json, f, indent=2)
        
        print(f"\n结果已保存到: {output_dir / 'replay_cshav_results.json'}")


def plot_training_contribution_evolution(results, output_path='replay_cshav_evolution.png'):
    """
    可视化训练过程中通信贡献的演化
    
    Args:
        results: analyze_all_agents_across_checkpoints的结果
        output_path: 输出图片路径
    """
    n_agents = len(results)
    
    fig, axes = plt.subplots(3, 1, figsize=(14, 12))
    
    # 子图1: 学习曲线对比（完整通信 vs 移除各agent）
    ax1 = axes[0]
    
    # 首先绘制完整通信的曲线（所有agent应该一样）
    agent_0_results = results[0]
    epochs = sorted(agent_0_results.keys())
    success_full = [agent_0_results[ep]['perf_full']['success_rate'] for ep in epochs]
    
    ax1.plot(epochs, success_full, label='Full Communication', 
            color='black', linewidth=2.5, marker='o', markersize=6)
    
    # 绘制移除每个agent后的曲线
    colors = plt.cm.Set2(np.linspace(0, 1, n_agents))
    for agent_i in range(n_agents):
        agent_results = results[agent_i]
        epochs = sorted(agent_results.keys())
        success_no_i = [agent_results[ep]['perf_no_i']['success_rate'] for ep in epochs]
        
        ax1.plot(epochs, success_no_i, label=f'w/o Agent {agent_i}', 
                linestyle='--', linewidth=1.5, marker='x', markersize=4,
                color=colors[agent_i], alpha=0.7)
    
    ax1.set_xlabel('Training Epoch', fontsize=12)
    ax1.set_ylabel('Success Rate', fontsize=12)
    ax1.set_title('Learning Curves: Impact of Removing Each Agent\'s Communication', 
                 fontsize=14, fontweight='bold')
    ax1.legend(loc='best', fontsize=9)
    ax1.grid(alpha=0.3)
    
    # 子图2: 边际贡献（成功率）随时间的变化
    ax2 = axes[1]
    
    for agent_i in range(n_agents):
        agent_results = results[agent_i]
        epochs = sorted(agent_results.keys())
        marginal_success = [agent_results[ep]['marginal_success'] for ep in epochs]
        
        ax2.plot(epochs, marginal_success, label=f'Agent {agent_i}', 
                marker='o', markersize=4, linewidth=1.5, color=colors[agent_i])
    
    ax2.axhline(y=0, color='red', linestyle=':', linewidth=2, alpha=0.5)
    ax2.set_xlabel('Training Epoch', fontsize=12)
    ax2.set_ylabel('Marginal Contribution (Success Rate)', fontsize=12)
    ax2.set_title('Evolution of Communication Contribution During Training', 
                 fontsize=14, fontweight='bold')
    ax2.legend(loc='best', fontsize=9)
    ax2.grid(alpha=0.3)
    
    # 子图3: 边际贡献（奖励）随时间的变化
    ax3 = axes[2]
    
    for agent_i in range(n_agents):
        agent_results = results[agent_i]
        epochs = sorted(agent_results.keys())
        marginal_reward = [agent_results[ep]['marginal_reward'] for ep in epochs]
        
        ax3.plot(epochs, marginal_reward, label=f'Agent {agent_i}', 
                marker='s', markersize=4, linewidth=1.5, color=colors[agent_i])
    
    ax3.axhline(y=0, color='red', linestyle=':', linewidth=2, alpha=0.5)
    ax3.set_xlabel('Training Epoch', fontsize=12)
    ax3.set_ylabel('Marginal Contribution (Reward)', fontsize=12)
    ax3.set_title('Reward Impact of Communication During Training', 
                 fontsize=14, fontweight='bold')
    ax3.legend(loc='best', fontsize=9)
    ax3.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"图表已保存到: {output_path}")
    plt.close()


def plot_contribution_heatmap(results, output_path='replay_cshav_heatmap.png'):
    """
    绘制贡献热力图：agent × epoch
    
    Args:
        results: analyze_all_agents_across_checkpoints的结果
        output_path: 输出图片路径
    """
    n_agents = len(results)
    
    # 准备数据
    agent_0_results = results[0]
    epochs = sorted(agent_0_results.keys())
    
    # 创建矩阵：[n_agents, n_epochs]
    contribution_matrix = np.zeros((n_agents, len(epochs)))
    
    for agent_i in range(n_agents):
        agent_results = results[agent_i]
        for idx, epoch in enumerate(epochs):
            contribution_matrix[agent_i, idx] = agent_results[epoch]['marginal_success']
    
    # 绘制热力图
    fig, ax = plt.subplots(figsize=(14, 6))
    
    im = ax.imshow(contribution_matrix, cmap='RdYlGn', aspect='auto', interpolation='nearest')
    
    # 设置坐标轴
    ax.set_xticks(np.arange(len(epochs)))
    ax.set_yticks(np.arange(n_agents))
    ax.set_xticklabels([f'{ep}' for ep in epochs], rotation=45)
    ax.set_yticklabels([f'Agent {i}' for i in range(n_agents)])
    
    ax.set_xlabel('Training Epoch', fontsize=12)
    ax.set_ylabel('Agent ID', fontsize=12)
    ax.set_title('Communication Contribution Heatmap (Marginal Success Rate)', 
                fontsize=14, fontweight='bold')
    
    # 添加颜色条
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Marginal Contribution', rotation=270, labelpad=20, fontsize=11)
    
    # 在每个格子中显示数值
    for i in range(n_agents):
        for j in range(len(epochs)):
            text = ax.text(j, i, f'{contribution_matrix[i, j]:.3f}',
                          ha="center", va="center", color="black", fontsize=8)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"热力图已保存到: {output_path}")
    plt.close()


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='Replay-Based Communication Shapley Value Analysis')
    
    # 基本参数
    parser.add_argument('--checkpoint_dir', type=str, required=True,
                       help='包含checkpoint的目录路径')
    parser.add_argument('--output_dir', type=str, default='./replay_cshav_results',
                       help='输出结果的目录')
    parser.add_argument('--n_episodes', type=int, default=20,
                       help='每次评估的episode数量')
    parser.add_argument('--compute_shapley', action='store_true',
                       help='是否计算精确的Shapley值（计算昂贵）')
    parser.add_argument('--shapley_samples', type=int, default=50,
                       help='计算Shapley值时的采样数量')
    
    # 环境和模型参数（需要从训练时的参数中恢复）
    parser.add_argument('--env_name', type=str, required=True,
                       help='环境名称')
    parser.add_argument('--nagents', type=int, required=True,
                       help='智能体数量')
    parser.add_argument('--hid_size', type=int, default=128,
                       help='隐藏层大小')
    parser.add_argument('--max_steps', type=int, default=40,
                       help='最大步数')
    
    # 从init_args_for_env导入环境特定参数
    from utils import init_args_for_env
    init_args_for_env(parser)
    
    args = parser.parse_args()
    
    # 初始化环境
    print(f"初始化环境: {args.env_name}")
    env = data.init(args.env_name, args, False)
    
    # 设置环境参数
    args.obs_size = env.observation_dim
    args.num_actions = env.num_actions
    if not isinstance(args.num_actions, (list, tuple)):
        args.num_actions = [args.num_actions]
    args.dim_actions = env.dim_actions
    args.nfriendly = args.nagents
    
    parse_action_args(args)
    
    # 创建分析器
    analyzer = ReplayBasedCShaVAnalyzer(args, args.checkpoint_dir, env)
    
    # 分析所有agent在所有checkpoint的贡献
    results = analyzer.analyze_all_agents_across_checkpoints(n_episodes=args.n_episodes)
    
    # 保存结果
    output_dir = Path(args.output_dir)
    analyzer.save_results(results, output_dir)
    
    # 可视化
    print("\n生成可视化图表...")
    plot_training_contribution_evolution(results, output_dir / 'replay_cshav_evolution.png')
    plot_contribution_heatmap(results, output_dir / 'replay_cshav_heatmap.png')
    
    # 如果需要，计算精确的Shapley值
    if args.compute_shapley:
        print("\n\n计算精确的Shapley值...")
        shapley_results = {}
        
        for checkpoint_info in analyzer.checkpoints:
            shapley_values = analyzer.compute_shapley_values(
                checkpoint_info,
                n_samples=args.shapley_samples,
                n_episodes=10
            )
            shapley_results[checkpoint_info['epoch']] = shapley_values.tolist()
        
        # 保存Shapley值
        with open(output_dir / 'shapley_values.json', 'w') as f:
            json.dump(shapley_results, f, indent=2)
        
        print(f"Shapley值已保存到: {output_dir / 'shapley_values.json'}")
    
    print("\n分析完成！")


if __name__ == '__main__':
    main()

