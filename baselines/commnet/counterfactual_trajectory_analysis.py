"""
Counterfactual Trajectory Analysis

这个脚本分析在训练的不同阶段，有/无特定智能体通信时的轨迹差异。
通过比较反事实轨迹，可以理解通信对行为模式的影响。
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
from action_utils import parse_action_args, select_action, translate_action


class CounterfactualTrajectoryAnalyzer:
    """反事实轨迹分析器"""
    
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
    
    def generate_trajectory(self, model, comm_mask, seed=None):
        """
        生成一条轨迹
        
        Args:
            model: 策略网络
            comm_mask: [n, n] 通信mask矩阵
            seed: 随机种子（用于复现）
            
        Returns:
            trajectory: 包含状态、动作、奖励等信息的字典
        """
        if seed is not None:
            np.random.seed(seed)
            torch.manual_seed(seed)
        
        # 重置环境
        state = self.env.reset()
        
        # 初始化轨迹记录
        trajectory = {
            'states': [],
            'actions': [],
            'rewards': [],
            'hidden_states': [],
            'comm_matrices': [],
            'agent_positions': [],  # 如果环境提供位置信息
            'total_reward': 0,
            'success': False,
            'steps': 0
        }
        
        # 初始化隐藏状态
        prev_hid = model.init_hidden(batch_size=state.shape[0])
        
        for t in range(self.args.max_steps):
            # 记录当前状态
            trajectory['states'].append(state.copy())
            
            # 准备info字典，包含通信mask
            info = {
                'comm_mask': comm_mask,
                'epoch': -1  # 评估模式
            }
            
            # 前向传播
            with torch.no_grad():
                x = [state, prev_hid]
                action_out, value, prev_hid = model(x, info)
                
                # 记录隐藏状态
                if isinstance(prev_hid, tuple):
                    trajectory['hidden_states'].append(prev_hid[0].cpu().numpy().copy())
                else:
                    trajectory['hidden_states'].append(prev_hid.cpu().numpy().copy())
                
                # 如果模型返回了通信矩阵，记录它
                if 'comm_matrix' in info:
                    trajectory['comm_matrices'].append(info['comm_matrix'].cpu().numpy().copy())
            
            # 选择动作
            action = select_action(self.args, action_out)
            action, actual = translate_action(self.args, self.env, action)
            
            trajectory['actions'].append(action.copy())
            
            # 执行动作
            next_state, reward, done, env_info = self.env.step(actual)
            
            trajectory['rewards'].append(reward.copy())
            trajectory['total_reward'] += reward[:self.args.nfriendly].sum()
            
            # 记录agent位置（如果可用）
            if 'agent_positions' in env_info:
                trajectory['agent_positions'].append(env_info['agent_positions'].copy())
            
            state = next_state
            trajectory['steps'] = t + 1
            
            if done or t == self.args.max_steps - 1:
                # 检查是否成功
                if hasattr(self.env, 'get_stat'):
                    stat = self.env.get_stat()
                    if 'success' in stat:
                        trajectory['success'] = bool(stat['success'])
                break
        
        return trajectory
    
    def analyze_counterfactual_trajectories(self, checkpoint_info, agent_i, n_episodes=20):
        """
        分析有/无agent i通信的轨迹差异
        
        Args:
            checkpoint_info: checkpoint信息字典
            agent_i: 目标agent的索引
            n_episodes: 生成的轨迹数量
            
        Returns:
            analysis: 包含各种差异度量的字典
        """
        print(f"  分析 Agent {agent_i} 在 Epoch {checkpoint_info['epoch']} 的轨迹差异...")
        
        # 加载模型
        model = self._load_model(checkpoint_info['path'])
        
        trajectories_with = []
        trajectories_without = []
        
        # 生成多条轨迹（使用相同的随机种子保证可比性）
        for ep in range(n_episodes):
            seed = ep * 1000 + checkpoint_info['epoch']
            
            # 1. 完整通信的轨迹
            comm_mask_full = torch.ones(self.n_agents, self.n_agents)
            comm_mask_full.fill_diagonal_(0)
            traj_with = self.generate_trajectory(model, comm_mask_full, seed)
            trajectories_with.append(traj_with)
            
            # 2. 移除agent i通信的轨迹
            comm_mask_no_i = comm_mask_full.clone()
            comm_mask_no_i[agent_i, :] = 0
            comm_mask_no_i[:, agent_i] = 0
            traj_without = self.generate_trajectory(model, comm_mask_no_i, seed)
            trajectories_without.append(traj_without)
        
        # 计算轨迹差异度量
        analysis = {
            'epoch': checkpoint_info['epoch'],
            'agent_i': agent_i,
            'n_episodes': n_episodes,
            'success_rate_with': np.mean([t['success'] for t in trajectories_with]),
            'success_rate_without': np.mean([t['success'] for t in trajectories_without]),
            'avg_reward_with': np.mean([t['total_reward'] for t in trajectories_with]),
            'avg_reward_without': np.mean([t['total_reward'] for t in trajectories_without]),
            'avg_steps_with': np.mean([t['steps'] for t in trajectories_with]),
            'avg_steps_without': np.mean([t['steps'] for t in trajectories_without]),
            'trajectory_divergence': self._compute_trajectory_divergence(
                trajectories_with, trajectories_without
            ),
            'action_divergence': self._compute_action_divergence(
                trajectories_with, trajectories_without
            ),
            'hidden_state_divergence': self._compute_hidden_state_divergence(
                trajectories_with, trajectories_without
            )
        }
        
        # 计算边际影响
        analysis['marginal_success'] = analysis['success_rate_with'] - analysis['success_rate_without']
        analysis['marginal_reward'] = analysis['avg_reward_with'] - analysis['avg_reward_without']
        
        return analysis, trajectories_with, trajectories_without
    
    def _compute_trajectory_divergence(self, trajs_with, trajs_without):
        """
        计算两组轨迹的状态散度
        
        使用平均L2距离度量轨迹差异
        """
        divergences = []
        
        for t_with, t_without in zip(trajs_with, trajs_without):
            states_with = np.array(t_with['states'])
            states_without = np.array(t_without['states'])
            
            # 取较短轨迹的长度
            min_len = min(len(states_with), len(states_without))
            
            # 计算每个时间步的L2距离
            step_divergences = []
            for step in range(min_len):
                l2_dist = np.linalg.norm(states_with[step] - states_without[step])
                step_divergences.append(l2_dist)
            
            # 平均散度
            if len(step_divergences) > 0:
                divergences.append(np.mean(step_divergences))
        
        return {
            'mean': np.mean(divergences) if len(divergences) > 0 else 0.0,
            'std': np.std(divergences) if len(divergences) > 0 else 0.0,
            'max': np.max(divergences) if len(divergences) > 0 else 0.0
        }
    
    def _compute_action_divergence(self, trajs_with, trajs_without):
        """
        计算动作选择的差异率
        
        返回有多少比例的时间步选择了不同的动作
        """
        divergence_rates = []
        
        for t_with, t_without in zip(trajs_with, trajs_without):
            actions_with = np.array(t_with['actions'])
            actions_without = np.array(t_without['actions'])
            
            min_len = min(len(actions_with), len(actions_without))
            
            if min_len > 0:
                # 计算不同动作的比例
                different_actions = np.sum(actions_with[:min_len] != actions_without[:min_len])
                divergence_rate = different_actions / (min_len * self.n_agents)
                divergence_rates.append(divergence_rate)
        
        return {
            'mean': np.mean(divergence_rates) if len(divergence_rates) > 0 else 0.0,
            'std': np.std(divergence_rates) if len(divergence_rates) > 0 else 0.0
        }
    
    def _compute_hidden_state_divergence(self, trajs_with, trajs_without):
        """
        计算隐藏状态的散度
        """
        divergences = []
        
        for t_with, t_without in zip(trajs_with, trajs_without):
            if len(t_with['hidden_states']) > 0 and len(t_without['hidden_states']) > 0:
                h_with = np.array(t_with['hidden_states'])
                h_without = np.array(t_without['hidden_states'])
                
                min_len = min(len(h_with), len(h_without))
                
                # 计算每个时间步的L2距离
                step_divergences = []
                for step in range(min_len):
                    l2_dist = np.linalg.norm(h_with[step] - h_without[step])
                    step_divergences.append(l2_dist)
                
                if len(step_divergences) > 0:
                    divergences.append(np.mean(step_divergences))
        
        return {
            'mean': np.mean(divergences) if len(divergences) > 0 else 0.0,
            'std': np.std(divergences) if len(divergences) > 0 else 0.0
        }
    
    def analyze_all_agents_across_checkpoints(self, n_episodes=20, checkpoint_indices=None):
        """
        分析所有agent在所有（或选定）checkpoint的轨迹差异
        
        Args:
            n_episodes: 每个分析的episode数量
            checkpoint_indices: 要分析的checkpoint索引列表（None表示全部）
            
        Returns:
            results: 嵌套字典，结构为 {agent_i: {epoch: analysis}}
        """
        results = defaultdict(dict)
        
        # 选择要分析的checkpoints
        if checkpoint_indices is not None:
            selected_checkpoints = [self.checkpoints[i] for i in checkpoint_indices 
                                   if i < len(self.checkpoints)]
        else:
            selected_checkpoints = self.checkpoints
        
        print(f"\n开始分析 {self.n_agents} 个agent在 {len(selected_checkpoints)} 个checkpoint的轨迹差异...")
        
        for checkpoint_info in selected_checkpoints:
            print(f"\n处理 Epoch {checkpoint_info['epoch']}...")
            
            for agent_i in range(self.n_agents):
                analysis, _, _ = self.analyze_counterfactual_trajectories(
                    checkpoint_info, 
                    agent_i, 
                    n_episodes
                )
                results[agent_i][checkpoint_info['epoch']] = analysis
        
        return dict(results)
    
    def save_results(self, results, output_dir):
        """保存分析结果"""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 保存为JSON
        results_json = {}
        for agent_i, agent_results in results.items():
            results_json[f'agent_{agent_i}'] = {}
            for epoch, analysis in agent_results.items():
                results_json[f'agent_{agent_i}'][str(epoch)] = {
                    'success_rate_with': float(analysis['success_rate_with']),
                    'success_rate_without': float(analysis['success_rate_without']),
                    'marginal_success': float(analysis['marginal_success']),
                    'marginal_reward': float(analysis['marginal_reward']),
                    'trajectory_divergence': {
                        'mean': float(analysis['trajectory_divergence']['mean']),
                        'std': float(analysis['trajectory_divergence']['std']),
                        'max': float(analysis['trajectory_divergence']['max'])
                    },
                    'action_divergence': {
                        'mean': float(analysis['action_divergence']['mean']),
                        'std': float(analysis['action_divergence']['std'])
                    },
                    'hidden_state_divergence': {
                        'mean': float(analysis['hidden_state_divergence']['mean']),
                        'std': float(analysis['hidden_state_divergence']['std'])
                    }
                }
        
        with open(output_dir / 'counterfactual_trajectory_results.json', 'w') as f:
            json.dump(results_json, f, indent=2)
        
        print(f"\n结果已保存到: {output_dir / 'counterfactual_trajectory_results.json'}")


def plot_trajectory_divergence_evolution(results, output_path='trajectory_divergence_evolution.png'):
    """
    可视化轨迹散度随训练的演化
    
    Args:
        results: analyze_all_agents_across_checkpoints的结果
        output_path: 输出图片路径
    """
    n_agents = len(results)
    
    fig, axes = plt.subplots(3, 1, figsize=(14, 12))
    
    colors = plt.cm.Set2(np.linspace(0, 1, n_agents))
    
    # 子图1: 状态轨迹散度
    ax1 = axes[0]
    for agent_i in range(n_agents):
        agent_results = results[agent_i]
        epochs = sorted(agent_results.keys())
        traj_div = [agent_results[ep]['trajectory_divergence']['mean'] for ep in epochs]
        traj_div_std = [agent_results[ep]['trajectory_divergence']['std'] for ep in epochs]
        
        ax1.plot(epochs, traj_div, label=f'Agent {agent_i}', 
                marker='o', markersize=4, linewidth=1.5, color=colors[agent_i])
        ax1.fill_between(epochs, 
                         np.array(traj_div) - np.array(traj_div_std),
                         np.array(traj_div) + np.array(traj_div_std),
                         alpha=0.2, color=colors[agent_i])
    
    ax1.set_xlabel('Training Epoch', fontsize=12)
    ax1.set_ylabel('State Trajectory Divergence (L2)', fontsize=12)
    ax1.set_title('State Trajectory Divergence: With vs Without Communication', 
                 fontsize=14, fontweight='bold')
    ax1.legend(loc='best', fontsize=9)
    ax1.grid(alpha=0.3)
    
    # 子图2: 动作散度
    ax2 = axes[1]
    for agent_i in range(n_agents):
        agent_results = results[agent_i]
        epochs = sorted(agent_results.keys())
        action_div = [agent_results[ep]['action_divergence']['mean'] for ep in epochs]
        
        ax2.plot(epochs, action_div, label=f'Agent {agent_i}', 
                marker='s', markersize=4, linewidth=1.5, color=colors[agent_i])
    
    ax2.set_xlabel('Training Epoch', fontsize=12)
    ax2.set_ylabel('Action Divergence Rate', fontsize=12)
    ax2.set_title('Action Selection Divergence: With vs Without Communication', 
                 fontsize=14, fontweight='bold')
    ax2.legend(loc='best', fontsize=9)
    ax2.grid(alpha=0.3)
    
    # 子图3: 隐藏状态散度
    ax3 = axes[2]
    for agent_i in range(n_agents):
        agent_results = results[agent_i]
        epochs = sorted(agent_results.keys())
        hidden_div = [agent_results[ep]['hidden_state_divergence']['mean'] for ep in epochs]
        
        ax3.plot(epochs, hidden_div, label=f'Agent {agent_i}', 
                marker='^', markersize=4, linewidth=1.5, color=colors[agent_i])
    
    ax3.set_xlabel('Training Epoch', fontsize=12)
    ax3.set_ylabel('Hidden State Divergence (L2)', fontsize=12)
    ax3.set_title('Hidden State Divergence: With vs Without Communication', 
                 fontsize=14, fontweight='bold')
    ax3.legend(loc='best', fontsize=9)
    ax3.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"图表已保存到: {output_path}")
    plt.close()


def plot_divergence_vs_contribution(results, output_path='divergence_vs_contribution.png'):
    """
    绘制散度与贡献的关系
    
    Args:
        results: analyze_all_agents_across_checkpoints的结果
        output_path: 输出图片路径
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    colors = plt.cm.Set2(np.linspace(0, 1, len(results)))
    
    # 收集所有数据点
    for agent_i, agent_results in results.items():
        traj_divs = []
        marginal_successes = []
        marginal_rewards = []
        
        for epoch, analysis in agent_results.items():
            traj_divs.append(analysis['trajectory_divergence']['mean'])
            marginal_successes.append(analysis['marginal_success'])
            marginal_rewards.append(analysis['marginal_reward'])
        
        # 子图1: 轨迹散度 vs 边际成功率
        axes[0].scatter(traj_divs, marginal_successes, 
                       label=f'Agent {agent_i}', 
                       color=colors[agent_i], 
                       alpha=0.6, s=50)
        
        # 子图2: 轨迹散度 vs 边际奖励
        axes[1].scatter(traj_divs, marginal_rewards, 
                       label=f'Agent {agent_i}', 
                       color=colors[agent_i], 
                       alpha=0.6, s=50)
    
    axes[0].set_xlabel('Trajectory Divergence (L2)', fontsize=12)
    axes[0].set_ylabel('Marginal Success Rate', fontsize=12)
    axes[0].set_title('Trajectory Divergence vs Communication Contribution (Success)', 
                     fontsize=13, fontweight='bold')
    axes[0].legend(loc='best', fontsize=9)
    axes[0].grid(alpha=0.3)
    axes[0].axhline(y=0, color='red', linestyle=':', alpha=0.5)
    
    axes[1].set_xlabel('Trajectory Divergence (L2)', fontsize=12)
    axes[1].set_ylabel('Marginal Reward', fontsize=12)
    axes[1].set_title('Trajectory Divergence vs Communication Contribution (Reward)', 
                     fontsize=13, fontweight='bold')
    axes[1].legend(loc='best', fontsize=9)
    axes[1].grid(alpha=0.3)
    axes[1].axhline(y=0, color='red', linestyle=':', alpha=0.5)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"图表已保存到: {output_path}")
    plt.close()


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='Counterfactual Trajectory Analysis')
    
    # 基本参数
    parser.add_argument('--checkpoint_dir', type=str, required=True,
                       help='包含checkpoint的目录路径')
    parser.add_argument('--output_dir', type=str, default='./trajectory_analysis_results',
                       help='输出结果的目录')
    parser.add_argument('--n_episodes', type=int, default=20,
                       help='每次分析的episode数量')
    parser.add_argument('--checkpoint_indices', type=int, nargs='+', default=None,
                       help='要分析的checkpoint索引（例如: 0 2 4）')
    
    # 环境和模型参数
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
    analyzer = CounterfactualTrajectoryAnalyzer(args, args.checkpoint_dir, env)
    
    # 分析轨迹差异
    results = analyzer.analyze_all_agents_across_checkpoints(
        n_episodes=args.n_episodes,
        checkpoint_indices=args.checkpoint_indices
    )
    
    # 保存结果
    output_dir = Path(args.output_dir)
    analyzer.save_results(results, output_dir)
    
    # 可视化
    print("\n生成可视化图表...")
    plot_trajectory_divergence_evolution(results, output_dir / 'trajectory_divergence_evolution.png')
    plot_divergence_vs_contribution(results, output_dir / 'divergence_vs_contribution.png')
    
    print("\n分析完成！")


if __name__ == '__main__':
    main()

