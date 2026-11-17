"""
Training Progress Analysis - 综合分析工具

这个脚本整合了Replay-Based CShaV和Counterfactual Trajectory Analysis，
提供一个统一的接口来分析训练过程中通信对性能的依赖程度。
"""

import os
import sys
import argparse
from pathlib import Path
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
import json

# 导入分析工具
from replay_based_cshav import ReplayBasedCShaVAnalyzer, plot_training_contribution_evolution, plot_contribution_heatmap
from counterfactual_trajectory_analysis import CounterfactualTrajectoryAnalyzer, plot_trajectory_divergence_evolution, plot_divergence_vs_contribution

# 导入训练相关模块
import data
from utils import *
from action_utils import parse_action_args


class TrainingProgressAnalyzer:
    """训练进度综合分析器"""
    
    def __init__(self, args, checkpoint_dir, env):
        """
        初始化综合分析器
        
        Args:
            args: 训练参数
            checkpoint_dir: checkpoint目录路径
            env: 环境实例
        """
        self.args = args
        self.checkpoint_dir = Path(checkpoint_dir)
        self.env = env
        self.n_agents = args.nagents
        
        # 创建子分析器
        self.cshav_analyzer = ReplayBasedCShaVAnalyzer(args, checkpoint_dir, env)
        self.trajectory_analyzer = CounterfactualTrajectoryAnalyzer(args, checkpoint_dir, env)
        
        print(f"初始化完成：找到 {len(self.cshav_analyzer.checkpoints)} 个checkpoint")
    
    def run_comprehensive_analysis(self, n_episodes_cshav=20, n_episodes_trajectory=20, 
                                   checkpoint_indices=None):
        """
        运行综合分析
        
        Args:
            n_episodes_cshav: CShaV分析的episode数量
            n_episodes_trajectory: 轨迹分析的episode数量
            checkpoint_indices: 要分析的checkpoint索引列表（None表示全部）
            
        Returns:
            results: 包含所有分析结果的字典
        """
        results = {}
        
        # 1. Replay-Based CShaV分析
        print("\n" + "="*80)
        print("第一部分: Replay-Based Communication Shapley Value 分析")
        print("="*80)
        results['cshav'] = self.cshav_analyzer.analyze_all_agents_across_checkpoints(
            n_episodes=n_episodes_cshav
        )
        
        # 2. Counterfactual Trajectory分析
        print("\n" + "="*80)
        print("第二部分: Counterfactual Trajectory Analysis")
        print("="*80)
        results['trajectory'] = self.trajectory_analyzer.analyze_all_agents_across_checkpoints(
            n_episodes=n_episodes_trajectory,
            checkpoint_indices=checkpoint_indices
        )
        
        # 3. 综合分析：计算相关性
        print("\n" + "="*80)
        print("第三部分: 综合指标相关性分析")
        print("="*80)
        results['correlations'] = self._compute_correlations(
            results['cshav'], 
            results['trajectory']
        )
        
        return results
    
    def _compute_correlations(self, cshav_results, trajectory_results):
        """
        计算不同指标之间的相关性
        
        Args:
            cshav_results: CShaV分析结果
            trajectory_results: 轨迹分析结果
            
        Returns:
            correlations: 相关性分析结果
        """
        from scipy.stats import pearsonr, spearmanr
        
        correlations = {}
        
        for agent_i in range(self.n_agents):
            agent_cshav = cshav_results[agent_i]
            agent_traj = trajectory_results[agent_i]
            
            # 找到共同的epochs
            common_epochs = sorted(set(agent_cshav.keys()) & set(agent_traj.keys()))
            
            if len(common_epochs) < 3:
                print(f"  Agent {agent_i}: 数据点不足，跳过相关性分析")
                continue
            
            # 提取数据
            marginal_success = [agent_cshav[ep]['marginal_success'] for ep in common_epochs]
            marginal_reward = [agent_cshav[ep]['marginal_reward'] for ep in common_epochs]
            traj_div = [agent_traj[ep]['trajectory_divergence']['mean'] for ep in common_epochs]
            action_div = [agent_traj[ep]['action_divergence']['mean'] for ep in common_epochs]
            hidden_div = [agent_traj[ep]['hidden_state_divergence']['mean'] for ep in common_epochs]
            
            # 计算相关系数
            correlations[agent_i] = {}
            
            # 边际成功率 vs 轨迹散度
            if np.std(marginal_success) > 1e-6 and np.std(traj_div) > 1e-6:
                pearson_r, pearson_p = pearsonr(marginal_success, traj_div)
                spearman_r, spearman_p = spearmanr(marginal_success, traj_div)
                correlations[agent_i]['success_vs_traj_div'] = {
                    'pearson_r': pearson_r,
                    'pearson_p': pearson_p,
                    'spearman_r': spearman_r,
                    'spearman_p': spearman_p
                }
                print(f"  Agent {agent_i} - 边际成功率 vs 轨迹散度:")
                print(f"    Pearson r = {pearson_r:.3f} (p = {pearson_p:.4f})")
                print(f"    Spearman r = {spearman_r:.3f} (p = {spearman_p:.4f})")
            
            # 边际奖励 vs 轨迹散度
            if np.std(marginal_reward) > 1e-6 and np.std(traj_div) > 1e-6:
                pearson_r, pearson_p = pearsonr(marginal_reward, traj_div)
                spearman_r, spearman_p = spearmanr(marginal_reward, traj_div)
                correlations[agent_i]['reward_vs_traj_div'] = {
                    'pearson_r': pearson_r,
                    'pearson_p': pearson_p,
                    'spearman_r': spearman_r,
                    'spearman_p': spearman_p
                }
                print(f"  Agent {agent_i} - 边际奖励 vs 轨迹散度:")
                print(f"    Pearson r = {pearson_r:.3f} (p = {pearson_p:.4f})")
            
            # 边际成功率 vs 动作散度
            if np.std(marginal_success) > 1e-6 and np.std(action_div) > 1e-6:
                pearson_r, pearson_p = pearsonr(marginal_success, action_div)
                spearman_r, spearman_p = spearmanr(marginal_success, action_div)
                correlations[agent_i]['success_vs_action_div'] = {
                    'pearson_r': pearson_r,
                    'pearson_p': pearson_p,
                    'spearman_r': spearman_r,
                    'spearman_p': spearman_p
                }
                print(f"  Agent {agent_i} - 边际成功率 vs 动作散度:")
                print(f"    Pearson r = {pearson_r:.3f} (p = {pearson_p:.4f})")
        
        return correlations
    
    def generate_comprehensive_report(self, results, output_dir):
        """
        生成综合分析报告
        
        Args:
            results: 综合分析结果
            output_dir: 输出目录
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        print("\n" + "="*80)
        print("生成综合分析报告")
        print("="*80)
        
        # 1. 保存JSON结果
        print("1. 保存JSON结果...")
        self.cshav_analyzer.save_results(results['cshav'], output_dir)
        self.trajectory_analyzer.save_results(results['trajectory'], output_dir)
        
        # 保存相关性结果
        correlations_json = {}
        for agent_i, agent_corr in results['correlations'].items():
            correlations_json[f'agent_{agent_i}'] = {}
            for key, value in agent_corr.items():
                correlations_json[f'agent_{agent_i}'][key] = {
                    k: float(v) for k, v in value.items()
                }
        
        with open(output_dir / 'correlations.json', 'w') as f:
            json.dump(correlations_json, f, indent=2)
        
        # 2. 生成可视化
        print("2. 生成可视化图表...")
        
        # CShaV可视化
        plot_training_contribution_evolution(
            results['cshav'], 
            output_dir / 'replay_cshav_evolution.png'
        )
        plot_contribution_heatmap(
            results['cshav'], 
            output_dir / 'replay_cshav_heatmap.png'
        )
        
        # 轨迹分析可视化
        plot_trajectory_divergence_evolution(
            results['trajectory'], 
            output_dir / 'trajectory_divergence_evolution.png'
        )
        plot_divergence_vs_contribution(
            results['trajectory'], 
            output_dir / 'divergence_vs_contribution.png'
        )
        
        # 综合对比可视化
        self._plot_comprehensive_comparison(results, output_dir)
        
        # 3. 生成文本报告
        print("3. 生成文本报告...")
        self._generate_text_report(results, output_dir / 'analysis_report.txt')
        
        print(f"\n报告已保存到: {output_dir}")
    
    def _plot_comprehensive_comparison(self, results, output_dir):
        """
        生成综合对比图表
        
        Args:
            results: 综合分析结果
            output_dir: 输出目录
        """
        n_agents = self.n_agents
        cshav_results = results['cshav']
        trajectory_results = results['trajectory']
        
        # 图1: 所有指标的演化对比
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        colors = plt.cm.Set2(np.linspace(0, 1, n_agents))
        
        # 获取共同的epochs
        agent_0_cshav = cshav_results[0]
        agent_0_traj = trajectory_results[0]
        common_epochs = sorted(set(agent_0_cshav.keys()) & set(agent_0_traj.keys()))
        
        # 子图1: 边际成功率贡献
        ax1 = axes[0, 0]
        for agent_i in range(n_agents):
            marginal_success = [cshav_results[agent_i][ep]['marginal_success'] 
                               for ep in common_epochs]
            ax1.plot(common_epochs, marginal_success, label=f'Agent {agent_i}', 
                    marker='o', linewidth=1.5, color=colors[agent_i])
        ax1.axhline(y=0, color='red', linestyle=':', alpha=0.5)
        ax1.set_xlabel('Training Epoch', fontsize=11)
        ax1.set_ylabel('Marginal Success Contribution', fontsize=11)
        ax1.set_title('Communication Contribution (Success Rate)', fontsize=12, fontweight='bold')
        ax1.legend(loc='best', fontsize=8)
        ax1.grid(alpha=0.3)
        
        # 子图2: 轨迹散度
        ax2 = axes[0, 1]
        for agent_i in range(n_agents):
            traj_div = [trajectory_results[agent_i][ep]['trajectory_divergence']['mean'] 
                       for ep in common_epochs]
            ax2.plot(common_epochs, traj_div, label=f'Agent {agent_i}', 
                    marker='s', linewidth=1.5, color=colors[agent_i])
        ax2.set_xlabel('Training Epoch', fontsize=11)
        ax2.set_ylabel('Trajectory Divergence (L2)', fontsize=11)
        ax2.set_title('State Trajectory Divergence', fontsize=12, fontweight='bold')
        ax2.legend(loc='best', fontsize=8)
        ax2.grid(alpha=0.3)
        
        # 子图3: 动作散度
        ax3 = axes[1, 0]
        for agent_i in range(n_agents):
            action_div = [trajectory_results[agent_i][ep]['action_divergence']['mean'] 
                         for ep in common_epochs]
            ax3.plot(common_epochs, action_div, label=f'Agent {agent_i}', 
                    marker='^', linewidth=1.5, color=colors[agent_i])
        ax3.set_xlabel('Training Epoch', fontsize=11)
        ax3.set_ylabel('Action Divergence Rate', fontsize=11)
        ax3.set_title('Action Selection Divergence', fontsize=12, fontweight='bold')
        ax3.legend(loc='best', fontsize=8)
        ax3.grid(alpha=0.3)
        
        # 子图4: 边际奖励贡献
        ax4 = axes[1, 1]
        for agent_i in range(n_agents):
            marginal_reward = [cshav_results[agent_i][ep]['marginal_reward'] 
                              for ep in common_epochs]
            ax4.plot(common_epochs, marginal_reward, label=f'Agent {agent_i}', 
                    marker='d', linewidth=1.5, color=colors[agent_i])
        ax4.axhline(y=0, color='red', linestyle=':', alpha=0.5)
        ax4.set_xlabel('Training Epoch', fontsize=11)
        ax4.set_ylabel('Marginal Reward Contribution', fontsize=11)
        ax4.set_title('Communication Contribution (Reward)', fontsize=12, fontweight='bold')
        ax4.legend(loc='best', fontsize=8)
        ax4.grid(alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_dir / 'comprehensive_comparison.png', dpi=300, bbox_inches='tight')
        print(f"  综合对比图已保存")
        plt.close()
        
        # 图2: 贡献 vs 散度的散点图矩阵
        self._plot_scatter_matrix(results, output_dir)
    
    def _plot_scatter_matrix(self, results, output_dir):
        """
        绘制散点图矩阵：展示不同指标之间的关系
        
        Args:
            results: 综合分析结果
            output_dir: 输出目录
        """
        cshav_results = results['cshav']
        trajectory_results = results['trajectory']
        
        # 收集所有数据
        all_data = {
            'Marginal Success': [],
            'Marginal Reward': [],
            'Trajectory Divergence': [],
            'Action Divergence': [],
            'Hidden State Divergence': [],
            'Agent': [],
            'Epoch': []
        }
        
        for agent_i in range(self.n_agents):
            agent_cshav = cshav_results[agent_i]
            agent_traj = trajectory_results[agent_i]
            common_epochs = sorted(set(agent_cshav.keys()) & set(agent_traj.keys()))
            
            for epoch in common_epochs:
                all_data['Marginal Success'].append(agent_cshav[epoch]['marginal_success'])
                all_data['Marginal Reward'].append(agent_cshav[epoch]['marginal_reward'])
                all_data['Trajectory Divergence'].append(agent_traj[epoch]['trajectory_divergence']['mean'])
                all_data['Action Divergence'].append(agent_traj[epoch]['action_divergence']['mean'])
                all_data['Hidden State Divergence'].append(agent_traj[epoch]['hidden_state_divergence']['mean'])
                all_data['Agent'].append(agent_i)
                all_data['Epoch'].append(epoch)
        
        # 使用seaborn创建散点图矩阵
        import pandas as pd
        df = pd.DataFrame(all_data)
        
        # 选择要绘制的列
        plot_cols = ['Marginal Success', 'Trajectory Divergence', 'Action Divergence']
        
        fig = plt.figure(figsize=(14, 14))
        
        # 创建3x3的子图
        n_cols = len(plot_cols)
        for i, col1 in enumerate(plot_cols):
            for j, col2 in enumerate(plot_cols):
                ax = plt.subplot(n_cols, n_cols, i * n_cols + j + 1)
                
                if i == j:
                    # 对角线：直方图
                    ax.hist(df[col1], bins=15, alpha=0.7, color='steelblue')
                    ax.set_ylabel('Frequency', fontsize=9)
                else:
                    # 非对角线：散点图
                    colors = plt.cm.Set2(df['Agent'])
                    ax.scatter(df[col2], df[col1], c=colors, alpha=0.6, s=30)
                    
                    # 添加趋势线
                    from scipy.stats import linregress
                    if len(df) > 2:
                        slope, intercept, r_value, p_value, std_err = linregress(df[col2], df[col1])
                        x_line = np.array([df[col2].min(), df[col2].max()])
                        y_line = slope * x_line + intercept
                        ax.plot(x_line, y_line, 'r--', alpha=0.5, linewidth=1)
                        ax.text(0.05, 0.95, f'R²={r_value**2:.3f}', 
                               transform=ax.transAxes, fontsize=8,
                               verticalalignment='top')
                
                # 设置标签
                if i == n_cols - 1:
                    ax.set_xlabel(col2, fontsize=9)
                else:
                    ax.set_xticklabels([])
                
                if j == 0 and i != j:
                    ax.set_ylabel(col1, fontsize=9)
                elif j != 0:
                    ax.set_yticklabels([])
                
                ax.grid(alpha=0.2)
        
        plt.tight_layout()
        plt.savefig(output_dir / 'scatter_matrix.png', dpi=300, bbox_inches='tight')
        print(f"  散点图矩阵已保存")
        plt.close()
    
    def _generate_text_report(self, results, output_path):
        """
        生成文本分析报告
        
        Args:
            results: 综合分析结果
            output_path: 输出文件路径
        """
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write("="*80 + "\n")
            f.write("训练过程通信依赖程度综合分析报告\n")
            f.write("="*80 + "\n\n")
            
            # 1. 基本信息
            f.write("1. 基本信息\n")
            f.write("-" * 80 + "\n")
            f.write(f"智能体数量: {self.n_agents}\n")
            f.write(f"环境: {self.args.env_name}\n")
            f.write(f"分析的Checkpoint数量: {len(self.cshav_analyzer.checkpoints)}\n\n")
            
            # 2. Replay-Based CShaV摘要
            f.write("2. Replay-Based Communication Shapley Value 分析摘要\n")
            f.write("-" * 80 + "\n")
            
            cshav_results = results['cshav']
            for agent_i in range(self.n_agents):
                agent_results = cshav_results[agent_i]
                epochs = sorted(agent_results.keys())
                
                marginal_successes = [agent_results[ep]['marginal_success'] for ep in epochs]
                marginal_rewards = [agent_results[ep]['marginal_reward'] for ep in epochs]
                
                f.write(f"\nAgent {agent_i}:\n")
                f.write(f"  平均边际成功率贡献: {np.mean(marginal_successes):.4f} ± {np.std(marginal_successes):.4f}\n")
                f.write(f"  平均边际奖励贡献: {np.mean(marginal_rewards):.4f} ± {np.std(marginal_rewards):.4f}\n")
                f.write(f"  最大边际成功率贡献: {np.max(marginal_successes):.4f} (Epoch {epochs[np.argmax(marginal_successes)]})\n")
                f.write(f"  最小边际成功率贡献: {np.min(marginal_successes):.4f} (Epoch {epochs[np.argmin(marginal_successes)]})\n")
            
            # 3. Counterfactual Trajectory分析摘要
            f.write("\n\n3. Counterfactual Trajectory Analysis 摘要\n")
            f.write("-" * 80 + "\n")
            
            trajectory_results = results['trajectory']
            for agent_i in range(self.n_agents):
                agent_results = trajectory_results[agent_i]
                epochs = sorted(agent_results.keys())
                
                traj_divs = [agent_results[ep]['trajectory_divergence']['mean'] for ep in epochs]
                action_divs = [agent_results[ep]['action_divergence']['mean'] for ep in epochs]
                
                f.write(f"\nAgent {agent_i}:\n")
                f.write(f"  平均轨迹散度: {np.mean(traj_divs):.4f} ± {np.std(traj_divs):.4f}\n")
                f.write(f"  平均动作散度: {np.mean(action_divs):.4f} ± {np.std(action_divs):.4f}\n")
                f.write(f"  最大轨迹散度: {np.max(traj_divs):.4f} (Epoch {epochs[np.argmax(traj_divs)]})\n")
            
            # 4. 相关性分析摘要
            if 'correlations' in results and len(results['correlations']) > 0:
                f.write("\n\n4. 指标相关性分析\n")
                f.write("-" * 80 + "\n")
                
                for agent_i, agent_corr in results['correlations'].items():
                    f.write(f"\nAgent {agent_i}:\n")
                    
                    if 'success_vs_traj_div' in agent_corr:
                        corr = agent_corr['success_vs_traj_div']
                        f.write(f"  边际成功率 vs 轨迹散度:\n")
                        f.write(f"    Pearson r = {corr['pearson_r']:.3f} (p = {corr['pearson_p']:.4f})\n")
                        f.write(f"    Spearman r = {corr['spearman_r']:.3f} (p = {corr['spearman_p']:.4f})\n")
                    
                    if 'success_vs_action_div' in agent_corr:
                        corr = agent_corr['success_vs_action_div']
                        f.write(f"  边际成功率 vs 动作散度:\n")
                        f.write(f"    Pearson r = {corr['pearson_r']:.3f} (p = {corr['pearson_p']:.4f})\n")
            
            # 5. 结论
            f.write("\n\n5. 主要发现\n")
            f.write("-" * 80 + "\n")
            f.write("（基于数据的自动总结）\n\n")
            
            # 找出贡献最大的agent
            avg_contributions = []
            for agent_i in range(self.n_agents):
                agent_results = cshav_results[agent_i]
                marginal_successes = [agent_results[ep]['marginal_success'] 
                                     for ep in agent_results.keys()]
                avg_contributions.append(np.mean(marginal_successes))
            
            max_contrib_agent = np.argmax(avg_contributions)
            min_contrib_agent = np.argmin(avg_contributions)
            
            f.write(f"- Agent {max_contrib_agent} 的通信贡献最大 (平均边际成功率: {avg_contributions[max_contrib_agent]:.4f})\n")
            f.write(f"- Agent {min_contrib_agent} 的通信贡献最小 (平均边际成功率: {avg_contributions[min_contrib_agent]:.4f})\n")
            
            # 分析趋势
            for agent_i in range(self.n_agents):
                agent_results = cshav_results[agent_i]
                epochs = sorted(agent_results.keys())
                if len(epochs) >= 3:
                    marginal_successes = [agent_results[ep]['marginal_success'] for ep in epochs]
                    
                    # 简单的趋势判断
                    early_avg = np.mean(marginal_successes[:len(epochs)//2])
                    late_avg = np.mean(marginal_successes[len(epochs)//2:])
                    
                    if late_avg > early_avg + 0.05:
                        f.write(f"- Agent {agent_i} 的通信重要性在训练过程中显著增加\n")
                    elif late_avg < early_avg - 0.05:
                        f.write(f"- Agent {agent_i} 的通信重要性在训练过程中显著降低\n")
            
            f.write("\n" + "="*80 + "\n")
            f.write("报告生成完成\n")
            f.write("="*80 + "\n")
        
        print(f"  文本报告已保存")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='Comprehensive Training Progress Analysis')
    
    # 基本参数
    parser.add_argument('--checkpoint_dir', type=str, required=True,
                       help='包含checkpoint的目录路径')
    parser.add_argument('--output_dir', type=str, default='./training_analysis_results',
                       help='输出结果的目录')
    parser.add_argument('--n_episodes_cshav', type=int, default=20,
                       help='CShaV分析的episode数量')
    parser.add_argument('--n_episodes_trajectory', type=int, default=20,
                       help='轨迹分析的episode数量')
    parser.add_argument('--checkpoint_indices', type=int, nargs='+', default=None,
                       help='要分析的checkpoint索引（用于轨迹分析，可减少计算量）')
    
    # 环境和模型参数
    parser.add_argument('--env_name', type=str, required=True,
                       help='环境名称 (例如: traffic_junction, predator_prey)')
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
    
    print("="*80)
    print("训练过程通信依赖程度综合分析")
    print("="*80)
    print(f"环境: {args.env_name}")
    print(f"Checkpoint目录: {args.checkpoint_dir}")
    print(f"输出目录: {args.output_dir}")
    print("="*80 + "\n")
    
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
    
    # 创建综合分析器
    analyzer = TrainingProgressAnalyzer(args, args.checkpoint_dir, env)
    
    # 运行综合分析
    results = analyzer.run_comprehensive_analysis(
        n_episodes_cshav=args.n_episodes_cshav,
        n_episodes_trajectory=args.n_episodes_trajectory,
        checkpoint_indices=args.checkpoint_indices
    )
    
    # 生成综合报告
    analyzer.generate_comprehensive_report(results, args.output_dir)
    
    print("\n" + "="*80)
    print("分析完成！所有结果已保存到:", args.output_dir)
    print("="*80)


if __name__ == '__main__':
    main()

