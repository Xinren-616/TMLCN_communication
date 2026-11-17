"""
Edge-Level Communication Shapley Value Analysis for MAGIC

这个脚本专门为MAGIC模型提供边级别通信链路分析：
- 正确处理MAGIC的图注意力通信机制
- 分析动态学习的通信拓扑
- 评估每条具体通信链路 (agent i → agent j) 的作用
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

# 导入MAGIC模型
from magic import MAGIC

# 导入训练相关模块
import data
from utils import *
from action_utils import parse_action_args, select_action, translate_action


class EdgeLevelCShaVAnalyzerMAGIC:
    """边级别（链路级别）通信Shapley值分析器 - MAGIC专用版本"""
    
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
        self.model_type = 'magic'
        
        # 加载所有checkpoint
        self.checkpoints = self._load_checkpoints()
        print(f"找到 {len(self.checkpoints)} 个checkpoint")
        
    def _load_checkpoints(self):
        """加载所有可用的checkpoint"""
        checkpoints = []
        
        for checkpoint_file in sorted(self.checkpoint_dir.glob('model_ep*.pt')):
            epoch = int(checkpoint_file.stem.split('ep')[1])
            checkpoints.append({
                'epoch': epoch,
                'path': checkpoint_file
            })
        
        checkpoints.sort(key=lambda x: x['epoch'])
        return checkpoints
    
    def _load_model(self, checkpoint_path):
        """加载指定checkpoint的模型"""
        # 加载checkpoint
        checkpoint = torch.load(checkpoint_path, weights_only=False)
        state_dict = checkpoint['policy_net']
        
        # 创建MAGIC模型实例
        policy_net = MAGIC(self.args)
        
        # 加载模型参数
        policy_net.load_state_dict(state_dict)
        policy_net.eval()
        
        # 将模型转换为double类型以匹配环境的观测类型
        policy_net = policy_net.double()
        
        return policy_net
    
    def extract_communication_topology(self, model, n_episodes=20):
        """
        提取MAGIC训练时的实际通信拓扑
        
        MAGIC使用图注意力机制动态学习通信拓扑
        
        Args:
            model: 策略网络
            n_episodes: 采样episode数量
            
        Returns:
            topology: [n, n] 矩阵，表示链路的使用频率（0-1之间）
        """
        print("  提取通信拓扑...")
        
        # 初始化拓扑统计
        topology_count = np.zeros((self.n_agents, self.n_agents))
        total_steps = 0
        
        for ep in range(n_episodes):
            # 重置环境
            state = self.env.reset(epoch=0)
            
            for t in range(self.args.max_steps):
                info = {}
                
                with torch.no_grad():
                    # MAGIC模型forward
                    action_out, value = model(state, info)
                
                # 提取通信图 - MAGIC在forward过程中会生成通信拓扑
                # 检查模型是否保存了注意力权重
                if hasattr(model, 'graph_attention') and model.graph_attention is not None:
                    # 获取第一轮通信的注意力权重
                    attn = model.graph_attention  # [batch, n, n] 或 [n, n]
                    if len(attn.shape) == 3:
                        attn = attn[0]  # 取第一个batch
                    elif len(attn.shape) == 2:
                        pass  # 已经是 [n, n]
                    topology_count += attn.detach().cpu().numpy()
                else:
                    # 如果没有保存attention，尝试从scheduler获取
                    if hasattr(model, 'scheduler') and hasattr(model.scheduler, 'adj'):
                        adj = model.scheduler.adj
                        if adj is not None:
                            if len(adj.shape) == 3:
                                adj = adj[0]
                            topology_count += adj.detach().cpu().numpy()
                    else:
                        # 默认假设完全图
                        full_graph = np.ones((self.n_agents, self.n_agents))
                        np.fill_diagonal(full_graph, 0)
                        topology_count += full_graph
                
                total_steps += 1
                
                # 执行动作（继续episode）
                action = select_action(self.args, action_out)
                action, actual = translate_action(self.args, self.env, action)
                next_state, reward, done, _ = self.env.step(actual)
                
                state = next_state
                
                if done or t == self.args.max_steps - 1:
                    break
        
        # 归一化为频率
        topology_freq = topology_count / total_steps if total_steps > 0 else topology_count
        
        print(f"  平均链路使用频率统计完成（{total_steps} 步）")
        
        return topology_freq
    
    def _evaluate_with_edge_mask(self, model, edge_mask, n_episodes=20):
        """
        在指定的边级别mask下评估模型性能
        
        Args:
            model: 策略网络
            edge_mask: [n, n] 边mask矩阵，1表示保留该边，0表示移除
            n_episodes: 评估episode数量
            
        Returns:
            performance_dict: 包含成功率、平均奖励等指标的字典
        """
        total_rewards = []
        total_steps = []
        successes = []
        
        for ep in range(n_episodes):
            # 重置环境
            state = self.env.reset(epoch=0)
            episode_reward = 0
            episode_steps = 0
            
            for t in range(self.args.max_steps):
                # 准备info字典，包含边级别的通信mask
                info = {
                    'comm_mask': edge_mask,  # [n, n] 边mask
                }
                
                with torch.no_grad():
                    # MAGIC模型forward
                    action_out, value = model(state, info)
                
                action = select_action(self.args, action_out)
                action, actual = translate_action(self.args, self.env, action)
                
                next_state, reward, done, info_step = self.env.step(actual)
                
                episode_reward += reward[:self.args.nfriendly].sum()
                episode_steps += 1
                
                state = next_state
                
                if done or t == self.args.max_steps - 1:
                    break
            
            total_rewards.append(episode_reward)
            total_steps.append(episode_steps)
            
            # 检查是否成功
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
    
    def compute_edge_contribution(self, checkpoint_info, edge_i_j, 
                                  base_topology=None, n_episodes=20):
        """
        计算特定边 (i→j) 的通信贡献
        
        Args:
            checkpoint_info: checkpoint信息字典
            edge_i_j: (i, j) 元组，表示要分析的边
            base_topology: 基础拓扑（如果为None，则提取）
            n_episodes: 评估episode数量
            
        Returns:
            contribution_dict: 包含边际贡献等信息的字典
        """
        i, j = edge_i_j
        print(f"  评估边 {i}→{j} 在 Epoch {checkpoint_info['epoch']} 的贡献...")
        
        # 加载模型
        model = self._load_model(checkpoint_info['path'])
        
        # 如果没有提供基础拓扑，提取之
        if base_topology is None:
            base_topology = self.extract_communication_topology(model, n_episodes=10)
        
        # 1. 使用基础拓扑的性能（包含边i→j）
        edge_mask_with = torch.tensor(base_topology, dtype=torch.float32)
        # 确保边i→j存在
        edge_mask_with[i, j] = 1.0
        perf_with = self._evaluate_with_edge_mask(model, edge_mask_with, n_episodes)
        
        # 2. 移除边i→j后的性能
        edge_mask_without = edge_mask_with.clone()
        edge_mask_without[i, j] = 0.0
        perf_without = self._evaluate_with_edge_mask(model, edge_mask_without, n_episodes)
        
        # 计算边际贡献
        marginal_success = perf_with['success_rate'] - perf_without['success_rate']
        marginal_reward = perf_with['avg_reward'] - perf_without['avg_reward']
        
        return {
            'epoch': checkpoint_info['epoch'],
            'edge': (i, j),
            'base_topology_freq': base_topology[i, j],  # 该边在基础拓扑中的使用频率
            'perf_with': perf_with,
            'perf_without': perf_without,
            'marginal_success': marginal_success,
            'marginal_reward': marginal_reward,
        }
    
    def analyze_all_edges_at_checkpoint(self, checkpoint_info, n_episodes=20,
                                       topology_threshold=0.1):
        """
        分析特定checkpoint下所有重要边的贡献
        
        Args:
            checkpoint_info: checkpoint信息
            n_episodes: 评估episode数量
            topology_threshold: 拓扑阈值，只分析使用频率>此值的边
            
        Returns:
            results: {(i,j): contribution_dict} 的字典
            topology: 基础拓扑矩阵
        """
        print(f"\n分析 Epoch {checkpoint_info['epoch']} 的所有边...")
        
        # 加载模型并提取拓扑
        model = self._load_model(checkpoint_info['path'])
        base_topology = self.extract_communication_topology(model, n_episodes=10)
        
        print(f"  基础拓扑使用频率统计:")
        print(f"    平均链路使用频率: {base_topology.mean():.4f}")
        print(f"    非零链路数量: {(base_topology > topology_threshold).sum()}")
        
        # 找出所有重要的边（使用频率超过阈值）
        important_edges = []
        for i in range(self.n_agents):
            for j in range(self.n_agents):
                if i != j and base_topology[i, j] > topology_threshold:
                    important_edges.append((i, j))
        
        print(f"  将分析 {len(important_edges)} 条重要链路（阈值={topology_threshold}）")
        
        # 分析每条边
        results = {}
        for edge in important_edges:
            contribution = self.compute_edge_contribution(
                checkpoint_info,
                edge,
                base_topology=base_topology,
                n_episodes=n_episodes
            )
            results[edge] = contribution
        
        return results, base_topology
    
    def analyze_edges_across_checkpoints(self, n_episodes=20, 
                                        topology_threshold=0.1,
                                        checkpoint_indices=None):
        """
        分析所有checkpoint的边级别贡献
        
        Args:
            n_episodes: 每次评估的episode数量
            topology_threshold: 拓扑阈值
            checkpoint_indices: 要分析的checkpoint索引列表
            
        Returns:
            results: {epoch: {(i,j): contribution_dict}}
            topologies: {epoch: topology_matrix}
        """
        # 选择要分析的checkpoints
        if checkpoint_indices is not None:
            selected_checkpoints = [self.checkpoints[i] for i in checkpoint_indices 
                                   if i < len(self.checkpoints)]
        else:
            selected_checkpoints = self.checkpoints
        
        print(f"\n开始边级别CShaV分析：{len(selected_checkpoints)} 个checkpoint")
        
        results = {}
        topologies = {}
        
        for checkpoint_info in selected_checkpoints:
            edge_results, topology = self.analyze_all_edges_at_checkpoint(
                checkpoint_info,
                n_episodes=n_episodes,
                topology_threshold=topology_threshold
            )
            
            results[checkpoint_info['epoch']] = edge_results
            topologies[checkpoint_info['epoch']] = topology
        
        return results, topologies
    
    def save_results(self, results, topologies, output_dir):
        """保存分析结果"""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 保存边级别贡献
        results_json = {}
        for epoch, edge_results in results.items():
            results_json[str(epoch)] = {}
            for edge, contribution in edge_results.items():
                edge_str = f"{edge[0]}->{edge[1]}"
                results_json[str(epoch)][edge_str] = {
                    'base_topology_freq': float(contribution['base_topology_freq']),
                    'marginal_success': float(contribution['marginal_success']),
                    'marginal_reward': float(contribution['marginal_reward']),
                    'success_with': float(contribution['perf_with']['success_rate']),
                    'success_without': float(contribution['perf_without']['success_rate']),
                }
        
        with open(output_dir / 'edge_level_cshav_results.json', 'w') as f:
            json.dump(results_json, f, indent=2)
        
        # 保存拓扑信息
        topologies_json = {}
        for epoch, topology in topologies.items():
            topologies_json[str(epoch)] = topology.tolist()
        
        with open(output_dir / 'communication_topologies.json', 'w') as f:
            json.dump(topologies_json, f, indent=2)
        
        print(f"\n结果已保存到: {output_dir}")


def plot_edge_contribution_matrix(results, topologies, epoch, output_path):
    """
    绘制特定epoch的边贡献矩阵
    
    Args:
        results: 分析结果
        topologies: 拓扑信息
        epoch: 要可视化的epoch
        output_path: 输出路径
    """
    if epoch not in results:
        print(f"Epoch {epoch} 不在结果中")
        return
    
    edge_results = results[epoch]
    topology = topologies[epoch]
    n = topology.shape[0]
    
    # 创建贡献矩阵
    contribution_matrix = np.zeros((n, n))
    for (i, j), contrib in edge_results.items():
        contribution_matrix[i, j] = contrib['marginal_success']
    
    # 创建图表
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # 子图1：拓扑频率
    im1 = axes[0].imshow(topology, cmap='YlOrRd', vmin=0, vmax=1)
    axes[0].set_title(f'Communication Topology Frequency\n(Epoch {epoch})', 
                     fontsize=13, fontweight='bold')
    axes[0].set_xlabel('Receiver Agent', fontsize=11)
    axes[0].set_ylabel('Sender Agent', fontsize=11)
    
    # 添加数值标注
    for i in range(n):
        for j in range(n):
            if i != j:
                text = axes[0].text(j, i, f'{topology[i, j]:.2f}',
                                   ha="center", va="center", color="black", fontsize=9)
    
    plt.colorbar(im1, ax=axes[0], label='Usage Frequency')
    
    # 子图2：边级别贡献
    im2 = axes[1].imshow(contribution_matrix, 
                        vmin=-0.1, vmax=0.1, cmap='bwr')
    axes[1].set_title(f'Edge-Level Communication Contribution\n(Marginal Success Rate)', 
                     fontsize=13, fontweight='bold')
    axes[1].set_xlabel('Receiver Agent', fontsize=11)
    axes[1].set_ylabel('Sender Agent', fontsize=11)
    
    # 添加数值标注
    for i in range(n):
        for j in range(n):
            if (i, j) in edge_results:
                val = contribution_matrix[i, j]
                color = 'white' if abs(val) > 0.05 else 'black'
                text = axes[1].text(j, i, f'{val:.3f}',
                                   ha="center", va="center", color=color, fontsize=9)
    
    plt.colorbar(im2, ax=axes[1], label='Marginal Contribution')
    
    # 子图3：贡献 × 频率（加权重要性）
    weighted_importance = contribution_matrix * topology
    im3 = axes[2].imshow(weighted_importance, cmap='YlOrRd')
    axes[2].set_title(f'Weighted Edge Importance\n(Contribution × Frequency)', 
                     fontsize=13, fontweight='bold')
    axes[2].set_xlabel('Receiver Agent', fontsize=11)
    axes[2].set_ylabel('Sender Agent', fontsize=11)
    
    # 添加数值标注
    for i in range(n):
        for j in range(n):
            if (i, j) in edge_results:
                val = weighted_importance[i, j]
                text = axes[2].text(j, i, f'{val:.3f}',
                                   ha="center", va="center", color="black", fontsize=9)
    
    plt.colorbar(im3, ax=axes[2], label='Weighted Importance')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"边贡献矩阵图已保存: {output_path}")
    plt.close()


def plot_edge_contribution_evolution(results, output_path):
    """
    绘制关键边的贡献随训练的演化
    
    Args:
        results: 分析结果
        output_path: 输出路径
    """
    # 收集所有边
    all_edges = set()
    for epoch, edge_results in results.items():
        all_edges.update(edge_results.keys())
    
    # 按边组织数据
    edge_evolution = defaultdict(lambda: {'epochs': [], 'contributions': []})
    
    for epoch in sorted(results.keys()):
        edge_results = results[epoch]
        for edge in all_edges:
            if edge in edge_results:
                edge_evolution[edge]['epochs'].append(epoch)
                edge_evolution[edge]['contributions'].append(
                    edge_results[edge]['marginal_success']
                )
    
    # 找出最重要的边（平均贡献最大）
    edge_importance = {}
    for edge, data in edge_evolution.items():
        if len(data['contributions']) > 0:
            edge_importance[edge] = np.mean(np.abs(data['contributions']))
    
    # 选择Top 10的边
    top_edges = sorted(edge_importance.items(), key=lambda x: x[1], reverse=True)[:10]
    
    # 绘图
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
    
    colors = plt.cm.tab20(np.linspace(0, 1, len(top_edges)))
    
    # 子图1：Top边的贡献演化
    for idx, (edge, _) in enumerate(top_edges):
        data = edge_evolution[edge]
        ax1.plot(data['epochs'], data['contributions'], 
                label=f'Edge {edge[0]}→{edge[1]}',
                marker='o', linewidth=2, color=colors[idx])
    
    ax1.axhline(y=0, color='red', linestyle=':', alpha=0.5)
    ax1.set_xlabel('Training Epoch', fontsize=12)
    ax1.set_ylabel('Marginal Success Contribution', fontsize=12)
    ax1.set_title('Top 10 Important Edges: Contribution Evolution', 
                 fontsize=14, fontweight='bold')
    ax1.legend(loc='best', fontsize=9, ncol=2)
    ax1.grid(alpha=0.3)
    
    # 子图2：所有边的热力图（epoch × edge）
    epochs = sorted(results.keys())
    edge_list = list(all_edges)
    contribution_heatmap = np.zeros((len(edge_list), len(epochs)))
    
    for j, epoch in enumerate(epochs):
        edge_results = results[epoch]
        for i, edge in enumerate(edge_list):
            if edge in edge_results:
                contribution_heatmap[i, j] = edge_results[edge]['marginal_success']
    
    im = ax2.imshow(contribution_heatmap, cmap='RdYlGn', aspect='auto', 
                   vmin=-0.1, vmax=0.1, interpolation='nearest')
    
    ax2.set_xlabel('Training Epoch', fontsize=12)
    ax2.set_ylabel('Communication Edge', fontsize=12)
    ax2.set_title('All Edges: Contribution Heatmap Across Training', 
                 fontsize=14, fontweight='bold')
    
    # 设置y轴标签（显示边）
    if len(edge_list) <= 20:
        ax2.set_yticks(np.arange(len(edge_list)))
        ax2.set_yticklabels([f'{e[0]}→{e[1]}' for e in edge_list], fontsize=8)
    
    # 设置x轴标签（显示epoch）
    ax2.set_xticks(np.arange(len(epochs)))
    ax2.set_xticklabels([str(e) for e in epochs], rotation=45)
    
    plt.colorbar(im, ax=ax2, label='Marginal Contribution')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"边贡献演化图已保存: {output_path}")
    plt.close()


def plot_network_graph(topology, contribution_matrix, epoch, output_path):
    """
    绘制通信网络图
    
    Args:
        topology: 拓扑频率矩阵
        contribution_matrix: 贡献矩阵
        epoch: epoch编号
        output_path: 输出路径
    """
    try:
        import networkx as nx
    except ImportError:
        print("需要安装networkx: pip install networkx")
        return
    
    n = topology.shape[0]
    
    # 创建有向图
    G = nx.DiGraph()
    G.add_nodes_from(range(n))
    
    # 添加边（只添加频率>0.1的边）
    for i in range(n):
        for j in range(n):
            if i != j and topology[i, j] > 0.1:
                # 边权重=贡献
                weight = contribution_matrix[i, j] if contribution_matrix is not None else topology[i, j]
                G.add_edge(i, j, weight=weight, frequency=topology[i, j])
    
    # 绘图
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
    
    # 布局
    pos = nx.spring_layout(G, seed=42, k=2)
    
    # 子图1：按频率着色
    ax1.set_title(f'Communication Network (Frequency-based)\nEpoch {epoch}', 
                 fontsize=13, fontweight='bold')
    
    # 绘制节点
    nx.draw_networkx_nodes(G, pos, node_color='lightblue', 
                          node_size=800, ax=ax1)
    nx.draw_networkx_labels(G, pos, font_size=12, ax=ax1)
    
    # 绘制边（按频率着色）
    edges = G.edges()
    frequencies = [G[u][v]['frequency'] for u, v in edges]
    
    nx.draw_networkx_edges(G, pos, edgelist=edges, 
                          edge_color=frequencies, edge_cmap=plt.cm.YlOrRd,
                          width=3, alpha=0.7, ax=ax1,
                          arrows=True, arrowsize=20, arrowstyle='->')
    
    sm1 = plt.cm.ScalarMappable(cmap=plt.cm.YlOrRd, 
                               norm=plt.Normalize(vmin=0, vmax=1))
    sm1.set_array([])
    plt.colorbar(sm1, ax=ax1, label='Usage Frequency')
    
    ax1.axis('off')
    
    # 子图2：按贡献着色
    if contribution_matrix is not None:
        ax2.set_title(f'Communication Network (Contribution-based)\nEpoch {epoch}', 
                     fontsize=13, fontweight='bold')
        
        nx.draw_networkx_nodes(G, pos, node_color='lightblue', 
                              node_size=800, ax=ax2)
        nx.draw_networkx_labels(G, pos, font_size=12, ax=ax2)
        
        # 绘制边（按贡献着色）
        weights = [G[u][v]['weight'] for u, v in edges]
        
        nx.draw_networkx_edges(G, pos, edgelist=edges, 
                              edge_color=weights, edge_cmap=plt.cm.RdYlGn,
                              width=3, alpha=0.7, ax=ax2,
                              arrows=True, arrowsize=20, arrowstyle='->')
        
        sm2 = plt.cm.ScalarMappable(cmap=plt.cm.RdYlGn, 
                                   norm=plt.Normalize(vmin=-0.1, vmax=0.1))
        sm2.set_array([])
        plt.colorbar(sm2, ax=ax2, label='Marginal Contribution')
        
        ax2.axis('off')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"网络图已保存: {output_path}")
    plt.close()


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='Edge-Level Communication Shapley Value Analysis for MAGIC')
    
    # 基本参数
    parser.add_argument('--checkpoint_dir', type=str, required=True,
                       help='包含checkpoint的目录路径')
    parser.add_argument('--output_dir', type=str, default='./edge_results_magic',
                       help='输出结果的目录')
    parser.add_argument('--n_episodes', type=int, default=20,
                       help='每次评估的episode数量')
    parser.add_argument('--topology_threshold', type=float, default=0.1,
                       help='拓扑阈值（只分析使用频率>此值的边）')
    parser.add_argument('--checkpoint_indices', type=int, nargs='+', default=None,
                       help='要分析的checkpoint索引（例如: 0 5 10）')
    
    # 环境和模型参数
    parser.add_argument('--env_name', type=str, required=True,
                       help='环境名称')
    parser.add_argument('--nagents', type=int, required=True,
                       help='智能体数量')
    parser.add_argument('--hid_size', type=int, default=64,
                       help='隐藏层大小')
    parser.add_argument('--max_steps', type=int, default=40,
                       help='最大步数')
    parser.add_argument('--nactions', default='1', type=str,
                       help='the number of agent actions (0 for continuous). Use N:M:K for multiple actions')
    
    # MAGIC specific args
    parser.add_argument('--directed', action='store_true', default=False,
                       help='whether the communication graph is directed')
    parser.add_argument('--self_loop_type1', default=2, type=int,
                       help='self loop type in the first gat layer')
    parser.add_argument('--self_loop_type2', default=2, type=int,
                       help='self loop type in the second gat layer')
    parser.add_argument('--gat_num_heads', default=1, type=int,
                       help='number of heads in gat layers')
    parser.add_argument('--gat_num_heads_out', default=1, type=int,
                       help='number of heads in output gat layer')
    parser.add_argument('--gat_hid_size', default=64, type=int,
                       help='hidden size of one head in gat')
    parser.add_argument('--ge_num_heads', default=4, type=int,
                       help='number of heads in the gat encoder')
    parser.add_argument('--first_gat_normalize', action='store_true', default=False,
                       help='whether normalize the coefficients in the first gat layer')
    parser.add_argument('--second_gat_normalize', action='store_true', default=False,
                       help='whether normalize the coefficients in the second gat layer')
    parser.add_argument('--gat_encoder_normalize', action='store_true', default=False,
                       help='whether normalize the coefficients in the gat encoder')
    parser.add_argument('--use_gat_encoder', action='store_true', default=False,
                       help='whether use the gat encoder')
    parser.add_argument('--gat_encoder_out_size', default=64, type=int,
                       help='hidden size of output of the gat encoder')
    parser.add_argument('--first_graph_complete', action='store_true', default=False,
                       help='whether the first communication graph is complete')
    parser.add_argument('--second_graph_complete', action='store_true', default=False,
                       help='whether the second communication graph is complete')
    parser.add_argument('--learn_second_graph', action='store_true', default=False,
                       help='whether learn a new graph at the second round')
    parser.add_argument('--message_encoder', action='store_true', default=False,
                       help='whether use the message encoder')
    parser.add_argument('--message_decoder', action='store_true', default=False,
                       help='whether use the message decoder')
    parser.add_argument('--mean_ratio', default=0, type=float,
                       help='how much cooperative')
    parser.add_argument('--comm_init', default='uniform', type=str,
                       help='how to initialise comm weights')
    parser.add_argument('--comm_mask_zero', action='store_true', default=False,
                       help='whether block communication')
    parser.add_argument('--detach_gap', default=10000, type=int,
                       help='detach gap')
    parser.add_argument('--advantages_per_action', default=False, action='store_true',
                       help='advantages per action')
    
    # 从init_args_for_env导入环境特定参数
    from utils import init_args_for_env
    init_args_for_env(parser)
    
    args = parser.parse_args()
    
    print("="*80)
    print(f"边级别（链路级别）通信Shapley值分析 - MAGIC")
    print("="*80)
    print(f"环境: {args.env_name}")
    print(f"智能体数量: {args.nagents}")
    print(f"可能的通信链路数量: {args.nagents * (args.nagents - 1)}")
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
    
    # 创建分析器
    analyzer = EdgeLevelCShaVAnalyzerMAGIC(args, args.checkpoint_dir, env)
    
    # 运行边级别分析
    results, topologies = analyzer.analyze_edges_across_checkpoints(
        n_episodes=args.n_episodes,
        topology_threshold=args.topology_threshold,
        checkpoint_indices=args.checkpoint_indices
    )
    
    # 保存结果
    output_dir = Path(args.output_dir)
    analyzer.save_results(results, topologies, output_dir)
    
    # 生成可视化
    print("\n生成可视化图表...")
    
    # 对每个checkpoint生成矩阵图
    for epoch in results.keys():
        plot_edge_contribution_matrix(
            results, topologies, epoch,
            output_dir / f'edge_contribution_matrix_epoch{epoch}.png'
        )
    
    # 生成演化图
    if len(results) > 1:
        plot_edge_contribution_evolution(
            results,
            output_dir / 'edge_contribution_evolution.png'
        )
    
    # 生成网络图（对每个checkpoint）
    for epoch in results.keys():
        contribution_matrix = np.zeros((args.nagents, args.nagents))
        for (i, j), contrib in results[epoch].items():
            contribution_matrix[i, j] = contrib['marginal_success']
        
        plot_network_graph(
            topologies[epoch],
            contribution_matrix,
            epoch,
            output_dir / f'network_graph_epoch{epoch}.png'
        )
    
    print("\n" + "="*80)
    print("边级别分析完成！")
    print(f"结果保存在: {output_dir}")
    print("="*80)


if __name__ == '__main__':
    main()

