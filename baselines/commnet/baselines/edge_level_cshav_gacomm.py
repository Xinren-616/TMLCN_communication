"""
Edge-Level Communication Shapley Value Analysis for GAComm
GAComm模型的边级别通信Shapley值分析
"""

import os
import sys
import argparse
from pathlib import Path
import numpy as np
import torch
import matplotlib.pyplot as plt
from collections import defaultdict
import json

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from baselines.ga_comm import GACommNetMLP
import data
from utils import *
from action_utils import parse_action_args, select_action, translate_action


class EdgeLevelCShaVAnalyzerGAComm:
    """GAComm边级别通信Shapley值分析器"""
    
    def __init__(self, args, checkpoint_dir, env):
        self.args = args
        self.checkpoint_dir = Path(checkpoint_dir)
        self.env = env
        self.n_agents = args.nagents
        self.checkpoints = self._load_checkpoints()
        print(f"✓ 找到 {len(self.checkpoints)} 个checkpoint")
        
    def _load_checkpoints(self):
        """加载所有checkpoint"""
        checkpoints = []
        for checkpoint_file in sorted(self.checkpoint_dir.glob('model_ep*.pt')):
            epoch = int(checkpoint_file.stem.split('ep')[1])
            checkpoints.append({'epoch': epoch, 'path': checkpoint_file})
        checkpoints.sort(key=lambda x: x['epoch'])
        return checkpoints
    
    def _infer_config_from_checkpoint(self, state_dict):
        """从checkpoint推断模型配置"""
        # 推断recurrent
        if 'f_module.weight_ih' in state_dict:
            self.args.recurrent = True
            print("  ✓ 检测到LSTM层")
        
        # 推断hid_size
        if 'encoder.weight' in state_dict:
            self.args.hid_size = state_dict['encoder.weight'].shape[0]
            print(f"  ✓ hid_size = {self.args.hid_size}")
        
        # 推断qk_hid_size
        if 'wq.weight' in state_dict:
            self.args.qk_hid_size = state_dict['wq.weight'].shape[0]
            print(f"  ✓ qk_hid_size = {self.args.qk_hid_size}")
    
    def _load_model(self, checkpoint_path):
        """加载模型"""
        checkpoint = torch.load(checkpoint_path, weights_only=False)
        state_dict = checkpoint['policy_net']
        
        # 推断配置
        self._infer_config_from_checkpoint(state_dict)
        
        # 创建模型
        num_inputs = self.env.observation_dim
        policy_net = GACommNetMLP(self.args, num_inputs)
        
        # 先转换为double，再加载参数
        policy_net = policy_net.double()
        
        # 转换state_dict为double
        state_dict_double = {}
        for key, value in state_dict.items():
            if isinstance(value, torch.Tensor) and value.dtype == torch.float32:
                state_dict_double[key] = value.double()
            else:
                state_dict_double[key] = value
        
        policy_net.load_state_dict(state_dict_double)
        policy_net.eval()
        
        return policy_net
    
    def extract_communication_topology(self, model, n_episodes=20):
        """提取通信拓扑"""
        topology_count = np.zeros((self.n_agents, self.n_agents))
        total_steps = 0
        
        for ep in range(n_episodes):
            state = self.env.reset(epoch=0)
            
            if self.args.recurrent:
                prev_hid = model.init_hidden(batch_size=state.shape[0])
                if isinstance(prev_hid, tuple):
                    prev_hid = tuple(h.double() for h in prev_hid)
            
            for t in range(self.args.max_steps):
                info = {}
                
                with torch.no_grad():
                    if self.args.recurrent:
                        x = [state, prev_hid]
                        action_out, value, prev_hid, comm_density = model(x, info)
                    else:
                        action_out, value, comm_density = model(state, info)
                
                # 提取注意力
                if hasattr(model, 'comm_attention') and model.comm_attention is not None:
                    attn = model.comm_attention
                    n = self.n_agents
                    topology_step = np.zeros((n, n))
                    
                    for i in range(n):
                        for j_idx in range(n - 1):
                            j = j_idx if j_idx < i else j_idx + 1
                            if i < attn.shape[0] and j_idx < attn.shape[1]:
                                topology_step[i, j] = attn[i, j_idx].item()
                    
                    topology_count += topology_step
                else:
                    full_graph = np.ones((self.n_agents, self.n_agents))
                    np.fill_diagonal(full_graph, 0)
                    topology_count += full_graph
                
                total_steps += 1
                
                action = select_action(self.args, action_out)
                action, actual = translate_action(self.args, self.env, action)
                next_state, reward, done, _ = self.env.step(actual)
                state = next_state
                
                if done or t == self.args.max_steps - 1:
                    break
        
        topology_freq = topology_count / total_steps if total_steps > 0 else topology_count
        return topology_freq
    
    def _evaluate_with_edge_mask(self, model, edge_mask, n_episodes=20):
        """评估性能"""
        total_rewards = []
        successes = []
        
        for ep in range(n_episodes):
            state = self.env.reset(epoch=0)
            episode_reward = 0
            
            if self.args.recurrent:
                prev_hid = model.init_hidden(batch_size=state.shape[0])
                if isinstance(prev_hid, tuple):
                    prev_hid = tuple(h.double() for h in prev_hid)
            
            for t in range(self.args.max_steps):
                info = {'comm_mask': edge_mask}
                
                with torch.no_grad():
                    if self.args.recurrent:
                        x = [state, prev_hid]
                        action_out, value, prev_hid, comm_density = model(x, info)
                    else:
                        action_out, value, comm_density = model(state, info)
                
                action = select_action(self.args, action_out)
                action, actual = translate_action(self.args, self.env, action)
                next_state, reward, done, _ = self.env.step(actual)
                
                episode_reward += reward[:self.args.nfriendly].sum()
                state = next_state
                
                if done or t == self.args.max_steps - 1:
                    break
            
            total_rewards.append(episode_reward)
            
            if hasattr(self.env, 'get_stat'):
                stat = self.env.get_stat()
                successes.append(stat.get('success', 1.0 if episode_reward > 0 else 0.0))
            else:
                successes.append(1.0 if episode_reward > 0 else 0.0)
        
        return {
            'success_rate': np.mean(successes),
            'avg_reward': np.mean(total_rewards),
        }
    
    def compute_edge_contribution(self, checkpoint_info, edge_i_j, base_topology, n_episodes):
        """计算边贡献"""
        i, j = edge_i_j
        model = self._load_model(checkpoint_info['path'])
        
        # 有该边
        edge_mask_with = torch.tensor(base_topology, dtype=torch.float64)
        edge_mask_with[i, j] = 1.0
        perf_with = self._evaluate_with_edge_mask(model, edge_mask_with, n_episodes)
        
        # 无该边
        edge_mask_without = edge_mask_with.clone()
        edge_mask_without[i, j] = 0.0
        perf_without = self._evaluate_with_edge_mask(model, edge_mask_without, n_episodes)
        
        return {
            'epoch': checkpoint_info['epoch'],
            'edge': (i, j),
            'base_topology_freq': base_topology[i, j],
            'marginal_success': perf_with['success_rate'] - perf_without['success_rate'],
            'marginal_reward': perf_with['avg_reward'] - perf_without['avg_reward'],
            'success_with': perf_with['success_rate'],
            'success_without': perf_without['success_rate'],
        }
    
    def analyze_all_edges_at_checkpoint(self, checkpoint_info, n_episodes, topology_threshold):
        """分析所有边"""
        print(f"\n分析 Epoch {checkpoint_info['epoch']}...")
        
        model = self._load_model(checkpoint_info['path'])
        base_topology = self.extract_communication_topology(model, n_episodes=10)
        
        print(f"  平均链路频率: {base_topology.mean():.4f}")
        
        important_edges = []
        for i in range(self.n_agents):
            for j in range(self.n_agents):
                if i != j and base_topology[i, j] > topology_threshold:
                    important_edges.append((i, j))
        
        print(f"  分析 {len(important_edges)} 条链路...")
        
        results = {}
        for idx, edge in enumerate(important_edges, 1):
            print(f"    [{idx}/{len(important_edges)}] 边 {edge[0]}→{edge[1]}", end='\r')
            contribution = self.compute_edge_contribution(
                checkpoint_info, edge, base_topology, n_episodes
            )
            results[edge] = contribution
        print()
        
        return results, base_topology
    
    def analyze_edges_across_checkpoints(self, n_episodes, topology_threshold, checkpoint_indices):
        """分析所有checkpoint"""
        if checkpoint_indices is not None:
            selected = [self.checkpoints[i] for i in checkpoint_indices if i < len(self.checkpoints)]
        else:
            selected = self.checkpoints
        
        results = {}
        topologies = {}
        
        for checkpoint_info in selected:
            edge_results, topology = self.analyze_all_edges_at_checkpoint(
                checkpoint_info, n_episodes, topology_threshold
            )
            results[checkpoint_info['epoch']] = edge_results
            topologies[checkpoint_info['epoch']] = topology
        
        return results, topologies
    
    def save_results(self, results, topologies, output_dir):
        """保存结果"""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 保存JSON
        results_json = {}
        for epoch, edge_results in results.items():
            results_json[str(epoch)] = {}
            for edge, contribution in edge_results.items():
                edge_str = f"{edge[0]}->{edge[1]}"
                results_json[str(epoch)][edge_str] = {
                    'base_topology_freq': float(contribution['base_topology_freq']),
                    'marginal_success': float(contribution['marginal_success']),
                    'marginal_reward': float(contribution['marginal_reward']),
                    'success_with': float(contribution['success_with']),
                    'success_without': float(contribution['success_without']),
                }
        
        with open(output_dir / 'edge_level_cshav_results.json', 'w') as f:
            json.dump(results_json, f, indent=2)
        
        topologies_json = {}
        for epoch, topology in topologies.items():
            topologies_json[str(epoch)] = topology.tolist()
        
        with open(output_dir / 'communication_topologies.json', 'w') as f:
            json.dump(topologies_json, f, indent=2)
        
        print(f"\n✓ 结果已保存到: {output_dir}")


def plot_edge_contribution_matrix(results, topologies, epoch, output_path):
    """绘制边贡献矩阵"""
    if epoch not in results:
        return
    
    edge_results = results[epoch]
    topology = topologies[epoch]
    n = topology.shape[0]
    
    contribution_matrix = np.zeros((n, n))
    for (i, j), contrib in edge_results.items():
        contribution_matrix[i, j] = contrib['marginal_success']
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # 拓扑频率
    im1 = axes[0].imshow(topology, cmap='YlOrRd', vmin=0, vmax=1)
    axes[0].set_title(f'Communication Topology\n(Epoch {epoch})', fontsize=13, fontweight='bold')
    axes[0].set_xlabel('Receiver Agent')
    axes[0].set_ylabel('Sender Agent')
    plt.colorbar(im1, ax=axes[0], label='Frequency')
    
    # 边贡献
    im2 = axes[1].imshow(contribution_matrix, vmin=-0.1, vmax=0.1, cmap='bwr')
    axes[1].set_title(f'Edge Contribution\n(Marginal Success)', fontsize=13, fontweight='bold')
    axes[1].set_xlabel('Receiver Agent')
    axes[1].set_ylabel('Sender Agent')
    plt.colorbar(im2, ax=axes[1], label='Contribution')
    
    # 加权重要性
    weighted = contribution_matrix * topology
    im3 = axes[2].imshow(weighted, cmap='YlOrRd')
    axes[2].set_title(f'Weighted Importance\n(Freq × Contrib)', fontsize=13, fontweight='bold')
    axes[2].set_xlabel('Receiver Agent')
    axes[2].set_ylabel('Sender Agent')
    plt.colorbar(im3, ax=axes[2], label='Importance')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ 保存图表: {output_path.name}")


def main():
    parser = argparse.ArgumentParser(description='GAComm Edge-Level Analysis')
    
    # 必需参数
    parser.add_argument('--checkpoint_dir', type=str, required=True)
    parser.add_argument('--env_name', type=str, required=True)
    parser.add_argument('--nagents', type=int, required=True)
    
    # 可选参数
    parser.add_argument('--output_dir', type=str, default='./edge_results_gacomm')
    parser.add_argument('--n_episodes', type=int, default=20)
    parser.add_argument('--topology_threshold', type=float, default=0.1)
    parser.add_argument('--checkpoint_indices', type=int, nargs='+', default=None)
    parser.add_argument('--hid_size', type=int, default=128)
    parser.add_argument('--qk_hid_size', type=int, default=64)
    parser.add_argument('--nactions', default='1', type=str)
    parser.add_argument('--recurrent', action='store_true', default=False)
    parser.add_argument('--continuous', action='store_true', default=False)
    parser.add_argument('--comm_mask_zero', action='store_true', default=False)
    parser.add_argument('--comm_action_one', action='store_true', default=False)
    parser.add_argument('--share_weights', action='store_true', default=False)
    parser.add_argument('--comm_init', default='uniform', type=str)
    parser.add_argument('--batch_size', default=1, type=int)
    
    # 先添加环境参数（包括difficulty, max_steps等）
    from utils import init_args_for_env
    init_args_for_env(parser)
    args = parser.parse_args()
    
    # 设置必需的默认参数
    if not hasattr(args, 'display'):
        args.display = False
    if not hasattr(args, 'render'):
        args.render = False
    if not hasattr(args, 'record'):
        args.record = False
    if not hasattr(args, 'max_steps'):
        args.max_steps = 20
    if not hasattr(args, 'rnn_type'):
        args.rnn_type = 'LSTM'
    if not hasattr(args, 'recurrent'):
        args.recurrent = False
    if not hasattr(args, 'continuous'):
        args.continuous = False
    if not hasattr(args, 'comm_mask_zero'):
        args.comm_mask_zero = False
    if not hasattr(args, 'comm_action_one'):
        args.comm_action_one = False
    if not hasattr(args, 'share_weights'):
        args.share_weights = False
    if not hasattr(args, 'comm_init'):
        args.comm_init = 'uniform'
    
    print("="*80)
    print("GAComm 边级别通信Shapley值分析")
    print("="*80)
    print(f"Checkpoint: {args.checkpoint_dir}")
    print(f"环境: {args.env_name}, Agents: {args.nagents}")
    print(f"输出: {args.output_dir}")
    print("="*80 + "\n")
    
    # 初始化环境
    print("初始化环境...")
    env = data.init(args.env_name, args, False)
    args.obs_size = env.observation_dim
    args.num_actions = env.num_actions
    if not isinstance(args.num_actions, (list, tuple)):
        args.num_actions = [args.num_actions]
    args.dim_actions = env.dim_actions
    args.nfriendly = args.nagents
    parse_action_args(args)
    print("✓ 环境初始化完成\n")
    
    # 创建分析器
    print("加载模型...")
    analyzer = EdgeLevelCShaVAnalyzerGAComm(args, args.checkpoint_dir, env)
    
    # 运行分析
    print("\n开始边级别分析...")
    results, topologies = analyzer.analyze_edges_across_checkpoints(
        n_episodes=args.n_episodes,
        topology_threshold=args.topology_threshold,
        checkpoint_indices=args.checkpoint_indices
    )
    
    # 保存结果
    output_dir = Path(args.output_dir)
    analyzer.save_results(results, topologies, output_dir)
    
    # 生成图表
    print("\n生成可视化...")
    for epoch in results.keys():
        plot_edge_contribution_matrix(
            results, topologies, epoch,
            output_dir / f'edge_matrix_epoch{epoch}.png'
        )
    
    print("\n" + "="*80)
    print("✓ 分析完成！")
    print(f"✓ 结果保存在: {output_dir}")
    print("="*80)


if __name__ == '__main__':
    main()

