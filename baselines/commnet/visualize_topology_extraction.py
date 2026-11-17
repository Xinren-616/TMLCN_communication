"""
可视化通信拓扑提取过程

这个脚本演示：
1. 如何从checkpoint加载模型
2. 如何提取通信拓扑
3. 拓扑在不同时间步/状态下的变化
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import argparse

# 导入必要的模块
import data
from utils import *
from action_utils import parse_action_args, select_action, translate_action


def visualize_topology_evolution_in_episode(model, env, args, episode_idx=0, save_path='topology_evolution.png'):
    """
    可视化单个episode中通信拓扑的演化
    
    展示拓扑如何随状态变化
    """
    print(f"\n可视化Episode {episode_idx}中的拓扑演化...")
    
    # 重置环境
    np.random.seed(episode_idx)
    torch.manual_seed(episode_idx)
    state = env.reset()
    prev_hid = model.init_hidden(batch_size=state.shape[0])
    
    # 收集拓扑
    topologies = []
    states_info = []
    
    for t in range(args.max_steps):
        info = {'epoch': -1}
        
        with torch.no_grad():
            x = [state, prev_hid]
            action_out, value, prev_hid = model(x, info)
        
        # 提取拓扑
        if 'adj' in info:
            adj = info['adj']
            if len(adj.shape) == 3:
                adj = adj[0]  # 取第一个batch
            topologies.append(adj.detach().cpu().numpy())
        else:
            # CommNet: 完全图
            n = args.nagents
            full_graph = np.ones((n, n)) - np.eye(n)
            topologies.append(full_graph)
        
        # 记录状态信息
        states_info.append({
            'step': t,
            'state_norm': np.linalg.norm(state),
        })
        
        # 执行动作
        action = select_action(args, action_out)
        action, actual = translate_action(args, env, action)
        next_state, reward, done, _ = env.step(actual)
        
        state = next_state
        
        if done or t == args.max_steps - 1:
            break
    
    # 可视化
    n_steps = len(topologies)
    n_agents = topologies[0].shape[0]
    
    # 选择要展示的时间步（最多9个）
    if n_steps > 9:
        step_indices = np.linspace(0, n_steps-1, 9, dtype=int)
    else:
        step_indices = range(n_steps)
    
    n_rows = int(np.ceil(len(step_indices) / 3))
    fig, axes = plt.subplots(n_rows, 3, figsize=(15, 5*n_rows))
    axes = axes.flatten() if n_rows > 1 else [axes] if n_rows == 1 else axes
    
    for idx, step in enumerate(step_indices):
        ax = axes[idx]
        
        # 绘制拓扑热力图
        im = ax.imshow(topologies[step], cmap='YlOrRd', vmin=0, vmax=1)
        ax.set_title(f'Step {step}', fontsize=12, fontweight='bold')
        ax.set_xlabel('Receiver', fontsize=10)
        ax.set_ylabel('Sender', fontsize=10)
        
        # 添加数值
        for i in range(n_agents):
            for j in range(n_agents):
                if i != j:
                    text = ax.text(j, i, f'{topologies[step][i, j]:.2f}',
                                 ha="center", va="center", 
                                 color="black" if topologies[step][i, j] < 0.5 else "white",
                                 fontsize=8)
        
        plt.colorbar(im, ax=ax)
    
    # 隐藏多余的子图
    for idx in range(len(step_indices), len(axes)):
        axes[idx].axis('off')
    
    plt.suptitle(f'Communication Topology Evolution in Episode {episode_idx}', 
                fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"可视化已保存: {save_path}")
    plt.close()
    
    return topologies


def visualize_topology_variance_across_episodes(model, env, args, n_episodes=10, save_path='topology_variance.png'):
    """
    可视化通信拓扑在不同episodes间的变化
    
    展示拓扑的稳定性/多样性
    """
    print(f"\n分析{n_episodes}个episodes的拓扑变化...")
    
    # 收集多个episodes的拓扑
    all_topologies = []
    
    for ep in range(n_episodes):
        np.random.seed(ep * 100)
        torch.manual_seed(ep * 100)
        state = env.reset()
        prev_hid = model.init_hidden(batch_size=state.shape[0])
        
        episode_topologies = []
        
        for t in range(args.max_steps):
            info = {'epoch': -1}
            
            with torch.no_grad():
                x = [state, prev_hid]
                action_out, value, prev_hid = model(x, info)
            
            if 'adj' in info:
                adj = info['adj']
                if len(adj.shape) == 3:
                    adj = adj[0]
                episode_topologies.append(adj.detach().cpu().numpy())
            else:
                n = args.nagents
                full_graph = np.ones((n, n)) - np.eye(n)
                episode_topologies.append(full_graph)
            
            action = select_action(args, action_out)
            action, actual = translate_action(args, env, action)
            next_state, reward, done, _ = env.step(actual)
            
            state = next_state
            
            if done or t == args.max_steps - 1:
                break
        
        all_topologies.append(episode_topologies)
    
    # 计算统计量
    # 1. 每条边的平均值和标准差
    n_agents = all_topologies[0][0].shape[0]
    
    # 展平所有拓扑
    flat_topologies = []
    for ep_topos in all_topologies:
        for topo in ep_topos:
            flat_topologies.append(topo)
    
    flat_topologies = np.array(flat_topologies)  # [total_steps, n, n]
    
    # 计算统计量
    mean_topology = flat_topologies.mean(axis=0)
    std_topology = flat_topologies.std(axis=0)
    
    # 可视化
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # 子图1：平均拓扑
    im1 = axes[0].imshow(mean_topology, cmap='YlOrRd', vmin=0, vmax=1)
    axes[0].set_title('Mean Topology\n(Average across all steps)', 
                     fontsize=13, fontweight='bold')
    axes[0].set_xlabel('Receiver', fontsize=11)
    axes[0].set_ylabel('Sender', fontsize=11)
    
    for i in range(n_agents):
        for j in range(n_agents):
            if i != j:
                text = axes[0].text(j, i, f'{mean_topology[i, j]:.2f}',
                                  ha="center", va="center", color="black", fontsize=9)
    
    plt.colorbar(im1, ax=axes[0], label='Average Probability')
    
    # 子图2：标准差
    im2 = axes[1].imshow(std_topology, cmap='Blues', vmin=0, vmax=0.3)
    axes[1].set_title('Topology Standard Deviation\n(Variability)', 
                     fontsize=13, fontweight='bold')
    axes[1].set_xlabel('Receiver', fontsize=11)
    axes[1].set_ylabel('Sender', fontsize=11)
    
    for i in range(n_agents):
        for j in range(n_agents):
            if i != j:
                text = axes[1].text(j, i, f'{std_topology[i, j]:.3f}',
                                  ha="center", va="center", color="black", fontsize=9)
    
    plt.colorbar(im2, ax=axes[1], label='Standard Deviation')
    
    # 子图3：变异系数 (CV = std / mean)
    # 避免除以0
    cv_topology = np.zeros_like(mean_topology)
    mask = mean_topology > 0.01
    cv_topology[mask] = std_topology[mask] / mean_topology[mask]
    
    im3 = axes[2].imshow(cv_topology, cmap='RdYlGn_r', vmin=0, vmax=1)
    axes[2].set_title('Coefficient of Variation\n(Stability Indicator)', 
                     fontsize=13, fontweight='bold')
    axes[2].set_xlabel('Receiver', fontsize=11)
    axes[2].set_ylabel('Sender', fontsize=11)
    
    for i in range(n_agents):
        for j in range(n_agents):
            if i != j and mean_topology[i, j] > 0.01:
                text = axes[2].text(j, i, f'{cv_topology[i, j]:.2f}',
                                  ha="center", va="center", color="black", fontsize=9)
    
    plt.colorbar(im3, ax=axes[2], label='CV (lower=more stable)')
    
    plt.suptitle(f'Topology Statistics Across {n_episodes} Episodes', 
                fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"统计可视化已保存: {save_path}")
    plt.close()
    
    # 打印分析
    print("\n拓扑统计分析:")
    print(f"  平均拓扑密度: {mean_topology[mean_topology > 0].mean():.3f}")
    print(f"  平均标准差: {std_topology[std_topology > 0].mean():.3f}")
    print(f"  平均变异系数: {cv_topology[cv_topology > 0].mean():.3f}")
    
    # 识别最稳定和最不稳定的链路
    stable_edges = []
    unstable_edges = []
    
    for i in range(n_agents):
        for j in range(n_agents):
            if i != j and mean_topology[i, j] > 0.1:
                if cv_topology[i, j] < 0.2:
                    stable_edges.append((i, j, mean_topology[i, j], cv_topology[i, j]))
                elif cv_topology[i, j] > 0.5:
                    unstable_edges.append((i, j, mean_topology[i, j], cv_topology[i, j]))
    
    if stable_edges:
        print("\n最稳定的链路（CV < 0.2）:")
        stable_edges.sort(key=lambda x: x[3])
        for i, j, mean, cv in stable_edges[:5]:
            print(f"  {i}→{j}: 均值={mean:.2f}, CV={cv:.3f}")
    
    if unstable_edges:
        print("\n最不稳定的链路（CV > 0.5）:")
        unstable_edges.sort(key=lambda x: x[3], reverse=True)
        for i, j, mean, cv in unstable_edges[:5]:
            print(f"  {i}→{j}: 均值={mean:.2f}, CV={cv:.3f}")


def compare_topology_at_different_epochs(checkpoint_dir, env, args, epoch_list, save_path='topology_comparison.png'):
    """
    比较不同训练阶段的拓扑
    """
    from magic import MAGIC
    
    checkpoint_dir = Path(checkpoint_dir)
    
    print(f"\n比较不同Epoch的拓扑...")
    
    # 加载不同epoch的模型并提取拓扑
    topologies = {}
    
    for epoch in epoch_list:
        checkpoint_file = checkpoint_dir / f'model_ep{epoch}.pt'
        
        if not checkpoint_file.exists():
            print(f"警告：未找到 {checkpoint_file}")
            continue
        
        # 加载模型
        model = MAGIC(args)
        checkpoint = torch.load(checkpoint_file)
        model.load_state_dict(checkpoint['policy_net'])
        model.eval()
        
        print(f"  提取 Epoch {epoch} 的拓扑...")
        
        # 运行几个episodes提取拓扑
        topology_count = np.zeros((args.nagents, args.nagents))
        total_steps = 0
        
        for ep in range(5):  # 5个episodes
            state = env.reset()
            prev_hid = model.init_hidden(batch_size=state.shape[0])
            
            for t in range(args.max_steps):
                info = {'epoch': -1}
                
                with torch.no_grad():
                    x = [state, prev_hid]
                    action_out, value, prev_hid = model(x, info)
                
                if 'adj' in info:
                    adj = info['adj']
                    if len(adj.shape) == 3:
                        adj = adj[0]
                    topology_count += adj.detach().cpu().numpy()
                else:
                    n = args.nagents
                    full_graph = np.ones((n, n)) - np.eye(n)
                    topology_count += full_graph
                
                total_steps += 1
                
                action = select_action(args, action_out)
                action, actual = translate_action(args, env, action)
                next_state, reward, done, _ = env.step(actual)
                
                state = next_state
                
                if done or t == args.max_steps - 1:
                    break
        
        topology_freq = topology_count / total_steps if total_steps > 0 else topology_count
        topologies[epoch] = topology_freq
    
    # 可视化对比
    n_epochs = len(topologies)
    fig, axes = plt.subplots(1, n_epochs, figsize=(6*n_epochs, 5))
    
    if n_epochs == 1:
        axes = [axes]
    
    for idx, (epoch, topology) in enumerate(sorted(topologies.items())):
        ax = axes[idx]
        
        im = ax.imshow(topology, cmap='YlOrRd', vmin=0, vmax=1)
        ax.set_title(f'Epoch {epoch}', fontsize=13, fontweight='bold')
        ax.set_xlabel('Receiver', fontsize=11)
        ax.set_ylabel('Sender', fontsize=11)
        
        # 添加数值
        for i in range(args.nagents):
            for j in range(args.nagents):
                if i != j:
                    text = ax.text(j, i, f'{topology[i, j]:.2f}',
                                 ha="center", va="center", 
                                 color="black" if topology[i, j] < 0.5 else "white",
                                 fontsize=9)
        
        plt.colorbar(im, ax=ax)
    
    plt.suptitle('Communication Topology Evolution Across Training', 
                fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"对比图已保存: {save_path}")
    plt.close()


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='Visualize Communication Topology Extraction')
    
    # 基本参数
    parser.add_argument('--checkpoint_file', type=str, required=True,
                       help='单个checkpoint文件路径（用于演化和方差分析）')
    parser.add_argument('--checkpoint_dir', type=str, default=None,
                       help='checkpoint目录（用于跨epoch对比）')
    parser.add_argument('--output_dir', type=str, default='./topology_visualization',
                       help='输出目录')
    
    # 环境和模型参数
    parser.add_argument('--env_name', type=str, required=True,
                       help='环境名称')
    parser.add_argument('--nagents', type=int, required=True,
                       help='智能体数量')
    parser.add_argument('--hid_size', type=int, default=128,
                       help='隐藏层大小')
    parser.add_argument('--max_steps', type=int, default=40,
                       help='最大步数')
    
    # 分析参数
    parser.add_argument('--n_episodes', type=int, default=10,
                       help='用于方差分析的episode数量')
    parser.add_argument('--epoch_list', type=int, nargs='+', default=[100, 1000, 2000],
                       help='要对比的epoch列表')
    
    # 环境特定参数
    from utils import init_args_for_env
    init_args_for_env(parser)
    
    args = parser.parse_args()
    
    # 创建输出目录
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*80)
    print("通信拓扑提取可视化")
    print("="*80)
    
    # 初始化环境
    print(f"\n初始化环境: {args.env_name}")
    env = data.init(args.env_name, args, False)
    
    args.obs_size = env.observation_dim
    args.num_actions = env.num_actions
    if not isinstance(args.num_actions, (list, tuple)):
        args.num_actions = [args.num_actions]
    args.dim_actions = env.dim_actions
    args.nfriendly = args.nagents
    
    parse_action_args(args)
    
    # 加载模型
    from magic import MAGIC
    model = MAGIC(args)
    checkpoint = torch.load(args.checkpoint_file)
    model.load_state_dict(checkpoint['policy_net'])
    model.eval()
    
    print(f"已加载模型: {args.checkpoint_file}")
    
    # 1. 可视化单个episode的拓扑演化
    print("\n" + "="*80)
    print("分析1：单个Episode内的拓扑演化")
    print("="*80)
    visualize_topology_evolution_in_episode(
        model, env, args, 
        episode_idx=0,
        save_path=output_dir / 'topology_evolution_episode.png'
    )
    
    # 2. 可视化多个episodes间的拓扑变化
    print("\n" + "="*80)
    print("分析2：多个Episodes间的拓扑统计")
    print("="*80)
    visualize_topology_variance_across_episodes(
        model, env, args,
        n_episodes=args.n_episodes,
        save_path=output_dir / 'topology_variance_analysis.png'
    )
    
    # 3. 如果提供了checkpoint目录，对比不同训练阶段
    if args.checkpoint_dir is not None:
        print("\n" + "="*80)
        print("分析3：不同训练阶段的拓扑对比")
        print("="*80)
        compare_topology_at_different_epochs(
            args.checkpoint_dir, env, args,
            epoch_list=args.epoch_list,
            save_path=output_dir / 'topology_comparison_epochs.png'
        )
    
    print("\n" + "="*80)
    print("所有可视化已完成！")
    print(f"结果保存在: {output_dir}")
    print("="*80)


if __name__ == '__main__':
    main()

