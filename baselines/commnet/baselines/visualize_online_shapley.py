"""
可视化在线Shapley值结果

这个脚本用于可视化在线计算的Shapley值结果，生成各种图表帮助分析。

使用方法：
python visualize_online_shapley.py --input_dir ./shapley_results --output_dir ./shapley_plots
"""

import argparse
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path


def load_shapley_results(input_dir):
    """加载Shapley值结果"""
    input_dir = Path(input_dir)
    
    # 加载Shapley值
    with open(input_dir / 'online_shapley_values.json', 'r') as f:
        shapley_data = json.load(f)
    
    # 加载拓扑
    with open(input_dir / 'online_topologies.json', 'r') as f:
        topology_data = json.load(f)
    
    # 加载统计信息
    with open(input_dir / 'online_shapley_stats.json', 'r') as f:
        stats = json.load(f)
    
    return shapley_data, topology_data, stats


def plot_shapley_matrix(shapley_data, epoch, n_agents, output_path):
    """绘制指定epoch的Shapley值矩阵热图"""
    # 创建矩阵
    shapley_matrix = np.zeros((n_agents, n_agents))
    
    epoch_data = shapley_data[str(epoch)]
    for edge_str, info in epoch_data.items():
        i, j = map(int, edge_str.split('->'))
        shapley_matrix[i, j] = info['shapley_value']
    
    # 绘图
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # 使用RdYlGn colormap，正值为绿色，负值为红色
    im = ax.imshow(shapley_matrix, cmap='RdYlGn', aspect='auto',
                   vmin=-0.1, vmax=0.1)
    
    # 添加colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Shapley Value', fontsize=12)
    
    # 添加数值标注
    for i in range(n_agents):
        for j in range(n_agents):
            if i != j and shapley_matrix[i, j] != 0:
                text_color = 'white' if abs(shapley_matrix[i, j]) > 0.05 else 'black'
                ax.text(j, i, f'{shapley_matrix[i, j]:.3f}',
                       ha="center", va="center", color=text_color, fontsize=10)
            elif i == j:
                ax.text(j, i, '-', ha="center", va="center", 
                       color='gray', fontsize=12)
    
    # 设置标签
    ax.set_xticks(np.arange(n_agents))
    ax.set_yticks(np.arange(n_agents))
    ax.set_xticklabels([f'Agent {i}' for i in range(n_agents)])
    ax.set_yticklabels([f'Agent {i}' for i in range(n_agents)])
    
    ax.set_xlabel('Receiver Agent', fontsize=12)
    ax.set_ylabel('Sender Agent', fontsize=12)
    ax.set_title(f'Edge-Level Shapley Values (Epoch {epoch})', 
                fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"  ✓ Shapley矩阵图已保存: {output_path}")
    plt.close()


def plot_shapley_evolution(shapley_data, output_path):
    """绘制Shapley值随训练的演化"""
    epochs = sorted([int(e) for e in shapley_data.keys()])
    
    # 收集所有边
    all_edges = set()
    for epoch_data in shapley_data.values():
        all_edges.update(epoch_data.keys())
    
    # 为每条边收集时间序列
    edge_evolution = {}
    for edge in all_edges:
        edge_evolution[edge] = {
            'epochs': [],
            'values': []
        }
        for epoch in epochs:
            if edge in shapley_data[str(epoch)]:
                edge_evolution[edge]['epochs'].append(epoch)
                edge_evolution[edge]['values'].append(
                    shapley_data[str(epoch)][edge]['shapley_value']
                )
    
    # 计算每条边的平均重要性
    edge_importance = {}
    for edge, data in edge_evolution.items():
        if len(data['values']) > 0:
            edge_importance[edge] = np.mean(np.abs(data['values']))
    
    # 选择Top 10重要的边
    top_edges = sorted(edge_importance.items(), key=lambda x: x[1], reverse=True)[:10]
    
    # 绘图
    fig, ax = plt.subplots(figsize=(14, 8))
    
    colors = plt.cm.tab20(np.linspace(0, 1, len(top_edges)))
    
    for idx, (edge, _) in enumerate(top_edges):
        data = edge_evolution[edge]
        ax.plot(data['epochs'], data['values'], 
               label=f'Edge {edge}',
               marker='o', linewidth=2, markersize=6,
               color=colors[idx])
    
    ax.axhline(y=0, color='red', linestyle='--', alpha=0.5, linewidth=1)
    ax.set_xlabel('Training Epoch', fontsize=12)
    ax.set_ylabel('Shapley Value', fontsize=12)
    ax.set_title('Top 10 Important Edges: Shapley Value Evolution', 
                fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=9, ncol=2)
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"  ✓ Shapley演化图已保存: {output_path}")
    plt.close()


def plot_shapley_statistics(shapley_data, output_path):
    """绘制Shapley值统计信息"""
    epochs = sorted([int(e) for e in shapley_data.keys()])
    
    # 收集统计数据
    stats = {
        'mean': [],
        'std': [],
        'max': [],
        'min': [],
        'positive_ratio': []
    }
    
    for epoch in epochs:
        epoch_data = shapley_data[str(epoch)]
        values = [info['shapley_value'] for info in epoch_data.values()]
        
        stats['mean'].append(np.mean(values))
        stats['std'].append(np.std(values))
        stats['max'].append(np.max(values))
        stats['min'].append(np.min(values))
        stats['positive_ratio'].append(sum(1 for v in values if v > 0) / len(values))
    
    # 绘图
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 子图1: 均值和标准差
    ax1 = axes[0, 0]
    ax1.plot(epochs, stats['mean'], 'b-o', linewidth=2, label='Mean')
    ax1.fill_between(epochs, 
                     np.array(stats['mean']) - np.array(stats['std']),
                     np.array(stats['mean']) + np.array(stats['std']),
                     alpha=0.3, color='blue')
    ax1.axhline(y=0, color='red', linestyle='--', alpha=0.5)
    ax1.set_xlabel('Epoch', fontsize=11)
    ax1.set_ylabel('Shapley Value', fontsize=11)
    ax1.set_title('Mean Shapley Value (± Std)', fontsize=12, fontweight='bold')
    ax1.legend()
    ax1.grid(alpha=0.3)
    
    # 子图2: 最大值和最小值
    ax2 = axes[0, 1]
    ax2.plot(epochs, stats['max'], 'g-o', linewidth=2, label='Max')
    ax2.plot(epochs, stats['min'], 'r-o', linewidth=2, label='Min')
    ax2.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax2.set_xlabel('Epoch', fontsize=11)
    ax2.set_ylabel('Shapley Value', fontsize=11)
    ax2.set_title('Max and Min Shapley Values', fontsize=12, fontweight='bold')
    ax2.legend()
    ax2.grid(alpha=0.3)
    
    # 子图3: 正贡献边比例
    ax3 = axes[1, 0]
    ax3.plot(epochs, stats['positive_ratio'], 'purple-o', linewidth=2)
    ax3.set_xlabel('Epoch', fontsize=11)
    ax3.set_ylabel('Ratio', fontsize=11)
    ax3.set_title('Ratio of Positive-Contribution Edges', fontsize=12, fontweight='bold')
    ax3.set_ylim([0, 1])
    ax3.grid(alpha=0.3)
    
    # 子图4: 标准差
    ax4 = axes[1, 1]
    ax4.plot(epochs, stats['std'], 'orange-o', linewidth=2)
    ax4.set_xlabel('Epoch', fontsize=11)
    ax4.set_ylabel('Standard Deviation', fontsize=11)
    ax4.set_title('Shapley Value Variability', fontsize=12, fontweight='bold')
    ax4.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"  ✓ 统计图已保存: {output_path}")
    plt.close()


def plot_topology_evolution(topology_data, output_path):
    """绘制通信拓扑的演化"""
    epochs = sorted([int(e) for e in topology_data.keys()])
    
    # 计算每个epoch的拓扑统计
    stats = {
        'mean_freq': [],
        'sparsity': [],
        'max_freq': []
    }
    
    for epoch in epochs:
        topology = np.array(topology_data[str(epoch)])
        n = topology.shape[0]
        
        # 去除对角线
        mask = ~np.eye(n, dtype=bool)
        values = topology[mask]
        
        stats['mean_freq'].append(np.mean(values))
        stats['sparsity'].append(np.sum(values < 0.1) / len(values))
        stats['max_freq'].append(np.max(values))
    
    # 绘图
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    # 子图1: 平均使用频率
    axes[0].plot(epochs, stats['mean_freq'], 'b-o', linewidth=2)
    axes[0].set_xlabel('Epoch', fontsize=11)
    axes[0].set_ylabel('Mean Frequency', fontsize=11)
    axes[0].set_title('Average Edge Usage Frequency', fontsize=12, fontweight='bold')
    axes[0].grid(alpha=0.3)
    
    # 子图2: 稀疏度
    axes[1].plot(epochs, stats['sparsity'], 'r-o', linewidth=2)
    axes[1].set_xlabel('Epoch', fontsize=11)
    axes[1].set_ylabel('Sparsity Ratio', fontsize=11)
    axes[1].set_title('Topology Sparsity (freq < 0.1)', fontsize=12, fontweight='bold')
    axes[1].set_ylim([0, 1])
    axes[1].grid(alpha=0.3)
    
    # 子图3: 最大频率
    axes[2].plot(epochs, stats['max_freq'], 'g-o', linewidth=2)
    axes[2].set_xlabel('Epoch', fontsize=11)
    axes[2].set_ylabel('Max Frequency', fontsize=11)
    axes[2].set_title('Maximum Edge Usage Frequency', fontsize=12, fontweight='bold')
    axes[2].grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"  ✓ 拓扑演化图已保存: {output_path}")
    plt.close()


def generate_summary_report(shapley_data, topology_data, stats, output_path):
    """生成文本摘要报告"""
    epochs = sorted([int(e) for e in shapley_data.keys()])
    n_agents = stats['n_agents']
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("="*80 + "\n")
        f.write("在线Shapley值分析 - 摘要报告\n")
        f.write("="*80 + "\n\n")
        
        # 基本信息
        f.write("基本信息:\n")
        f.write(f"  - 智能体数量: {n_agents}\n")
        f.write(f"  - 可能的边数: {n_agents * (n_agents - 1)}\n")
        f.write(f"  - 分析的epochs: {epochs}\n")
        f.write(f"  - 评估episodes: {stats['n_eval_episodes']}\n")
        f.write(f"  - 拓扑阈值: {stats['topology_threshold']}\n")
        f.write(f"  - 平均计算时间: {stats['avg_computation_time']:.2f}秒\n\n")
        
        # 每个epoch的详细信息
        f.write("各Epoch详细信息:\n")
        f.write("-"*80 + "\n")
        
        for epoch in epochs:
            epoch_data = shapley_data[str(epoch)]
            values = [info['shapley_value'] for info in epoch_data.values()]
            
            f.write(f"\nEpoch {epoch}:\n")
            f.write(f"  分析的边数: {len(values)}\n")
            f.write(f"  Shapley值统计:\n")
            f.write(f"    - 平均值: {np.mean(values):.6f}\n")
            f.write(f"    - 标准差: {np.std(values):.6f}\n")
            f.write(f"    - 最大值: {np.max(values):.6f}\n")
            f.write(f"    - 最小值: {np.min(values):.6f}\n")
            f.write(f"    - 正贡献边: {sum(1 for v in values if v > 0)}/{len(values)}\n")
            f.write(f"    - 负贡献边: {sum(1 for v in values if v < 0)}/{len(values)}\n")
            
            # Top 5重要边
            sorted_edges = sorted(epoch_data.items(), 
                                key=lambda x: abs(x[1]['shapley_value']), 
                                reverse=True)[:5]
            f.write(f"  Top 5重要边:\n")
            for edge, info in sorted_edges:
                f.write(f"    - {edge}: {info['shapley_value']:.6f}\n")
        
        f.write("\n" + "="*80 + "\n")
        f.write("报告生成完成\n")
        f.write("="*80 + "\n")
    
    print(f"  ✓ 摘要报告已保存: {output_path}")


def main():
    parser = argparse.ArgumentParser(description='可视化在线Shapley值结果')
    parser.add_argument('--input_dir', type=str, required=True,
                       help='输入目录（包含Shapley值JSON文件）')
    parser.add_argument('--output_dir', type=str, default='./shapley_plots',
                       help='输出目录（保存图表）')
    
    args = parser.parse_args()
    
    # 创建输出目录
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*80)
    print("在线Shapley值结果可视化")
    print("="*80)
    print(f"输入目录: {args.input_dir}")
    print(f"输出目录: {args.output_dir}")
    print("="*80 + "\n")
    
    # 加载数据
    print("[1/6] 加载数据...")
    shapley_data, topology_data, stats = load_shapley_results(args.input_dir)
    print(f"  ✓ 加载了 {len(shapley_data)} 个epochs的数据\n")
    
    n_agents = stats['n_agents']
    epochs = sorted([int(e) for e in shapley_data.keys()])
    
    # 生成各种图表
    print("[2/6] 生成Shapley值矩阵图...")
    for epoch in epochs:
        plot_shapley_matrix(shapley_data, epoch, n_agents,
                          output_dir / f'shapley_matrix_epoch{epoch}.png')
    
    print("\n[3/6] 生成Shapley值演化图...")
    plot_shapley_evolution(shapley_data, 
                          output_dir / 'shapley_evolution.png')
    
    print("\n[4/6] 生成统计图...")
    plot_shapley_statistics(shapley_data,
                           output_dir / 'shapley_statistics.png')
    
    print("\n[5/6] 生成拓扑演化图...")
    plot_topology_evolution(topology_data,
                           output_dir / 'topology_evolution.png')
    
    print("\n[6/6] 生成摘要报告...")
    generate_summary_report(shapley_data, topology_data, stats,
                           output_dir / 'summary_report.txt')
    
    print("\n" + "="*80)
    print("可视化完成！")
    print("="*80)
    print(f"所有图表已保存到: {output_dir}")
    print("\n生成的文件:")
    print(f"  - shapley_matrix_epoch*.png: 各epoch的Shapley值矩阵")
    print(f"  - shapley_evolution.png: Top 10边的Shapley值演化")
    print(f"  - shapley_statistics.png: Shapley值统计信息")
    print(f"  - topology_evolution.png: 通信拓扑演化")
    print(f"  - summary_report.txt: 文本摘要报告")
    print("="*80)


if __name__ == '__main__':
    main()



