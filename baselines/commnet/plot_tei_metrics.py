#!/usr/bin/env python3
"""
TEI (Total Edge Influence) 指标分析
TEI = Success Rate / 通信边数 × 100
综合评估通信效率
"""

import re
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from collections import defaultdict

# 配色方案
BASELINE_COLORS = {
    'commnet': '#1f77b4',
    'ic3net': '#ff7f0e',
    'tarmac': '#2ca02c',
    'magic': '#d62728',
    'gacomm': '#9467bd',
}

BASELINE_NAMES = {
    'commnet': 'CommNet',
    'ic3net': 'IC3Net',
    'tarmac': 'TarMAC',
    'magic': 'MAGIC',
    'gacomm': 'GAComm',
}

def parse_log_file(log_file):
    """解析训练日志，提取success rate"""
    epochs = []
    successes = []
    
    try:
        with open(log_file, 'r') as f:
            current_epoch = None
            for line in f:
                epoch_match = re.match(r'Epoch (\d+)/\d+', line)
                if epoch_match:
                    current_epoch = int(epoch_match.group(1))
                
                if current_epoch and '  Success:' in line:
                    success_match = re.search(r'Success:\s+([\d.]+)', line)
                    if success_match:
                        successes.append(float(success_match.group(1)))
                        epochs.append(current_epoch)
    
    except FileNotFoundError:
        return None, None
    
    return np.array(epochs), np.array(successes)

def parse_shapley_topology(shapley_dir):
    """从shapley结果中提取通信边数"""
    topology_file = shapley_dir / 'online_topologies.json'
    
    if not topology_file.exists():
        return {}
    
    try:
        with open(topology_file, 'r') as f:
            topologies = json.load(f)
        
        edge_counts = {}
        for epoch_str, topology in topologies.items():
            epoch = int(epoch_str)
            topology_array = np.array(topology)
            # 计算非零边数（通信边数）
            edge_count = np.sum(topology_array > 0.1)  # 阈值0.1
            edge_counts[epoch] = edge_count
        
        return edge_counts
    
    except Exception as e:
        print(f"    警告: 解析topology文件失败: {e}")
        return {}

def calculate_tei(successes, edge_counts):
    """计算TEI指标"""
    if edge_counts == 0:
        return 0
    return (successes / edge_counts) * 100

def plot_tei_evolution(data_dict, output_dir, args):
    """绘制TEI随epoch的演化"""
    fig, ax = plt.subplots(figsize=(12, 6))
    
    for model, data in data_dict.items():
        if data['tei_epochs'] is not None and len(data['tei_epochs']) > 0:
            ax.plot(data['tei_epochs'], data['tei_values'],
                   label=BASELINE_NAMES[model],
                   color=BASELINE_COLORS[model],
                   linewidth=2,
                   marker='o',
                   markersize=4,
                   alpha=0.8)
    
    # 添加剪枝开始的垂直线
    ax.axvline(x=500, color='red', linestyle='--', linewidth=1.5,
              alpha=0.5, label='Pruning Starts')
    
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('TEI (Success Rate / Edge Count × 100)', fontsize=12)
    ax.set_title('TEI Evolution During Training', fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    output_file = output_dir / 'tei_evolution'
    if args.format in ['pdf', 'both']:
        plt.savefig(f'{output_file}.pdf', dpi=args.dpi, bbox_inches='tight')
    if args.format in ['png', 'both']:
        plt.savefig(f'{output_file}.png', dpi=args.dpi, bbox_inches='tight')
    
    print(f"  ✓ 已保存: {output_file}.{args.format}")
    plt.close()

def plot_tei_improvement(data_dict, output_dir, args):
    """绘制TEI提升百分比"""
    models = []
    tei_before = []
    tei_after = []
    improvements = []
    
    for model, data in data_dict.items():
        if data['tei_before'] is not None and data['tei_after'] is not None:
            models.append(BASELINE_NAMES[model])
            tei_before.append(data['tei_before'])
            tei_after.append(data['tei_after'])
            improvement = ((data['tei_after'] - data['tei_before']) / data['tei_before']) * 100
            improvements.append(improvement)
    
    if not models:
        print("  警告: 没有足够的数据绘制TEI提升图")
        return
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # 左图：剪枝前后TEI对比
    x = np.arange(len(models))
    width = 0.35
    
    bars1 = ax1.bar(x - width/2, tei_before, width, label='Before Pruning (Epoch 500)',
                    alpha=0.8)
    bars2 = ax1.bar(x + width/2, tei_after, width, label='After Pruning (Epoch 2000)',
                    alpha=0.8)
    
    # 为每个柱子设置颜色
    for i, (bar1, bar2) in enumerate(zip(bars1, bars2)):
        model_key = list(data_dict.keys())[i]
        bar1.set_color(BASELINE_COLORS[model_key])
        bar2.set_color(BASELINE_COLORS[model_key])
        bar2.set_alpha(0.6)
    
    ax1.set_xlabel('Model', fontsize=12)
    ax1.set_ylabel('TEI Value', fontsize=12)
    ax1.set_title('TEI: Before vs After Pruning', fontsize=14, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(models, rotation=15, ha='right')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3, axis='y')
    
    # 右图：TEI提升百分比
    colors = [BASELINE_COLORS[list(data_dict.keys())[i]] for i in range(len(models))]
    bars = ax2.bar(models, improvements, color=colors, alpha=0.8)
    
    # 添加数值标签
    for bar, imp in zip(bars, improvements):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{imp:+.1f}%',
                ha='center', va='bottom' if imp > 0 else 'top',
                fontsize=10)
    
    ax2.set_xlabel('Model', fontsize=12)
    ax2.set_ylabel('TEI Improvement (%)', fontsize=12)
    ax2.set_title('TEI Improvement After Pruning', fontsize=14, fontweight='bold')
    ax2.set_xticklabels(models, rotation=15, ha='right')
    ax2.grid(True, alpha=0.3, axis='y')
    ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
    
    plt.tight_layout()
    
    output_file = output_dir / 'tei_improvement'
    if args.format in ['pdf', 'both']:
        plt.savefig(f'{output_file}.pdf', dpi=args.dpi, bbox_inches='tight')
    if args.format in ['png', 'both']:
        plt.savefig(f'{output_file}.png', dpi=args.dpi, bbox_inches='tight')
    
    print(f"  ✓ 已保存: {output_file}.{args.format}")
    plt.close()

def plot_tei_ranking(data_dict, output_dir, args):
    """绘制最终TEI排名"""
    models = []
    final_tei = []
    
    for model, data in data_dict.items():
        if data['tei_after'] is not None:
            models.append(BASELINE_NAMES[model])
            final_tei.append(data['tei_after'])
    
    if not models:
        print("  警告: 没有足够的数据绘制TEI排名图")
        return
    
    # 排序
    sorted_indices = np.argsort(final_tei)[::-1]  # 降序
    sorted_models = [models[i] for i in sorted_indices]
    sorted_tei = [final_tei[i] for i in sorted_indices]
    sorted_colors = [BASELINE_COLORS[list(data_dict.keys())[i]] 
                     for i in sorted_indices]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    bars = ax.barh(sorted_models, sorted_tei, color=sorted_colors, alpha=0.8)
    
    # 添加数值标签
    for bar, tei in zip(bars, sorted_tei):
        width = bar.get_width()
        ax.text(width, bar.get_y() + bar.get_height()/2.,
                f'{tei:.2f}',
                ha='left', va='center',
                fontsize=11,
                fontweight='bold')
    
    ax.set_xlabel('TEI Value (Higher is Better)', fontsize=12)
    ax.set_ylabel('Model', fontsize=12)
    ax.set_title('Final TEI Ranking (Epoch 2000)', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='x')
    
    # 添加排名标签
    for i, (model, tei) in enumerate(zip(sorted_models, sorted_tei)):
        ax.text(-0.5, i, f'#{i+1}',
               ha='right', va='center',
               fontsize=12,
               fontweight='bold',
               color='gray')
    
    plt.tight_layout()
    
    output_file = output_dir / 'tei_ranking'
    if args.format in ['pdf', 'both']:
        plt.savefig(f'{output_file}.pdf', dpi=args.dpi, bbox_inches='tight')
    if args.format in ['png', 'both']:
        plt.savefig(f'{output_file}.png', dpi=args.dpi, bbox_inches='tight')
    
    print(f"  ✓ 已保存: {output_file}.{args.format}")
    plt.close()

def main():
    parser = argparse.ArgumentParser(description='TEI指标分析')
    parser.add_argument('--date', type=str, default='20260122',
                       help='日志文件日期')
    parser.add_argument('--format', type=str, default='pdf',
                       choices=['pdf', 'png', 'both'],
                       help='输出格式')
    parser.add_argument('--dpi', type=int, default=300,
                       help='图片分辨率')
    parser.add_argument('--style', type=str, default='seaborn-v0_8-paper',
                       help='绘图风格')
    args = parser.parse_args()
    
    # 设置绘图风格
    try:
        plt.style.use(args.style)
    except:
        print(f"警告: 无法使用风格 {args.style}，使用默认风格")
    
    # 创建输出目录
    output_dir = Path('figures') / 'tei_metrics'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*80)
    print("TEI指标分析")
    print("="*80)
    print()
    
    # 解析数据
    print("解析数据...")
    baselines = ['commnet', 'ic3net', 'tarmac', 'magic', 'gacomm']
    data_dict = {}
    
    for model in baselines:
        print(f"  - {BASELINE_NAMES[model]}")
        
        # 日志文件
        log_file = f'train_tj_{model}_prune_online_shapley_{args.date}.log'
        if model == 'tarmac':
            log_file = f'train_tj_tarcomm_prune_online_shapley_{args.date}.log'
        
        # Shapley结果目录
        shapley_dir = Path('saved') / f'tj_{model}_prune_online_shapley_{args.date}' / 'shapley_results'
        if model == 'tarmac':
            shapley_dir = Path('saved') / f'tj_tarcomm_prune_online_shapley_{args.date}' / 'shapley_results'
        
        # 解析success rate
        epochs, successes = parse_log_file(log_file)
        
        if successes is None:
            print(f"    ✗ 解析失败")
            continue
        
        # 解析edge counts
        edge_counts_dict = parse_shapley_topology(shapley_dir)
        
        if not edge_counts_dict:
            print(f"    ⚠ 无法获取通信边数，跳过")
            continue
        
        # 计算TEI
        tei_epochs = []
        tei_values = []
        
        for epoch in sorted(edge_counts_dict.keys()):
            # 找到对应的success rate
            idx = np.where(epochs == epoch)[0]
            if len(idx) > 0:
                success = successes[idx[0]]
                edge_count = edge_counts_dict[epoch]
                if edge_count > 0:
                    tei = calculate_tei(success, edge_count)
                    tei_epochs.append(epoch)
                    tei_values.append(tei)
        
        if not tei_epochs:
            print(f"    ✗ 无法计算TEI")
            continue
        
        # 计算剪枝前后的TEI
        tei_before = None
        tei_after = None
        
        for i, epoch in enumerate(tei_epochs):
            if epoch >= 400 and epoch <= 500 and tei_before is None:
                tei_before = tei_values[i]
            if epoch >= 1900 and tei_after is None:
                tei_after = tei_values[i]
        
        data_dict[model] = {
            'tei_epochs': np.array(tei_epochs),
            'tei_values': np.array(tei_values),
            'tei_before': tei_before,
            'tei_after': tei_after,
        }
        
        print(f"    ✓ 计算了 {len(tei_epochs)} 个epochs的TEI")
        if tei_before and tei_after:
            improvement = ((tei_after - tei_before) / tei_before) * 100
            print(f"      TEI提升: {improvement:+.1f}%")
    
    print()
    
    if not data_dict:
        print("错误: 没有成功解析任何数据")
        return 1
    
    # 生成图表
    print("生成图表...")
    print()
    
    print("[1/3] TEI演化曲线...")
    plot_tei_evolution(data_dict, output_dir, args)
    
    print("[2/3] TEI提升百分比...")
    plot_tei_improvement(data_dict, output_dir, args)
    
    print("[3/3] 最终TEI排名...")
    plot_tei_ranking(data_dict, output_dir, args)
    
    print()
    print("="*80)
    print(f"完成！图表已保存到: {output_dir.absolute()}/")
    print("="*80)
    
    return 0

if __name__ == '__main__':
    import sys
    sys.exit(main())






