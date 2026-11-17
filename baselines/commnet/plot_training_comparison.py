#!/usr/bin/env python3
"""
训练性能对比图
对比各baseline在剪枝前后的Success Rate和Reward变化
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
    'commnet': '#1f77b4',   # 蓝色
    'ic3net': '#ff7f0e',    # 橙色
    'tarmac': '#2ca02c',    # 绿色
    'magic': '#d62728',     # 红色
    'gacomm': '#9467bd',    # 紫色
}

BASELINE_NAMES = {
    'commnet': 'CommNet',
    'ic3net': 'IC3Net',
    'tarmac': 'TarMAC',
    'magic': 'MAGIC',
    'gacomm': 'GAComm',
}

def parse_log_file(log_file):
    """解析训练日志文件"""
    epochs = []
    successes = []
    rewards = []
    
    try:
        with open(log_file, 'r') as f:
            current_epoch = None
            for line in f:
                # 匹配 "Epoch 1160/2000"
                epoch_match = re.match(r'Epoch (\d+)/\d+', line)
                if epoch_match:
                    current_epoch = int(epoch_match.group(1))
                
                # 匹配 "  Success: 0.9820"
                if current_epoch and '  Success:' in line:
                    success_match = re.search(r'Success:\s+([\d.]+)', line)
                    if success_match:
                        successes.append(float(success_match.group(1)))
                        epochs.append(current_epoch)
                
                # 匹配 "  Reward: -0.7586"
                if current_epoch and '  Reward:' in line:
                    reward_match = re.search(r'Reward:\s+([-\d.]+)', line)
                    if reward_match:
                        rewards.append(float(reward_match.group(1)))
    
    except FileNotFoundError:
        print(f"警告: 找不到文件 {log_file}")
        return None, None, None
    
    return np.array(epochs), np.array(successes), np.array(rewards)

def plot_success_rate_full(data_dict, output_dir, args):
    """绘制完整的Success Rate对比曲线"""
    fig, ax = plt.subplots(figsize=(12, 6))
    
    for model, (epochs, successes, _) in data_dict.items():
        if successes is not None and len(successes) > 0:
            ax.plot(epochs, successes, 
                   label=BASELINE_NAMES[model],
                   color=BASELINE_COLORS[model],
                   linewidth=2,
                   alpha=0.8)
    
    # 添加剪枝开始的垂直线
    ax.axvline(x=500, color='red', linestyle='--', linewidth=1.5, 
              alpha=0.5, label='Pruning Starts')
    
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Success Rate', fontsize=12)
    ax.set_title('Success Rate Comparison (Full Training)', fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 2000)
    ax.set_ylim(0, 1.05)
    
    plt.tight_layout()
    
    # 保存
    output_file = output_dir / 'success_rate_full'
    if args.format in ['pdf', 'both']:
        plt.savefig(f'{output_file}.pdf', dpi=args.dpi, bbox_inches='tight')
    if args.format in ['png', 'both']:
        plt.savefig(f'{output_file}.png', dpi=args.dpi, bbox_inches='tight')
    
    print(f"  ✓ 已保存: {output_file}.{args.format}")
    plt.close()

def plot_success_rate_phases(data_dict, output_dir, args):
    """绘制剪枝前后的Success Rate对比"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # 左图：剪枝前 (0-500)
    for model, (epochs, successes, _) in data_dict.items():
        if successes is not None and len(successes) > 0:
            mask = epochs <= 500
            ax1.plot(epochs[mask], successes[mask],
                    label=BASELINE_NAMES[model],
                    color=BASELINE_COLORS[model],
                    linewidth=2,
                    alpha=0.8)
    
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Success Rate', fontsize=12)
    ax1.set_title('Before Pruning (Epoch 0-500)', fontsize=14, fontweight='bold')
    ax1.legend(loc='best', fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(0, 500)
    ax1.set_ylim(0, 1.05)
    
    # 右图：剪枝期间 (500-2000)
    for model, (epochs, successes, _) in data_dict.items():
        if successes is not None and len(successes) > 0:
            mask = epochs >= 500
            ax2.plot(epochs[mask], successes[mask],
                    label=BASELINE_NAMES[model],
                    color=BASELINE_COLORS[model],
                    linewidth=2,
                    alpha=0.8)
    
    ax2.axvline(x=500, color='red', linestyle='--', linewidth=1.5,
               alpha=0.5, label='Pruning Starts')
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Success Rate', fontsize=12)
    ax2.set_title('During Pruning (Epoch 500-2000)', fontsize=14, fontweight='bold')
    ax2.legend(loc='best', fontsize=10)
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(500, 2000)
    ax2.set_ylim(0, 1.05)
    
    plt.tight_layout()
    
    # 保存
    output_file = output_dir / 'success_rate_phases'
    if args.format in ['pdf', 'both']:
        plt.savefig(f'{output_file}.pdf', dpi=args.dpi, bbox_inches='tight')
    if args.format in ['png', 'both']:
        plt.savefig(f'{output_file}.png', dpi=args.dpi, bbox_inches='tight')
    
    print(f"  ✓ 已保存: {output_file}.{args.format}")
    plt.close()

def plot_success_milestones(data_dict, output_dir, args):
    """绘制关键节点的Success Rate柱状图"""
    milestones = [500, 1000, 1500, 2000]
    models = list(data_dict.keys())
    
    # 提取各节点的成功率
    milestone_data = {m: [] for m in milestones}
    
    for milestone in milestones:
        for model in models:
            epochs, successes, _ = data_dict[model]
            if successes is not None and len(successes) > 0:
                # 找到最接近milestone的epoch
                idx = np.argmin(np.abs(epochs - milestone))
                if abs(epochs[idx] - milestone) <= 10:  # 容忍10个epoch的误差
                    milestone_data[milestone].append(successes[idx])
                else:
                    milestone_data[milestone].append(np.nan)
            else:
                milestone_data[milestone].append(np.nan)
    
    # 绘图
    fig, ax = plt.subplots(figsize=(12, 6))
    
    x = np.arange(len(milestones))
    width = 0.15
    
    for i, model in enumerate(models):
        values = [milestone_data[m][i] for m in milestones]
        offset = (i - len(models)/2) * width
        ax.bar(x + offset, values, width,
              label=BASELINE_NAMES[model],
              color=BASELINE_COLORS[model],
              alpha=0.8)
    
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Success Rate', fontsize=12)
    ax.set_title('Success Rate at Key Milestones', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(milestones)
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim(0, 1.05)
    
    # 添加剪枝开始的标注
    ax.axvline(x=0, color='red', linestyle='--', linewidth=1.5,
              alpha=0.5)
    ax.text(0, 1.02, 'Pruning Starts', ha='center', fontsize=10, color='red')
    
    plt.tight_layout()
    
    # 保存
    output_file = output_dir / 'success_rate_milestones'
    if args.format in ['pdf', 'both']:
        plt.savefig(f'{output_file}.pdf', dpi=args.dpi, bbox_inches='tight')
    if args.format in ['png', 'both']:
        plt.savefig(f'{output_file}.png', dpi=args.dpi, bbox_inches='tight')
    
    print(f"  ✓ 已保存: {output_file}.{args.format}")
    plt.close()

def plot_reward_comparison(data_dict, output_dir, args):
    """绘制Reward对比曲线"""
    fig, ax = plt.subplots(figsize=(12, 6))
    
    for model, (epochs, _, rewards) in data_dict.items():
        if rewards is not None and len(rewards) > 0:
            ax.plot(epochs, rewards,
                   label=BASELINE_NAMES[model],
                   color=BASELINE_COLORS[model],
                   linewidth=2,
                   alpha=0.8)
    
    # 添加剪枝开始的垂直线
    ax.axvline(x=500, color='red', linestyle='--', linewidth=1.5,
              alpha=0.5, label='Pruning Starts')
    
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Average Reward', fontsize=12)
    ax.set_title('Reward Comparison (Full Training)', fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 2000)
    
    plt.tight_layout()
    
    # 保存
    output_file = output_dir / 'reward_comparison'
    if args.format in ['pdf', 'both']:
        plt.savefig(f'{output_file}.pdf', dpi=args.dpi, bbox_inches='tight')
    if args.format in ['png', 'both']:
        plt.savefig(f'{output_file}.png', dpi=args.dpi, bbox_inches='tight')
    
    print(f"  ✓ 已保存: {output_file}.{args.format}")
    plt.close()

def main():
    parser = argparse.ArgumentParser(description='训练性能对比图')
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
    output_dir = Path('figures') / 'training_comparison'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*80)
    print("训练性能对比图")
    print("="*80)
    print()
    
    # 解析日志文件
    print("解析日志文件...")
    baselines = ['commnet', 'ic3net', 'tarmac', 'magic', 'gacomm']
    data_dict = {}
    
    for model in baselines:
        log_file = f'train_tj_{model}_prune_online_shapley_{args.date}.log'
        if model == 'tarmac':
            log_file = f'train_tj_tarcomm_prune_online_shapley_{args.date}.log'
        
        print(f"  - {BASELINE_NAMES[model]}: {log_file}")
        epochs, successes, rewards = parse_log_file(log_file)
        
        if successes is not None:
            print(f"    ✓ 解析成功: {len(epochs)} epochs")
            data_dict[model] = (epochs, successes, rewards)
        else:
            print(f"    ✗ 解析失败")
    
    print()
    
    if not data_dict:
        print("错误: 没有成功解析任何日志文件")
        return 1
    
    # 生成图表
    print("生成图表...")
    print()
    
    print("[1/4] Success Rate 完整曲线...")
    plot_success_rate_full(data_dict, output_dir, args)
    
    print("[2/4] Success Rate 分阶段对比...")
    plot_success_rate_phases(data_dict, output_dir, args)
    
    print("[3/4] Success Rate 关键节点...")
    plot_success_milestones(data_dict, output_dir, args)
    
    print("[4/4] Reward 对比曲线...")
    plot_reward_comparison(data_dict, output_dir, args)
    
    print()
    print("="*80)
    print(f"完成！图表已保存到: {output_dir.absolute()}/")
    print("="*80)
    
    return 0

if __name__ == '__main__':
    import sys
    sys.exit(main())






