import matplotlib.pyplot as plt
import matplotlib.colors as colors
import numpy as np
import sys
import glob
import argparse
import re

plt.rcParams.update({'font.size': 12})

# 颜色映射
colors_map = {
    'IC3Net': '#fca503',
    'CommNet': '#b0b0b0',
    'TarMAC-IC3Net': '#b700ff',
    'GA-Comm': '#77ab3f',
    'MAGIC': '#0040ff',
    'MAGIC w/o Scheduler': '#ff6373'
}

def parse_ci_matrix(line_list, start_idx):
    """
    从日志行列表中解析 CI Matrix
    
    Args:
        line_list: 日志文件的所有行
        start_idx: "CI Matrix (i→j):" 这一行的索引
    
    Returns:
        matrix: numpy array of shape (n, n), 对角线为 NaN
    """
    # 跳过标题行，开始读取矩阵数据
    matrix_lines = []
    idx = start_idx + 1
    
    while idx < len(line_list):
        line = line_list[idx].strip()
        # 如果遇到空行或者新的section，停止
        if not line or line.startswith('Epoch') or line.startswith('Episode'):
            break
        
        # 解析矩阵行，格式如: "  [  -  , 0.0003, 0.0003, 0.0003, 0.0003]"
        if '[' in line and ']' in line:
            # 提取方括号内的内容
            content = line[line.find('[') + 1:line.find(']')]
            # 分割并处理每个值
            values = []
            for val in content.split(','):
                val = val.strip()
                if val == '-':
                    values.append(np.nan)  # 对角线用 NaN 表示
                else:
                    try:
                        values.append(float(val))
                    except ValueError:
                        values.append(np.nan)
            matrix_lines.append(values)
        
        idx += 1
    
    if not matrix_lines:
        return None
    
    return np.array(matrix_lines)

def read_ci_matrices(file_name):
    """
    读取日志文件中所有的 CI Matrix
    
    Returns:
        epochs: list of epoch numbers
        matrices: list of numpy arrays
    """
    print(f"读取文件: {file_name}")
    epochs = []
    matrices = []
    
    with open(file_name, 'r') as f:
        lines = f.readlines()
        
        for idx, line in enumerate(lines):
            if 'CI Matrix (i→j):' in line:
                # 向前查找最近的 Epoch 行
                epoch = None
                for back_idx in range(idx - 1, max(0, idx - 200), -1):
                    epoch_line = lines[back_idx].strip()
                    if epoch_line.startswith('Epoch '):
                        try:
                            parts = epoch_line.split()
                            if len(parts) >= 2:
                                epoch = int(parts[1])
                                break
                        except (ValueError, IndexError):
                            pass
                
                if epoch is None:
                    continue
                
                # 解析矩阵
                matrix = parse_ci_matrix(lines, idx)
                if matrix is not None:
                    epochs.append(epoch)
                    matrices.append(matrix)
    
    print(f"成功读取 {len(epochs)} 个 epoch 的 CI 矩阵")
    return epochs, matrices

def plot_combined_ci_analysis(files, save_path=None, key_epochs=None):
    """
    绘制组合图：上半部分为关键epoch的热力图，下半部分为均值和标准差的折线图
    
    Args:
        files: 日志文件列表
        save_path: 保存路径前缀
        key_epochs: 要展示热力图的关键epoch列表，如果为None则自动选择
    """
    # 读取所有文件的数据
    all_data = {}
    
    for fname in files:
        # 确定算法标签
        fname_lower = fname.lower()
        if 'ic3net' in fname_lower and 'tar' not in fname_lower:
            label = 'IC3Net'
        elif 'commnet' in fname_lower and 'gacomm' not in fname_lower:
            label = 'CommNet'
        elif 'magic' in fname_lower or 'gacomm' in fname_lower or 'ga-comm' in fname_lower:
            if 'scheduler' in fname_lower or 'complete' in fname_lower:
                label = 'MAGIC w/o Scheduler'
            else:
                label = 'MAGIC'
        elif 'tarmac' in fname_lower or 'tar' in fname_lower:
            label = 'TarMAC-IC3Net'
        else:
            label = fname.split('/')[-1].replace('.log', '')
        
        epochs, matrices = read_ci_matrices(fname)
        if epochs:
            all_data[label] = {'epochs': epochs, 'matrices': matrices}
    
    if not all_data:
        print("错误：没有找到任何 CI 矩阵数据")
        return
    
    # 为每个算法创建一个组合图
    for label, data in all_data.items():
        epochs = data['epochs']
        matrices = data['matrices']
        
        if not epochs:
            continue
        
        # 自动选择关键 epoch（如果未指定）
        if key_epochs is None:
            max_epoch = max(epochs)
            if max_epoch >= 1800:
                selected_epochs = [1, 50, 100, 300, 500, 750, 1000, 1500, max_epoch]
            elif max_epoch >= 1000:
                selected_epochs = [1, 20, 30, 40, 50, 100, 300, 800, 1000]
            elif max_epoch >= 500:
                selected_epochs = [1, 50, 150, 300, 450, max_epoch]
            else:
                selected_epochs = [1, 50, 150, 160, 170, 200, 280, 450, max_epoch]
        else:
            selected_epochs = key_epochs
        
        # 计算每个epoch的统计信息（均值和标准差）
        mean_ci = []
        std_ci = []
        
        for matrix in matrices:
            # 提取非对角线元素（非NaN）
            off_diagonal = matrix[~np.isnan(matrix)]
            if len(off_diagonal) > 0:
                mean_ci.append(np.mean(off_diagonal))
                std_ci.append(np.std(off_diagonal))
            else:
                mean_ci.append(0)
                std_ci.append(0)
        
        mean_ci = np.array(mean_ci)
        std_ci = np.array(std_ci)
        
        # 创建组合图
        n_heatmaps = len(selected_epochs)
        n_cols = 3
        n_rows_heatmap = (n_heatmaps + n_cols - 1) // n_cols  # 向上取整
        
        fig = plt.figure(figsize=(16, 12))
        
        # 创建gridspec布局：热力图行 + 均值图 + 标准差图
        from matplotlib.gridspec import GridSpec
        gs = GridSpec(n_rows_heatmap + 2, n_cols, figure=fig,
                     height_ratios=[1]*n_rows_heatmap + [1.2, 0.8],
                     hspace=0.35, wspace=0.3)
        
        # 首先找到所有选定epoch的全局最小值和最大值（用于统一颜色范围）
        global_vmin = float('inf')
        global_vmax = float('-inf')
        selected_matrices = []
        
        for epoch in selected_epochs:
            # 找到最接近的epoch
            if epoch in epochs:
                idx = epochs.index(epoch)
            else:
                idx = min(range(len(epochs)), key=lambda x: abs(epochs[x] - epoch))
            
            matrix = matrices[idx]
            selected_matrices.append((epochs[idx], matrix))
            
            # 更新全局最小值和最大值（忽略NaN）
            valid_values = matrix[~np.isnan(matrix)]
            if len(valid_values) > 0:
                global_vmin = min(global_vmin, valid_values.min())
                global_vmax = max(global_vmax, valid_values.max())
        
        print(f"\n热力图颜色范围: [{global_vmin:.6f}, {global_vmax:.6f}]")
        
        # 上半部分：热力图（使用统一的颜色范围）
        for i, (actual_epoch, matrix) in enumerate(selected_matrices):
            
            # 创建子图（使用gridspec）
            row = i // n_cols
            col = i % n_cols
            ax = fig.add_subplot(gs[row, col])
            
            # 绘制热力图
            # 创建mask来隐藏对角线
            mask = np.isnan(matrix)
            plot_matrix = np.copy(matrix)
            plot_matrix[mask] = 0  # 对角线设为0以便绘图
            
            # 使用统一的颜色范围
            im = ax.imshow(plot_matrix, cmap='YlOrRd', aspect='auto', 
                          vmin=global_vmin, vmax=global_vmax)
            
            # 设置标题
            ax.set_title(f'Epoch {actual_epoch}', fontsize=14, fontweight='bold')
            
            # 设置坐标轴
            n_agents = matrix.shape[0]
            ax.set_xticks(np.arange(n_agents))
            ax.set_yticks(np.arange(n_agents))
            ax.set_xticklabels([f'{i}' for i in range(n_agents)])
            ax.set_yticklabels([f'{i}' for i in range(n_agents)])
            ax.set_xlabel('Agent j (receiver)', fontsize=10)
            ax.set_ylabel('Agent i (sender)', fontsize=10)
            
            # 在每个格子中显示数值
            for row in range(n_agents):
                for col in range(n_agents):
                    if not mask[row, col]:
                        text = ax.text(col, row, f'{matrix[row, col]:.4f}',
                                     ha="center", va="center", color="black", fontsize=8)
                    else:
                        # 对角线显示 "-"
                        text = ax.text(col, row, '-',
                                     ha="center", va="center", color="gray", fontsize=10)
            
            # 添加颜色条
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        
        # 下半部分：使用两个子图，一个显示均值，一个显示标准差
        # 均值图（占据倒数第二行的所有列）
        ax_mean = fig.add_subplot(gs[n_rows_heatmap, :])
        
        color = colors_map.get(label, '#0040ff')
        
        # 绘制均值曲线和标准差区域
        ax_mean.fill_between(epochs, 
                             mean_ci - std_ci, 
                             mean_ci + std_ci,
                             color=colors.to_rgba(color, alpha=0.3),
                             label=f'±1 Std Dev')
        
        ax_mean.plot(epochs, mean_ci, linewidth=2.5, label=f'Mean CI', 
                    color=color, marker='o', markersize=3, markevery=max(1, len(epochs)//50))
        
        ax_mean.plot(epochs, mean_ci + std_ci, linewidth=1.0, 
                    linestyle='--', color=color, alpha=0.5)
        ax_mean.plot(epochs, mean_ci - std_ci, linewidth=1.0, 
                    linestyle='--', color=color, alpha=0.5)
        
        ax_mean.set_ylabel('Mean CI ± Std', fontsize=12, fontweight='bold')
        ax_mean.set_title('Mean CI of Matrix Over Training', fontsize=13, fontweight='bold')
        ax_mean.legend(loc='best', fontsize=10)
        ax_mean.grid(True, linestyle='--', alpha=0.5)
        
        # 标准差图（占据最后一行的所有列，共享x轴）
        ax_std = fig.add_subplot(gs[n_rows_heatmap + 1, :], sharex=ax_mean)
        
        ax_std.plot(epochs, std_ci, linewidth=2.0, label=f'Std Dev', 
                   color='darkred', marker='s', markersize=2, markevery=max(1, len(epochs)//50))
        ax_std.fill_between(epochs, 0, std_ci, color='darkred', alpha=0.2)
        
        ax_std.set_xlabel('Training Epochs', fontsize=12, fontweight='bold')
        ax_std.set_ylabel('Std Deviation', fontsize=12, fontweight='bold')
        ax_std.set_title('Standard Deviation of CI Matrix', fontsize=13, fontweight='bold')
        ax_std.legend(loc='best', fontsize=10)
        ax_std.grid(True, linestyle='--', alpha=0.5)
        
        # 添加统计信息文本到均值图
        max_mean_ci = np.max(mean_ci)
        min_mean_ci = np.min(mean_ci)
        final_mean_ci = mean_ci[-1]
        final_std_ci = std_ci[-1]
        avg_std_ci = np.mean(std_ci)
        max_std_ci = np.max(std_ci)
        min_std_ci = np.min(std_ci)
        
        stats_text = f'Mean CI:\n'
        stats_text += f'  Max: {max_mean_ci:.6f}\n'
        stats_text += f'  Min: {min_mean_ci:.6f}\n'
        stats_text += f'  Final: {final_mean_ci:.6f}'
        
        ax_mean.text(0.02, 0.98, stats_text, transform=ax_mean.transAxes,
                    fontsize=9, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.6))
        
        # 添加统计信息到标准差图
        std_stats_text = f'Std Dev:\n'
        std_stats_text += f'  Avg: {avg_std_ci:.6f}\n'
        std_stats_text += f'  Max: {max_std_ci:.6f}\n'
        std_stats_text += f'  Final: {final_std_ci:.6f}'
        
        ax_std.text(0.02, 0.98, std_stats_text, transform=ax_std.transAxes,
                   fontsize=9, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.6))
        
        # 整体标题
        fig.suptitle(f'CI Matrix Analysis - {label}', fontsize=16, fontweight='bold', y=0.995)
        
        plt.tight_layout(rect=[0, 0, 1, 0.99])
        
        if save_path:
            filename = f"{save_path}_{label.replace(' ', '_')}_combined.png"
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            print(f"图表已保存到: {filename}")
        
        plt.show()

def plot_multi_algorithm_comparison(files, save_path=None):
    """
    绘制多算法的 CI 均值对比图
    """
    all_data = {}
    
    for fname in files:
        # 确定算法标签
        fname_lower = fname.lower()
        if 'ic3net' in fname_lower and 'tar' not in fname_lower:
            label = 'IC3Net'
        elif 'commnet' in fname_lower and 'gacomm' not in fname_lower:
            label = 'CommNet'
        elif 'magic' in fname_lower or 'gacomm' in fname_lower or 'ga-comm' in fname_lower:
            if 'scheduler' in fname_lower or 'complete' in fname_lower:
                label = 'MAGIC w/o Scheduler'
            else:
                label = 'MAGIC'
        elif 'tarmac' in fname_lower or 'tar' in fname_lower:
            label = 'TarMAC-IC3Net'
        else:
            label = fname.split('/')[-1].replace('.log', '')
        
        epochs, matrices = read_ci_matrices(fname)
        if epochs:
            all_data[label] = {'epochs': epochs, 'matrices': matrices}
    
    if not all_data:
        print("错误：没有找到任何 CI 矩阵数据")
        return
    
    plt.figure(figsize=(12, 7))
    
    for label, data in all_data.items():
        epochs = data['epochs']
        matrices = data['matrices']
        
        # 计算均值和标准差
        mean_ci = []
        std_ci = []
        
        for matrix in matrices:
            off_diagonal = matrix[~np.isnan(matrix)]
            if len(off_diagonal) > 0:
                mean_ci.append(np.mean(off_diagonal))
                std_ci.append(np.std(off_diagonal))
            else:
                mean_ci.append(0)
                std_ci.append(0)
        
        mean_ci = np.array(mean_ci)
        std_ci = np.array(std_ci)
        
        color = colors_map.get(label, np.random.rand(3,))
        
        # 绘制均值曲线
        plt.plot(epochs, mean_ci, linewidth=2.5, label=label, color=color,
                marker='o', markersize=3, markevery=max(1, len(epochs)//50))
        
        # 绘制标准差阴影
        plt.fill_between(epochs, mean_ci - std_ci, mean_ci + std_ci,
                        color=colors.to_rgba(color, alpha=0.15))
        
        # 打印统计信息
        print(f"\n{label} CI 统计:")
        print(f"  最大均值: {np.max(mean_ci):.6f}")
        print(f"  最小均值: {np.min(mean_ci):.6f}")
        print(f"  最终均值: {mean_ci[-1]:.6f} ± {std_ci[-1]:.6f}")
    
    plt.xlabel('Training Epochs', fontsize=14, fontweight='bold')
    plt.ylabel('Average CI (off-diagonal)', fontsize=14, fontweight='bold')
    plt.title('CI Matrix Mean Comparison Across Algorithms', fontsize=16, fontweight='bold')
    plt.legend(loc='best', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.5)
    
    if save_path:
        filename = f"{save_path}_comparison.png"
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"\n图表已保存到: {filename}")
    
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='绘制 CI Matrix 可视化图表')
    parser.add_argument('log_path', type=str, help='日志文件路径(可使用通配符)')
    parser.add_argument('--mode', type=str, default='combined', 
                       choices=['combined', 'comparison'],
                       help='绘图模式: combined=单算法组合图(默认), comparison=多算法对比')
    parser.add_argument('--epochs', type=str, default=None,
                       help='关键epoch列表，用逗号分隔，如: 1,100,300,500,800,1000')
    parser.add_argument('--save', type=str, default=None, help='保存图表的路径前缀')
    
    args = parser.parse_args()
    
    files = glob.glob(args.log_path)
    files = list(filter(lambda x: '.pt' not in x, files))
    
    if not files:
        print(f"错误：找不到匹配的日志文件: {args.log_path}")
        sys.exit(1)
    
    print(f"找到{len(files)}个日志文件:")
    for f in files:
        print(f"- {f}")
    
    # 解析关键epoch
    key_epochs = None
    if args.epochs:
        try:
            key_epochs = [int(e.strip()) for e in args.epochs.split(',')]
            print(f"使用指定的关键epoch: {key_epochs}")
        except ValueError:
            print("警告：epoch参数格式错误，将使用默认值")
    
    if args.mode == 'combined':
        plot_combined_ci_analysis(files, args.save, key_epochs)
    else:
        plot_multi_algorithm_comparison(files, args.save)

