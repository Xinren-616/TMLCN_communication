import matplotlib.pyplot as plt
import matplotlib.colors as colors
import numpy as np
import sys
import glob
import argparse
import re

plt.rcParams.update({'font.size': 16})

# 更新颜色映射，可以根据需要添加更多算法
colors_map = {
    'IC3Net': '#fca503',
    'CommNet': '#b0b0b0',
    'TarMAC-IC3Net': '#b700ff',
    'GA-Comm': '#77ab3f',
    'MAGIC': '#0040ff',  # 您的方法
    'MAGIC w/o Scheduler': '#ff6373'
}

def read_file(vec, file_name, term):
    print(f"读取文件: {file_name}")
    with open(file_name, 'r') as f:
        lines = f.readlines()
        if len(lines) < 2:
            return vec

        for idx, line in enumerate(lines):
            if term not in line:
                continue
            epoch_idx = idx
            epoch_line = line
            while 'Epoch' not in epoch_line and epoch_idx > 0:
                epoch_idx -= 1
                epoch_line = lines[epoch_idx]

            if 'Epoch' not in epoch_line:
                continue

            epoch = int(epoch_line.split(' ')[1].split('\t')[0])

            # 提取Steps-Taken值
            value_part = line.split(': ')[1].strip()
            try:
                value = float(value_part)
                
                while epoch > len(vec):
                    vec.append([])
                if epoch <= len(vec):
                    vec[epoch - 1].append(value)
            except Exception as e:
                print(f"处理Steps-Taken时出错: {e}")
                continue

    return vec

def read_comm_stats(file_name):
    """
    读取通信统计数据
    """
    print(f"读取通信统计数据: {file_name}")
    
    epochs = []
    total_comm_num = []
    active_channels = []
    entropy = []
    total_max_entropy = []
    total_effective_bandwidth = []
    total_similarity = []
    total_elements = []
    comm_prob = []
    
    with open(file_name, 'r', encoding='utf-8', errors='ignore') as f:
        lines = f.readlines()
        print(f"读取到 {len(lines)} 行数据")
        
        current_epoch = None
        epoch_found = False
        is_round_data = False
        
        # 多种可能的Epoch行格式
        epoch_patterns = [
            r'Epoch\s+(\d+)',                # Epoch 123
            r'Epoch[\s:]+(\d+)',             # Epoch: 123
            r'Epoch[^\d]*(\d+)',             # Epoch[任何非数字字符]123
            r'.*?Epoch.*?(\d+).*?'           # 任何包含Epoch和数字的行
        ]
        
        # 统计模式
        stat_patterns = {
            'total_comm_num': [
                r'Total\s+Comm\s+Num.*?:\s*([\d\.]+)',
                r'Total\s+Comm\s+Num\s*:\s*([\d\.]+)',
                r'Total\s+Comm\s+Num.*?Overall.*?:\s*([\d\.]+)'
            ],
            'active_channels': [
                r'Active\s+Channels.*?:\s*([\d\.]+)',
                r'Active\s+Channels\s*:\s*([\d\.]+)'
            ],
            'entropy': [
                r'Entropy\s*:\s*([\d\.]+)',
                r'Entropy.*?:\s*([\d\.]+)'
            ],
            'total_max_entropy': [
                r'Max\s+Entropy\s*:\s*([\d\.]+)',
                r'Total\s+Max\s+Entropy.*?:\s*([\d\.]+)',
                r'Max\s+Entropy.*?:\s*([\d\.]+)'
            ],
            'total_effective_bandwidth': [
                r'Effective\s+Bandwidth\s*:\s*([\d\.]+)',
                r'Total\s+Effective\s+Bandwidth.*?:\s*([\d\.]+)'
            ],
            'total_similarity': [
                r'Message\s+Similarity\s*:\s*([\d\.]+)',
                r'Total\s+Message\s+Similarity.*?:\s*([\d\.]+)'
            ],
            'total_elements': [
                r'Total\s+Elements\s*:\s*([\d\.]+)',
                r'Total\s+Elements.*?:\s*([\d\.]+)'
            ],
            'comm_prob': [
                r'Current\s+Comm\s+Prob.*?:\s*([\d\.]+)',
                r'Comm\s+Prob\s*:\s*([\d\.]+)'
            ]
        }
        
        i = 0
        while i < len(lines):
            line = lines[i]
            
            # 检查是否是ROUND1标记的开始
            if 'ROUND1:' in line:
                is_round_data = True
                i += 1
                continue
            
            # 尝试用多种模式匹配Epoch
            for pattern in epoch_patterns:
                match = re.search(pattern, line)
                if match:
                    try:
                        current_epoch = int(match.group(1))
                        epoch_found = True
                        print(f"找到 Epoch {current_epoch}")
                        break
                    except Exception as e:
                        print(f"解析Epoch时出错: {e}, 行内容: {line}")
                        
            # 如果当前行找到了Epoch信息，检查之前的行是否包含通信统计数据
            if epoch_found and current_epoch is not None and current_epoch not in epochs and is_round_data:
                # 当找到一个新的Epoch，我们收集之前几行中的所有通信统计信息
                # 通常通信统计出现在Epoch之前的几行
                start_idx = max(0, i - 10)  # 向上查找最多10行
                found_stats = False
                
                for j in range(start_idx, i):
                    check_line = lines[j]
                    
                    # 收集该行的所有统计数据
                    for stat_name, patterns in stat_patterns.items():
                        for pattern in patterns:
                            match = re.search(pattern, check_line)
                            if match:
                                try:
                                    value = float(match.group(1))
                                    
                                    if stat_name == 'total_comm_num':
                                        total_comm_num.append(value)
                                        found_stats = True
                                        print(f"Epoch {current_epoch} 的 Total Comm Num: {value}")
                                    elif stat_name == 'active_channels':
                                        active_channels.append(value)
                                        print(f"Epoch {current_epoch} 的 Active Channels: {value}")
                                    elif stat_name == 'entropy':
                                        entropy.append(value)
                                        print(f"Epoch {current_epoch} 的 Entropy: {value}")
                                    elif stat_name == 'total_max_entropy':
                                        total_max_entropy.append(value)
                                        print(f"Epoch {current_epoch} 的 Max Entropy: {value}")
                                    elif stat_name == 'total_effective_bandwidth':
                                        total_effective_bandwidth.append(value)
                                        print(f"Epoch {current_epoch} 的 Effective Bandwidth: {value}")
                                    elif stat_name == 'total_similarity':
                                        total_similarity.append(value)
                                        print(f"Epoch {current_epoch} 的 Message Similarity: {value}")
                                    elif stat_name == 'total_elements':
                                        total_elements.append(value)
                                        print(f"Epoch {current_epoch} 的 Total Elements: {value}")
                                    elif stat_name == 'comm_prob':
                                        comm_prob.append(value)
                                        print(f"Epoch {current_epoch} 的 Comm Prob: {value}")
                                    
                                except Exception as e:
                                    print(f"处理{stat_name}时出错: {e}, 行内容: {check_line}")
                
                # 如果找到了统计数据，将当前epoch添加到列表中
                if found_stats:
                    epochs.append(current_epoch)
                    is_round_data = False  # 重置标志，等待下一个ROUND1
                    
            # 即使不在ROUND1块中，也尝试匹配各种统计数据
            # 有些日志可能没有ROUND1标记
            if not is_round_data:
                for stat_name, patterns in stat_patterns.items():
                    for pattern in patterns:
                        match = re.search(pattern, line)
                        if match:
                            try:
                                value = float(match.group(1))
                                
                                # 针对不同的统计类型执行不同的操作
                                if stat_name == 'total_comm_num':
                                    if current_epoch is not None and current_epoch not in epochs:
                                        epochs.append(current_epoch)
                                        total_comm_num.append(value)
                                        print(f"Epoch {current_epoch} 的 Total Comm Num: {value}")
                                elif stat_name == 'active_channels' and len(active_channels) < len(epochs):
                                    active_channels.append(value)
                                    print(f"Epoch {current_epoch} 的 Active Channels: {value}")
                                elif stat_name == 'entropy' and len(entropy) < len(epochs):
                                    entropy.append(value)
                                    print(f"Epoch {current_epoch} 的 Entropy: {value}")
                                elif stat_name == 'total_max_entropy' and len(total_max_entropy) < len(epochs):
                                    total_max_entropy.append(value)
                                    print(f"Epoch {current_epoch} 的 Max Entropy: {value}")
                                elif stat_name == 'total_effective_bandwidth' and len(total_effective_bandwidth) < len(epochs):
                                    total_effective_bandwidth.append(value)
                                    print(f"Epoch {current_epoch} 的 Effective Bandwidth: {value}")
                                elif stat_name == 'total_similarity' and len(total_similarity) < len(epochs):
                                    total_similarity.append(value)
                                    print(f"Epoch {current_epoch} 的 Message Similarity: {value}")
                                elif stat_name == 'total_elements' and len(total_elements) < len(epochs):
                                    total_elements.append(value)
                                    print(f"Epoch {current_epoch} 的 Total Elements: {value}")
                                elif stat_name == 'comm_prob' and len(comm_prob) < len(epochs):
                                    comm_prob.append(value)
                                    print(f"Epoch {current_epoch} 的 Comm Prob: {value}")
                                
                                break  # 匹配到一个模式后就跳出模式循环
                            except Exception as e:
                                print(f"处理{stat_name}时出错: {e}, 行内容: {line}")
            
            i += 1
    
    # 尝试手动扫描文件中的关键词
    if not epochs:
        print("通过正则表达式没有匹配到数据，尝试手动扫描...")
        with open(file_name, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
            
            # 直接搜索整个文件中的通信指标
            comm_num_matches = re.findall(r'Total Comm Num\s*:\s*([\d\.]+)', content)
            if comm_num_matches:
                print(f"找到 {len(comm_num_matches)} 个 Total Comm Num 值")
                # 如果没有epoch信息，则使用索引作为epoch
                for i, val in enumerate(comm_num_matches):
                    epochs.append(i+1)
                    total_comm_num.append(float(val))
            
            active_channels_matches = re.findall(r'Active Channels\s*:\s*([\d\.]+)', content)
            if active_channels_matches:
                print(f"找到 {len(active_channels_matches)} 个 Active Channels 值")
                active_channels = [float(val) for val in active_channels_matches]
            
            entropy_matches = re.findall(r'Entropy\s*:\s*([\d\.]+)', content)
            if entropy_matches:
                print(f"找到 {len(entropy_matches)} 个 Entropy 值")
                entropy = [float(val) for val in entropy_matches]
            
            max_entropy_matches = re.findall(r'Max Entropy\s*:\s*([\d\.]+)', content)
            if max_entropy_matches:
                print(f"找到 {len(max_entropy_matches)} 个 Max Entropy 值")
                total_max_entropy = [float(val) for val in max_entropy_matches]
            
            effective_bandwidth_matches = re.findall(r'Effective Bandwidth\s*:\s*([\d\.]+)', content)
            if effective_bandwidth_matches:
                print(f"找到 {len(effective_bandwidth_matches)} 个 Effective Bandwidth 值")
                total_effective_bandwidth = [float(val) for val in effective_bandwidth_matches]
            
            message_similarity_matches = re.findall(r'Message Similarity\s*:\s*([\d\.]+)', content)
            if message_similarity_matches:
                print(f"找到 {len(message_similarity_matches)} 个 Message Similarity 值")
                total_similarity = [float(val) for val in message_similarity_matches]
                
            total_elements_matches = re.findall(r'Total Elements\s*:\s*([\d\.]+)', content)
            if total_elements_matches:
                print(f"找到 {len(total_elements_matches)} 个 Total Elements 值")
                total_elements = [float(val) for val in total_elements_matches]
            
            comm_prob_matches = re.findall(r'Comm Prob\s*:\s*([\d\.]+)', content)
            if comm_prob_matches:
                print(f"找到 {len(comm_prob_matches)} 个 Comm Prob 值")
                comm_prob = [float(val) for val in comm_prob_matches]
    
    # 确保所有数组长度一致
    min_length = min(len(epochs), len(total_comm_num))
    if min_length == 0:
        print("警告：找不到匹配的epoch和通信数据！")
        # 如果没有找到匹配的epoch和通信数据，则尝试使用索引作为epoch
        if total_comm_num:
            epochs = list(range(1, len(total_comm_num) + 1))
            min_length = len(total_comm_num)
            print(f"使用索引作为epoch，共 {min_length} 个数据点")
    
    print(f"数据统计: 找到 {len(epochs)} 个epoch, {len(total_comm_num)} 个总通信数, " 
          f"{len(active_channels)} 个活跃通道, {len(entropy)} 个总熵, "
          f"{len(total_max_entropy)} 个最大熵, {len(total_effective_bandwidth)} 个有效带宽, "
          f"{len(total_elements)} 个总元素数, "
          f"{len(total_similarity)} 个消息相似度, {len(comm_prob)} 个通信概率")
    
    if not epoch_found and not epochs:
        print("警告：未找到任何Epoch信息！")
    
    # 处理数组长度不一致的情况
    # 处理新增指标
    if total_elements and len(total_elements) != min_length:
        if len(total_elements) > min_length:
            total_elements = total_elements[:min_length]
        else:
            last_value = total_elements[-1] if total_elements else 0
            total_elements.extend([last_value] * (min_length - len(total_elements)))
            
    if total_max_entropy and len(total_max_entropy) != min_length:
        if len(total_max_entropy) > min_length:
            total_max_entropy = total_max_entropy[:min_length]
        else:
            last_value = total_max_entropy[-1] if total_max_entropy else 0
            total_max_entropy.extend([last_value] * (min_length - len(total_max_entropy)))
    
    if total_effective_bandwidth and len(total_effective_bandwidth) != min_length:
        if len(total_effective_bandwidth) > min_length:
            total_effective_bandwidth = total_effective_bandwidth[:min_length]
        else:
            last_value = total_effective_bandwidth[-1] if total_effective_bandwidth else 0
            total_effective_bandwidth.extend([last_value] * (min_length - len(total_effective_bandwidth)))
    
    if active_channels and len(active_channels) != min_length:
        if len(active_channels) > min_length:
            active_channels = active_channels[:min_length]
        else:
            # 如果长度不够，则用最后一个值填充
            last_value = active_channels[-1] if active_channels else 0
            active_channels.extend([last_value] * (min_length - len(active_channels)))
    
    if entropy and len(entropy) != min_length:
        if len(entropy) > min_length:
            entropy = entropy[:min_length]
        else:
            last_value = entropy[-1] if entropy else 0
            entropy.extend([last_value] * (min_length - len(entropy)))
    
    if total_similarity and len(total_similarity) != min_length:
        if len(total_similarity) > min_length:
            total_similarity = total_similarity[:min_length]
        else:
            last_value = total_similarity[-1] if total_similarity else 0
            total_similarity.extend([last_value] * (min_length - len(total_similarity)))
    
    if comm_prob and len(comm_prob) != min_length:
        if len(comm_prob) > min_length:
            comm_prob = comm_prob[:min_length]
        else:
            last_value = comm_prob[-1] if comm_prob else 1.0
            comm_prob.extend([last_value] * (min_length - len(comm_prob)))
    
    return {
        'epochs': epochs[:min_length],
        'total_comm_num': total_comm_num[:min_length],
        'active_channels': active_channels[:min_length] if active_channels else [],
        'entropy': entropy[:min_length] if entropy else [],
        'total_max_entropy': total_max_entropy[:min_length] if total_max_entropy else [],
        'total_effective_bandwidth': total_effective_bandwidth[:min_length] if total_effective_bandwidth else [],
        'total_elements': total_elements[:min_length] if total_elements else [],
        'total_similarity': total_similarity[:min_length] if total_similarity else [],
        'comm_prob': comm_prob[:min_length] if comm_prob else []
    }

def plot_single_stat(file_name, stat_name, save_path=None):
    """
    绘制单个通信统计数据的双轴图
    
    Arguments:
        file_name (str): 日志文件路径
        stat_name (str): 要绘制的统计数据名称
        save_path (str): 保存图表的路径
    """
    stats = read_comm_stats(file_name)
    
    # 检查是否有数据
    if not stats['epochs'] or stat_name not in stats or not stats[stat_name]:
        print(f"错误：没有找到有效的{stat_name}数据！")
        return
    
    print(f"准备绘制 {len(stats['epochs'])} 个epoch的{stat_name}数据")
    
    # 创建图表
    fig, ax1 = plt.subplots(figsize=(15, 8))
    
    # 获取数据
    data = stats[stat_name]
    epochs = stats['epochs']
    min_len = min(len(epochs), len(data))
    
    # 设置颜色映射
    colors_dict = {
        'total_comm_num': '#0040ff',
        'active_channels': '#77ab3f',
        'entropy': '#b700ff',
        'total_max_entropy': '#ff6373',
        'total_effective_bandwidth': '#fca503',
        'total_message_similarity': '#00a651',
        'total_elements': '#8e44ad',
        'total_similarity': '#00a651'
    }
    
    # 获取颜色
    color = colors_dict.get(stat_name, '#0040ff')
    
    # 绘制左轴 - 通信指标
    ax1.plot(epochs[:min_len], data[:min_len], color=color, linewidth=2.5, marker='o', markersize=5)
    ax1.set_xlabel('Epochs', fontsize=14)
    ax1.set_ylabel(stat_name.replace('_', ' ').title(), color=color, fontsize=14)
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.grid(True, linestyle='--', alpha=0.7)
    
    # 绘制右轴 - 通信概率
    ax2 = ax1.twinx()
    comm_prob_data = stats['comm_prob']
    if comm_prob_data and len(comm_prob_data) > 0:
        # 确保长度匹配
        cp_min_len = min(min_len, len(comm_prob_data))
        print(f"通信概率有效数据点: {cp_min_len}")
        
        if cp_min_len > 0:
            ax2.plot(epochs[:cp_min_len], comm_prob_data[:cp_min_len], 
                    color='red', linestyle='--', linewidth=2.5, marker='x', markersize=5)
            ax2.set_ylabel('Comm Prob', color='red', fontsize=14)
            ax2.tick_params(axis='y', labelcolor='red')
            ax2.set_ylim(0, 1.1)  # 通信概率在0-1之间
    
    # 设置标题
    plt.title(f'{stat_name.replace("_", " ").title()} vs Communication Probability', fontsize=16)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(f"{save_path}_{stat_name}.png", dpi=300, bbox_inches='tight')
        print(f"图表已保存到: {save_path}_{stat_name}.png")
    
    plt.show()  # 确保显示图表

def plot_comm_stats(file_name, save_path=None):
    """
    绘制通信统计数据的双轴图
    """
    stats = read_comm_stats(file_name)
    
    # 检查是否有数据
    if not stats['epochs'] or not stats['total_comm_num']:
        print(f"错误：没有找到有效的通信统计数据！")
        return
    
    print(f"准备绘制 {len(stats['epochs'])} 个epoch的通信统计数据")
    
    # 创建8个子图 (2x4布局)
    fig, axes = plt.subplots(2, 4, figsize=(40, 12))
    fig.suptitle('Index of Communication Statistics', fontsize=20)
    
    # 扁平化axes数组以便迭代
    axes_flat = axes.flatten()
    
    # 定义要绘制的统计数据
    plot_data = [
        {'name': 'Total Comm Num', 'data': stats['total_comm_num'], 'color': '#0040ff', 'idx': 0},
        {'name': 'Active Channels', 'data': stats['active_channels'], 'color': '#77ab3f', 'idx': 1},
        {'name': 'Entropy', 'data': stats['entropy'], 'color': '#b700ff', 'idx': 2},
        {'name': 'Total Max Entropy', 'data': stats['total_max_entropy'], 'color': '#ff6373', 'idx': 3},
        {'name': 'Total Effective Bandwidth', 'data': stats['total_effective_bandwidth'], 'color': '#fca503', 'idx': 4},
        {'name': 'Total Message Similarity', 'data': stats['total_similarity'], 'color': '#00a651', 'idx': 5},
        {'name': 'Total Elements', 'data': stats['total_elements'], 'color': '#8e44ad', 'idx': 6}
    ]
    
    # 为每个指标创建双轴图
    for item in plot_data:
        ax1 = axes_flat[item['idx']]
        ax2 = ax1.twinx()
        
        # 获取数据
        data = item['data']
        print(f"处理 {item['name']} 数据, 长度: {len(data)}")
        
        # 检查是否有数据
        if len(data) > 0:
            # 确保数据和epoch长度一致
            epochs = stats['epochs']
            min_len = min(len(epochs), len(data))
            print(f"{item['name']} 有效数据点: {min_len}")
            
            if min_len > 0:
                # 绘制左轴 - 通信指标
                ax1.plot(epochs[:min_len], data[:min_len], color=item['color'], linewidth=2.0, marker='o', markersize=4)
                ax1.set_xlabel('Epochs')
                ax1.set_ylabel(item['name'], color=item['color'])
                ax1.tick_params(axis='y', labelcolor=item['color'])
                ax1.grid(True, linestyle='--', alpha=0.7)
                
                # 绘制右轴 - 通信概率
                comm_prob_data = stats['comm_prob']
                if comm_prob_data and len(comm_prob_data) > 0:
                    # 确保长度匹配
                    cp_min_len = min(min_len, len(comm_prob_data))
                    print(f"通信概率有效数据点: {cp_min_len}")
                    
                    if cp_min_len > 0:
                        ax2.plot(epochs[:cp_min_len], comm_prob_data[:cp_min_len], 
                                color='red', linestyle='--', linewidth=2.0, marker='x', markersize=4)
                        ax2.set_ylabel('Comm Prob', color='red')
                        ax2.tick_params(axis='y', labelcolor='red')
                        ax2.set_ylim(0, 1.1)  # 通信概率在0-1之间
                
                ax1.set_title(f'{item["name"]}')
            else:
                ax1.set_title(f'No data for {item["name"]}')
        else:
            ax1.set_title(f'No data for {item["name"]}')
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])  # 调整布局，给suptitle留出空间
    
    if save_path:
        plt.savefig(f"{save_path}_comm_stats.png", dpi=300, bbox_inches='tight')
        print(f"图表已保存到: {save_path}_comm_stats.png")
    
    plt.show()  # 确保显示图表

def parse_comm_stats(files, save_path=None):
    """解析并绘制通信统计数据
    
    Args:
        files: 日志文件列表
        save_path: 保存图表的路径前缀
    """
    # 通信指标列表
    metrics = [
        'total_elements',
        'total_comm_num', 
        'active_channels', 
        'entropy', 
        'total_max_entropy', 
        'total_effective_bandwidth',
        'total_similarity'
    ]
    
    # 指标显示名称映射
    metric_names = {
        'total_elements': 'Total Elements',
        'total_comm_num': 'Total Comm Num', 
        'active_channels': 'Active Channels', 
        'entropy': 'Entropy', 
        'total_max_entropy': 'Max Entropy', 
        'total_effective_bandwidth': 'Effective Bandwidth',
        'total_similarity': 'Message Similarity'
    }
    
    # 颜色映射
    colors_dict = {
        'total_elements': '#8e44ad',
        'total_comm_num': '#0040ff',
        'active_channels': '#77ab3f',
        'entropy': '#b700ff',
        'total_max_entropy': '#ff6373',
        'total_effective_bandwidth': '#fca503',
        'total_similarity': '#00a651'
    }
    
    for fname in files:
        # 读取通信统计数据
        stats = read_comm_stats(fname)
        
        # 检查是否有数据
        if not stats['epochs']:
            print(f"错误：在文件 {fname} 中找不到有效的epoch数据！")
            continue
            
        # 文件名（不含路径和扩展名）
        base_name = fname.split('/')[-1].replace('.log', '')
        
        # 为每个指标创建单独的图表
        for metric in metrics:
            if metric not in stats or not stats[metric]:
                print(f"跳过指标 {metric}，未找到数据")
                continue
                
            # 创建图表
            fig, ax = plt.subplots(figsize=(12, 6))
            
            # 获取数据
            data = stats[metric]
            epochs = stats['epochs']
            
            # 确保数据长度一致
            min_len = min(len(epochs), len(data))
            
            # 打印数据统计
            print(f"\n{base_name} 的 {metric_names[metric]} 统计:")
            print(f"数据点数量: {min_len}")
            if min_len > 0:
                print(f"最大值: {np.max(data[:min_len]):.4f}")
                print(f"最小值: {np.min(data[:min_len]):.4f}")
                print(f"平均值: {np.mean(data[:min_len]):.4f}")
            
            # 绘制数据
            ax.plot(epochs[:min_len], data[:min_len], 
                    linewidth=2.0, color=colors_dict[metric], marker='o', markersize=4)
            
            # 设置图表属性
            ax.set_xlabel('Epochs')
            ax.set_ylabel(metric_names[metric])
            ax.grid(True, linestyle='--', alpha=0.7)
            
            plt.title(f'{metric_names[metric]} Throughout Training - {base_name}')
            
            # 保存图表
            if save_path:
                save_file = f"{save_path}_{base_name}_{metric}.png"
                plt.savefig(save_file, dpi=300, bbox_inches='tight')
                print(f"图表已保存到: {save_file}")
            
            plt.show()  # 确保显示图表
        
        # 创建所有指标的组合图
        fig, axes = plt.subplots(2, 4, figsize=(20, 10))
        fig.suptitle(f'Communication Statistics - {base_name}', fontsize=16)
        
        # 扁平化axes数组以便迭代
        axes_flat = axes.flatten()
        
        # 绘制每个指标
        for i, metric in enumerate(metrics):
            if i >= len(axes_flat):
                break
                
            ax = axes_flat[i]
            
            if metric in stats and stats[metric]:
                data = stats[metric]
                epochs = stats['epochs']
                min_len = min(len(epochs), len(data))
                
                ax.plot(epochs[:min_len], data[:min_len], 
                        linewidth=2.0, color=colors_dict[metric], marker='o', markersize=3)
                ax.set_xlabel('Epochs')
                ax.set_ylabel(metric_names[metric])
                ax.grid(True, linestyle='--', alpha=0.7)
                ax.set_title(metric_names[metric])
            else:
                ax.set_title(f'No data for {metric_names[metric]}')
        
        # 隐藏多余的子图
        for i in range(len(metrics), len(axes_flat)):
            axes_flat[i].axis('off')
        
        plt.tight_layout(rect=[0, 0, 1, 0.96])  # 调整布局，给suptitle留出空间
        
        # 保存组合图
        if save_path:
            save_file = f"{save_path}_{base_name}_all_metrics.png"
            plt.savefig(save_file, dpi=300, bbox_inches='tight')
            print(f"组合图表已保存到: {save_file}")
        
        plt.show()  # 确保显示图表

def parse_plot(files, save_path=None):
    coll = dict()
    
    for fname in files:
        # 确定算法标签
        if 'ic3net' in fname.lower() and not 'tar' in fname.lower():
            label = 'IC3Net'
        elif 'commnet' in fname.lower():
            label = 'CommNet'
        elif 'magic' in fname.lower() or 'gcomm' in fname.lower():
            if 'scheduler' in fname.lower() or 'complete' in fname.lower():
                label = 'MAGIC w/o Scheduler'
            else:
                label = 'MAGIC'
        else:
            # 从文件名提取算法名称
            label = fname.split('/')[-1].replace('.log', '')
            
        if label not in coll:
            coll[label] = []

        coll[label] = read_file(coll[label], fname, 'Steps-Taken')

    # 创建图形
    fig, ax = plt.subplots(figsize=(12, 6))
    
    for label in coll.keys():
        if not coll[label]:
            print(f"警告：{label}没有Steps-Taken数据")
            continue
            
        # 计算Steps-Taken的均值和标准差
        epochs = min(2000, len(coll[label]))
        mean_values = []
        max_values = []
        min_values = []

        for i in range(epochs):
            if i < len(coll[label]) and coll[label][i]:
                mean = np.mean(coll[label][i])
                mean_values.append(mean)
                
                std = np.std(coll[label][i]) / np.sqrt(len(coll[label][i]))
                std = min(std, 5)  # 限制标准差范围
                max_values.append(mean + std)
                min_values.append(mean - std)
            else:
                if mean_values:
                    mean_values.append(mean_values[-1])
                    max_values.append(max_values[-1])
                    min_values.append(min_values[-1])
                else:
                    mean_values.append(0)
                    max_values.append(0)
                    min_values.append(0)

        # 打印统计信息
        print(f"\n{label}算法的Steps-Taken统计:")
        print(f"最大值: {np.max(mean_values):.4f}")
        print(f"最小值: {np.min(mean_values):.4f}")
        print(f"平均值: {np.mean(mean_values):.4f}")
        
        # 绘制Steps-Taken曲线
        color = colors_map.get(label, np.random.rand(3,))
        ax.plot(np.arange(len(mean_values)), mean_values, 
                linewidth=2.0, label=label, color=color)
        ax.fill_between(np.arange(len(mean_values)), min_values, max_values, 
                       color=colors.to_rgba(color, alpha=0.2))

    # 设置图表属性
    ax.set_xlabel('Epochs')
    ax.set_ylabel('Steps-Taken')
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.legend(loc='best')
    
    plt.title('Steps-Taken Throughout Training')
    
    if save_path:
        plt.savefig(f"{save_path}_steps_taken.png", dpi=300, bbox_inches='tight')
        print(f"图表已保存到: {save_path}_steps_taken.png")
    
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Plot Steps-Taken and Communication Statistics')
    parser.add_argument('log_path', type=str, help='日志文件路径(可使用通配符)')
    parser.add_argument('--save', type=str, default=None, help='保存图表的路径前缀')
    parser.add_argument('--comm_stats', action='store_true', help='绘制所有通信统计数据')
    parser.add_argument('--single_stat', type=str, default=None, help='绘制单个指标，可选值: total_comm_num, active_channels, entropy, total_max_entropy, total_effective_bandwidth, total_similarity, total_elements')
    parser.add_argument('--comm_metrics', action='store_true', help='绘制所有通信指标')
    
    args = parser.parse_args()
    
    files = glob.glob(args.log_path)
    # 过滤掉.pt文件
    files = list(filter(lambda x: '.pt' not in x, files))
    
    if not files:
        print(f"错误：找不到匹配的日志文件: {args.log_path}")
        sys.exit(1)
    
    print(f"找到{len(files)}个日志文件:")
    for f in files:
        print(f"- {f}")
    
    if args.single_stat:
        # 绘制单个指标
        for f in files:
            plot_single_stat(f, args.single_stat, args.save)
    elif args.comm_stats:
        # 绘制所有通信统计数据
        for f in files:
            plot_comm_stats(f, args.save)
    elif args.comm_metrics:
        # 绘制所有通信指标
        parse_comm_stats(files, args.save)
    else:
        # 绘制Steps-Taken
        parse_plot(files, args.save)