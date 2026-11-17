"""
Topology Pruning Module

基于Shapley值的拓扑剪枝：
1. 在warm-up期后开始剪枝
2. 根据Shapley值threshold剪除低贡献边
3. 每100 epoch更新剪枝mask（方案A：动态更新，可以"复活"边）
4. 记录剪枝历史和统计
"""

import numpy as np
import json
from pathlib import Path


class TopologyPruner:
    """基于Shapley值的拓扑剪枝器"""
    
    def __init__(self, args, n_agents):
        """
        初始化拓扑剪枝器
        
        Args:
            args: 训练参数
                - pruning_threshold: Shapley值阈值（默认-0.3）
                - pruning_start_epoch: 开始剪枝的epoch（默认500）
                - pruning_interval: 剪枝更新频率（默认100）
            n_agents: 智能体数量
        """
        self.threshold = getattr(args, 'pruning_threshold', -0.3)
        self.start_epoch = getattr(args, 'pruning_start_epoch', 500)
        self.update_interval = getattr(args, 'pruning_interval', 100)
        self.n_agents = n_agents
        self.max_edges = n_agents * (n_agents - 1)
        
        # 当前使用的mask（None表示不剪枝，全连接）
        self.current_mask = None
        
        # 剪枝历史记录
        self.pruning_history = {
            'epoch': [],
            'num_pruned_edges': [],
            'num_kept_edges': [],
            'pruning_ratio': [],
            'shapley_stats': []
        }
        
        print(f"\n{'='*80}")
        print(f"拓扑剪枝器已初始化")
        print(f"  - Shapley值阈值: {self.threshold}")
        print(f"  - 开始剪枝epoch: {self.start_epoch}")
        print(f"  - 更新间隔: {self.update_interval} epochs")
        print(f"  - 智能体数量: {n_agents}")
        print(f"  - 最大边数: {self.max_edges}")
        print(f"  - 策略: 动态剪枝（每{self.update_interval}epoch重新评估）")
        print(f"{'='*80}\n")
    
    def should_prune(self, epoch):
        """
        判断是否应该执行剪枝
        
        Args:
            epoch: 当前epoch编号
        
        Returns:
            bool: True表示应该剪枝
        """
        return epoch >= self.start_epoch
    
    def should_update_mask(self, epoch):
        """
        判断是否应该更新剪枝mask
        
        Args:
            epoch: 当前epoch编号
        
        Returns:
            bool: True表示应该更新mask
        """
        # 在开始剪枝后，每隔update_interval个epoch更新一次
        return (epoch >= self.start_epoch and 
                epoch % self.update_interval == 0)
    
    def update_mask(self, shapley_matrix, epoch):
        """
        根据Shapley值更新剪枝mask
        
        Args:
            shapley_matrix: [n, n] Shapley值矩阵
            epoch: 当前epoch编号
        
        Returns:
            mask: [n, n] 剪枝mask，1表示保留，0表示剪枝
        """
        if epoch < self.start_epoch:
            # Warm-up期，不剪枝，返回全1 mask
            self.current_mask = np.ones((self.n_agents, self.n_agents))
            np.fill_diagonal(self.current_mask, 0)  # 对角线为0
            return self.current_mask
        
        # 根据threshold生成mask
        # mask[i,j] = 1 if shapley_matrix[i,j] >= threshold else 0
        mask = (shapley_matrix >= self.threshold).astype(float)
        
        # 确保对角线为0（自己不与自己通信）
        np.fill_diagonal(mask, 0)
        
        # 统计信息
        num_kept_edges = int(mask.sum())
        num_pruned_edges = self.max_edges - num_kept_edges
        pruning_ratio = num_pruned_edges / self.max_edges
        
        # Shapley值统计
        valid_shapley = shapley_matrix[~np.eye(self.n_agents, dtype=bool)]
        shapley_stats = {
            'mean': float(np.mean(valid_shapley)),
            'std': float(np.std(valid_shapley)),
            'min': float(np.min(valid_shapley)),
            'max': float(np.max(valid_shapley)),
            'median': float(np.median(valid_shapley))
        }
        
        # 记录历史
        self.pruning_history['epoch'].append(epoch)
        self.pruning_history['num_pruned_edges'].append(num_pruned_edges)
        self.pruning_history['num_kept_edges'].append(num_kept_edges)
        self.pruning_history['pruning_ratio'].append(pruning_ratio)
        self.pruning_history['shapley_stats'].append(shapley_stats)
        
        # 更新当前mask
        self.current_mask = mask
        
        # 打印剪枝信息
        self._print_pruning_info(epoch, shapley_matrix, mask, num_kept_edges, 
                                 num_pruned_edges, pruning_ratio, shapley_stats)
        
        return self.current_mask
    
    def get_mask(self, epoch=None):
        """
        获取当前epoch应使用的mask
        
        Args:
            epoch: 当前epoch编号（可选）
        
        Returns:
            mask: [n, n] 剪枝mask，如果为None则表示不剪枝
        """
        if epoch is not None and epoch < self.start_epoch:
            # Warm-up期，不使用剪枝
            return None
        
        return self.current_mask
    
    def _print_pruning_info(self, epoch, shapley_matrix, mask, 
                           num_kept_edges, num_pruned_edges, 
                           pruning_ratio, shapley_stats):
        """打印剪枝信息"""
        print(f"\n{'#'*80}")
        print(f"# Epoch {epoch}: 拓扑剪枝更新")
        print(f"{'#'*80}")
        print(f"\n  Shapley值统计:")
        print(f"    - 平均值: {shapley_stats['mean']:.6f}")
        print(f"    - 标准差: {shapley_stats['std']:.6f}")
        print(f"    - 范围: [{shapley_stats['min']:.6f}, {shapley_stats['max']:.6f}]")
        print(f"    - 中位数: {shapley_stats['median']:.6f}")
        
        print(f"\n  剪枝结果:")
        print(f"    - 阈值: {self.threshold}")
        print(f"    - 保留边数: {num_kept_edges}/{self.max_edges}")
        print(f"    - 剪枝边数: {num_pruned_edges}/{self.max_edges}")
        print(f"    - 剪枝比例: {pruning_ratio:.2%}")
        
        # 统计各个Shapley值区间的边数
        valid_shapley = shapley_matrix[~np.eye(self.n_agents, dtype=bool)]
        below_threshold = (valid_shapley < self.threshold).sum()
        above_threshold = (valid_shapley >= self.threshold).sum()
        
        print(f"\n  Shapley值分布:")
        print(f"    - Shapley < {self.threshold}: {below_threshold} 条边（被剪枝）")
        print(f"    - Shapley >= {self.threshold}: {above_threshold} 条边（保留）")
        
        print(f"\n{'#'*80}\n")
    
    def save_history(self, save_dir, filename='pruning_history.json'):
        """
        保存剪枝历史
        
        Args:
            save_dir: 保存目录
            filename: 文件名
        """
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        save_path = save_dir / filename
        with open(save_path, 'w') as f:
            json.dump(self.pruning_history, f, indent=2)
        
        print(f"剪枝历史已保存: {save_path}")
    
    def get_summary(self):
        """获取剪枝统计摘要"""
        if len(self.pruning_history['epoch']) == 0:
            return "没有剪枝历史"
        
        summary = {
            'total_updates': len(self.pruning_history['epoch']),
            'avg_pruning_ratio': np.mean(self.pruning_history['pruning_ratio']),
            'final_kept_edges': self.pruning_history['num_kept_edges'][-1],
            'final_pruned_edges': self.pruning_history['num_pruned_edges'][-1],
        }
        return summary







