"""
Communication Statistics Module

用于统计和分析通信开销：
1. 每个epoch的总通信次数
2. 通信密度
3. TEI (Task Efficiency Index) 指标
4. 活跃边统计
"""

import numpy as np
import json
from pathlib import Path
from collections import defaultdict


class CommunicationStatsCollector:
    """通信统计收集器"""
    
    def __init__(self, n_agents, save_dir=None):
        """
        初始化通信统计收集器
        
        Args:
            n_agents: 智能体数量
            save_dir: 统计结果保存目录
        """
        self.n_agents = n_agents
        self.max_edges = n_agents * (n_agents - 1)  # 最大可能的通信边数
        self.save_dir = Path(save_dir) if save_dir else None
        
        # 存储历史统计
        self.history = {
            'epoch': [],
            'total_comm_count': [],
            'avg_comm_per_step': [],
            'comm_density': [],
            'active_edges': [],
            'success_rate': [],
            'tei': []
        }
        
        if self.save_dir:
            self.save_dir.mkdir(parents=True, exist_ok=True)
            
        print(f"\n{'='*80}")
        print(f"通信统计收集器已初始化")
        print(f"  - 智能体数量: {n_agents}")
        print(f"  - 最大通信边数: {self.max_edges}")
        if self.save_dir:
            print(f"  - 保存目录: {self.save_dir}")
        print(f"{'='*80}\n")
    
    def compute_epoch_stats(self, comm_history, success_rate, epoch):
        """
        计算一个epoch的通信统计
        
        Args:
            comm_history: List of communication graphs
                         每个graph是 [n, n] 的矩阵，1表示通信，0表示未通信
                         来自所有episodes的所有steps
            success_rate: 该epoch的成功率
            epoch: 当前epoch编号
        
        Returns:
            stats: 统计字典
        """
        if len(comm_history) == 0:
            return self._get_zero_stats()
        
        # 将所有step的通信图求和
        total_comm_matrix = np.zeros((self.n_agents, self.n_agents))
        for comm_graph in comm_history:
            total_comm_matrix += comm_graph
        
        # 计算各项统计
        total_comm_count = int(total_comm_matrix.sum())
        avg_comm_per_step = float(total_comm_count / len(comm_history))
        comm_density = float(total_comm_count / (self.max_edges * len(comm_history)))
        active_edges = int((total_comm_matrix > 0).sum())
        
        # 计算TEI (Task Efficiency Index)
        tei = self.compute_TEI(success_rate, comm_density)
        
        stats = {
            'epoch': epoch,
            'total_comm_count': total_comm_count,
            'avg_comm_per_step': avg_comm_per_step,
            'comm_density': comm_density,
            'active_edges': active_edges,
            'success_rate': success_rate,
            'tei': tei
        }
        
        # 保存到历史记录
        for key, value in stats.items():
            self.history[key].append(value)
        
        return stats
    
    def compute_TEI(self, success_rate, comm_density):
        """
        计算TEI (Task Efficiency Index)
        
        TEI = Success_Rate / Communication_Density
        
        越高越好：表示用更少的通信达到更高的成功率
        
        Args:
            success_rate: 成功率 (0-1)
            comm_density: 通信密度 (0-1)
        
        Returns:
            tei: TEI值
        """
        if comm_density <= 0:
            return 0.0 if success_rate <= 0 else float('inf')
        return success_rate / comm_density
    
    def _get_zero_stats(self):
        """返回零统计（当没有通信历史时）"""
        return {
            'total_comm_count': 0,
            'avg_comm_per_step': 0.0,
            'comm_density': 0.0,
            'active_edges': 0,
            'success_rate': 0.0,
            'tei': 0.0
        }
    
    def print_stats(self, stats):
        """打印统计信息"""
        print(f"\n  {'='*76}")
        print(f"  通信统计 (Epoch {stats.get('epoch', '?')})")
        print(f"    - 总通信次数: {stats['total_comm_count']}")
        print(f"    - 平均每步通信: {stats['avg_comm_per_step']:.2f}")
        print(f"    - 通信密度: {stats['comm_density']:.4f}")
        print(f"    - 活跃边数: {stats['active_edges']}/{self.max_edges}")
        print(f"    - 成功率: {stats['success_rate']:.4f}")
        print(f"    - TEI指标: {stats['tei']:.4f}")
        print(f"  {'='*76}\n")
    
    def save_stats(self, filename='communication_stats.json'):
        """保存统计历史到文件"""
        if self.save_dir is None:
            return
        
        save_path = self.save_dir / filename
        with open(save_path, 'w') as f:
            json.dump(self.history, f, indent=2)
        
        print(f"通信统计已保存: {save_path}")
    
    def get_summary(self):
        """获取统计摘要"""
        if len(self.history['epoch']) == 0:
            return "没有统计数据"
        
        summary = {
            'total_epochs': len(self.history['epoch']),
            'avg_comm_density': np.mean(self.history['comm_density']),
            'avg_success_rate': np.mean(self.history['success_rate']),
            'avg_tei': np.mean([t for t in self.history['tei'] if t != float('inf')]),
            'final_tei': self.history['tei'][-1] if self.history['tei'] else 0,
        }
        return summary


class CommunicationLogger:
    """用于在训练过程中收集通信图"""
    
    def __init__(self):
        self.current_epoch_comm = []
    
    def log_communication(self, comm_graph):
        """
        记录一次通信图
        
        Args:
            comm_graph: [n, n] numpy数组，表示当前step的通信图
        """
        self.current_epoch_comm.append(comm_graph.copy())
    
    def get_and_reset(self):
        """获取当前epoch的所有通信记录，并重置"""
        comm_history = self.current_epoch_comm
        self.current_epoch_comm = []
        return comm_history
    
    def reset(self):
        """重置记录"""
        self.current_epoch_comm = []







