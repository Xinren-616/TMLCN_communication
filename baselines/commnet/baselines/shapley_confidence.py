"""
Shapley Confidence Analyzer
时序Shapley值置信度分析器

用于分析边的Shapley值在多个epoch上的一致性，
识别高置信度的负贡献边。
"""

import numpy as np
from collections import defaultdict
from pathlib import Path
import json


class ShapleyConfidenceAnalyzer:
    """
    时序Shapley值置信度分析器
    
    核心思想：
    - 只剪枝那些在多个epoch上持续表现为负贡献的边
    - 避免因偶然的负值而误剪有价值的边
    """
    
    def __init__(self, n_agents, window_size=5, confidence_prob=0.8):
        """
        Args:
            n_agents: 智能体数量
            window_size: 观测窗口大小（w）- 保留最近w个epoch的Shapley值
            confidence_prob: 置信度阈值（p_0）- 需要多少比例的观测为负才认为是"高置信度负贡献"
        """
        self.n_agents = n_agents
        self.window_size = window_size
        self.confidence_prob = confidence_prob
        
        # 存储历史Shapley值: {(i,j): [{'value': phi, 'epoch': t}, ...]}
        self.shapley_history = defaultdict(list)
        
        # 存储置信度得分历史
        self.confidence_history = []
    
    def update(self, shapley_matrix, epoch):
        """
        更新Shapley值历史
        
        Args:
            shapley_matrix: [n, n] 当前epoch的Shapley矩阵
            epoch: 当前epoch
        """
        for i in range(self.n_agents):
            for j in range(self.n_agents):
                if i == j:
                    continue
                
                key = (i, j)
                self.shapley_history[key].append({
                    'value': float(shapley_matrix[i, j]),
                    'epoch': int(epoch)
                })
                
                # 只保留最近window_size个
                if len(self.shapley_history[key]) > self.window_size:
                    self.shapley_history[key].pop(0)
    
    def compute_confidence_scores(self, epoch):
        """
        计算每条边的负贡献置信度
        
        置信度得分定义：
        Conf_{ij} = |mean(phi_{ij})| / (1 + std(phi_{ij}))
        
        只有当边的Shapley值在>=confidence_prob比例的观测中为负时，
        才计算置信度得分，否则为0（不剪枝）。
        
        Returns:
            confidence_matrix: [n, n] 置信度矩阵（值越大越应该剪枝）
            stats: 统计信息字典
        """
        confidence_matrix = np.zeros((self.n_agents, self.n_agents))
        
        stats = {
            'high_confidence_negative': 0,  # 高置信度负边数量
            'uncertain_edges': 0,           # 不确定边数量
            'positive_edges': 0,            # 正贡献边数量
            'total_valid_edges': 0          # 总有效边数
        }
        
        for i in range(self.n_agents):
            for j in range(self.n_agents):
                if i == j:
                    continue
                
                stats['total_valid_edges'] += 1
                
                key = (i, j)
                history = self.shapley_history[key]
                
                if len(history) < 3:  # 至少需要3个观测
                    stats['uncertain_edges'] += 1
                    continue
                
                # 提取值
                values = [h['value'] for h in history]
                
                # 计算一致性概率 P_hat（负值的比例）
                negative_count = sum(1 for v in values if v < 0)
                p_hat = negative_count / len(values)
                
                if p_hat >= self.confidence_prob:
                    # 高置信度负贡献
                    mean_value = np.mean(values)
                    std_value = np.std(values)
                    
                    # 置信度得分 = |均值| / (1 + 标准差)
                    # 分子：负贡献的幅度（越大越应该剪）
                    # 分母：不确定性（越小越有信心）
                    confidence_score = abs(mean_value) / (1 + std_value)
                    
                    confidence_matrix[i, j] = confidence_score
                    stats['high_confidence_negative'] += 1
                
                elif p_hat <= (1 - self.confidence_prob):
                    # 高置信度正贡献（大部分观测为正）
                    stats['positive_edges'] += 1
                    confidence_matrix[i, j] = 0  # 不剪枝
                
                else:
                    # 不确定（正负混合）
                    stats['uncertain_edges'] += 1
                    confidence_matrix[i, j] = 0  # 保守策略：不剪枝
        
        # 记录历史统计
        self.confidence_history.append({
            'epoch': int(epoch),
            'stats': stats.copy(),
            'mean_confidence': float(confidence_matrix[confidence_matrix > 0].mean()) 
                              if (confidence_matrix > 0).any() else 0.0,
            'max_confidence': float(confidence_matrix.max()),
            'num_nonzero_confidence': int((confidence_matrix > 0).sum())
        })
        
        return confidence_matrix, stats
    
    def get_negative_probability(self, i, j):
        """
        获取边(i,j)为负贡献的概率估计
        
        Returns:
            float: 负贡献概率 [0, 1]
        """
        key = (i, j)
        history = self.shapley_history[key]
        
        if len(history) == 0:
            return 0.5  # 未知，返回中性概率
        
        values = [h['value'] for h in history]
        negative_count = sum(1 for v in values if v < 0)
        
        return negative_count / len(values)
    
    def get_edge_statistics(self, i, j):
        """
        获取特定边的详细统计信息
        
        Returns:
            dict: 包含mean, std, min, max, negative_prob等
        """
        key = (i, j)
        history = self.shapley_history[key]
        
        if len(history) == 0:
            return None
        
        values = [h['value'] for h in history]
        
        return {
            'mean': float(np.mean(values)),
            'std': float(np.std(values)),
            'min': float(np.min(values)),
            'max': float(np.max(values)),
            'negative_prob': self.get_negative_probability(i, j),
            'sample_size': len(values),
            'recent_values': values[-3:] if len(values) >= 3 else values
        }
    
    def get_statistics_report(self):
        """生成统计报告"""
        if not self.confidence_history:
            return "No data yet."
        
        latest = self.confidence_history[-1]
        
        report = f"""
╔══════════════════════════════════════════════════════════════╗
║          Shapley Confidence Analysis Report                  ║
╠══════════════════════════════════════════════════════════════╣
║ Epoch: {latest['epoch']:<52} ║
║ Total valid edges: {latest['stats']['total_valid_edges']:<42} ║
║ High-confidence negative edges: {latest['stats']['high_confidence_negative']:<31} ║
║ Uncertain edges: {latest['stats']['uncertain_edges']:<44} ║
║ Positive edges: {latest['stats']['positive_edges']:<45} ║
║ Mean confidence score: {latest['mean_confidence']:<40.4f} ║
║ Max confidence score: {latest['max_confidence']:<41.4f} ║
╚══════════════════════════════════════════════════════════════╝
        """
        
        return report
    
    def save_history(self, save_path):
        """
        保存历史数据到文件
        
        Args:
            save_path: 保存路径（JSON格式）
        """
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        data = {
            'config': {
                'n_agents': self.n_agents,
                'window_size': self.window_size,
                'confidence_prob': self.confidence_prob
            },
            'confidence_history': self.confidence_history,
            # Shapley history太大，只保存最近的统计信息
            'shapley_summary': {
                f"edge_{i}_{j}": self.get_edge_statistics(i, j)
                for i in range(self.n_agents)
                for j in range(self.n_agents)
                if i != j and self.get_edge_statistics(i, j) is not None
            }
        }
        
        with open(save_path, 'w') as f:
            json.dump(data, f, indent=2)
        
        print(f"✅ Confidence analysis saved to {save_path}")


if __name__ == "__main__":
    # 测试代码
    print("Testing ShapleyConfidenceAnalyzer...")
    
    n_agents = 5
    analyzer = ShapleyConfidenceAnalyzer(n_agents=n_agents, window_size=5, confidence_prob=0.8)
    
    # 模拟5个epoch的Shapley值
    for epoch in range(1, 6):
        # 生成模拟Shapley矩阵
        shapley_matrix = np.random.randn(n_agents, n_agents) * 0.1
        # 边(0,1)持续为负
        shapley_matrix[0, 1] = -0.3 - 0.05 * np.random.rand()
        # 边(1,2)持续为正
        shapley_matrix[1, 2] = 0.2 + 0.05 * np.random.rand()
        # 边(2,3)不确定
        shapley_matrix[2, 3] = 0.1 * np.random.randn()
        
        analyzer.update(shapley_matrix, epoch)
        
        if epoch >= 3:
            confidence_matrix, stats = analyzer.compute_confidence_scores(epoch)
            print(f"\nEpoch {epoch}:")
            print(analyzer.get_statistics_report())
            print(f"\nEdge (0,1) statistics: {analyzer.get_edge_statistics(0, 1)}")
    
    print("\n✅ Test completed!")


