"""
TEI Optimizer
基于任务效率指数(TEI)反馈的阈值优化器

TEI = Success_Rate / Communication_Density

目标：动态调整剪枝threshold以最大化TEI
"""

import numpy as np
from pathlib import Path
import json


class TEIOptimizer:
    """
    基于TEI反馈的threshold优化器
    
    核心思想：
    - 跟踪TEI的变化趋势
    - 当TEI提升时，继续当前策略
    - 当TEI下降时，调整threshold方向
    - 最终收敛到最优的剪枝强度
    """
    
    def __init__(self, initial_threshold=-0.3, 
                 min_threshold=-0.5, max_threshold=0.2,
                 learning_rate=0.05, patience=3):
        """
        Args:
            initial_threshold: 初始阈值
            min_threshold: 最小阈值（最激进剪枝）
            max_threshold: 最大阈值（最保守剪枝）
            learning_rate: 调整步长
            patience: 容忍TEI不提升的次数
        """
        self.current_threshold = initial_threshold
        self.min_threshold = min_threshold
        self.max_threshold = max_threshold
        self.learning_rate = learning_rate
        self.patience = patience
        
        # 历史记录
        self.tei_history = []
        self.threshold_history = []
        self.best_tei = -np.inf
        self.best_threshold = initial_threshold
        self.no_improve_count = 0
        
        # 梯度估计
        self.tei_gradient_window = 3  # 用于估计梯度的窗口
        
        # 优化状态
        self.optimization_phase = "warmup"  # warmup, exploration, exploitation
    
    def update(self, current_tei, current_success, current_comm_density, epoch):
        """
        更新threshold基于TEI反馈
        
        Args:
            current_tei: 当前TEI值
            current_success: 当前成功率
            current_comm_density: 当前通信密度
            epoch: 当前epoch
        
        Returns:
            new_threshold: 更新后的threshold
            adjustment_info: 调整信息字典
        """
        # 记录历史
        self.tei_history.append({
            'epoch': int(epoch),
            'tei': float(current_tei),
            'success': float(current_success),
            'comm_density': float(current_comm_density),
            'threshold': float(self.current_threshold)
        })
        self.threshold_history.append(self.current_threshold)
        
        # 检查是否是最佳
        if current_tei > self.best_tei:
            self.best_tei = current_tei
            self.best_threshold = self.current_threshold
            self.no_improve_count = 0
            action = "BEST_SO_FAR"
        else:
            self.no_improve_count += 1
            action = "NO_IMPROVE"
        
        # 决策：是否调整threshold
        if len(self.tei_history) < self.tei_gradient_window:
            # 数据不足，保持不变
            adjustment = 0
            reason = "Insufficient data for gradient estimation"
            gradient_estimate = 0.0
        
        else:
            # 估计TEI梯度
            recent_teis = [h['tei'] for h in self.tei_history[-self.tei_gradient_window:]]
            recent_thresholds = self.threshold_history[-self.tei_gradient_window:]
            
            # 简单线性回归估计梯度: dTEI/dThreshold
            tei_trend = recent_teis[-1] - recent_teis[0]
            threshold_change = recent_thresholds[-1] - recent_thresholds[0]
            
            if abs(threshold_change) < 1e-6:
                gradient_estimate = 0.0
            else:
                gradient_estimate = tei_trend / threshold_change
            
            # 基于梯度和当前状态决策
            if current_tei < self.best_tei * 0.95:
                # TEI显著下降（>5%），回退到最佳点
                adjustment = (self.best_threshold - self.current_threshold) * 0.5
                reason = "TEI_DROP_ROLLBACK"
                action = "ROLLBACK"
            
            elif self.no_improve_count >= self.patience:
                # 多次未改善，尝试探索
                if gradient_estimate > 0:
                    # 梯度为正：threshold增加导致TEI增加
                    # 继续增加threshold（减少剪枝）
                    adjustment = self.learning_rate
                    reason = "EXPLORE_INCREASE"
                elif gradient_estimate < 0:
                    # 梯度为负：threshold减少导致TEI增加
                    # 继续减少threshold（增加剪枝）
                    adjustment = -self.learning_rate
                    reason = "EXPLORE_DECREASE"
                else:
                    # 梯度为0，随机探索
                    adjustment = self.learning_rate * (np.random.rand() - 0.5) * 2
                    reason = "RANDOM_EXPLORE"
                action = "EXPLORE"
                
                # 探索时重置no_improve_count
                self.no_improve_count = 0
            
            elif gradient_estimate > 0.01:
                # 明显正梯度，继续当前方向（增加threshold）
                adjustment = self.learning_rate * 0.5
                reason = "FOLLOW_GRADIENT_UP"
                action = "FOLLOW"
            
            elif gradient_estimate < -0.01:
                # 明显负梯度，反向调整（减少threshold）
                adjustment = -self.learning_rate * 0.5
                reason = "FOLLOW_GRADIENT_DOWN"
                action = "FOLLOW"
            
            else:
                # 梯度接近0，保持不变（可能已收敛）
                adjustment = 0
                reason = "STABLE"
                action = "STAY"
        
        # 应用调整并限制在范围内
        new_threshold = np.clip(
            self.current_threshold + adjustment,
            self.min_threshold,
            self.max_threshold
        )
        
        self.current_threshold = new_threshold
        
        # 更新优化阶段
        self._update_phase()
        
        adjustment_info = {
            'action': action,
            'reason': reason,
            'adjustment': float(adjustment),
            'new_threshold': float(new_threshold),
            'best_tei': float(self.best_tei),
            'best_threshold': float(self.best_threshold),
            'gradient_estimate': float(gradient_estimate) if isinstance(gradient_estimate, (int, float, np.number)) else 0.0,
            'no_improve_count': self.no_improve_count,
            'optimization_phase': self.optimization_phase
        }
        
        return new_threshold, adjustment_info
    
    def _update_phase(self):
        """更新优化阶段"""
        if len(self.tei_history) < 5:
            self.optimization_phase = "warmup"
        elif self.no_improve_count >= self.patience:
            self.optimization_phase = "exploration"
        else:
            self.optimization_phase = "exploitation"
    
    def get_threshold(self):
        """获取当前threshold"""
        return self.current_threshold
    
    def get_best_threshold(self):
        """获取历史最佳threshold"""
        return self.best_threshold
    
    def get_tei_trend(self, window=5):
        """
        获取TEI的趋势
        
        Args:
            window: 计算趋势的窗口大小
        
        Returns:
            str: "increasing", "decreasing", "stable"
        """
        if len(self.tei_history) < window:
            return "insufficient_data"
        
        recent_teis = [h['tei'] for h in self.tei_history[-window:]]
        trend = recent_teis[-1] - recent_teis[0]
        
        if trend > 0.01:
            return "increasing"
        elif trend < -0.01:
            return "decreasing"
        else:
            return "stable"
    
    def get_optimization_report(self):
        """生成优化报告"""
        if not self.tei_history:
            return "No data yet."
        
        latest = self.tei_history[-1]
        trend = self.get_tei_trend()
        
        report = f"""
╔══════════════════════════════════════════════════════════════╗
║              TEI Optimization Report                         ║
╠══════════════════════════════════════════════════════════════╣
║ Current TEI: {latest['tei']:<48.4f} ║
║ Best TEI: {self.best_tei:<51.4f} ║
║ Current Threshold: {self.current_threshold:<42.3f} ║
║ Best Threshold: {self.best_threshold:<45.3f} ║
║ No-improve count: {self.no_improve_count:<43} ║
║ TEI Trend (last 5): {trend:<41} ║
║ Optimization Phase: {self.optimization_phase:<41} ║
╚══════════════════════════════════════════════════════════════╝
        """
        
        return report
    
    def save_history(self, save_path):
        """
        保存优化历史到文件
        
        Args:
            save_path: 保存路径（JSON格式）
        """
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        data = {
            'config': {
                'initial_threshold': self.threshold_history[0] if self.threshold_history else None,
                'min_threshold': self.min_threshold,
                'max_threshold': self.max_threshold,
                'learning_rate': self.learning_rate,
                'patience': self.patience
            },
            'best': {
                'tei': float(self.best_tei),
                'threshold': float(self.best_threshold)
            },
            'history': self.tei_history
        }
        
        with open(save_path, 'w') as f:
            json.dump(data, f, indent=2)
        
        print(f"✅ TEI optimization history saved to {save_path}")


if __name__ == "__main__":
    # 测试代码
    print("Testing TEIOptimizer...")
    
    optimizer = TEIOptimizer(initial_threshold=-0.3, learning_rate=0.05, patience=3)
    
    # 模拟优化过程
    for epoch in range(1, 21):
        # 模拟TEI（假设在threshold=-0.2时最优）
        optimal_threshold = -0.2
        current_threshold = optimizer.get_threshold()
        
        # TEI是一个二次函数，在optimal_threshold处最大
        tei = 1.0 - 0.5 * (current_threshold - optimal_threshold) ** 2 + 0.1 * np.random.rand()
        success = 0.9 + 0.05 * np.random.rand()
        comm_density = 0.7 + 0.1 * np.random.rand()
        
        new_threshold, info = optimizer.update(tei, success, comm_density, epoch)
        
        print(f"\nEpoch {epoch}:")
        print(f"  Threshold: {current_threshold:.3f} -> {new_threshold:.3f}")
        print(f"  TEI: {tei:.4f} (Best: {optimizer.best_tei:.4f})")
        print(f"  Action: {info['action']}, Reason: {info['reason']}")
        print(f"  Gradient: {info['gradient_estimate']:.4f}")
    
    print("\n" + optimizer.get_optimization_report())
    print("\n✅ Test completed!")

