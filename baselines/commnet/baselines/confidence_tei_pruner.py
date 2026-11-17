"""
CTAP: Confidence-guided TEI-maximizing Adaptive Pruning
置信度引导的TEI最大化自适应剪枝器

结合时序Shapley值置信度建模和TEI优化的动态拓扑剪枝策略
"""

import numpy as np
from pathlib import Path
import json

try:
    from .shapley_confidence import ShapleyConfidenceAnalyzer
    from .tei_optimizer import TEIOptimizer
except ImportError:
    # 如果相对导入失败，尝试绝对导入
    from shapley_confidence import ShapleyConfidenceAnalyzer
    from tei_optimizer import TEIOptimizer


class CTAPruner:
    """
    CTAP剪枝器：结合置信度建模和TEI优化
    
    核心特点：
    1. 置信度建模：只剪枝持续多个epoch表现为负贡献的边
    2. TEI优化：动态调整threshold以最大化任务效率
    3. 渐进式：从保守到激进，确保训练稳定性
    
    支持模式：
    - full: 完整CTAP（置信度 + TEI优化）
    - conf_only: 仅置信度建模（固定threshold）
    - tei_only: 仅TEI优化（不考虑置信度）
    """
    
    def __init__(self, args, n_agents):
        """
        Args:
            args: 命令行参数
            n_agents: 智能体数量
        """
        self.args = args
        self.n_agents = n_agents
        
        # CTAP模式
        self.mode = getattr(args, 'ctap_mode', 'full')  # full, conf_only, tei_only
        
        # 置信度分析器
        if self.mode in ['full', 'conf_only']:
            self.confidence_analyzer = ShapleyConfidenceAnalyzer(
                n_agents=n_agents,
                window_size=getattr(args, 'confidence_window', 5),
                confidence_prob=getattr(args, 'confidence_prob', 0.8)
            )
        else:
            self.confidence_analyzer = None
        
        # TEI优化器
        if self.mode in ['full', 'tei_only']:
            self.tei_optimizer = TEIOptimizer(
                initial_threshold=getattr(args, 'initial_threshold', -0.3),
                min_threshold=getattr(args, 'min_threshold', -0.5),
                max_threshold=getattr(args, 'max_threshold', 0.2),
                learning_rate=getattr(args, 'tei_learning_rate', 0.05),
                patience=getattr(args, 'tei_patience', 3)
            )
        else:
            self.tei_optimizer = None
            # conf_only模式使用固定threshold
            self.fixed_threshold = getattr(args, 'initial_threshold', -0.3)
        
        # 当前剪枝mask
        self.current_mask = np.ones((n_agents, n_agents))
        np.fill_diagonal(self.current_mask, 0)  # 自己不与自己通信
        
        # 统计信息
        self.update_history = []
        
        print(f"✅ CTAP Pruner initialized in '{self.mode}' mode")
    
    def should_update(self, epoch):
        """
        判断是否应该更新mask
        
        Args:
            epoch: 当前epoch
        
        Returns:
            bool: 是否应该更新
        """
        if epoch < self.args.pruning_start_epoch:
            return False
        
        if epoch == self.args.pruning_start_epoch:
            return True
        
        return (epoch - self.args.pruning_start_epoch) % self.args.pruning_interval == 0
    
    def update_mask(self, shapley_matrix, epoch, success_rate, comm_density):
        """
        更新剪枝mask
        
        Args:
            shapley_matrix: [n, n] Shapley值矩阵
            epoch: 当前epoch
            success_rate: 当前成功率
            comm_density: 当前通信密度
        
        Returns:
            update_info: 更新信息字典
        """
        # Step 1: 计算/更新置信度（如果使用置信度模式）
        confidence_matrix = None
        conf_stats = None
        
        if self.mode in ['full', 'conf_only']:
            self.confidence_analyzer.update(shapley_matrix, epoch)
            confidence_matrix, conf_stats = self.confidence_analyzer.compute_confidence_scores(epoch)
        
        # Step 2: 获取当前threshold
        if self.mode in ['full', 'tei_only']:
            # 计算TEI并优化threshold
            current_tei = success_rate / (comm_density + 1e-6) if comm_density > 0 else 0
            new_threshold, adjustment_info = self.tei_optimizer.update(
                current_tei, success_rate, comm_density, epoch
            )
        else:
            # conf_only模式使用固定threshold
            new_threshold = self.fixed_threshold
            adjustment_info = {
                'action': 'FIXED',
                'reason': 'conf_only mode',
                'adjustment': 0,
                'new_threshold': new_threshold,
                'best_tei': None,
                'best_threshold': None,
                'gradient_estimate': None
            }
            current_tei = success_rate / (comm_density + 1e-6) if comm_density > 0 else 0
        
        # Step 3: 生成新mask
        new_mask = np.ones((self.n_agents, self.n_agents))
        np.fill_diagonal(new_mask, 0)
        
        pruned_count = 0
        pruning_details = []
        
        for i in range(self.n_agents):
            for j in range(self.n_agents):
                if i == j:
                    continue
                
                should_prune = False
                prune_reason = ""
                
                if self.mode == 'full':
                    # 完整CTAP：高置信度负贡献，且均值为负（放宽：不强制低于动态threshold）
                    # 理由：当comm_density未收敛时，threshold可能停留在初始值附近，
                    # 过于严格的双重条件会导致永远不剪枝。改为：置信度高且均值<0即可剪枝。
                    if confidence_matrix[i, j] > 0:
                        # 有高置信度负贡献
                        key = (i, j)
                        history = self.confidence_analyzer.shapley_history[key]
                        if len(history) > 0:
                            recent_values = [h['value'] for h in history]
                            mean_shapley = np.mean(recent_values)
                            
                            # 主条件：均值低于threshold（优先）
                            # 备用条件：均值为负且置信度高（防止threshold过低时失效）
                            if mean_shapley < new_threshold or mean_shapley < 0:
                                should_prune = True
                                prune_reason = "high_confidence_below_threshold"
                
                elif self.mode == 'conf_only':
                    # 仅置信度：有高置信度且均值为负（放宽固定threshold限制）
                    if confidence_matrix[i, j] > 0:
                        key = (i, j)
                        history = self.confidence_analyzer.shapley_history[key]
                        if len(history) > 0:
                            recent_values = [h['value'] for h in history]
                            mean_shapley = np.mean(recent_values)
                            
                            if mean_shapley < new_threshold or mean_shapley < 0:
                                should_prune = True
                                prune_reason = "confidence_based"
                
                elif self.mode == 'tei_only':
                    # 仅TEI：直接基于threshold剪枝（不考虑置信度）
                    if shapley_matrix[i, j] < new_threshold:
                        should_prune = True
                        prune_reason = "tei_threshold_based"
                
                if should_prune:
                    new_mask[i, j] = 0
                    pruned_count += 1
                    
                    # 记录剪枝详情（最多记录前5个）
                    if len(pruning_details) < 5:
                        detail = {
                            'edge': (i, j),
                            'shapley': float(shapley_matrix[i, j]),
                            'threshold': float(new_threshold),
                            'reason': prune_reason
                        }
                        
                        if confidence_matrix is not None:
                            detail['confidence'] = float(confidence_matrix[i, j])
                        
                        pruning_details.append(detail)
        
        # 更新当前mask
        self.current_mask = new_mask
        
        # 记录更新历史
        update_info = {
            'epoch': int(epoch),
            'mode': self.mode,
            'pruned_count': int(pruned_count),
            'remaining_edges': int(new_mask.sum()),
            'total_possible_edges': self.n_agents * (self.n_agents - 1),
            'pruning_ratio': float(pruned_count) / (self.n_agents * (self.n_agents - 1)),
            'threshold': float(new_threshold),
            'tei': float(current_tei),
            'success_rate': float(success_rate),
            'comm_density': float(comm_density),
            'confidence_stats': conf_stats if conf_stats else {},
            'adjustment_info': adjustment_info,
            'pruning_details': pruning_details
        }
        
        self.update_history.append(update_info)
        
        return update_info
    
    def get_mask(self):
        """
        获取当前mask
        
        Returns:
            np.ndarray: [n, n] 剪枝mask（1=保留，0=剪枝）
        """
        return self.current_mask.copy()
    
    def get_detailed_report(self, epoch):
        """
        生成详细报告
        
        Args:
            epoch: 当前epoch
        
        Returns:
            str: 格式化的报告字符串
        """
        if not self.update_history:
            return "No updates yet."
        
        latest = self.update_history[-1]
        
        # 基础报告
        report = f"""
╔══════════════════════════════════════════════════════════════════╗
║           CTAP Update Report - Epoch {epoch:<27} ║
╠══════════════════════════════════════════════════════════════════╣
║ Mode: {latest['mode']:<60} ║
║                                                                  ║
║ 📊 Performance Metrics                                           ║
║    Success Rate: {latest['success_rate']:<45.4f} ║
║    Comm Density: {latest['comm_density']:<45.4f} ║
║    TEI: {latest['tei']:<58.4f} ║
║                                                                  ║
║ ✂️  Pruning Statistics                                           ║
║    Pruned Edges: {latest['pruned_count']:<47} ║
║    Remaining Edges: {latest['remaining_edges']:<44} ║
║    Total Possible: {latest['total_possible_edges']:<45} ║
║    Pruning Ratio: {latest['pruning_ratio']:<44.2%} ║
║    Current Threshold: {latest['threshold']:<42.3f} ║
"""
        
        # 添加置信度分析（如果有）
        if latest['confidence_stats']:
            report += f"""║                                                                  ║
║ 🔍 Confidence Analysis                                           ║
║    High-confidence negative: {latest['confidence_stats']['high_confidence_negative']:<34} ║
║    Uncertain edges: {latest['confidence_stats']['uncertain_edges']:<44} ║
║    Positive edges: {latest['confidence_stats']['positive_edges']:<45} ║
"""
        
        # 添加TEI优化信息（如果有）
        if latest['adjustment_info']['best_tei'] is not None:
            report += f"""║                                                                  ║
║ 🎯 TEI Optimization                                              ║
║    Action: {latest['adjustment_info']['action']:<53} ║
║    Reason: {latest['adjustment_info']['reason']:<53} ║
║    Best TEI so far: {latest['adjustment_info']['best_tei']:<42.4f} ║
║    Best Threshold: {latest['adjustment_info']['best_threshold']:<43.3f} ║
"""
            if latest['adjustment_info']['gradient_estimate'] is not None:
                report += f"""║    Gradient Estimate: {latest['adjustment_info']['gradient_estimate']:<40.4f} ║
"""
        
        report += "╚══════════════════════════════════════════════════════════════════╝\n"
        
        # 添加示例剪枝边
        if latest['pruning_details']:
            report += "\n📋 Sample Pruned Edges:\n"
            for detail in latest['pruning_details']:
                report += f"  Edge {detail['edge']}: Shapley={detail['shapley']:.4f}"
                if 'confidence' in detail:
                    report += f", Confidence={detail['confidence']:.4f}"
                report += f", Reason={detail['reason']}\n"
        
        return report
    
    def get_summary_statistics(self):
        """
        获取整体统计信息
        
        Returns:
            dict: 包含各种统计信息
        """
        if not self.update_history:
            return {}
        
        teis = [h['tei'] for h in self.update_history]
        success_rates = [h['success_rate'] for h in self.update_history]
        pruning_ratios = [h['pruning_ratio'] for h in self.update_history]
        
        return {
            'num_updates': len(self.update_history),
            'tei': {
                'mean': float(np.mean(teis)),
                'std': float(np.std(teis)),
                'max': float(np.max(teis)),
                'min': float(np.min(teis)),
                'current': float(teis[-1])
            },
            'success_rate': {
                'mean': float(np.mean(success_rates)),
                'std': float(np.std(success_rates)),
                'current': float(success_rates[-1])
            },
            'pruning_ratio': {
                'mean': float(np.mean(pruning_ratios)),
                'std': float(np.std(pruning_ratios)),
                'current': float(pruning_ratios[-1])
            }
        }
    
    def save_analysis(self, save_dir, epoch):
        """
        保存分析结果
        
        Args:
            save_dir: 保存目录
            epoch: 当前epoch
        """
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # 保存更新历史
        history_file = save_dir / f'ctap_history_epoch{epoch}.json'
        with open(history_file, 'w') as f:
            json.dump(self.update_history, f, indent=2)
        
        # 保存当前mask
        mask_file = save_dir / f'pruning_mask_epoch{epoch}.npy'
        np.save(mask_file, self.current_mask)
        
        # 保存置信度历史（如果有）
        if self.confidence_analyzer is not None:
            conf_file = save_dir / f'confidence_history_epoch{epoch}.json'
            self.confidence_analyzer.save_history(conf_file)
        
        # 保存TEI优化历史（如果有）
        if self.tei_optimizer is not None:
            tei_file = save_dir / f'tei_optimization_epoch{epoch}.json'
            self.tei_optimizer.save_history(tei_file)
        
        # 保存汇总统计
        summary_file = save_dir / f'ctap_summary_epoch{epoch}.json'
        with open(summary_file, 'w') as f:
            summary = {
                'mode': self.mode,
                'epoch': epoch,
                'statistics': self.get_summary_statistics(),
                'latest_update': self.update_history[-1] if self.update_history else {}
            }
            json.dump(summary, f, indent=2)
        
        print(f"✅ CTAP analysis saved to {save_dir}")


if __name__ == "__main__":
    # 测试代码
    print("Testing CTAPruner...")
    
    # 模拟args
    class Args:
        pruning_start_epoch = 500
        pruning_interval = 100
        ctap_mode = 'full'
        confidence_window = 5
        confidence_prob = 0.8
        initial_threshold = -0.3
        min_threshold = -0.5
        max_threshold = 0.2
        tei_learning_rate = 0.05
        tei_patience = 3
    
    args = Args()
    n_agents = 5
    
    # 测试三种模式
    for mode in ['full', 'conf_only', 'tei_only']:
        print(f"\n{'='*60}")
        print(f"Testing mode: {mode}")
        print('='*60)
        
        args.ctap_mode = mode
        pruner = CTAPruner(args, n_agents)
        
        # 模拟几个epoch
        for epoch in [500, 600, 700]:
            if pruner.should_update(epoch):
                # 生成模拟Shapley矩阵
                shapley_matrix = np.random.randn(n_agents, n_agents) * 0.1
                # 边(0,1)持续为负
                shapley_matrix[0, 1] = -0.3
                
                # 模拟性能指标
                success_rate = 0.9 + 0.05 * np.random.rand()
                comm_density = 0.7 - 0.05 * (epoch - 500) / 200  # 逐渐降低
                
                update_info = pruner.update_mask(
                    shapley_matrix, epoch, success_rate, comm_density
                )
                
                print(f"\n{pruner.get_detailed_report(epoch)}")
        
        print(f"\nSummary Statistics:")
        print(json.dumps(pruner.get_summary_statistics(), indent=2))
    
    print("\n✅ Test completed!")

