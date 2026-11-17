"""
Epoch-level Causal Influence Calculator using Welford's online algorithm
用于计算基于 KL 散度的 Epoch 级别因果影响
"""
import torch
import numpy as np


class EpochCICalculator:
    """
    在线计算 Epoch 级别的因果影响（基于 KL 散度）
    使用 Welford 算法高效计算均值和方差
    """
    
    def __init__(self, n_agents, hid_size):
        """
        初始化计算器
        
        Args:
            n_agents: 智能体数量
            hid_size: 隐藏状态维度
        """
        self.n = n_agents
        self.hid_size = hid_size
        self.reset()
    
    def reset(self):
        """重置所有统计量"""
        # Welford's online algorithm for mean and variance
        # 对每个 (i, j) 对维护统计量
        self.count = torch.zeros(self.n, self.n)
        self.mean_with = torch.zeros(self.n, self.n, self.hid_size)
        self.mean_without = torch.zeros(self.n, self.n, self.hid_size)
        self.M2_with = torch.zeros(self.n, self.n, self.hid_size)  # For variance calculation
        self.M2_without = torch.zeros(self.n, self.n, self.hid_size)
    
    def update(self, h_with, h_without, i, j):
        """
        在线更新统计量（Welford's algorithm）
        
        Args:
            h_with: 有 agent i 消息时的隐藏状态 [hid_size]
            h_without: 无 agent i 消息时的隐藏状态 [hid_size]
            i: 发送者 agent 索引
            j: 接收者 agent 索引
        """
        self.count[i, j] += 1
        n = self.count[i, j]
        
        # 更新 with 的均值和方差
        delta_with = h_with - self.mean_with[i, j]
        self.mean_with[i, j] += delta_with / n
        delta2_with = h_with - self.mean_with[i, j]
        self.M2_with[i, j] += delta_with * delta2_with
        
        # 更新 without 的均值和方差
        delta_without = h_without - self.mean_without[i, j]
        self.mean_without[i, j] += delta_without / n
        delta2_without = h_without - self.mean_without[i, j]
        self.M2_without[i, j] += delta_without * delta2_without
    
    def compute_ci(self, epsilon=1e-8):
        """
        计算整个 Epoch 的 CI（基于 KL 散度）
        
        Args:
            epsilon: 数值稳定性常数
            
        Returns:
            CI_matrix: [n, n] 矩阵，CI_matrix[i, j] = agent i 对 agent j 的因果影响
            total_ci: 标量，所有 agent 对的平均 CI
            statistics: 字典，包含详细统计信息
        """
        CI_matrix = torch.zeros(self.n, self.n)
        valid_pairs = 0
        
        for i in range(self.n):
            for j in range(self.n):
                if i != j and self.count[i, j] > 1:
                    # 计算方差（无偏估计）
                    var_with = self.M2_with[i, j] / (self.count[i, j] - 1) + epsilon
                    var_without = self.M2_without[i, j] / (self.count[i, j] - 1) + epsilon
                    
                    # 计算对角高斯分布的 KL 散度
                    # D_KL(N(μ1, Σ1) || N(μ0, Σ0))
                    mean_diff = self.mean_with[i, j] - self.mean_without[i, j]
                    
                    kl_div = 0.5 * torch.sum(
                        var_with / var_without +
                        (mean_diff ** 2) / var_without -
                        1 +
                        torch.log(var_without / var_with)
                    )
                    
                    CI_matrix[i, j] = kl_div.item()
                    valid_pairs += 1
        
        # 计算总体 CI（所有有效 agent 对的平均）
        total_ci = CI_matrix.sum().item() / valid_pairs if valid_pairs > 0 else 0.0
        
        # 计算每个 agent 的总影响力
        per_agent_influence = CI_matrix.sum(dim=1).numpy()  # agent i 对所有其他 agents 的总影响
        
        # 统计信息
        statistics = {
            'total_ci': total_ci,
            'ci_matrix': CI_matrix.numpy().tolist(),
            'per_agent_influence': per_agent_influence.tolist(),
            'valid_pairs': valid_pairs,
            'total_samples': self.count.sum().item(),
            'mean_samples_per_pair': self.count.sum().item() / valid_pairs if valid_pairs > 0 else 0,
            # 添加方差信息
            'mean_var_with': (self.M2_with.sum() / (self.count.sum() - self.n * (self.n - 1))).item() if self.count.sum() > self.n * (self.n - 1) else 0,
            'mean_var_without': (self.M2_without.sum() / (self.count.sum() - self.n * (self.n - 1))).item() if self.count.sum() > self.n * (self.n - 1) else 0,
        }
        
        return CI_matrix, total_ci, statistics


class BatchCICollector:
    """
    在 batch 级别收集样本（用于计算 epoch CI 前的中间收集）
    """
    
    def __init__(self, n_agents, hid_size):
        self.n = n_agents
        self.hid_size = hid_size
        self.reset()
    
    def reset(self):
        """重置收集的样本"""
        # 存储所有样本
        self.samples_with = {(i, j): [] for i in range(self.n) for j in range(self.n) if i != j}
        self.samples_without = {(i, j): [] for i in range(self.n) for j in range(self.n) if i != j}
    
    def add_sample(self, h_with, h_without, i, j):
        """
        添加一个样本
        
        Args:
            h_with: 有消息时的隐藏状态 [hid_size]
            h_without: 无消息时的隐藏状态 [hid_size]
            i: 发送者索引
            j: 接收者索引
        """
        if i != j:
            self.samples_with[(i, j)].append(h_with.cpu().detach().numpy())
            self.samples_without[(i, j)].append(h_without.cpu().detach().numpy())
    
    def compute_batch_ci(self, epsilon=1e-8):
        """
        计算当前 batch 收集的样本的 CI
        
        Returns:
            CI_matrix: [n, n] 矩阵
            total_ci: 标量
        """
        CI_matrix = np.zeros((self.n, self.n))
        valid_pairs = 0
        
        for i in range(self.n):
            for j in range(self.n):
                if i != j and len(self.samples_with[(i, j)]) > 1:
                    # 转换为 numpy 数组
                    samples_with = np.array(self.samples_with[(i, j)])  # [num_samples, hid_size]
                    samples_without = np.array(self.samples_without[(i, j)])
                    
                    # 计算均值和方差
                    mu_with = samples_with.mean(axis=0)
                    mu_without = samples_without.mean(axis=0)
                    var_with = samples_with.var(axis=0, ddof=1) + epsilon
                    var_without = samples_without.var(axis=0, ddof=1) + epsilon
                    
                    # 计算 KL 散度
                    kl_div = 0.5 * np.sum(
                        var_with / var_without +
                        ((mu_with - mu_without) ** 2) / var_without -
                        1 +
                        np.log(var_without / var_with)
                    )
                    
                    CI_matrix[i, j] = kl_div
                    valid_pairs += 1
        
        total_ci = CI_matrix.sum() / valid_pairs if valid_pairs > 0 else 0.0
        
        return CI_matrix, total_ci



