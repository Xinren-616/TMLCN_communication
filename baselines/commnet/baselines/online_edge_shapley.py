"""
Online Edge-Level Shapley Value Computation Module

这个模块在训练过程中在线计算每条通信边的Shapley值。
每隔指定的epoch数（如100），会额外运行几个episodes来评估每条边的贡献。

主要特点：
1. 在线计算：在训练过程中实时计算，不需要离线加载checkpoint
2. 高效采样：使用较少的episodes进行估计，减少计算开销
3. 边级别分析：为每条通信边(i→j)计算其对性能的边际贡献
4. 结果记录：保存每个epoch的Shapley值，用于后续分析和优化
"""

import os
import numpy as np
import torch
import json
from pathlib import Path
from collections import defaultdict
import time


def _to_device(tensor_or_tuple, device):
    """将 tensor 或 (tensor, tensor) tuple 移到指定 device。"""
    if isinstance(tensor_or_tuple, (list, tuple)):
        return type(tensor_or_tuple)(t.to(device) for t in tensor_or_tuple)
    return tensor_or_tuple.to(device)


class OnlineEdgeShapleyAnalyzer:
    """在线边级别Shapley值分析器"""
    
    def __init__(self, args, env, n_eval_episodes=10, topology_threshold=0.1):
        """
        初始化在线Shapley值分析器
        
        Args:
            args: 训练参数
            env: 环境实例
            n_eval_episodes: 每次评估使用的episode数量（默认10，可调整）
            topology_threshold: 拓扑阈值，只分析使用频率>此值的边
        """
        self.args = args
        self.env = env
        self.n_agents = args.nagents
        self.n_eval_episodes = n_eval_episodes
        self.topology_threshold = topology_threshold
        
        # 存储历史Shapley值
        self.shapley_history = {}  # {epoch: {(i,j): shapley_value}}
        self.topology_history = {}  # {epoch: topology_matrix}
        
        # 统计信息
        self.computation_times = []  # 记录每次计算的时间
        
        print(f"\n{'='*80}")
        print(f"在线边级别Shapley值分析器已初始化")
        print(f"  - 智能体数量: {self.n_agents}")
        print(f"  - 可能的通信边数: {self.n_agents * (self.n_agents - 1)}")
        print(f"  - 每次评估episodes: {self.n_eval_episodes}")
        print(f"  - 拓扑阈值: {self.topology_threshold}")
        print(f"{'='*80}\n")
    
    def extract_communication_topology(self, model, n_episodes=None):
        """
        提取当前模型的通信拓扑
        
        Args:
            model: 当前的策略网络
            n_episodes: 采样episode数量（如果为None，使用self.n_eval_episodes）
            
        Returns:
            topology_freq: [n, n] 矩阵，表示每条边的使用频率
        """
        if n_episodes is None:
            n_episodes = self.n_eval_episodes
        
        # 初始化拓扑统计
        topology_count = np.zeros((self.n_agents, self.n_agents))
        total_steps = 0
        
        model.eval()
        
        with torch.no_grad():
            for ep in range(n_episodes):
                # 重置环境
                state = self.env.reset(epoch=0)
                device = next(model.parameters()).device
                state = state.to(device)
                
                # 初始化隐藏状态（总是初始化，因为comm.py总是需要）
                prev_hid = model.init_hidden(batch_size=state.shape[0])
                # MAGIC模型返回(hidden, cell)元组，需要转换为double
                if self.args.model_type == 'magic':
                    prev_hid = (prev_hid[0].double().to(device), prev_hid[1].double().to(device))
                elif self.args.recurrent and hasattr(self.args, 'rnn_type') and self.args.rnn_type == 'LSTM':
                    prev_hid = (prev_hid[0].double().to(device), prev_hid[1].double().to(device))
                elif isinstance(prev_hid, tuple):
                    # 其他返回元组的模型
                    prev_hid = (prev_hid[0].double().to(device), prev_hid[1].double().to(device))
                else:
                    # Non-recurrent模式或返回单个tensor
                    if isinstance(prev_hid, torch.Tensor):
                        prev_hid = prev_hid.double().to(device)
                    else:
                        prev_hid = torch.zeros(1, self.args.nagents, self.args.hid_size).double().to(device)
                
                for t in range(self.args.max_steps):
                    info = {}
                    
                    # 前向传播（不同模型返回值数量不同）
                    x = [state, prev_hid]
                    result = model(x, info)
                    # 根据返回值数量灵活处理
                    # GAComm返回4个值：action, value, prev_hid, comm_density
                    # MAGIC/TarMAC/IC3Net返回3个值：action, value, prev_hid
                    if isinstance(result, (list, tuple)):
                        if len(result) == 4:
                            action_out, value, prev_hid, _ = result  # 忽略comm_density
                        elif len(result) == 3:
                            action_out, value, prev_hid = result
                        else:
                            raise ValueError(f"模型返回了意外数量的值: {len(result)}")
                    else:
                        raise ValueError(f"模型返回了意外的类型: {type(result)}")
                    
                    # 提取通信图（根据模型类型）
                    comm_graph = self._extract_comm_graph_from_model(model)
                    if comm_graph is not None:
                        topology_count += comm_graph
                    else:
                        # 如果无法提取，假设完全图
                        full_graph = np.ones((self.n_agents, self.n_agents))
                        np.fill_diagonal(full_graph, 0)
                        topology_count += full_graph
                    
                    total_steps += 1
                    
                    # 执行动作
                    from action_utils import select_action, translate_action
                    action = select_action(self.args, action_out)
                    action, actual = translate_action(self.args, self.env, action)
                    next_state, reward, done, _ = self.env.step(actual)
                    
                    state = next_state.to(device)
                    
                    if done or t == self.args.max_steps - 1:
                        break
                
                # **关键修复**: 显式清理隐藏状态，防止计算图累积
                del prev_hid
        
        # 归一化为频率
        topology_freq = topology_count / total_steps if total_steps > 0 else topology_count
        
        # **关键修复**: 恢复模型到训练模式
        # 强制设置模型及其所有子模块为训练模式
        model.train()
        for module in model.modules():
            module.train()
        
        return topology_freq
    
    def _extract_comm_graph_from_model(self, model):
        """
        从模型中提取通信图
        
        Args:
            model: 策略网络
            
        Returns:
            comm_graph: [n, n] numpy数组，或None
        """
        # 尝试从不同类型的模型中提取通信图
        
        # TarMAC: 软注意力
        if hasattr(model, 'comm_attention') and model.comm_attention is not None:
            attn = model.comm_attention
            if len(attn.shape) == 3:
                attn = attn[0]  # 取第一个batch
            return attn.detach().cpu().numpy()
        
        # IC3Net: hard attention
        if hasattr(model, 'comm_mask') and model.comm_mask is not None:
            if isinstance(model.comm_mask, torch.Tensor):
                comm_mask = model.comm_mask
                if len(comm_mask.shape) == 2:  # [batch, n]
                    # comm_mask[i] = 1 表示 agent i 发送消息
                    batch_comm_mask = comm_mask[0].detach().cpu().numpy()
                    comm_graph = np.zeros((self.n_agents, self.n_agents))
                    for i in range(self.n_agents):
                        if batch_comm_mask[i] > 0.5:
                            for j in range(self.n_agents):
                                if i != j:
                                    comm_graph[i, j] = 1
                    return comm_graph
        
        # GAComm: 图注意力
        if hasattr(model, 'graph_attn') and model.graph_attn is not None:
            attn = model.graph_attn
            if len(attn.shape) == 3:
                attn = attn[0]
            return attn.detach().cpu().numpy()

        # MAT: Transformer attention weights [batch, nhead, n, n]
        if hasattr(model, '_last_attn_weights') and model._last_attn_weights is not None:
            attn = model._last_attn_weights  # [batch, nhead, n, n]
            avg = attn.mean(dim=(0, 1)).cpu().numpy()  # [n, n]
            np.fill_diagonal(avg, 0)
            return avg

        # BVME: use comm_mask (full graph) as topology
        if hasattr(model, 'comm_mask') and isinstance(model.comm_mask, torch.Tensor):
            mask = model.comm_mask
            if mask.shape == (self.n_agents, self.n_agents):
                return mask.detach().cpu().numpy()

        return None
    
    def _evaluate_with_edge_mask(self, model, edge_mask, n_episodes=None):
        """
        在指定的边mask下评估模型性能
        
        对于 CommNet 等使用固定 comm_mask 属性的模型，直接临时替换 model.comm_mask；
        对于 MAGIC 等通过 info['comm_mask'] 接收动态 mask 的模型，通过 info dict 传入。
        
        Args:
            model: 策略网络
            edge_mask: [n, n] 边mask矩阵 (torch.Tensor, float32)，1表示保留该边，0表示移除
            n_episodes: 评估episode数量
            
        Returns:
            performance_dict: 包含成功率、平均奖励等指标的字典
        """
        if n_episodes is None:
            n_episodes = self.n_eval_episodes
        
        total_rewards = []
        total_steps = []
        successes = []
        
        # 如果模型有固定的 comm_mask 属性（CommNet 等），临时替换它
        device = next(model.parameters()).device
        original_comm_mask = None
        if hasattr(model, 'comm_mask') and isinstance(model.comm_mask, torch.Tensor):
            original_comm_mask = model.comm_mask.clone()
            model.comm_mask = edge_mask.to(device)
        
        model.eval()
        
        with torch.no_grad():
            for ep in range(n_episodes):
                # 重置环境
                state = self.env.reset(epoch=0)
                device = next(model.parameters()).device
                state = state.to(device)
                episode_reward = 0
                episode_steps = 0
                
                # 初始化隐藏状态（总是初始化）
                prev_hid = model.init_hidden(batch_size=state.shape[0])
                # MAGIC模型返回(hidden, cell)元组，需要转换为double
                if self.args.model_type == 'magic':
                    prev_hid = (prev_hid[0].double().to(device), prev_hid[1].double().to(device))
                elif self.args.recurrent and hasattr(self.args, 'rnn_type') and self.args.rnn_type == 'LSTM':
                    prev_hid = (prev_hid[0].double().to(device), prev_hid[1].double().to(device))
                elif isinstance(prev_hid, tuple):
                    # 其他返回元组的模型
                    prev_hid = (prev_hid[0].double().to(device), prev_hid[1].double().to(device))
                else:
                    # Non-recurrent模式或返回单个tensor
                    if isinstance(prev_hid, torch.Tensor):
                        prev_hid = prev_hid.double().to(device)
                    else:
                        prev_hid = torch.zeros(1, self.args.nagents, self.args.hid_size).double().to(device)
                
                for t in range(self.args.max_steps):
                    
                    # 准备info字典，包含边级别的通信mask
                    info = {
                        'comm_mask': edge_mask,  # [n, n] 边mask
                    }
                    
                    # 前向传播（不同模型返回值数量不同）
                    x = [state, prev_hid]
                    result = model(x, info)
                    # 根据返回值数量灵活处理
                    # GAComm返回4个值：action, value, prev_hid, comm_density
                    # MAGIC/TarMAC/IC3Net返回3个值：action, value, prev_hid
                    if isinstance(result, (list, tuple)):
                        if len(result) == 4:
                            action_out, value, prev_hid, _ = result  # 忽略comm_density
                        elif len(result) == 3:
                            action_out, value, prev_hid = result
                        else:
                            raise ValueError(f"模型返回了意外数量的值: {len(result)}")
                    else:
                        raise ValueError(f"模型返回了意外的类型: {type(result)}")
                    
                    # 执行动作
                    from action_utils import select_action, translate_action
                    action = select_action(self.args, action_out)
                    action, actual = translate_action(self.args, self.env, action)
                    
                    next_state, reward, done, info_step = self.env.step(actual)
                    
                    episode_reward += reward[:self.args.nfriendly].sum()
                    episode_steps += 1
                    
                    state = next_state.to(device)
                    
                    if done or t == self.args.max_steps - 1:
                        break
                
                total_rewards.append(episode_reward)
                total_steps.append(episode_steps)
                
                # 检查是否成功
                if hasattr(self.env, 'get_stat'):
                    stat = self.env.get_stat()
                    if 'success' in stat:
                        successes.append(stat['success'])
                    else:
                        successes.append(1.0 if episode_reward > 0 else 0.0)
                else:
                    successes.append(1.0 if episode_reward > 0 else 0.0)
                
                # **关键修复**: 显式清理隐藏状态，防止计算图累积
                # 这对于LSTM等RNN模型非常重要
                del prev_hid
        
        # 恢复原始 comm_mask（如果被临时替换过）
        if original_comm_mask is not None:
            model.comm_mask = original_comm_mask
        
        # **关键修复**: 恢复模型到训练模式
        # 强制设置模型及其所有子模块为训练模式
        model.train()
        for module in model.modules():
            module.train()
        
        return {
            'success_rate': np.mean(successes),
            'avg_reward': np.mean(total_rewards),
            'avg_steps': np.mean(total_steps),
            'std_reward': np.std(total_rewards),
            'std_success': np.std(successes)
        }
    
    def compute_edge_shapley_value(self, model, edge_i_j, base_topology):
        """
        计算特定边(i→j)的Shapley值（边际贡献）
        
        Args:
            model: 当前的策略网络
            edge_i_j: (i, j) 元组，表示要分析的边
            base_topology: 基础拓扑矩阵
            
        Returns:
            shapley_value: 该边的Shapley值（基于平均奖励的边际贡献）
        """
        i, j = edge_i_j
        
        # 1. 包含边i→j的性能
        edge_mask_with = torch.tensor(base_topology, dtype=torch.float32)
        edge_mask_with[i, j] = 1.0
        perf_with = self._evaluate_with_edge_mask(model, edge_mask_with)
        
        # 2. 移除边i→j的性能
        edge_mask_without = edge_mask_with.clone()
        edge_mask_without[i, j] = 0.0
        perf_without = self._evaluate_with_edge_mask(model, edge_mask_without)
        
        # 用平均奖励差值作为 Shapley 值：奖励是负数时仍能反映方向
        marginal_reward = perf_with['avg_reward'] - perf_without['avg_reward']
        
        return {
            'shapley_value': marginal_reward,
            'reward_with': perf_with['avg_reward'],
            'reward_without': perf_without['avg_reward'],
            'success_with': perf_with['success_rate'],
            'success_without': perf_without['success_rate'],
            'topology_freq': base_topology[i, j],
        }
    
    def compute_all_edges_shapley(self, model, epoch):
        """
        计算当前epoch下所有重要边的Shapley值
        
        Args:
            model: 当前的策略网络
            epoch: 当前epoch数
            
        Returns:
            edge_shapley_dict: {(i,j): shapley_info} 字典
            topology: 当前的通信拓扑矩阵
            computation_time: 计算耗时（秒）
        """
        start_time = time.time()
        
        print(f"\n{'='*80}")
        print(f"开始在线计算 Epoch {epoch} 的边级别Shapley值...")
        print(f"{'='*80}")
        
        # 1. 提取当前通信拓扑
        print("  [1/3] 提取通信拓扑...")
        topology = self.extract_communication_topology(model)
        
        print(f"    - 平均链路使用频率: {topology.mean():.4f}")
        print(f"    - 非零链路数量: {(topology > self.topology_threshold).sum()}")
        
        # 2. 找出所有重要的边
        important_edges = []
        for i in range(self.n_agents):
            for j in range(self.n_agents):
                if i != j and topology[i, j] > self.topology_threshold:
                    important_edges.append((i, j))
        
        print(f"  [2/3] 识别重要边: {len(important_edges)} 条（阈值={self.topology_threshold}）")
        
        # 3. 计算每条边的Shapley值
        print(f"  [3/3] 计算Shapley值...")
        edge_shapley_dict = {}
        
        for idx, edge in enumerate(important_edges):
            if (idx + 1) % 5 == 0 or idx == 0:
                print(f"    - 进度: {idx+1}/{len(important_edges)} 边 {edge[0]}→{edge[1]}")
            
            shapley_info = self.compute_edge_shapley_value(model, edge, topology)
            edge_shapley_dict[edge] = shapley_info
        
        computation_time = time.time() - start_time
        self.computation_times.append(computation_time)
        
        # 4. 保存到历史记录
        self.shapley_history[epoch] = edge_shapley_dict
        self.topology_history[epoch] = topology
        
        # 5. 打印统计信息
        shapley_values = [info['shapley_value'] for info in edge_shapley_dict.values()]
        reward_withs = [info.get('reward_with', 0.0) for info in edge_shapley_dict.values()]
        reward_withouts = [info.get('reward_without', 0.0) for info in edge_shapley_dict.values()]
        
        print(f"\n  {'='*76}")
        print(f"  Shapley值统计 (基于奖励差: reward_with - reward_without):")
        print(f"    - 平均值: {np.mean(shapley_values):.6f}")
        print(f"    - 标准差: {np.std(shapley_values):.6f}")
        if shapley_values:
            best_edge = max(edge_shapley_dict.items(), key=lambda x: x[1]['shapley_value'])
            worst_edge = min(edge_shapley_dict.items(), key=lambda x: x[1]['shapley_value'])
            print(f"    - 最大值: {np.max(shapley_values):.6f} (边 {best_edge[0]}, reward_with={best_edge[1].get('reward_with',0):.4f}, reward_without={best_edge[1].get('reward_without',0):.4f})")
            print(f"    - 最小值: {np.min(shapley_values):.6f} (边 {worst_edge[0]}, reward_with={worst_edge[1].get('reward_with',0):.4f}, reward_without={worst_edge[1].get('reward_without',0):.4f})")
        print(f"    - 正贡献边数: {sum(1 for v in shapley_values if v > 0)}/{len(shapley_values)}")
        print(f"    - 负贡献边数: {sum(1 for v in shapley_values if v < 0)}/{len(shapley_values)}")
        print(f"    - 参考: avg reward_with={np.mean(reward_withs):.4f}, avg reward_without={np.mean(reward_withouts):.4f}")
        print(f"  {'='*76}")
        print(f"  计算耗时: {computation_time:.2f}秒")
        print(f"  平均每条边: {computation_time/len(important_edges):.2f}秒")
        print(f"{'='*80}\n")
        
        # **关键修复**: 恢复模型到训练模式
        # Shapley值计算过程中模型被设置为eval模式，必须恢复到train模式
        # 强制设置模型及其所有子模块为训练模式
        model.train()
        for module in model.modules():
            module.train()
        
        # **关键修复**: 清理LSTM/RNN的计算图和缓存
        # 在Shapley计算过程中，RNN的隐藏状态可能累积了计算图
        # 需要显式清理以防止数值不稳定和内存泄漏
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # 强制进行垃圾回收
        import gc
        gc.collect()
        
        return edge_shapley_dict, topology, computation_time
    
    def get_shapley_matrix(self, epoch):
        """
        获取指定epoch的Shapley值矩阵
        
        Args:
            epoch: epoch编号
            
        Returns:
            shapley_matrix: [n, n] numpy数组，未计算的边为0
        """
        if epoch not in self.shapley_history:
            return None
        
        shapley_matrix = np.zeros((self.n_agents, self.n_agents))
        for (i, j), info in self.shapley_history[epoch].items():
            shapley_matrix[i, j] = info['shapley_value']
        
        return shapley_matrix
    
    def save_results(self, output_dir):
        """
        保存所有历史结果到文件
        
        Args:
            output_dir: 输出目录路径
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 1. 保存Shapley值历史
        shapley_json = {}
        for epoch, edge_dict in self.shapley_history.items():
            shapley_json[str(epoch)] = {}
            for (i, j), info in edge_dict.items():
                edge_str = f"{i}->{j}"
                shapley_json[str(epoch)][edge_str] = {
                    'shapley_value': float(info['shapley_value']),
                    'reward_with': float(info.get('reward_with', 0.0)),
                    'reward_without': float(info.get('reward_without', 0.0)),
                    'success_with': float(info.get('success_with', 0.0)),
                    'success_without': float(info.get('success_without', 0.0)),
                    'topology_freq': float(info['topology_freq']),
                }
        
        with open(output_dir / 'online_shapley_values.json', 'w') as f:
            json.dump(shapley_json, f, indent=2)
        
        # 2. 保存拓扑历史
        topology_json = {}
        for epoch, topology in self.topology_history.items():
            topology_json[str(epoch)] = topology.tolist()
        
        with open(output_dir / 'online_topologies.json', 'w') as f:
            json.dump(topology_json, f, indent=2)
        
        # 3. 保存统计信息
        stats = {
            'n_agents': self.n_agents,
            'n_eval_episodes': self.n_eval_episodes,
            'topology_threshold': self.topology_threshold,
            'epochs_analyzed': list(self.shapley_history.keys()),
            'computation_times': self.computation_times,
            'avg_computation_time': np.mean(self.computation_times) if self.computation_times else 0,
        }
        
        with open(output_dir / 'online_shapley_stats.json', 'w') as f:
            json.dump(stats, f, indent=2)
        
        print(f"\n在线Shapley值结果已保存到: {output_dir}")
        print(f"  - online_shapley_values.json: Shapley值历史")
        print(f"  - online_topologies.json: 拓扑历史")
        print(f"  - online_shapley_stats.json: 统计信息")
    
    def print_shapley_matrix(self, epoch):
        """
        打印指定epoch的Shapley值矩阵（用于调试）
        
        Args:
            epoch: epoch编号
        """
        if epoch not in self.shapley_history:
            print(f"Epoch {epoch} 没有Shapley值数据")
            return
        
        shapley_matrix = self.get_shapley_matrix(epoch)
        
        print(f"\nShapley值矩阵 (Epoch {epoch}):")
        print("  (i→j表示从agent i到agent j的通信边)")
        print("  " + "-" * (self.n_agents * 9 + 5))
        
        for i in range(self.n_agents):
            row_str = f"  {i} | "
            for j in range(self.n_agents):
                if i == j:
                    row_str += "    -   "
                else:
                    val = shapley_matrix[i, j]
                    if val == 0 and (i, j) not in self.shapley_history.get(epoch, {}):
                        # 该边未被计算（拓扑中不存在）
                        row_str += "   --   "
                    else:
                        row_str += f"{val:+.4f} "
            print(row_str)
        print("  " + "-" * (self.n_agents * 9 + 5))


def integrate_with_training(trainer, shapley_analyzer, epoch, shapley_interval=100):
    """
    将Shapley值计算集成到训练循环中
    
    这个函数应该在训练循环的每个epoch结束时调用。
    当epoch是shapley_interval的倍数时，会触发Shapley值计算。
    
    Args:
        trainer: Trainer对象或MultiProcessTrainer对象
        shapley_analyzer: OnlineEdgeShapleyAnalyzer对象
        epoch: 当前epoch数（从1开始）
        shapley_interval: 计算Shapley值的间隔（默认100）
        
    Returns:
        shapley_results: 如果计算了Shapley值，返回结果字典；否则返回None
    """
    # 检查是否需要计算Shapley值
    if epoch % shapley_interval == 0:
        print(f"\n{'#'*80}")
        print(f"# Epoch {epoch}: 触发在线Shapley值计算")
        print(f"{'#'*80}")
        
        # 获取当前模型（处理MultiProcessTrainer的情况）
        if hasattr(trainer, 'policy_net'):
            model = trainer.policy_net
        elif hasattr(trainer, 'trainer') and hasattr(trainer.trainer, 'policy_net'):
            model = trainer.trainer.policy_net
        else:
            raise AttributeError("无法从trainer中获取policy_net")
        
        # **关键修复**: 在计算Shapley值前，保存训练模式状态
        training_mode = model.training
        
        # 计算Shapley值
        edge_shapley_dict, topology, computation_time = \
            shapley_analyzer.compute_all_edges_shapley(model, epoch)
        
        # **关键修复**: 强制恢复训练模式，确保所有模块都被设置
        if training_mode:
            model.train()
            for module in model.modules():
                module.train()
        else:
            model.eval()
            for module in model.modules():
                module.eval()
        
        # 返回结果供后续使用
        return {
            'epoch': epoch,
            'edge_shapley': edge_shapley_dict,
            'topology': topology,
            'computation_time': computation_time,
        }
    
    return None

