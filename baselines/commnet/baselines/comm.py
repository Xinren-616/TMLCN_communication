import torch
import torch.nn.functional as F
from torch import nn
import numpy as np

from models import MLP
import sys 
sys.path.append("..") 
from action_utils import select_action, translate_action

class CommNetMLP(nn.Module):
    """
    MLP based CommNet. Uses communication vector to communicate info
    between agents
    """
    def __init__(self, args, num_inputs):
        """Initialization method for this class, setup various internal networks
        and weights

        Arguments:
            MLP {object} -- Self
            args {Namespace} -- Parse args namespace
            num_inputs {number} -- Environment observation dimension for agents
        """

        super(CommNetMLP, self).__init__()
        self.args = args
        self.nagents = args.nagents
        self.hid_size = args.hid_size
        self.comm_passes = 1  # 修改为只通信一轮
        self.recurrent = args.recurrent

        # 添加通信信息量统计变量 - 修改为与MAGIC相似的格式，只保留一轮通信
        self.comm_bits_stats = {
            'round1': {
                'total_elements': 0,         # 传输的消息数(元素数)
                'active_channels': 0,        # 活跃通信通道数（边数）
                'entropy': 0.0,              # 通信信息熵
                'max_entropy': 0.0,          # 最大熵
                'effective_bandwidth': 0.0,  # 有效带宽利用率
                'message_similarity': 0.0,   # 消息相似度
                'total_comm_num': 0,         # 总通信次数
            }
        }

        self.continuous = args.continuous
        if self.continuous:
            self.action_mean = nn.Linear(args.hid_size, args.dim_actions)
            self.action_log_std = nn.Parameter(torch.zeros(1, args.dim_actions))
        else:
            self.heads = nn.ModuleList([nn.Linear(args.hid_size, o)
                                        for o in args.naction_heads])
        self.init_std = args.init_std if hasattr(args, 'comm_init_std') else 0.2

        # Mask for communication
        if self.args.comm_mask_zero:
            self.comm_mask = torch.zeros(self.nagents, self.nagents)
        else:
            self.comm_mask = torch.ones(self.nagents, self.nagents) \
                            - torch.eye(self.nagents, self.nagents)


        # Since linear layers in PyTorch now accept * as any number of dimensions
        # between last and first dim, num_agents dimension will be covered.
        # The network below is function r in the paper for encoding
        # initial environment stage
        self.encoder = nn.Linear(num_inputs, args.hid_size)

        # if self.args.env_name == 'starcraft':
        #     self.state_encoder = nn.Linear(num_inputs, num_inputs)
        #     self.encoder = nn.Linear(num_inputs * 2, args.hid_size)
        if args.recurrent:
            self.hidd_encoder = nn.Linear(args.hid_size, args.hid_size)

        if args.recurrent:
            self.init_hidden(args.batch_size)
            self.f_module = nn.LSTMCell(args.hid_size, args.hid_size)

        else:
            if args.share_weights:
                self.f_module = nn.Linear(args.hid_size, args.hid_size)
                self.f_modules = nn.ModuleList([self.f_module
                                                for _ in range(self.comm_passes)])
            else:
                self.f_modules = nn.ModuleList([nn.Linear(args.hid_size, args.hid_size)
                                                for _ in range(self.comm_passes)])
        # else:
            # raise RuntimeError("Unsupported RNN type.")

        # Our main function for converting current hidden state to next state
        # self.f = nn.Linear(args.hid_size, args.hid_size)
        if args.share_weights:
            self.C_module = nn.Linear(args.hid_size, args.hid_size)
            self.C_modules = nn.ModuleList([self.C_module
                                            for _ in range(self.comm_passes)])
        else:
            self.C_modules = nn.ModuleList([nn.Linear(args.hid_size, args.hid_size)
                                            for _ in range(self.comm_passes)])
        # self.C = nn.Linear(args.hid_size, args.hid_size)

        # initialise weights as 0
        if args.comm_init == 'zeros':
            for i in range(self.comm_passes):
                self.C_modules[i].weight.data.zero_()
        self.tanh = nn.Tanh()

        # print(self.C)
        # self.C.weight.data.zero_()
        # Init weights for linear layers
        # self.apply(self.init_weights)

        self.value_head = nn.Linear(self.hid_size, 1)


    def get_agent_mask(self, batch_size, info):
        n = self.nagents

        if 'alive_mask' in info:
            agent_mask = torch.from_numpy(info['alive_mask'])
            num_agents_alive = agent_mask.sum()
        else:
            agent_mask = torch.ones(n)
            num_agents_alive = n

        agent_mask = agent_mask.view(1, 1, n)
        agent_mask = agent_mask.expand(batch_size, n, n).unsqueeze(-1).clone() # clone gives the full tensor and avoid the error

        return num_agents_alive, agent_mask

    def forward_state_encoder(self, x):
        hidden_state, cell_state = None, None

        if self.args.recurrent:
            x, extras = x
            x = self.encoder(x)

            if self.args.rnn_type == 'LSTM':
                hidden_state, cell_state = extras
            else:
                hidden_state = extras
            # hidden_state = self.tanh( self.hidd_encoder(prev_hidden_state) + x)
        else:
            # 兼容trainer.py总是传递[state, prev_hid]的情况
            if isinstance(x, (list, tuple)):
                x = x[0]  # 取第一个元素（state）
            x = self.encoder(x)
            x = self.tanh(x)
            hidden_state = x

        return x, hidden_state, cell_state

    def _compute_gaussian_kl_divergence(self, mu1, var1, mu0, var0, epsilon=1e-6):
        """
        计算两个对角高斯分布之间的KL散度：D_KL(N(mu1, var1) || N(mu0, var0))
        
        对于对角协方差矩阵，KL散度可以按维度分解：
        D_KL = 0.5 * sum_k [(mu1_k - mu0_k)^2 / var0_k + var1_k / var0_k - 1 - log(var1_k / var0_k)]
        
        Arguments:
            mu1: 分布P的均值 [..., dim]
            var1: 分布P的方差 [..., dim]
            mu0: 分布Q的均值 [..., dim]
            var0: 分布Q的方差 [..., dim]
            epsilon: 数值稳定性常数
            
        Returns:
            kl_div: KL散度值 [...]
        """
        # 确保方差为正
        var1 = var1 + epsilon
        var0 = var0 + epsilon
        
        # 计算KL散度的各项
        mean_diff_term = ((mu1 - mu0) ** 2) / var0  # 均值差异项
        var_ratio_term = var1 / var0  # 方差比
        log_var_term = torch.log(var1 / var0)  # 对数方差比
        
        # KL散度 = 0.5 * sum[(mean_diff + var_ratio - 1 - log_var)]
        kl_div = 0.5 * (mean_diff_term + var_ratio_term - 1 - log_var_term).sum(dim=-1)
        
        return kl_div

    def _compute_counterfactual_influence(self, comm, x, hidden_state_before, cell_state, active_mask, n, batch_size):
        """
        计算反事实因果影响：C_i→j = D_KL(p(h_j | m_i) || p(h_j | no m_i))
        
        使用KL散度衡量消息i对接收者j的隐藏状态分布的影响。
        假设隐藏状态服从对角高斯分布，通过batch中的样本估计均值和方差。
        
        对每个智能体i，计算移除其消息后，接收者j的隐藏状态分布变化
        
        Arguments:
            comm: 通信矩阵 [batch_size, n, n, hid_size]
            x: 编码后的输入 [batch_size, n, hid_size] 或 [batch_size*n, hid_size]
            hidden_state_before: 通信前隐藏状态 [batch_size, n, hid_size]
            cell_state: LSTM cell state (if recurrent)
            active_mask: 通信掩码 [batch_size, n, n, 1]
            n: 智能体数量
            batch_size: 批次大小
            
        Returns:
            influences: [batch_size, n, n] - influences[b, i, j]表示agent i对agent j的因果影响（瞬时CI）
            counterfactual_samples: 字典 - 包含用于 epoch 级别 CI 计算的样本
                {'with': [batch, n, n, hid], 'without': [batch, n, n, hid]}
        """
        with torch.no_grad():
            influences = torch.zeros(batch_size, n, n)
            
            # 存储所有agent的"有消息"和"无消息"的隐藏状态
            # 用于后续计算KL散度
            hidden_states_with = torch.zeros(batch_size, n, n, self.hid_size)  # [batch, sender, receiver, hid]
            hidden_states_without = torch.zeros(batch_size, n, n, self.hid_size)
            
            # 对每个发送者i
            for i in range(n):
                # 创建移除agent i消息的反事实通信矩阵
                comm_without_i = comm.clone()
                comm_without_i[:, i, :, :] = 0  # 移除agent i发送的所有消息
                
                # 计算反事实的comm_sum
                comm_sum_without_i = comm_without_i.sum(dim=1)  # [batch_size, n, hid_size]
                c_without_i = self.C_modules[0](comm_sum_without_i)
                
                # 计算反事实的隐藏状态
                if self.args.recurrent:
                    inp_without_i = x + c_without_i
                    inp_without_i = inp_without_i.view(batch_size * n, self.hid_size)
                    output_without_i = self.f_module(inp_without_i, (hidden_state_before.view(batch_size * n, self.hid_size), 
                                                                      cell_state))
                    hidden_state_without_i = output_without_i[0].view(batch_size, n, self.hid_size)
                else:
                    hidden_state_without_i = sum([x, self.f_modules[0](hidden_state_before), c_without_i])
                    hidden_state_without_i = self.tanh(hidden_state_without_i)
                
                # 原始的隐藏状态（有agent i的消息）
                comm_sum = comm.sum(dim=1)
                c = self.C_modules[0](comm_sum)
                
                if self.args.recurrent:
                    inp = x + c
                    inp = inp.view(batch_size * n, self.hid_size)
                    output = self.f_module(inp, (hidden_state_before.view(batch_size * n, self.hid_size), 
                                                 cell_state))
                    hidden_state_with_i = output[0].view(batch_size, n, self.hid_size)
                else:
                    hidden_state_with_i = sum([x, self.f_modules[0](hidden_state_before), c])
                    hidden_state_with_i = self.tanh(hidden_state_with_i)
                
                # 存储隐藏状态供后续计算
                hidden_states_with[:, i, :, :] = hidden_state_with_i
                hidden_states_without[:, i, :, :] = hidden_state_without_i
            
            # 计算KL散度形式的因果影响
            # 对每个发送者i和接收者j
            for i in range(n):
                for j in range(n):
                    if i != j:  # agent i不能影响自己
                        # 检查是否有有效通信
                        has_communication = active_mask[:, i, j, 0] > 0  # [batch_size]
                        
                        # 获取有/无消息i时，接收者j的隐藏状态
                        h_with = hidden_states_with[:, i, j, :]  # [batch_size, hid_size]
                        h_without = hidden_states_without[:, i, j, :]  # [batch_size, hid_size]
                        
                        # 方法1：使用简化KL散度（假设固定方差σ²=1）
                        # D_KL ≈ 0.5 * ||mu_with - mu_without||²
                        # 归一化：除以 hid_size 以获得每维度的平均影响
                        influence_l2 = 0.5 * torch.sum((h_with - h_without) ** 2, dim=-1) / self.hid_size  # [batch_size]
                        
                        # 方法2：使用完整的高斯KL散度（从batch中估计方差）
                        # 注意：这需要batch_size > 1才有意义
                        if batch_size > 1 and self.args.use_full_kl_divergence:
                            # 计算均值和方差（沿batch维度）
                            mu_with = h_with.mean(dim=0, keepdim=True)  # [1, hid_size]
                            var_with = h_with.var(dim=0, keepdim=True, unbiased=False)  # [1, hid_size]
                            
                            mu_without = h_without.mean(dim=0, keepdim=True)  # [1, hid_size]
                            var_without = h_without.var(dim=0, keepdim=True, unbiased=False)  # [1, hid_size]
                            
                            # 计算KL散度（广播到所有batch）
                            kl_div = self._compute_gaussian_kl_divergence(
                                mu_with.expand(batch_size, -1),
                                var_with.expand(batch_size, -1),
                                mu_without.expand(batch_size, -1),
                                var_without.expand(batch_size, -1)
                            )  # [batch_size]
                            
                            influence = kl_div
                        else:
                            # 默认使用简化版本（固定方差）
                            influence = influence_l2
                        
                        # 只在有通信时记录影响
                        influence = influence * has_communication.float()
                        influences[:, i, j] = influence
            
            # 计算通信掩码（哪些链路有通信）
            # active_mask: [batch_size, n, n, 1]
            comm_mask = active_mask.squeeze(-1)  # [batch_size, n, n]
            
            # 返回瞬时 CI 和反事实样本（用于 epoch 级别计算）
            counterfactual_samples = {
                'with': hidden_states_with,      # [batch, n, n, hid_size]
                'without': hidden_states_without, # [batch, n, n, hid_size]
                'comm_mask': comm_mask           # [batch, n, n] - 1表示有通信，0表示无通信
            }
            
            return influences, counterfactual_samples

    def _calculate_comm_stats(self, comm, n, batch_size, round_name, hidden_state_before=None, hidden_state_after=None, counterfactual_influences=None, active_mask_param=None):
        """
        计算通信信息量统计指标，并添加因果影响度量
        
        Arguments:
            comm (tensor): 已应用掩码的通信消息 [batch_size * n * n * hid_size]
            n (int): 智能体数量
            batch_size (int): 批次大小
            round_name (str): 通信轮次名称 ('round1')
            hidden_state_before (tensor): 通信前的隐藏状态 [batch_size, n, hid_size] (可选)
            hidden_state_after (tensor): 通信后的隐藏状态 [batch_size, n, hid_size] (可选)
            counterfactual_influences (tensor): 反事实因果影响 [batch_size, n, n] (可选)
            active_mask_param (tensor): 通信掩码 [batch_size, n, n, 1] (可选)
        """
        #print('comm:', comm)
        with torch.no_grad():
            # 获取通信维度
            hid_size = self.hid_size
            
            # 检测每一次通信是否有效（向量非全0）
            # 对于每个智能体对(i,j)，判断i发送给j的消息是否有效
            active_mask = (comm.abs().sum(dim=-1) > 1e-8).float()  # [batch_size, n, n]
            
            # 计算有效通信次数
            active_channels = active_mask.sum().item()
            
            # 计算通信元素总数 = 有效通信次数 × 每次通信的元素数(hid_size)
            total_elements = active_channels * hid_size
            
            # 计算信息熵和消息相似度
            eps = 1e-10
            entropy = 0.0
            max_entropy = 0.0
            total_valid_messages = 0
            
            # 收集所有有效消息，用于后续计算相似度
            valid_messages = []
            
            # 遍历每个批次和每对智能体之间的通信
            for b in range(batch_size):
                for i in range(n):
                    for j in range(n):
                        # 检查i到j的通信是否有效
                        if active_mask[b, i, j] > 0:
                            # 获取i发送给j的消息
                            message = comm[b, i, j]  # [hid_size]
                            
                            # 保存有效消息用于计算相似度
                            valid_messages.append(message.detach().clone())
                            
                            # 计算消息的概率分布（绝对值并归一化）
                            message_sum = message.abs().sum() + eps
                            message_prob = message.abs() / message_sum
                            
                            # 过滤掉太小的概率值以提高数值稳定性
                            threshold = 1e-6
                            valid_probs = message_prob[message_prob > threshold]
                            
                            if len(valid_probs) > 0:
                                # 重新归一化
                                valid_probs = valid_probs / torch.sum(valid_probs)
                                
                                # 计算该消息的熵
                                message_entropy = -torch.sum(valid_probs * torch.log2(valid_probs + eps)).item()
                                
                                # 最大熵：均匀分布时的熵，为log2(hid_size)
                                message_max_entropy = np.log2(hid_size)
                                
                                # 累加总熵和总最大熵
                                entropy += message_entropy
                                max_entropy += message_max_entropy
                                total_valid_messages += 1
            #print('valid_messages:', valid_messages)
            # 计算消息相似度（基于余弦相似度）
            message_similarity = 0.0
            if len(valid_messages) >= 2:
                # 计算所有消息对之间的余弦相似度
                similarity_sum = 0.0
                pair_count = 0
                
                for i in range(len(valid_messages)):
                    for j in range(i+1, len(valid_messages)):
                        vec1 = valid_messages[i]
                        vec2 = valid_messages[j]
                        # 计算余弦相似度
                        cos_sim = F.cosine_similarity(vec1.unsqueeze(0), vec2.unsqueeze(0)).item()
                        similarity_sum += abs(cos_sim)  # 取绝对值，关注相似度大小而不是方向
                        pair_count += 1
                
                if pair_count > 0:
                    message_similarity = similarity_sum / pair_count
            
            # 计算平均熵和平均最大熵
            if total_valid_messages > 0:
                entropy = entropy / total_valid_messages
                max_entropy = max_entropy / total_valid_messages
            else:
                entropy = 0.0
                max_entropy = np.log2(hid_size)
            
            # 计算有效带宽利用率
            effective_bandwidth = entropy / max_entropy if max_entropy > 0 else 0.0
            
            # ========== 新增：计算因果影响指标 (Causal Influence Metrics) ==========
            causal_influence = 0.0
            hidden_state_change_norm = 0.0
            message_impact_ratio = 0.0
            per_agent_influence = np.zeros(n)  # 每个智能体的因果影响
            pairwise_influence_matrix = np.zeros((n, n))  # 智能体间的成对影响矩阵
            
            if hidden_state_before is not None and hidden_state_after is not None:
                # 计算隐藏状态变化
                state_change = hidden_state_after - hidden_state_before  # [batch_size, n, hid_size]
                
                # 1. 总体隐藏状态变化范数（衡量通信导致的状态更新程度）
                hidden_state_change_norm = torch.norm(state_change, p=2, dim=-1).mean().item()
                
                # 2. 使用反事实影响计算真实的因果影响
                if counterfactual_influences is not None:
                    # counterfactual_influences: [batch_size, n, n]
                    # influences[b, i, j] = agent i对agent j的因果影响
                    
                    # 对每个智能体i，计算其总因果影响 = Σ_j C_i→j
                    for i in range(n):
                        # agent i对所有其他agent的影响之和
                        agent_i_total_influence = 0.0
                        for j in range(n):
                            if i != j:
                                # 平均所有batch
                                c_ij = counterfactual_influences[:, i, j].mean().item()
                                agent_i_total_influence += c_ij
                                pairwise_influence_matrix[i, j] = c_ij
                        
                        per_agent_influence[i] = agent_i_total_influence
                    
                    # 总体因果影响 = 所有智能体的平均影响
                    causal_influence = per_agent_influence.mean()
                
                else:
                    # 如果没有反事实影响，使用简化估计（向后兼容）
                    # 需要使用active_mask_param或从comm计算
                    if active_mask_param is not None:
                        influence_active_mask = active_mask_param
                    else:
                        influence_active_mask = (comm.abs().sum(dim=-1) > 1e-8).unsqueeze(-1).float()
                    
                    for b in range(batch_size):
                        for i in range(n):
                            agent_i_messages = comm[b, i, :, :]
                            influence_mask = influence_active_mask[b, i, :, 0] > 0
                            
                            if influence_mask.sum() > 0:
                                for j in range(n):
                                    if influence_mask[j] and i != j:
                                        m_ij = agent_i_messages[j]
                                        delta_h_j = state_change[b, j]
                                        message_norm = torch.norm(m_ij, p=2).item() + eps
                                        state_change_norm = torch.norm(delta_h_j, p=2).item()
                                        per_agent_influence[i] += state_change_norm * (message_norm / (message_norm + 1.0))
                    
                    per_agent_influence = per_agent_influence / batch_size
                    causal_influence = per_agent_influence.mean()
                
                # 3. 消息影响比率：状态变化 vs 消息强度
                total_message_strength = torch.norm(comm, p=2, dim=-1).sum().item() + eps
                total_state_change = torch.norm(state_change, p=2, dim=-1).sum().item()
                message_impact_ratio = total_state_change / total_message_strength
            
            # 更新通信统计信息
            self.comm_bits_stats[round_name] = {
                'total_elements': total_elements,        # 通信元素总数
                'active_channels': active_channels,      # 有效通信次数
                'total_comm_num': active_channels,       # 与active_channels相同
                'entropy': entropy,                      # 平均消息熵
                'max_entropy': max_entropy,              # 平均最大熵
                'effective_bandwidth': effective_bandwidth, # 有效带宽利用率
                'message_similarity': message_similarity,   # 平均消息相似度
                # 新增因果影响指标（基于反事实推理）
                'causal_influence': causal_influence,    # 平均因果影响: C_i = Σ_j C_i→j
                'hidden_state_change': hidden_state_change_norm,  # 隐藏状态变化范数
                'message_impact_ratio': message_impact_ratio,     # 消息影响效率比率
                'per_agent_influence': per_agent_influence.tolist() if isinstance(per_agent_influence, np.ndarray) else per_agent_influence,  # 每个智能体的总因果影响
                'pairwise_influence_matrix': pairwise_influence_matrix.tolist() if isinstance(pairwise_influence_matrix, np.ndarray) else pairwise_influence_matrix,  # 成对影响矩阵 C_i→j
            }

    def forward(self, x, info={}):
        """Forward function for CommNet class, expects state, previous hidden
        and communication tensor.
        B: Batch Size: Normally 1 in case of episode
        N: number of agents

        Arguments:
            x {tensor} -- State of the agents (N x num_inputs)
            prev_hidden_state {tensor} -- Previous hidden state for the networks in
            case of multiple passes (1 x N x hid_size)
            comm_in {tensor} -- Communication tensor for the network. (1 x N x N x hid_size)

        Returns:
            tuple -- Contains
                next_hidden {tensor}: Next hidden state for network
                comm_out {tensor}: Next communication tensor
                action_data: Data needed for taking next action (Discrete values in
                case of discrete, mean and std in case of continuous)
                v: value head
        """

        x, hidden_state, cell_state = self.forward_state_encoder(x)

        batch_size = x.size()[0]
        n = self.nagents

        num_agents_alive, agent_mask = self.get_agent_mask(batch_size, info)

        # Hard Attention - action whether an agent communicates or not
        if self.args.hard_attn:
            # 如果info中没有comm_action，提供默认值（所有智能体都通信）
            if 'comm_action' not in info:
                if hasattr(self.args, 'comm_action_one') and self.args.comm_action_one:
                    comm_action = np.ones(self.nagents, dtype=int)
                else:
                    comm_action = np.zeros(self.nagents, dtype=int)
                info['comm_action'] = comm_action
            comm_action = torch.tensor(info['comm_action'])
            comm_action_mask = comm_action.expand(batch_size, n, n).unsqueeze(-1)
            # action 1 is talk, 0 is silent i.e. act as dead for comm purposes.
            agent_mask *= comm_action_mask.double()

        agent_mask_transpose = agent_mask.transpose(1, 2)
        
        # 一轮通信
        # Choose current or prev depending on recurrent
        comm = hidden_state.view(batch_size, n, self.hid_size) if self.args.recurrent else hidden_state
        
        # 保存通信前的隐藏状态用于计算因果影响
        hidden_state_before_comm = comm.clone().detach()  # [batch_size, n, hid_size]

        # Get the next communication vector based on next hidden state
        comm = comm.unsqueeze(-2).expand(-1, n, n, self.hid_size)

        # Create mask for masking self communication
        mask = self.comm_mask.view(1, n, n)
        mask = mask.expand(comm.shape[0], n, n)
        mask = mask.unsqueeze(-1)

        mask = mask.expand_as(comm)
        comm = comm * mask

        if hasattr(self.args, 'comm_mode') and self.args.comm_mode == 'avg' \
            and num_agents_alive > 1:
            comm = comm / (num_agents_alive - 1)

        # Mask comm_in
        # Mask communcation from dead agents
        comm = comm * agent_mask
        # Mask communication to dead agents
        comm = comm * agent_mask_transpose
        
        # Apply pruning mask if provided
        if 'pruning_mask' in info and info['pruning_mask'] is not None:
            pruning_mask = info['pruning_mask']  # shape: [n, n]
            # Convert to tensor if needed and expand to match comm shape
            if not isinstance(pruning_mask, torch.Tensor):
                pruning_mask = torch.tensor(pruning_mask, dtype=torch.float32, device=comm.device)
            # Expand pruning_mask to [batch_size, n, n, hid_size]
            pruning_mask = pruning_mask.view(1, n, n, 1).expand_as(comm)
            comm = comm * pruning_mask

        # Combine all of C_j for an ith agent which essentially are h_j
        comm_sum = comm.sum(dim=1)
        c = self.C_modules[0](comm_sum)

        if self.args.recurrent:
            # skip connection - combine comm. matrix and encoded input for all agents
            inp = x + c

            inp = inp.view(batch_size * n, self.hid_size)

            output = self.f_module(inp, (hidden_state, cell_state))

            hidden_state = output[0]
            cell_state = output[1]
        else: # MLP|RNN
            # Get next hidden state from f node
            # and Add skip connection from start and sum them
            hidden_state = sum([x, self.f_modules[0](hidden_state), c])
            hidden_state = self.tanh(hidden_state)
        
        # 保存通信后的隐藏状态用于计算因果影响
        if self.args.recurrent:
            hidden_state_after_comm = hidden_state.view(batch_size, n, self.hid_size).clone().detach()
        else:
            hidden_state_after_comm = hidden_state.clone().detach()
        
        # ========== 计算反事实因果影响 (Counterfactual Causal Influence) ==========
        # 为每个智能体i计算其对其他智能体j的因果影响
        # C_i→j = ||h_j^{with m_i} - h_j^{without m_i}||
        counterfactual_influences, counterfactual_samples = self._compute_counterfactual_influence(
            comm, x, hidden_state_before_comm, cell_state, agent_mask, n, batch_size
        )
        
        # 统计通信信息量并获取结果（传入通信前后的隐藏状态和反事实影响）
        self._calculate_comm_stats(comm, n, batch_size, 'round1', 
                                   hidden_state_before=hidden_state_before_comm,
                                   hidden_state_after=hidden_state_after_comm,
                                   counterfactual_influences=counterfactual_influences,
                                   active_mask_param=agent_mask)
        #print('comm_bits_stats:', self.comm_bits_stats)
        # 将通信统计信息添加到info字典中
        if isinstance(info, dict):
            info['comm_stats'] = self.comm_bits_stats
            # 添加反事实样本供 epoch 级别 CI 计算使用
            info['counterfactual_samples'] = counterfactual_samples

        # v = torch.stack([self.value_head(hidden_state[:, i, :]) for i in range(n)])
        # v = v.view(hidden_state.size(0), n, -1)
        value_head = self.value_head(hidden_state)
        h = hidden_state.view(batch_size, n, self.hid_size)

        if self.continuous:
            action_mean = self.action_mean(h)
            action_log_std = self.action_log_std.expand_as(action_mean)
            action_std = torch.exp(action_log_std)
            # will be used later to sample
            action = (action_mean, action_log_std, action_std)
        else:
            # discrete actions
            action = [F.log_softmax(head(h), dim=-1) for head in self.heads]

        if self.args.recurrent:
            return action, value_head, (hidden_state.clone(), cell_state.clone())
        else:
            # 兼容trainer.py总是期望3个返回值的情况
            # 在non-recurrent模式下，返回hidden_state作为第三个值
            return action, value_head, hidden_state

    def init_weights(self, m):
        if type(m) == nn.Linear:
            m.weight.data.normal_(0, self.init_std)

    def init_hidden(self, batch_size):
        # dim 0 = num of layers * num of direction
        # **修复**: 根据模型的训练/评估模式动态设置requires_grad
        # 在eval模式下不需要梯度，可以避免计算图累积
        requires_grad = self.training
        return tuple(( torch.zeros(batch_size * self.nagents, self.hid_size, requires_grad=requires_grad),
                       torch.zeros(batch_size * self.nagents, self.hid_size, requires_grad=requires_grad)))

