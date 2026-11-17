# Slightly adapted from a version from Dr. Abhishek Das

import math

import torch
import torch.nn.functional as F
from torch import nn
import numpy as np

from models import MLP
from action_utils import select_action, translate_action

class TarCommNetMLP(nn.Module):
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

        super(TarCommNetMLP, self).__init__()
        self.args = args
        self.nagents = args.nagents
        self.hid_size = args.hid_size
        self.comm_passes = 1  # 修改为只通信一轮
        self.recurrent = args.recurrent

        # 添加通信信息量统计变量 - 修改为只保留一轮通信
        self.comm_bits_stats = {
            'round1': {
                'total_elements': 0,         # 传输的消息数(元素数)
                'active_channels': 0,        # 活跃通信通道数（边数）
                'total_comm_num': 0,         # 总通信次数
                'entropy': 0.0,              # 通信信息熵
                'max_entropy': 0.0,          # 最大熵
                'effective_bandwidth': 0.0,  # 有效带宽利用率
                'message_similarity': 0.0,   # 消息相似度
                'causal_influence': 0.0,     # 因果影响
                'hidden_state_change': 0.0,  # 隐藏状态变化
                'message_impact_ratio': 0.0, # 消息影响比率
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

        ######################################################
        # [TarMAC changeset] Attentional communication modules
        ######################################################

        self.state2query = nn.Linear(args.hid_size, 16)
        self.state2key = nn.Linear(args.hid_size, 16)
        self.state2value = nn.Linear(args.hid_size, args.hid_size)


    def get_agent_mask(self, batch_size, info):
        n = self.nagents

        if 'alive_mask' in info:
            agent_mask = torch.from_numpy(info['alive_mask'])
            num_agents_alive = agent_mask.sum()
        else:
            agent_mask = torch.ones(n)
            num_agents_alive = n
        agent_mask = agent_mask.view(1, 1, n)
        agent_mask = agent_mask.expand(batch_size, n, n).unsqueeze(-1).clone()

        return num_agents_alive, agent_mask

    def forward_state_encoder(self, x):
        hidden_state, cell_state = None, None

        # x通常是一个列表 [state, prev_hid]
        if isinstance(x, list) or isinstance(x, tuple):
            if len(x) == 2:
                state, extras = x
                x = state
            else:
                # 如果只有一个元素，就直接使用
                x = x[0] if len(x) == 1 else x
                extras = None
        else:
            extras = None

        if self.args.recurrent and extras is not None:
            x = self.encoder(x)

            if self.args.rnn_type == 'LSTM':
                hidden_state, cell_state = extras
            else:
                hidden_state = extras
            # hidden_state = self.tanh( self.hidd_encoder(prev_hidden_state) + x)
        else:
            x = self.encoder(x)
            x = self.tanh(x)
            hidden_state = x

        return x, hidden_state, cell_state

    def _compute_gaussian_kl_divergence(self, mu1, var1, mu0, var0, epsilon=1e-6):
        """
        计算两个对角高斯分布之间的KL散度：D_KL(N(mu1, var1) || N(mu0, var0))
        """
        var1 = var1 + epsilon
        var0 = var0 + epsilon
        
        mean_diff_term = ((mu1 - mu0) ** 2) / var0
        var_ratio_term = var1 / var0
        log_var_term = torch.log(var1 / var0)
        
        kl_div = 0.5 * (mean_diff_term + var_ratio_term - 1 - log_var_term).sum(dim=-1)
        
        return kl_div

    def _compute_counterfactual_influence(self, comm, hidden_state_before, attn, agent_mask, n, batch_size):
        """
        计算反事实因果影响：C_i→j = D_KL(p(h_j | m_i) || p(h_j | no m_i))
        
        对于TarMAC，移除agent i的影响意味着：
        1. 将注意力矩阵中agent j对agent i的注意力设为0
        2. 重新归一化注意力权重
        3. 重新计算聚合的通信消息
        
        Arguments:
            comm: 原始通信消息 [batch_size, n, hid_size] (value向量)
            hidden_state_before: 通信前的隐藏状态 [batch_size, n, hid_size]
            attn: 注意力权重矩阵 [batch_size, n, n]
            agent_mask: agent掩码 [batch_size, n, n, 1]
            n: 智能体数量
            batch_size: 批次大小
            
        Returns:
            influences: [batch_size, n, n] - influences[b, i, j]表示agent i对agent j的因果影响（瞬时CI）
            counterfactual_samples: 字典 - 包含用于 epoch 级别 CI 计算的样本
        """
        with torch.no_grad():
            influences = torch.zeros(batch_size, n, n)
            
            # 存储反事实样本
            comm_with_all = torch.zeros(batch_size, n, n, self.hid_size)
            comm_without_all = torch.zeros(batch_size, n, n, self.hid_size)
            
            # 计算有所有agent通信的最终状态（使用原始注意力）
            comm_aggregated_with_all = torch.matmul(attn, comm)  # [batch_size, n, hid_size]
            
            # 对每个发送者i，计算移除其影响后的通信结果
            for i in range(n):
                # 创建反事实注意力矩阵（移除对agent i的注意）
                attn_without_i = attn.clone()  # [batch_size, n, n]
                
                # 对每个接收者j，将其对agent i的注意力设为0
                attn_without_i[:, :, i] = 0  # 所有agent对agent i的注意力设为0
                
                # 重新归一化注意力（确保每行和为1）
                attn_sum = attn_without_i.sum(dim=-1, keepdim=True)  # [batch_size, n, 1]
                # 避免除零
                attn_without_i = torch.where(
                    attn_sum > 1e-8,
                    attn_without_i / attn_sum,
                    attn_without_i
                )
                
                # 使用修改后的注意力重新聚合通信
                comm_aggregated_without_i = torch.matmul(attn_without_i, comm)  # [batch_size, n, hid_size]
                
                # 计算因果影响：比较有/无agent i时的聚合结果
                for j in range(n):
                    if i != j:
                        # 提取agent j的聚合通信结果
                        comm_with = comm_aggregated_with_all[:, j, :]  # [batch_size, hid_size]
                        comm_without = comm_aggregated_without_i[:, j, :]  # [batch_size, hid_size]
                        
                        # 使用简化的KL散度（L2距离的平方，假设方差=1）
                        # 归一化：除以 hid_size 以获得每维度的平均影响
                        influence = 0.5 * torch.sum((comm_with - comm_without) ** 2, dim=-1) / self.hid_size
                        influences[:, i, j] = influence
                        
                        # 存储样本供 epoch 级别计算
                        comm_with_all[:, i, j, :] = comm_with
                        comm_without_all[:, i, j, :] = comm_without
            
            # 计算通信掩码（基于注意力矩阵）
            # attn: [batch_size, n, n]
            # 注意：TarMAC 允许自注意力，但我们排除它
            comm_mask = attn.clone()
            for b in range(batch_size):
                for i in range(n):
                    comm_mask[b, i, i] = 0  # 排除自注意力
            # 将注意力权重二值化：> 阈值则认为有通信
            comm_mask = (comm_mask > 1e-6).float()
            
            # 返回瞬时 CI 和反事实样本
            counterfactual_samples = {
                'with': comm_with_all,       # [batch, n, n, hid_size]
                'without': comm_without_all, # [batch, n, n, hid_size]
                'comm_mask': comm_mask       # [batch, n, n] - 1表示有通信，0表示无通信
            }
            
            return influences, counterfactual_samples

    def forward(self, x, info={}):
        # TODO: Update dimensions
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

        # if self.args.env_name == 'starcraft':
        #     maxi = x.max(dim=-2)[0]
        #     x = self.state_encoder(x)
        #     x = x.sum(dim=-2)
        #     x = torch.cat([x, maxi], dim=-1)
        #     x = self.tanh(x)

        x, hidden_state, cell_state = self.forward_state_encoder(x)

        batch_size = x.size()[0]
        n = self.nagents

        num_agents_alive, agent_mask = self.get_agent_mask(batch_size, info)
        agent_mask_alive = agent_mask.clone()

        # Hard Attention - action whether an agent communicates or not
        if self.args.hard_attn:
            comm_action = torch.tensor(info['comm_action'])
            comm_action_mask = comm_action.expand(batch_size, n, n).unsqueeze(-1)
            # action 1 is talk, 0 is silent i.e. act as dead for comm purposes.
            agent_mask *= comm_action_mask.double()

        agent_mask_transpose = agent_mask.transpose(1, 2)

        # 保存通信前的隐藏状态（用于计算因果影响）
        hidden_state_before_comm = hidden_state.view(batch_size, n, self.hid_size).clone().detach() if self.args.recurrent else hidden_state.view(batch_size, n, self.hid_size).clone().detach()

        # 只进行一轮通信
        # Choose current or prev depending on recurrent
        comm = hidden_state.view(batch_size, n, self.hid_size) if self.args.recurrent else hidden_state

        if self.args.comm_mask_zero:
            comm_mask = torch.zeros_like(comm)
            comm = comm * comm_mask
        #########################################################
        # [TarMAC changeset] Don't expand same comm vector to all
        #########################################################
        # Get the next communication vector based on next hidden state
        # comm = comm.unsqueeze(-2).expand(-1, n, n, self.hid_size)
        #########################################################

        ########################################################
        # [TarMAC changeset] Removing self-communication masking
        ########################################################
        # # Create mask for masking self communication
        # mask = self.comm_mask.view(1, n, n)
        # mask = mask.expand(comm.shape[0], n, n)
        # mask = mask.unsqueeze(-1)
        # mask = mask.expand_as(comm)
        # comm = comm * mask
        ########################################################

        ############################################################
        # [TarMAC changeset] Replacing averaging with soft-attention
        ############################################################
        # if hasattr(self.args, 'comm_mode') and self.args.comm_mode == 'avg' \
        #     and num_agents_alive > 1:
        #     comm = comm / (num_agents_alive - 1)
        ############################################################

        # if info['comm_action'].sum() != 0:
        #     import pdb; pdb.set_trace()

        #########################################################
        # [TarMAC changeset] Attentional communication b/w agents
        #########################################################
        # compute q, k, c
        query = self.state2query(comm)
        key = self.state2key(comm)
        value = self.state2value(comm)

        # scores
        scores = torch.matmul(query, key.transpose(
            -2, -1)) / math.sqrt(self.hid_size)
        # scores = scores.masked_fill(comm_action_mask.squeeze(-1) == 0, -1e9)
        # Use agent_mask instead of comm_action_mask to make this work in tj env
        scores = scores.masked_fill(agent_mask.squeeze(-1) == 0, -1e9)

        # softmax + weighted sum
        attn = F.softmax(scores, dim=-1)
        # if the scores are all -1e9 for all agents, the attns should be all 0 (fixed from the original version)
        attn = attn * agent_mask.squeeze(-1) # cannot use inplace operation *=
        #print('attn:', attn)
        
        # 保存用于因果影响计算的注意力和value
        attn_for_causal = attn.clone().detach()
        value_for_causal = value.clone().detach()
        
        comm = torch.matmul(attn, value)
        
        ####################################################
        # [TarMAC changeset] Incorporated this masking above
        ####################################################
        # # Mask comm_in
        # # Mask communcation from dead agents
        # comm = comm * agent_mask
        # # Mask communication to dead agents
        # comm = comm * agent_mask_transpose
        ###########################################################

        ###########################################################
        # [TarMAC changeset] Replaced this averaging with attention
        ###########################################################
        # # Combine all of C_j for an ith agent which essentially are h_j
        # comm_sum = comm.sum(dim=1)
        ###########################################################
        # for tj: dead agents do not receive messages
        # for tj: alive agents with no comm actions can receive messages (align with tarmac+ic3net in pp)
        comm *= agent_mask_alive.squeeze(-1)[:, 0].unsqueeze(-1).expand(batch_size, n, self.hid_size)
        #print('comm:', comm)
        
        c = self.C_modules[0](comm)

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
        
        # 保存通信后的隐藏状态（用于计算因果影响）
        hidden_state_after_comm = hidden_state.view(batch_size, n, self.hid_size).clone().detach() if self.args.recurrent else hidden_state.view(batch_size, n, self.hid_size).clone().detach()
        
        # ========== 计算反事实因果影响 (Counterfactual Causal Influence) ==========
        counterfactual_influences, counterfactual_samples = self._compute_counterfactual_influence(
            value_for_causal,
            hidden_state_before_comm,
            attn_for_causal,
            agent_mask,
            n,
            batch_size
        )
        
        # 计算通信统计指标（传入通信前后的隐藏状态和反事实影响）
        with torch.no_grad():
            self._calculate_comm_stats(
                comm, 
                attn_for_causal, 
                n, 
                'round1',
                hidden_state_before=hidden_state_before_comm,
                hidden_state_after=hidden_state_after_comm,
                counterfactual_influences=counterfactual_influences
            )
        
        # 将通信统计信息添加到info字典中以供外部访问
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
            # 在non-recurrent模式下，也返回hidden_state作为第三个值以保持接口一致
            return action, value_head, hidden_state

    def init_weights(self, m):
        if type(m) == nn.Linear:
            m.weight.data.normal_(0, self.init_std)

    def init_hidden(self, batch_size):
        # dim 0 = num of layers * num of direction
        return tuple(( torch.zeros(batch_size * self.nagents, self.hid_size, requires_grad=True),
                       torch.zeros(batch_size * self.nagents, self.hid_size, requires_grad=True)))

    def _calculate_comm_stats(self, comm, attn, n, round_name, hidden_state_before=None, hidden_state_after=None, counterfactual_influences=None):
        """
        计算通信信息量统计指标，包括因果影响度量
        
        Arguments:
            comm (tensor): 通信消息 [batch_size, n, hid_size]
            attn (tensor): 注意力矩阵 [batch_size, n, n]
            n (int): 智能体数量
            round_name (str): 通信轮次名称 ('round1')
            hidden_state_before (tensor): 通信前的隐藏状态 [batch_size, n, hid_size] (可选)
            hidden_state_after (tensor): 通信后的隐藏状态 [batch_size, n, hid_size] (可选)
            counterfactual_influences (tensor): 反事实因果影响 [batch_size, n, n] (可选)
        """
        # 重置统计信息
        total_elements = 0
        active_agents = 0
        active_channels = 0
        
        # 创建注意力矩阵的副本，将对角线元素置为0，去除自身通信
        attn_no_self = attn.clone()
        batch_size = attn.size(0)
        for b in range(batch_size):
            for i in range(n):
                attn_no_self[b, i, i] = 0
        #print('attn_no_self:', attn_no_self)
        # 遍历每个智能体，检查它是否接收到信息
        for b in range(batch_size):
            for i in range(n):
                # 检查该智能体是否接收到信息(comm是否不全为0)
                comm_row = comm[b, i]
                if torch.sum(torch.abs(comm_row) > 1e-6) > 0:
                    active_agents += 1
                    # 按照定义，如果一个智能体能接收信息，它从(n-1)个其他智能体接收
                    # 所以每个活跃智能体贡献(n-1)个活跃通道
                    active_channels += (n - 1)
                    # 总传输元素数 = 活跃通道数 * 消息大小
                    total_elements += (n - 1) * self.hid_size
        
        # 计算信息熵: -sum(p*log(p))
        eps = 1e-10
        entropy = 0.0
        max_entropy = 0.0
        total_valid_agents = 0
        
        # 收集所有有效智能体的消息，用于后续计算相似度
        valid_messages = []
        
        for b in range(batch_size):
            for i in range(n):
                comm_row = comm[b, i]
                if torch.sum(torch.abs(comm_row) > 1e-6) > 0:
                    # 计算该智能体消息的概率分布
                    agent_sum = torch.sum(torch.abs(comm_row)) + eps
                    agent_prob_dist = torch.abs(comm_row) / agent_sum
                    
                    # 保存有效消息用于计算相似度
                    valid_messages.append(comm_row.detach().clone())
                    
                    # 过滤掉太小的概率值
                    threshold = 1e-6
                    agent_valid_probs = agent_prob_dist[agent_prob_dist > threshold]
                    
                    if len(agent_valid_probs) > 0:
                        # 重新归一化
                        agent_valid_probs = agent_valid_probs / torch.sum(agent_valid_probs)
                        # 计算该智能体消息的熵
                        agent_entropy = -torch.sum(agent_valid_probs * torch.log2(agent_valid_probs + eps)).item()
                        
                        # 最大熵计算：如果所有元素均匀分布时的熵
                        # 对于长度为L的消息，最大熵是log2(L)
                        message_length = self.hid_size
                        agent_max_entropy = np.log2(message_length)
                        
                        # 累加总熵和总最大熵
                        entropy += agent_entropy
                        max_entropy += agent_max_entropy
                        total_valid_agents += 1
        #print('valid_messages:', valid_messages)
        # 计算智能体间消息相似度 (基于余弦相似度)
        message_similarity = 0.0
        if len(valid_messages) >= 2:
            # 构建所有智能体对之间的余弦相似度
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
        
        # 如果有通信，计算平均熵
        if total_valid_agents > 0:
            # 按发送消息的智能体数量计算平均熵和平均最大熵
            entropy = entropy / total_valid_agents
            max_entropy = max_entropy / total_valid_agents
        else:
            # 如果没有有效通信，设置默认值
            entropy = 0.0
            max_entropy = 1.0  # 避免除零
            
        # 计算有效带宽利用率
        effective_bandwidth = entropy / max_entropy if max_entropy > 0 else 0.0
        
        # ========== 新增：计算因果影响指标 (Causal Influence Metrics) ==========
        causal_influence = 0.0
        hidden_state_change_norm = 0.0
        message_impact_ratio = 0.0
        per_agent_influence = np.zeros(n)
        pairwise_influence_matrix = np.zeros((n, n))
        
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
                # 如果没有反事实影响，使用简化估计
                for b in range(batch_size):
                    for i in range(n):
                        if torch.sum(torch.abs(comm[b, i])) > 1e-6:
                            for j in range(n):
                                if i != j and attn[b, j, i] > 1e-6:
                                    m_i = comm[b, i]
                                    delta_h_j = state_change[b, j]
                                    message_norm = torch.norm(m_i, p=2).item() + eps
                                    state_change_norm_j = torch.norm(delta_h_j, p=2).item()
                                    per_agent_influence[i] += state_change_norm_j * (message_norm / (message_norm + 1.0))
                
                per_agent_influence = per_agent_influence / batch_size if batch_size > 0 else per_agent_influence
                causal_influence = per_agent_influence.mean()
            
            # 3. 消息影响比率：状态变化 vs 消息强度
            total_message_strength = torch.norm(comm, p=2, dim=-1).sum().item() + eps
            total_state_change = torch.norm(state_change, p=2, dim=-1).sum().item()
            message_impact_ratio = total_state_change / total_message_strength
        
        # 更新特定轮次的统计信息
        self.comm_bits_stats[round_name]['total_elements'] = total_elements
        self.comm_bits_stats[round_name]['active_channels'] = active_channels
        self.comm_bits_stats[round_name]['total_comm_num'] = active_channels
        self.comm_bits_stats[round_name]['entropy'] = entropy
        self.comm_bits_stats[round_name]['max_entropy'] = max_entropy
        self.comm_bits_stats[round_name]['effective_bandwidth'] = effective_bandwidth
        self.comm_bits_stats[round_name]['message_similarity'] = message_similarity
        # 新增因果影响指标
        self.comm_bits_stats[round_name]['causal_influence'] = causal_influence
        self.comm_bits_stats[round_name]['hidden_state_change'] = hidden_state_change_norm
        self.comm_bits_stats[round_name]['message_impact_ratio'] = message_impact_ratio
        self.comm_bits_stats[round_name]['per_agent_influence'] = per_agent_influence.tolist() if isinstance(per_agent_influence, np.ndarray) else per_agent_influence
        self.comm_bits_stats[round_name]['pairwise_influence_matrix'] = pairwise_influence_matrix.tolist() if isinstance(pairwise_influence_matrix, np.ndarray) else pairwise_influence_matrix

