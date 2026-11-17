import torch
import torch.nn.functional as F
from torch import nn
import numpy as np
from action_utils import select_action, translate_action
from gnn_layers import GraphAttention

class MAGIC(nn.Module):
    """
    The communication protocol of Multi-Agent Graph AttentIon Communication (MAGIC)
    """
    def __init__(self, args):
        super(MAGIC, self).__init__()
        """
        Initialization method for the MAGIC communication protocol (1 round of communication)

        Arguements:
            args (Namespace): Parse arguments
        """

        self.args = args
        self.nagents = args.nagents
        self.hid_size = args.hid_size

        # 添加通信信息量统计变量
        self.comm_bits_stats = {
            'round1': {
                'total_elements': 0,         # 传输的消息数(元素数)
                'active_channels': 0,        # 活跃通信通道数（边数）
                'entropy': 0.0,              # 通信信息熵
                'max_entropy': 0.0,          # 最大熵
                'effective_bandwidth': 0.0,  # 有效带宽利用率
                'message_similarity': 0.0,   # 消息相似度
                'total_comm_num': 0,         # 总通信次数
                'causal_influence': 0.0,     # 因果影响（新增）
                'hidden_state_change': 0.0,  # 隐藏状态变化（新增）
                'message_impact_ratio': 0.0, # 消息影响效率比率（新增）
            }
        }

        dropout = 0
        negative_slope = 0.2

        # initialize sub-processor
        self.sub_processor = GraphAttention(args.hid_size, args.hid_size, dropout=dropout, negative_slope=negative_slope, num_heads=args.gat_num_heads, self_loop_type=args.self_loop_type1, average=True, normalize=args.first_gat_normalize)
        
        # initialize the gat encoder for the Scheduler
        if args.use_gat_encoder:
            self.gat_encoder = GraphAttention(args.hid_size, args.gat_encoder_out_size, dropout=dropout, negative_slope=negative_slope, num_heads=args.ge_num_heads, self_loop_type=1, average=True, normalize=args.gat_encoder_normalize)

        self.obs_encoder = nn.Linear(args.obs_size, args.hid_size)

        self.init_hidden(args.batch_size)
        self.lstm_cell= nn.LSTMCell(args.hid_size, args.hid_size)

        # initialize mlp layers for the scheduler
        if not args.first_graph_complete:
            if args.use_gat_encoder:
                self.sub_scheduler_mlp = nn.Sequential(
                    nn.Linear(args.gat_encoder_out_size*2, args.gat_encoder_out_size//2),
                    nn.ReLU(),
                    nn.Linear(args.gat_encoder_out_size//2, args.gat_encoder_out_size//2),
                    nn.ReLU(),
                    nn.Linear(args.gat_encoder_out_size//2, 2))
            else:
                self.sub_scheduler_mlp = nn.Sequential(
                    nn.Linear(self.hid_size*2, self.hid_size//2),
                    nn.ReLU(),
                    nn.Linear(self.hid_size//2, self.hid_size//8),
                    nn.ReLU(),
                    nn.Linear(self.hid_size//8, 2))

        if args.message_encoder:
            self.message_encoder = nn.Linear(args.hid_size, args.hid_size)
        if args.message_decoder:
            self.message_decoder = nn.Linear(args.hid_size, args.hid_size)

        # initialize weights as 0
        if args.comm_init == 'zeros':
            if args.message_encoder:
                self.message_encoder.weight.data.zero_()
            if args.message_decoder:
                self.message_decoder.weight.data.zero_()
            if not args.first_graph_complete:
                self.sub_scheduler_mlp.apply(self.init_linear)
                   
        # initialize the action head (in practice, one action head is used)
        self.action_heads = nn.ModuleList([nn.Linear(2*args.hid_size, o)
                                        for o in args.naction_heads])
        # initialize the value head
        self.value_head = nn.Linear(2 * self.hid_size, 1)


    def forward(self, x, info={}):
        """
        Forward function of MAGIC (one round of communication)

        Arguments:
            x (list): a list for the input of the communication protocol [observations, (previous hidden states, previous cell states)]
            observations (tensor): the observations for all agents [1 (batch_size) * n * obs_size]
            previous hidden/cell states (tensor): the hidden/cell states from the previous time steps [n * hid_size]

        Returns:
            action_out (list): a list of tensors of size [1 (batch_size) * n * num_actions] that represent output policy distributions
            value_head (tensor): estimated values [n * 1]
            next hidden/cell states (tensor): next hidden/cell states [n * hid_size]
        """

        # n: number of agents

        obs, extras = x

        # encoded_obs: [1 (batch_size) * n * hid_size]
        encoded_obs = self.obs_encoder(obs)
        hidden_state, cell_state = extras

        batch_size = encoded_obs.size()[0]
        n = self.nagents

        num_agents_alive, agent_mask = self.get_agent_mask(batch_size, info)

        # if self.args.comm_mask_zero == True, block the communiction (can also comment out the protocol to make training faster)
        if self.args.comm_mask_zero:
            agent_mask *= torch.zeros(n, 1)

        hidden_state, cell_state = self.lstm_cell(encoded_obs.squeeze(), (hidden_state, cell_state))

        # 保存通信前的隐藏状态（用于计算因果影响）
        hidden_state_before_comm = hidden_state.clone().detach()

        # comm: [n * hid_size]
        comm = hidden_state
        if self.args.message_encoder:
            comm = self.message_encoder(comm)
            
        # mask communcation from dead agents (only effective in Traffic Junction)
        comm = comm * agent_mask

        # scheduler
        # if args.first_graph_complete == True, scheduler will be disabled
        if not self.args.first_graph_complete:
            if self.args.use_gat_encoder:
                adj_complete = self.get_complete_graph(agent_mask)
                encoded_state = self.gat_encoder(comm, adj_complete)
                adj = self.sub_scheduler(self.sub_scheduler_mlp, encoded_state, agent_mask, self.args.directed)
            else:
                adj = self.sub_scheduler(self.sub_scheduler_mlp, comm, agent_mask, self.args.directed)
        else:
            adj = self.get_complete_graph(agent_mask)

        # sub-processor
        comm = F.elu(self.sub_processor(comm, adj))
        
        # 保存通信后的隐藏状态（用于计算因果影响）
        # 注意：在MAGIC中，通信后的状态就是经过sub_processor处理后的comm
        hidden_state_after_comm = comm.clone().detach()
        
        # ========== 计算反事实因果影响 (Counterfactual Causal Influence) ==========
        counterfactual_influences, counterfactual_samples = self._compute_counterfactual_influence(
            hidden_state,
            adj,
            agent_mask,
            n
        )
        
        # 计算通信的信息量指标（传入通信前后的隐藏状态和反事实影响）
        with torch.no_grad():
            self._calculate_comm_stats(
                comm, 
                adj, 
                'round1', 
                n,
                hidden_state_before=hidden_state_before_comm,
                hidden_state_after=hidden_state_after_comm,
                counterfactual_influences=counterfactual_influences
            )
        
        # 将通信统计信息添加到info字典中以供外部访问
        if isinstance(info, dict):
            info['comm_stats'] = self.comm_bits_stats
            # 添加反事实样本供 epoch 级别 CI 计算使用
            info['counterfactual_samples'] = counterfactual_samples
            
        if self.args.message_decoder:
            comm = self.message_decoder(comm)

        value_head = self.value_head(torch.cat((hidden_state, comm), dim=-1))
        h = hidden_state.view(batch_size, n, self.hid_size)
        c = comm.view(batch_size, n, self.hid_size)

        action_out = [F.log_softmax(action_head(torch.cat((h, c), dim=-1)), dim=-1) for action_head in self.action_heads]

        return action_out, value_head, (hidden_state.clone(), cell_state.clone())
        
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

    def _compute_counterfactual_influence(self, hidden_state, adj, agent_mask, n):
        """
        计算反事实因果影响：C_i→j = ||h_j^{with m_i} - h_j^{without m_i}||
        
        对于MAGIC，移除agent i的影响意味着：
        1. 在邻接矩阵中将所有指向agent i的边设为0
        2. 重新运行sub_processor
        3. 比较有/无agent i时的最终通信状态
        
        Arguments:
            hidden_state: LSTM后的隐藏状态 [n, hid_size]
            adj: 邻接矩阵 [n, n]
            agent_mask: agent掩码 [n, 1]
            n: 智能体数量
            
        Returns:
            influences: [1, n, n] - influences[0, i, j]表示agent i对agent j的因果影响（瞬时CI）
            counterfactual_samples: 字典 - 包含用于 epoch 级别 CI 计算的样本
        """
        with torch.no_grad():
            # batch_size = 1 in the environment
            batch_size = 1
            influences = torch.zeros(batch_size, n, n)
            
            # 存储反事实样本
            hidden_states_with = torch.zeros(batch_size, n, n, self.hid_size)
            hidden_states_without = torch.zeros(batch_size, n, n, self.hid_size)
            
            # 准备通信消息
            comm = hidden_state.clone()
            if self.args.message_encoder:
                comm = self.message_encoder(comm)
            comm = comm * agent_mask
            
            # 计算有所有agent通信的最终状态（使用原始邻接矩阵）
            comm_with_all = F.elu(self.sub_processor(comm, adj))  # [n, hid_size]
            
            # 对每个发送者i，计算移除其影响后的通信结果
            for i in range(n):
                # 创建反事实邻接矩阵（移除对agent i的所有通信链路）
                adj_without_i = adj.clone()
                # 将所有agent对agent i的通信设为0（即第i列）
                adj_without_i[:, i] = 0
                
                # 使用修改后的邻接矩阵重新计算通信
                comm_without_i = F.elu(self.sub_processor(comm, adj_without_i))  # [n, hid_size]
                
                # 计算因果影响：比较有/无agent i时每个agent j的状态
                for j in range(n):
                    if i != j:
                        # 提取agent j的状态
                        state_with = comm_with_all[j]  # [hid_size]
                        state_without = comm_without_i[j]  # [hid_size]
                        
                        # 使用简化的KL散度（L2距离的平方，假设方差=1）
                        # 归一化：除以 hid_size 以获得每维度的平均影响
                        influence = 0.5 * torch.sum((state_with - state_without) ** 2) / self.hid_size
                        influences[0, i, j] = influence
                        
                        # 存储样本供 epoch 级别计算
                        hidden_states_with[0, i, j, :] = state_with
                        hidden_states_without[0, i, j, :] = state_without
            
            # 计算通信掩码（基于邻接矩阵）
            comm_mask = adj.clone().unsqueeze(0)  # [1, n, n]
            # 排除自身通信
            for idx in range(n):
                comm_mask[0, idx, idx] = 0
            
            # 返回瞬时 CI 和反事实样本
            counterfactual_samples = {
                'with': hidden_states_with,      # [1, n, n, hid_size]
                'without': hidden_states_without, # [1, n, n, hid_size]
                'comm_mask': comm_mask           # [1, n, n] - 1表示有通信，0表示无通信
            }
            
            return influences, counterfactual_samples

    def _calculate_comm_stats(self, comm, adj, round_name, n, hidden_state_before=None, hidden_state_after=None, counterfactual_influences=None):
        """
        计算通信信息量统计指标，包括因果影响度量
        
        Arguments:
            comm (tensor): 通信消息 [n * hid_size]
            adj (tensor): 邻接矩阵 [n * n]
            round_name (str): 通信轮次名称 ('round1')
            n (int): 智能体数量
            hidden_state_before (tensor): 通信前的隐藏状态 [n, hid_size] (可选)
            hidden_state_after (tensor): 通信后的隐藏状态 [n, hid_size] (可选)
            counterfactual_influences (tensor): 反事实因果影响 [1, n, n] (可选)
        """
        # 创建邻接矩阵的副本，将对角线元素置为0，去除自身通信
        adj_no_self = adj.clone()
        for i in range(n):
            adj_no_self[i, i] = 0
            
        # 计算通信图中实际存在的边数（排除自身通信）
        active_channels = torch.sum(adj_no_self > 0).item()
        
        # 通信图密度（实际边数与可能的最大边数之比）
        max_edges = n * (n - 1) if self.args.directed else (n * (n - 1) / 2)
        
        graph_density = active_channels / max_edges if max_edges > 0 else 0
        
        # 计算参与通信的智能体数量
        active_agents = torch.sum(torch.sum(adj_no_self > 0, dim=1) > 0).item()
        
        # 通过邻接矩阵过滤实际传输的通信内容
        total_elements = 0
        # 计算消息大小
        msg_size = self.hid_size
                
        for i in range(n):
            # 获取节点i的所有接收方（排除自身）
            receivers = torch.where(adj_no_self[i] > 0)[0]
            if len(receivers) > 0 and torch.sum(torch.abs(comm[i]) > 1e-6) > 0:
                # 计算总共发送的元素数量
                total_elements += len(receivers) * msg_size
                
        # 计算信息熵: -sum(p*log(p))
        # 考虑通信图结构的信息熵计算
        eps = 1e-10
        entropy = 0.0
        max_entropy = 0.0
        total_valid_agents = 0  # 用于计算平均熵
        
        # 收集所有有效智能体的消息，用于后续计算相似度
        valid_messages = []
        valid_agent_indices = []
        
        # 对每个发送消息的智能体计算信息熵
        for i in range(n):
            # 获取节点i的所有接收方（排除自身）
            receivers = torch.where(adj_no_self[i] > 0)[0]
            if len(receivers) > 0 and torch.sum(torch.abs(comm[i]) > 1e-6) > 0:
                # 计算该智能体消息的概率分布
                agent_comm = comm[i]
                agent_sum = torch.sum(torch.abs(agent_comm)) + eps
                agent_prob_dist = torch.abs(agent_comm) / agent_sum
                
                # 保存有效消息用于计算相似度
                valid_messages.append(agent_comm.detach().clone())
                valid_agent_indices.append(i)
                
                # 过滤掉太小的概率值
                threshold = 1e-6
                agent_valid_probs = agent_prob_dist[agent_prob_dist > threshold]
                
                if len(agent_valid_probs) > 0:
                    # 重新归一化
                    agent_valid_probs = agent_valid_probs / torch.sum(agent_valid_probs)
                    # 计算该智能体消息的熵
                    agent_entropy = -torch.sum(agent_valid_probs * torch.log2(agent_valid_probs)).item()
                    
                    # 最大熵计算：如果所有元素均匀分布时的熵
                    # 对于长度为L的消息，最大熵是log2(L)
                    message_length = msg_size  # 使用实际消息大小， 也就是均匀分布的熵
                    agent_max_entropy = np.log2(message_length)
                    
                    # 累加总熵和总最大熵 - 不再乘以接收者数量
                    entropy += agent_entropy
                    max_entropy += agent_max_entropy
                    total_valid_agents += 1
        
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
        
        # 统计总接收者数量 - 用于其他计算
        total_receivers = 0
        for i in range(n):
            receivers = torch.where(adj_no_self[i] > 0)[0]
            if len(receivers) > 0 and torch.sum(torch.abs(comm[i]) > 1e-6) > 0:
                total_receivers += len(receivers)
        
        # ========== 新增：计算因果影响指标 (Causal Influence Metrics) ==========
        causal_influence = 0.0
        hidden_state_change_norm = 0.0
        message_impact_ratio = 0.0
        per_agent_influence = np.zeros(n)  # 每个智能体的因果影响
        pairwise_influence_matrix = np.zeros((n, n))  # 智能体间的成对影响矩阵
        
        if hidden_state_before is not None and hidden_state_after is not None:
            # 计算隐藏状态变化
            state_change = hidden_state_after - hidden_state_before  # [n, hid_size]
            
            # 1. 总体隐藏状态变化范数（衡量通信导致的状态更新程度）
            hidden_state_change_norm = torch.norm(state_change, p=2, dim=-1).mean().item()
            
            # 2. 使用反事实影响计算真实的因果影响
            if counterfactual_influences is not None:
                # counterfactual_influences: [1, n, n]
                # influences[0, i, j] = agent i对agent j的因果影响
                
                # 对每个智能体i，计算其总因果影响 = Σ_j C_i→j
                for i in range(n):
                    # agent i对所有其他agent的影响之和
                    agent_i_total_influence = 0.0
                    for j in range(n):
                        if i != j:
                            c_ij = counterfactual_influences[0, i, j].item()
                            agent_i_total_influence += c_ij
                            pairwise_influence_matrix[i, j] = c_ij
                    
                    per_agent_influence[i] = agent_i_total_influence
                
                # 总体因果影响 = 所有智能体的平均影响
                causal_influence = per_agent_influence.mean()
            else:
                # 如果没有反事实影响，使用简化估计（基于邻接矩阵和状态变化）
                eps = 1e-10
                for i in range(n):
                    # 检查agent i是否有发送消息（通过邻接矩阵检查）
                    has_comm = adj_no_self[i].sum() > 0
                    
                    if has_comm and torch.sum(torch.abs(comm[i])) > 1e-6:
                        for j in range(n):
                            if i != j and adj_no_self[i, j] > 0:
                                # 使用消息强度和状态变化的相关性估计影响
                                m_i = comm[i]
                                delta_h_j = state_change[j]
                                message_norm = torch.norm(m_i, p=2).item() + eps
                                state_change_norm_j = torch.norm(delta_h_j, p=2).item()
                                per_agent_influence[i] += state_change_norm_j * (message_norm / (message_norm + 1.0))
                
                causal_influence = per_agent_influence.mean()
            
            # 3. 消息影响比率：状态变化 vs 消息强度
            total_message_strength = torch.norm(comm, p=2, dim=-1).sum().item() + eps
            total_state_change = torch.norm(state_change, p=2, dim=-1).sum().item()
            message_impact_ratio = total_state_change / total_message_strength if total_message_strength > eps else 0.0
        
        # 更新统计信息
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

    def get_agent_mask(self, batch_size, info):
        """
        Function to generate agent mask to mask out inactive agents (only effective in Traffic Junction)

        Returns:
            num_agents_alive (int): number of active agents
            agent_mask (tensor): [n, 1]
        """

        n = self.nagents

        if 'alive_mask' in info:
            agent_mask = torch.from_numpy(info['alive_mask'])
            num_agents_alive = agent_mask.sum()
        else:
            agent_mask = torch.ones(n)
            num_agents_alive = n

        agent_mask = agent_mask.view(n, 1).clone()

        return num_agents_alive, agent_mask

    def init_linear(self, m):
        """
        Function to initialize the parameters in nn.Linear as o 
        """
        if type(m) == nn.Linear:
            m.weight.data.fill_(0.)
            m.bias.data.fill_(0.)
        
    def init_hidden(self, batch_size):
        """
        Function to initialize the hidden states and cell states
        """
        return tuple(( torch.zeros(batch_size * self.nagents, self.hid_size, requires_grad=True),
                       torch.zeros(batch_size * self.nagents, self.hid_size, requires_grad=True)))
    
    
    def sub_scheduler(self, sub_scheduler_mlp, hidden_state, agent_mask, directed=True):
        """
        Function to perform a sub-scheduler

        Arguments: 
            sub_scheduler_mlp (nn.Sequential): the MLP layers in a sub-scheduler
            hidden_state (tensor): the encoded messages input to the sub-scheduler [n * hid_size]
            agent_mask (tensor): [n * 1]
            directed (bool): decide if generate directed graphs

        Return:
            adj (tensor): a adjacency matrix which is the communication graph [n * n]  
        """

        # hidden_state: [n * hid_size]
        n = self.args.nagents
        hid_size = hidden_state.size(-1)
        # hard_attn_input: [n * n * (2*hid_size)]
        hard_attn_input = torch.cat([hidden_state.repeat(1, n).view(n * n, -1), hidden_state.repeat(n, 1)], dim=1).view(n, -1, 2 * hid_size)
        # hard_attn_output: [n * n * 2]
        if directed:
            hard_attn_output = F.gumbel_softmax(sub_scheduler_mlp(hard_attn_input), hard=True)
        else:
            hard_attn_output = F.gumbel_softmax(0.5*sub_scheduler_mlp(hard_attn_input)+0.5*sub_scheduler_mlp(hard_attn_input.permute(1,0,2)), hard=True)
        # hard_attn_output: [n * n * 1]
        hard_attn_output = torch.narrow(hard_attn_output, 2, 1, 1)
        # agent_mask and agent_mask_transpose: [n * n]
        agent_mask = agent_mask.expand(n, n)
        agent_mask_transpose = agent_mask.transpose(0, 1)
        # adj: [n * n]
        adj = hard_attn_output.squeeze() * agent_mask * agent_mask_transpose
        
        return adj
    
    def get_complete_graph(self, agent_mask):
        """
        Function to generate a complete graph, and mask it with agent_mask
        """
        n = self.args.nagents
        adj = torch.ones(n, n)
        agent_mask = agent_mask.expand(n, n)
        agent_mask_transpose = agent_mask.transpose(0, 1)
        adj = adj * agent_mask * agent_mask_transpose
        
        return adj
