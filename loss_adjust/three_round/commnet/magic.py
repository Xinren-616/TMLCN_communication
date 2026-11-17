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
        Initialization method for the MAGIC communication protocol (2 rounds of communication)

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
            },
            'round2': {
                'total_elements': 0,         # 传输的消息数(元素数)
                'active_channels': 0,        # 活跃通信通道数（边数）
                'entropy': 0.0,              # 通信信息熵
                'max_entropy': 0.0,          # 最大熵
                'effective_bandwidth': 0.0,  # 有效带宽利用率
                'message_similarity': 0.0,   # 消息相似度
            },
            'total_elements': 0,             # 总传输消息数
            'total_entropy': 0.0,            # 总信息熵
            'total_max_entropy': 0.0,        # 总最大熵
            'total_effective_bandwidth': 0.0, # 总有效带宽利用率
            'total_message_similarity': 0.0,  # 总消息相似度
        }

        dropout = 0
        negative_slope = 0.2

        # initialize sub-processors
        self.sub_processor1 = GraphAttention(args.hid_size, args.gat_hid_size, dropout=dropout, negative_slope=negative_slope, num_heads=args.gat_num_heads, self_loop_type=args.self_loop_type1, average=False, normalize=args.first_gat_normalize)
        self.sub_processor2 = GraphAttention(args.gat_hid_size*args.gat_num_heads, args.hid_size, dropout=dropout, negative_slope=negative_slope, num_heads=args.gat_num_heads_out, self_loop_type=args.self_loop_type2, average=True, normalize=args.second_gat_normalize)
        # initialize the gat encoder for the Scheduler
        if args.use_gat_encoder:
            self.gat_encoder = GraphAttention(args.hid_size, args.gat_encoder_out_size, dropout=dropout, negative_slope=negative_slope, num_heads=args.ge_num_heads, self_loop_type=1, average=True, normalize=args.gat_encoder_normalize)

        self.obs_encoder = nn.Linear(args.obs_size, args.hid_size)

        self.init_hidden(args.batch_size)
        self.lstm_cell= nn.LSTMCell(args.hid_size, args.hid_size)

        # initialize mlp layers for the sub-schedulers
        if not args.first_graph_complete:
            if args.use_gat_encoder:
                self.sub_scheduler_mlp1 = nn.Sequential(
                    nn.Linear(args.gat_encoder_out_size*2, args.gat_encoder_out_size//2),
                    nn.ReLU(),
                    nn.Linear(args.gat_encoder_out_size//2, args.gat_encoder_out_size//2),
                    nn.ReLU(),
                    nn.Linear(args.gat_encoder_out_size//2, 2))
            else:
                self.sub_scheduler_mlp1 = nn.Sequential(
                    nn.Linear(self.hid_size*2, self.hid_size//2),
                    nn.ReLU(),
                    nn.Linear(self.hid_size//2, self.hid_size//8),
                    nn.ReLU(),
                    nn.Linear(self.hid_size//8, 2))
                
        if args.learn_second_graph and not args.second_graph_complete:
            if args.use_gat_encoder:
                self.sub_scheduler_mlp2 = nn.Sequential(
                    nn.Linear(args.gat_encoder_out_size*2, args.gat_encoder_out_size//2),
                    nn.ReLU(),
                    nn.Linear(args.gat_encoder_out_size//2, args.gat_encoder_out_size//2),
                    nn.ReLU(),
                    nn.Linear(args.gat_encoder_out_size//2, 2))
            else:
                self.sub_scheduler_mlp2 = nn.Sequential(
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
                self.sub_scheduler_mlp1.apply(self.init_linear)
            if args.learn_second_graph and not args.second_graph_complete:
                self.sub_scheduler_mlp2.apply(self.init_linear)
                   
        # initialize the action head (in practice, one action head is used)
        self.action_heads = nn.ModuleList([nn.Linear(2*args.hid_size, o)
                                        for o in args.naction_heads])
        # initialize the value head
        self.value_head = nn.Linear(2 * self.hid_size, 1)


    def forward(self, x, info={}):
        """
        Forward function of MAGIC (two rounds of communication)

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

        # comm: [n * hid_size]
        comm = hidden_state
        if self.args.message_encoder:
            comm = self.message_encoder(comm)
            
        # mask communcation from dead agents (only effective in Traffic Junction)
        comm = comm * agent_mask
        comm_ori = comm.clone()

        # sub-scheduler 1
        # if args.first_graph_complete == True, sub-scheduler 1 will be disabled
        if not self.args.first_graph_complete:
            if self.args.use_gat_encoder:
                adj_complete = self.get_complete_graph(agent_mask)
                encoded_state1 = self.gat_encoder(comm, adj_complete)
                adj1 = self.sub_scheduler(self.sub_scheduler_mlp1, encoded_state1, agent_mask, self.args.directed)
            else:
                adj1 = self.sub_scheduler(self.sub_scheduler_mlp1, comm, agent_mask, self.args.directed)
        else:
            adj1 = self.get_complete_graph(agent_mask)

        # sub-processor 1
        comm_round1 = F.elu(self.sub_processor1(comm, adj1))
        print('comm_round1:', comm_round1)
        print('adj1:', adj1)
        # 计算第一轮通信的信息量指标
        with torch.no_grad():
            self._calculate_comm_stats(comm_round1, adj1, 'round1', n)

        # sub-scheduler 2
        if self.args.learn_second_graph and not self.args.second_graph_complete:
            if self.args.use_gat_encoder:
                if self.args.first_graph_complete:
                    adj_complete = self.get_complete_graph(agent_mask)
                    encoded_state2 = self.gat_encoder(comm_ori, adj_complete)
                else:
                    encoded_state2 = encoded_state1
                adj2 = self.sub_scheduler(self.sub_scheduler_mlp2, encoded_state2, agent_mask, self.args.directed)
            else:
                adj2 = self.sub_scheduler(self.sub_scheduler_mlp2, comm_ori, agent_mask, self.args.directed)
        elif not self.args.learn_second_graph and not self.args.second_graph_complete:
            adj2 = adj1
        else:
            adj2 = self.get_complete_graph(agent_mask)
            
        # sub-processor 2
        comm = self.sub_processor2(comm_round1, adj2)
        print('comm:', comm)
        print('adj2:', adj2)
        # 计算第二轮通信的信息量指标
        with torch.no_grad():
            self._calculate_comm_stats(comm, adj2, 'round2', n)
        # mask communication to dead agents (only effective in Traffic Junction)
        # 计算总通信成本（基于边数和传输的比特数）
        self.comm_bits_stats['total_elements'] = self.comm_bits_stats['round1']['total_elements'] + self.comm_bits_stats['round2']['total_elements']
        # 总通信次数和信息熵已在_calculate_comm_stats方法的round2部分更新
        
        # 将通信统计信息添加到info字典中以供外部访问
        if isinstance(info, dict):
            info['comm_stats'] = self.comm_bits_stats

        if self.args.message_decoder:
            comm = self.message_decoder(comm)

        value_head = self.value_head(torch.cat((hidden_state, comm), dim=-1))
        h = hidden_state.view(batch_size, n, self.hid_size)
        c = comm.view(batch_size, n, self.hid_size)

        action_out = [F.log_softmax(action_head(torch.cat((h, c), dim=-1)), dim=-1) for action_head in self.action_heads]

        return action_out, value_head, (hidden_state.clone(), cell_state.clone())
    def _calculate_comm_stats(self, comm, adj, round_name, n):
        """
        计算通信信息量统计指标
        
        Arguments:
            comm (tensor): 通信消息 [n * hid_size]
            adj (tensor): 邻接矩阵 [n * n]
            round_name (str): 通信轮次名称 ('round1' 或 'round2')
            n (int): 智能体数量
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
        # 根据不同的通信轮次，计算消息大小
        if round_name == 'round2':
            # 第二轮通信：传输comm[i]的长度 (hid_size)
            msg_size = self.hid_size
        else:
            # 第一轮通信：传输comm_round1[i]的长度 (gat_hid_size * gat_num_heads)
            msg_size = self.args.gat_hid_size * self.args.gat_num_heads
                
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
        
        # 更新统计信息
        self.comm_bits_stats[round_name]['total_elements'] = total_elements
        self.comm_bits_stats[round_name]['active_channels'] = active_channels
        self.comm_bits_stats[round_name]['total_comm_num'] = active_channels
        self.comm_bits_stats[round_name]['entropy'] = entropy
        self.comm_bits_stats[round_name]['max_entropy'] = max_entropy
        self.comm_bits_stats[round_name]['effective_bandwidth'] = effective_bandwidth
        self.comm_bits_stats[round_name]['message_similarity'] = message_similarity
        
        # 在round2结束后，更新总指标
        if round_name == 'round2':
            self.comm_bits_stats['total_elements'] = self.comm_bits_stats['round1']['total_elements'] + self.comm_bits_stats['round2']['total_elements']
            self.comm_bits_stats['total_comm_num'] = self.comm_bits_stats['round1']['total_comm_num'] + self.comm_bits_stats['round2']['total_comm_num']
            self.comm_bits_stats['active_channels'] = (self.comm_bits_stats['round1']['active_channels'] + self.comm_bits_stats['round2']['active_channels']) / 2.0
            self.comm_bits_stats['total_entropy'] = (self.comm_bits_stats['round1']['entropy'] + self.comm_bits_stats['round2']['entropy']) / 2.0
            self.comm_bits_stats['total_max_entropy'] = (self.comm_bits_stats['round1']['max_entropy'] + self.comm_bits_stats['round2']['max_entropy']) / 2.0
            self.comm_bits_stats['total_effective_bandwidth'] = (self.comm_bits_stats['round1']['effective_bandwidth'] + self.comm_bits_stats['round2']['effective_bandwidth']) / 2.0
            self.comm_bits_stats['total_message_similarity'] = (self.comm_bits_stats['round1']['message_similarity'] + self.comm_bits_stats['round2']['message_similarity']) / 2.0

        #print('self.comm_bits_stats:', self.comm_bits_stats)

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
