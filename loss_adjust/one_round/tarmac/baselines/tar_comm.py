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
        self.comm_stats = {
            'round1': {
                'total_elements': 0,         # 传输的消息数(元素数)
                'active_channels': 0,        # 活跃通信通道数（边数）
                'total_comm_num': 0,         # 总通信次数
                'entropy': 0.0,              # 通信信息熵
                'max_entropy': 0.0,          # 最大熵
                'effective_bandwidth': 0.0,  # 有效带宽利用率
                'message_similarity': 0.0,   # 消息相似度
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

        if self.args.recurrent:
            x, extras = x
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
        
        # 计算通信统计指标 - 只保留一轮
        with torch.no_grad():
            self._calculate_comm_stats(comm, attn, n, 'round1')
        #print('self.comm_stats:', self.comm_stats)
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
        
        # 将通信统计信息添加到info字典中以供外部访问
        if isinstance(info, dict):
            info['comm_stats'] = self.comm_stats

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
            return action, value_head

    def init_weights(self, m):
        if type(m) == nn.Linear:
            m.weight.data.normal_(0, self.init_std)

    def init_hidden(self, batch_size):
        # dim 0 = num of layers * num of direction
        return tuple(( torch.zeros(batch_size * self.nagents, self.hid_size, requires_grad=True),
                       torch.zeros(batch_size * self.nagents, self.hid_size, requires_grad=True)))

    def _calculate_comm_stats(self, comm, attn, n, round_name):
        """
        计算通信信息量统计指标
        
        Arguments:
            comm (tensor): 通信消息 [batch_size * n * hid_size]
            attn (tensor): 注意力矩阵 [batch_size * n * n]
            n (int): 智能体数量
            round_name (str): 通信轮次名称 ('round1')
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
        
        # 更新特定轮次的统计信息
        self.comm_stats[round_name]['total_elements'] = total_elements
        self.comm_stats[round_name]['active_channels'] = active_channels
        self.comm_stats[round_name]['total_comm_num'] = active_channels
        self.comm_stats[round_name]['entropy'] = entropy
        self.comm_stats[round_name]['max_entropy'] = max_entropy
        self.comm_stats[round_name]['effective_bandwidth'] = effective_bandwidth
        self.comm_stats[round_name]['message_similarity'] = message_similarity

