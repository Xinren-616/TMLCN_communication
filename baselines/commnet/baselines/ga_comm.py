"""
Revised from CommNetMLP
"""
import torch
import torch.nn.functional as F
from torch import nn
import numpy as np

from models import MLP
import sys 
sys.path.append("..") 
from action_utils import select_action, translate_action

class GACommNetMLP(nn.Module):
    def __init__(self, args, num_inputs):
        super(GACommNetMLP, self).__init__()
        self.args = args
        self.nagents = args.nagents
        self.hid_size = args.hid_size
        self.comm_passes = 1  # 修改为只通信一轮
        self.recurrent = args.recurrent
        self.qk_hid_size = args.qk_hid_size
        
        # 添加通信信息量统计变量 - 修改为只保留一轮通信的统计
        self.comm_bits_stats = {
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
            # support multi action
            self.heads = nn.ModuleList([nn.Linear(args.hid_size*2, o)
                                        for o in args.naction_heads])
        self.init_std = args.init_std if hasattr(args, 'comm_init_std') else 0.2

        # Mask for communication
        if self.args.comm_mask_zero:
            self.comm_mask = torch.zeros(self.nagents, self.nagents)
        else:
            self.comm_mask = torch.ones(self.nagents, self.nagents) \
                            - torch.eye(self.nagents, self.nagents)

        self.encoder = nn.Linear(num_inputs, args.hid_size)

        if args.recurrent:
            self.hidd_encoder = nn.Linear(args.hid_size, args.hid_size)

        if args.recurrent:
            self.init_hidden(args.batch_size)
            self.f_module = nn.LSTMCell(args.hid_size, args.hid_size)

        else:
            if args.share_weights:
                self.f_module = nn.Linear(args.hid_size, args.hid_size)
                self.f_modules = nn.ModuleList([self.f_module])
            else:
                self.f_modules = nn.ModuleList([nn.Linear(args.hid_size, args.hid_size)])
        
        # Our main function for converting current hidden state to next state
        if args.share_weights:
            self.C_module = nn.Linear(args.hid_size, args.hid_size)
            self.C_modules = nn.ModuleList([self.C_module])
        else:
            self.C_modules = nn.ModuleList([nn.Linear(args.hid_size, args.hid_size)])

        # initialise weights as 0
        if args.comm_init == 'zeros':
            self.C_modules[0].weight.data.zero_()
        self.tanh = nn.Tanh()

        self.value_head = nn.Linear(2*self.hid_size, 1)

        # hard attention layers to form the graph 
        self.lstm = nn.LSTM(args.hid_size * 2, args.hid_size * 2, bidirectional=True)
        self.linear = nn.Linear(args.hid_size * 4, 2) # *4: after bidirectional output
        # soft attention layers 
        self.wq = nn.Linear(args.hid_size, args.qk_hid_size)
        self.wk = nn.Linear(args.hid_size, args.qk_hid_size)

    def get_agent_mask(self, batch_size, info):
        n = self.nagents

        if 'alive_mask' in info:
            agent_mask = torch.from_numpy(info['alive_mask'])
            num_agents_alive = agent_mask.sum()
        else:
            agent_mask = torch.ones(n)
            num_agents_alive = n

        agent_mask = agent_mask.view(n, 1).clone()

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
        else:
            # 兼容trainer.py总是传递[state, prev_hid]的情况
            if isinstance(x, (list, tuple)):
                x = x[0]  # 取第一个元素（state）
            x = self.encoder(x)
            x = self.tanh(x)
            hidden_state = x

        return x, hidden_state, cell_state


    def forward(self, x, info={}):
        """Forward function for GACommNetMLP class, expects state, previous hidden
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

        if self.args.recurrent:
            inp = x 
            inp = inp.view(batch_size * n, self.hid_size)
            output = self.f_module(inp, (hidden_state, cell_state))
            hidden_state = output[0]
            cell_state = output[1]
        else: 
            # 在非循环模式下，这里需要处理c变量未定义的问题
            # 由于我们移除了两轮通信，这里暂时不使用c
            # 确保hidden_state的形状是[batch_size * n, hid_size]
            if len(hidden_state.shape) == 3:
                # 如果是[batch_size, n, hid_size]，需要reshape
                hidden_state = hidden_state.view(batch_size * n, self.hid_size)
            hidden_state = self.f_modules[0](hidden_state) # 这里不一样
            hidden_state = self.tanh(hidden_state)

        if not self.args.comm_mask_zero:
            if not self.args.comm_action_one:
                # TO DO: should do batch-wise
                # 使用与hidden_state相同的dtype和device
                dtype = hidden_state.dtype
                device = hidden_state.device
                h0 = torch.zeros(2, batch_size * n, self.hid_size * 2, requires_grad=True, dtype=dtype, device=device)
                c0 = torch.zeros(2, batch_size * n, self.hid_size * 2, requires_grad=True, dtype=dtype, device=device)

                # according to the paper, there is no self-attend 
                list1 = []
                for a in range(batch_size * n):
                    list2 = [torch.cat([hidden_state[a], hidden_state[b]]) for b in range(batch_size * n) if b != a]
                    list1.append(torch.stack(list2))
                # hard_attn_input: size of (N-1) x N x (hid_size*2)
                hard_attn_input = torch.stack(list1, dim=1)
                # hard_attn_output: size of (N-1) x N x 2, the third dimension is one-hot vector
                hard_attn_output = self.lstm(hard_attn_input, (h0, c0))[0]
                hard_attn_output = self.linear(hard_attn_output)
                hard_attn_output = F.gumbel_softmax(hard_attn_output, hard=True) 
                # hard_attn_output: size of (N-1) x N x 1
                hard_attn_output = torch.narrow(hard_attn_output, 2, 1, 1)
                # hard_attn_output: size of N x (N-1)
                hard_attn_output = hard_attn_output.permute(1, 0, 2).squeeze()
            else:
                hard_attn_output = self.get_hard_attn_one(agent_mask)
            
            # 计算query和key用于软注意力
            q = self.wq(hidden_state)
            k = self.wk(hidden_state)
            # size of N x N
            soft_attn = torch.matmul(q, k.transpose(0, 1)) / np.sqrt(self.qk_hid_size)
            # size of N x (N-1)
            soft_attn = torch.stack([torch.cat([soft_attn[l][:l], soft_attn[l][l+1:batch_size*n]]) for l in range(batch_size*n)])
            soft_attn = soft_attn * hard_attn_output
            soft_attn = F.softmax(soft_attn, dim=1)
            attn = soft_attn * hard_attn_output

            # 选择当前状态或上一状态，取决于是否使用循环
            comm = hidden_state.view(batch_size, n, self.hid_size) if self.args.recurrent else hidden_state
            comm = comm * agent_mask
            comm = comm.view(batch_size * n, self.hid_size)
            # 聚合通信信息
            comm = torch.stack([(attn[l].reshape(batch_size*n-1,1)*torch.cat([comm[:l], comm[l+1:batch_size*n]], dim=0)).sum(dim=0) for l in range(batch_size*n)], dim=0)
            comm = comm.view(batch_size, n, self.hid_size)
            comm = comm * agent_mask
            #print('comm:', comm)
            # 计算通信指标
            self._calculate_comm_stats(comm.view(batch_size*n, self.hid_size), hard_attn_output, 'round1', n)
        else:
            comm = torch.zeros(batch_size, n, self.hid_size) if self.args.recurrent else torch.zeros(hidden_state.size())
            # 重置所有通信指标为0
            self._reset_comm_stats()
        #print('comm_bits_stats:', self.comm_bits_stats)
        # 计算通信密度
        comm_density = self.comm_bits_stats['round1']['active_channels'] / (n*(n-1)) if n > 1 else 0
            
        # 使用hidden_state和comm计算价值
        value_head = self.value_head(torch.cat((hidden_state, comm.view(batch_size*n, self.hid_size)), dim=-1))
        h = hidden_state.view(batch_size, n, self.hid_size)
        
        if self.continuous:
            action_mean = self.action_mean(h)
            action_log_std = self.action_log_std.expand_as(action_mean)
            action_std = torch.exp(action_log_std)
            # will be used later to sample
            action = (action_mean, action_log_std, action_std)
        else:
            # discrete actions
            action = [F.log_softmax(head(torch.cat((h, comm), dim=-1)), dim=-1) for head in self.heads]

        # 将通信统计信息添加到info字典
        if isinstance(info, dict):
            info['comm_stats'] = self.comm_bits_stats
            
        if self.args.recurrent:
            return action, value_head, (hidden_state.clone(), cell_state.clone()), comm_density
        else:
            # 兼容trainer.py总是期望4个返回值的情况（action, value, prev_hid, comm_density）
            return action, value_head, None, comm_density

    def init_weights(self, m):
        if type(m) == nn.Linear:
            m.weight.data.normal_(0, self.init_std)

    def init_hidden(self, batch_size):
        # dim 0 = num of layers * num of direction
        return tuple(( torch.zeros(batch_size * self.nagents, self.hid_size, requires_grad=True),
                       torch.zeros(batch_size * self.nagents, self.hid_size, requires_grad=True)))
    
    def get_hard_attn_one(self, agent_mask):
        n = self.args.nagents
        adj = torch.ones(n, n)
        agent_mask = agent_mask.expand(n, n)
        agent_mask_transpose = agent_mask.transpose(0, 1)
        adj = adj * agent_mask * agent_mask_transpose
        hard_attn = torch.stack([torch.cat([adj[x][:x], adj[x][x+1:n]]) for x in range(n)])

        return hard_attn
        
    def _calculate_comm_stats(self, comm, hard_attn_output, round_name, n):
        """
        计算通信指标
        
        Arguments:
            comm (tensor): 通信消息 [n * hid_size]
            hard_attn_output (tensor): 硬注意力输出 [n * (n-1)]
            round_name (str): 通信轮次名称 ('round1')
            n (int): 智能体数量
        """
        # 计算活跃通信通道数
        active_channels = hard_attn_output.nonzero().size(0)
        
        # 计算总元素数 = 通信频次 * 隐藏层大小
        total_elements = active_channels * self.hid_size
        
        # 计算信息熵
        entropy = 0.0
        max_entropy = 0.0
        valid_messages = []
        eps = 1e-10
        total_valid_agents = 0
        #print('hard_attn_output:', hard_attn_output)
        # 收集所有消息，用于计算相似度
        for i in range(n):
            # 获取节点i的所有接收方（排除自身）
            receivers = torch.where(hard_attn_output[i] > 0)[0]
            if len(receivers) > 0 and torch.sum(torch.abs(comm[i])) > 1e-6:
                agent_comm = comm[i]
                valid_messages.append(agent_comm.detach().clone())
                
                # 计算消息的概率分布
                agent_sum = torch.sum(torch.abs(agent_comm)) + eps
                agent_prob_dist = torch.abs(agent_comm) / agent_sum
                
                # 过滤太小的概率值
                threshold = 1e-6
                agent_valid_probs = agent_prob_dist[agent_prob_dist > threshold]
                
                if len(agent_valid_probs) > 0:
                    # 重新归一化
                    agent_valid_probs = agent_valid_probs / torch.sum(agent_valid_probs)
                    # 计算熵
                    agent_entropy = -torch.sum(agent_valid_probs * torch.log2(agent_valid_probs)).item()
                    # 最大熵
                    agent_max_entropy = np.log2(self.hid_size)
                    
                    entropy += agent_entropy
                    max_entropy += agent_max_entropy
                    total_valid_agents += 1
        
        # 计算平均熵
        if total_valid_agents > 0:
            entropy = entropy / total_valid_agents
            max_entropy = max_entropy / total_valid_agents
        else:
            entropy = 0.0
            max_entropy = 1.0  # 避免除零
        
        # 计算有效带宽利用率
        effective_bandwidth = entropy / max_entropy if max_entropy > 0 else 0.0
        
        # 计算消息相似度
        message_similarity = 0.0
        if len(valid_messages) >= 2:
            similarity_sum = 0.0
            pair_count = 0
            
            for i in range(len(valid_messages)):
                for j in range(i+1, len(valid_messages)):
                    vec1 = valid_messages[i]
                    vec2 = valid_messages[j]
                    # 计算余弦相似度
                    cos_sim = F.cosine_similarity(vec1.unsqueeze(0), vec2.unsqueeze(0)).item()
                    similarity_sum += abs(cos_sim)
                    pair_count += 1
            
            if pair_count > 0:
                message_similarity = similarity_sum / pair_count
        
        # 更新统计信息
        self.comm_bits_stats[round_name]['total_elements'] = total_elements
        self.comm_bits_stats[round_name]['active_channels'] = active_channels
        self.comm_bits_stats[round_name]['total_comm_num'] = active_channels
        self.comm_bits_stats[round_name]['entropy'] = entropy
        self.comm_bits_stats[round_name]['max_entropy'] = max_entropy
        self.comm_bits_stats[round_name]['effective_bandwidth'] = effective_bandwidth
        self.comm_bits_stats[round_name]['message_similarity'] = message_similarity
    
    def _reset_comm_stats(self):
        """重置所有通信统计指标为0"""
        for metric_key in self.comm_bits_stats['round1']:
            self.comm_bits_stats['round1'][metric_key] = 0.0
