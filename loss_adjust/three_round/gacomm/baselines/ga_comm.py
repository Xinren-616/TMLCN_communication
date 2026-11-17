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
        self.comm_passes = args.comm_passes
        self.recurrent = args.recurrent
        self.qk_hid_size = args.qk_hid_size
        
        # 添加通信信息量统计变量
        self.comm_bits_stats = {
            'round1': {
                'total_elements': 0,         # 传输的消息数(元素数)
                'active_channels': 0,        # 活跃通信通道数（边数）
                'total_comm_num': 0,         # 总通信次数
                'entropy': 0.0,              # 通信信息熵
                'max_entropy': 0.0,          # 最大熵
                'effective_bandwidth': 0.0,  # 有效带宽利用率
                'message_similarity': 0.0,   # 消息相似度
            },
            'round2': {
                'total_elements': 0,         # 传输的消息数(元素数)
                'active_channels': 0,        # 活跃通信通道数（边数）
                'total_comm_num': 0,         # 总通信次数
                'entropy': 0.0,              # 通信信息熵
                'max_entropy': 0.0,          # 最大熵
                'effective_bandwidth': 0.0,  # 有效带宽利用率
                'message_similarity': 0.0,   # 消息相似度
            },
            'round3': {
                'total_elements': 0,         # 传输的消息数(元素数)
                'active_channels': 0,        # 活跃通信通道数（边数）
                'total_comm_num': 0,         # 总通信次数
                'entropy': 0.0,              # 通信信息熵
                'max_entropy': 0.0,          # 最大熵
                'effective_bandwidth': 0.0,  # 有效带宽利用率
                'message_similarity': 0.0,   # 消息相似度
            },
            'total_elements': 0,             # 总传输消息数
            'total_comm_num': 0,             # 总通信次数
            'active_channels': 0,            # 平均活跃通道数
            'total_entropy': 0.0,            # 总信息熵
            'total_max_entropy': 0.0,        # 总最大熵
            'total_effective_bandwidth': 0.0, # 总有效带宽利用率
            'total_message_similarity': 0.0,  # 总消息相似度
        }

        # 添加是否使用三轮通信的参数（use_two_rounds 保留向下兼容）
        self.use_two_rounds = args.use_two_rounds if hasattr(args, 'use_two_rounds') else False
        self.use_three_rounds = args.use_three_rounds if hasattr(args, 'use_three_rounds') else False
        # 开启三轮必须同时开启两轮
        if self.use_three_rounds:
            self.use_two_rounds = True
        
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

        self.value_head = nn.Linear(2*self.hid_size, 1)

        # hard attention layers to form the graph 
        self.lstm = nn.LSTM(args.hid_size * 2, args.hid_size * 2, bidirectional=True)
        self.linear = nn.Linear(args.hid_size * 4, 2) # *4: after bidirectional output
        # soft attention layers 
        self.wq = nn.Linear(args.hid_size, args.qk_hid_size)
        self.wk = nn.Linear(args.hid_size, args.qk_hid_size)

        # 添加第二轮通信的组件
        if self.use_two_rounds:
            if args.share_weights:
                self.wq2 = self.wq
                self.wk2 = self.wk
                self.lstm2 = self.lstm
                self.linear2 = self.linear
            else:
                self.wq2 = nn.Linear(args.hid_size, args.qk_hid_size)
                self.wk2 = nn.Linear(args.hid_size, args.qk_hid_size)
                self.lstm2 = nn.LSTM(args.hid_size * 2, args.hid_size * 2, bidirectional=True)
                self.linear2 = nn.Linear(args.hid_size * 4, 2)

        # 添加第三轮通信的组件
        if self.use_three_rounds:
            if args.share_weights:
                self.wq3 = self.wq
                self.wk3 = self.wk
                self.lstm3 = self.lstm
                self.linear3 = self.linear
            else:
                self.wq3 = nn.Linear(args.hid_size, args.qk_hid_size)
                self.wk3 = nn.Linear(args.hid_size, args.qk_hid_size)
                self.lstm3 = nn.LSTM(args.hid_size * 2, args.hid_size * 2, bidirectional=True)
                self.linear3 = nn.Linear(args.hid_size * 4, 2)

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

        # # Hard Attention - action whether an agent communicates or not
        # if self.args.hard_attn:
        #     comm_action = torch.tensor(info['comm_action'])
        #     comm_action_mask = comm_action.expand(batch_size, n).unsqueeze(-1)
        #     # action 1 is talk, 0 is silent i.e. act as dead for comm purposes. ???
        #     agent_mask *= comm_action_mask.double()            

        if self.args.recurrent:
            inp = x 

            inp = inp.view(batch_size * n, self.hid_size)

            output = self.f_module(inp, (hidden_state, cell_state))

            hidden_state = output[0]
            cell_state = output[1]
        else: # MLP|RNN
            # Get next hidden state from f node
            # and Add skip connection from start and sum them
            # bugs to be fixed 
            hidden_state = sum([x, self.f_modules[i](hidden_state), c])
            hidden_state = self.tanh(hidden_state)

        if not self.args.comm_mask_zero:
            if not self.args.comm_action_one:
                # TO DO: should do batch-wise
                h0 = torch.zeros(2, batch_size * n, self.hid_size * 2, requires_grad=True)
                c0 = torch.zeros(2, batch_size * n, self.hid_size * 2, requires_grad=True)

                # according to the paper, there is no self-attend 
                list1 = []
                for a in range(batch_size * n):
                    list2 = [torch.cat([hidden_state[a], hidden_state[b]]) for b in range(batch_size * n) if b != a]
                    # for b in range(batch_size * n):
                    #     if a != b:
                    #         list2.append(torch.cat([hidden_state[a], hidden_state[b]]))
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
            # 聚合第一轮通信信息
            comm_round1 = torch.stack([(attn[l].reshape(batch_size*n-1,1)*torch.cat([comm[:l], comm[l+1:batch_size*n]], dim=0)).sum(dim=0) for l in range(batch_size*n)], dim=0)
            comm_round1 = comm_round1.view(batch_size, n, self.hid_size)
            comm_round1 = comm_round1 * agent_mask
            
            # 修正：更新第一轮通信的指标计算，使用聚合后的comm_round1
            self._calculate_comm_stats(comm_round1.view(batch_size*n, self.hid_size), hard_attn_output, 'round1', n)

            # 是否进行第二轮通信
            if self.use_two_rounds:
                # 更新隐藏状态，结合第一轮通信的结果
                hidden_state_round2 = hidden_state
                if not self.args.recurrent:
                    hidden_state_round2 = self.f_module(hidden_state_round2) + comm_round1.view(batch_size*n, self.hid_size)
                    hidden_state_round2 = self.tanh(hidden_state_round2)
                else:
                    hidden_state_round2 = hidden_state + comm_round1.view(batch_size*n, self.hid_size)
                
                # 第二轮通信的硬注意力
                if not self.args.comm_action_one:
                    h0 = torch.zeros(2, batch_size * n, self.hid_size * 2, requires_grad=True)
                    c0 = torch.zeros(2, batch_size * n, self.hid_size * 2, requires_grad=True)
                    
                    list1 = []
                    for a in range(batch_size * n):
                        list2 = [torch.cat([hidden_state_round2[a], hidden_state_round2[b]]) for b in range(batch_size * n) if b != a]
                        list1.append(torch.stack(list2))
                    
                    hard_attn_input = torch.stack(list1, dim=1)
                    hard_attn_output2 = self.lstm2(hard_attn_input, (h0, c0))[0]
                    hard_attn_output2 = self.linear2(hard_attn_output2)
                    hard_attn_output2 = F.gumbel_softmax(hard_attn_output2, hard=True)
                    hard_attn_output2 = torch.narrow(hard_attn_output2, 2, 1, 1)
                    hard_attn_output2 = hard_attn_output2.permute(1, 0, 2).squeeze()
                else:
                    hard_attn_output2 = self.get_hard_attn_one(agent_mask)
                
                # 第二轮通信的软注意力
                q2 = self.wq2(hidden_state_round2)
                k2 = self.wk2(hidden_state_round2)
                soft_attn2 = torch.matmul(q2, k2.transpose(0, 1)) / np.sqrt(self.qk_hid_size)
                soft_attn2 = torch.stack([torch.cat([soft_attn2[l][:l], soft_attn2[l][l+1:batch_size*n]]) for l in range(batch_size*n)])
                soft_attn2 = soft_attn2 * hard_attn_output2
                soft_attn2 = F.softmax(soft_attn2, dim=1)
                attn2 = soft_attn2 * hard_attn_output2
                
                # 聚合第二轮通信信息
                comm2 = hidden_state_round2.view(batch_size, n, self.hid_size) if self.args.recurrent else hidden_state_round2
                comm2 = comm2 * agent_mask
                comm2 = comm2.view(batch_size * n, self.hid_size)
                comm_round2 = torch.stack([(attn2[l].reshape(batch_size*n-1,1)*torch.cat([comm2[:l], comm2[l+1:batch_size*n]], dim=0)).sum(dim=0) for l in range(batch_size*n)], dim=0)
                comm_round2 = comm_round2.view(batch_size, n, self.hid_size)
                comm_round2 = comm_round2 * agent_mask

                self._calculate_comm_stats(comm_round2.view(batch_size*n, self.hid_size), hard_attn_output2, 'round2', n)

                # 是否进行第三轮通信
                if self.use_three_rounds:
                    # 更新隐藏状态，结合第二轮通信的结果
                    hidden_state_round3 = hidden_state_round2 + comm_round2.view(batch_size*n, self.hid_size)

                    # 第三轮通信的硬注意力
                    if not self.args.comm_action_one:
                        h0 = torch.zeros(2, batch_size * n, self.hid_size * 2, requires_grad=True)
                        c0 = torch.zeros(2, batch_size * n, self.hid_size * 2, requires_grad=True)

                        list1 = []
                        for a in range(batch_size * n):
                            list2 = [torch.cat([hidden_state_round3[a], hidden_state_round3[b]]) for b in range(batch_size * n) if b != a]
                            list1.append(torch.stack(list2))

                        hard_attn_input = torch.stack(list1, dim=1)
                        hard_attn_output3 = self.lstm3(hard_attn_input, (h0, c0))[0]
                        hard_attn_output3 = self.linear3(hard_attn_output3)
                        hard_attn_output3 = F.gumbel_softmax(hard_attn_output3, hard=True)
                        hard_attn_output3 = torch.narrow(hard_attn_output3, 2, 1, 1)
                        hard_attn_output3 = hard_attn_output3.permute(1, 0, 2).squeeze()
                    else:
                        hard_attn_output3 = self.get_hard_attn_one(agent_mask)

                    # 第三轮通信的软注意力
                    q3 = self.wq3(hidden_state_round3)
                    k3 = self.wk3(hidden_state_round3)
                    soft_attn3 = torch.matmul(q3, k3.transpose(0, 1)) / np.sqrt(self.qk_hid_size)
                    soft_attn3 = torch.stack([torch.cat([soft_attn3[l][:l], soft_attn3[l][l+1:batch_size*n]]) for l in range(batch_size*n)])
                    soft_attn3 = soft_attn3 * hard_attn_output3
                    soft_attn3 = F.softmax(soft_attn3, dim=1)
                    attn3 = soft_attn3 * hard_attn_output3

                    # 聚合第三轮通信信息
                    comm3 = hidden_state_round3.view(batch_size, n, self.hid_size) if self.args.recurrent else hidden_state_round3
                    comm3 = comm3 * agent_mask
                    comm3 = comm3.view(batch_size * n, self.hid_size)
                    comm = torch.stack([(attn3[l].reshape(batch_size*n-1,1)*torch.cat([comm3[:l], comm3[l+1:batch_size*n]], dim=0)).sum(dim=0) for l in range(batch_size*n)], dim=0)
                    comm = comm.view(batch_size, n, self.hid_size)
                    comm = comm * agent_mask

                    self._calculate_comm_stats(comm.view(batch_size*n, self.hid_size), hard_attn_output3, 'round3', n)
                    # _calculate_comm_stats 在 round3 时会自动调用 _update_total_comm_stats
                else:
                    # 不使用三轮，第二轮结果作为最终结果
                    comm = comm_round2
                    # 复制第二轮指标到第三轮，再汇总
                    for key in self.comm_bits_stats['round2']:
                        self.comm_bits_stats['round3'][key] = self.comm_bits_stats['round2'][key]
                    self._update_total_comm_stats()
            else:
                # 如果不使用两轮通信，第一轮的结果就是最终结果
                comm = comm_round1
                # 复制第一轮指标到第二、三轮，再汇总
                for key in self.comm_bits_stats['round1']:
                    self.comm_bits_stats['round2'][key] = self.comm_bits_stats['round1'][key]
                    self.comm_bits_stats['round3'][key] = self.comm_bits_stats['round1'][key]
                self._update_total_comm_stats()
        else:
            comm = torch.zeros(batch_size, n, self.hid_size) if self.args.recurrent else torch.zeros(hidden_state.size())
            # 重置所有通信指标为0
            self._reset_comm_stats()
            
            comm_density1 = 0
            comm_density2 = 0
            
        # 删除原来的print语句
        # print('comm:', comm)
        # print('comm_density1:', comm_density1)
        # print('comm_density2:', comm_density2)
        
        # 根据通信轮次，选择最终隐藏状态计算价值函数
        if self.use_three_rounds and not self.args.comm_mask_zero:
            final_hidden = hidden_state_round3
        elif self.use_two_rounds and not self.args.comm_mask_zero:
            final_hidden = hidden_state_round2
        else:
            final_hidden = hidden_state

        value_head = self.value_head(torch.cat((final_hidden, comm.view(batch_size*n, self.hid_size)), dim=-1))
        h = final_hidden.view(batch_size, n, self.hid_size)
        
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
            
        # 更新返回值中的通信密度统计
        comm_density1 = self.comm_bits_stats['round1']['active_channels'] / (n*(n-1)) if n > 1 else 0
        comm_density2 = self.comm_bits_stats['round2']['active_channels'] / (n*(n-1)) if n > 1 else 0
        comm_density3 = self.comm_bits_stats['round3']['active_channels'] / (n*(n-1)) if n > 1 else 0

        if self.args.recurrent:
            return action, value_head, (final_hidden.clone(), cell_state.clone()), [comm_density1, comm_density2, comm_density3]
        else:
            return action, value_head, [comm_density1, comm_density2, comm_density3]

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
            round_name (str): 通信轮次名称 ('round1' 或 'round2')
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
        
        # 在 round3 结束后，更新总指标
        if round_name == 'round3':
            self._update_total_comm_stats()
    
    def _update_total_comm_stats(self):
        """更新总体通信统计指标（三轮）"""
        # 累加型指标：直接累加
        self.comm_bits_stats['total_elements'] = (self.comm_bits_stats['round1']['total_elements']
                                                  + self.comm_bits_stats['round2']['total_elements']
                                                  + self.comm_bits_stats['round3']['total_elements'])
        self.comm_bits_stats['total_comm_num'] = (self.comm_bits_stats['round1']['total_comm_num']
                                                  + self.comm_bits_stats['round2']['total_comm_num']
                                                  + self.comm_bits_stats['round3']['total_comm_num'])
        # 平均型指标：对三轮求平均
        self.comm_bits_stats['active_channels'] = (self.comm_bits_stats['round1']['active_channels']
                                                   + self.comm_bits_stats['round2']['active_channels']
                                                   + self.comm_bits_stats['round3']['active_channels']) / 3.0
        self.comm_bits_stats['total_entropy'] = (self.comm_bits_stats['round1']['entropy']
                                                 + self.comm_bits_stats['round2']['entropy']
                                                 + self.comm_bits_stats['round3']['entropy']) / 3.0
        self.comm_bits_stats['total_max_entropy'] = (self.comm_bits_stats['round1']['max_entropy']
                                                     + self.comm_bits_stats['round2']['max_entropy']
                                                     + self.comm_bits_stats['round3']['max_entropy']) / 3.0
        self.comm_bits_stats['total_effective_bandwidth'] = (self.comm_bits_stats['round1']['effective_bandwidth']
                                                              + self.comm_bits_stats['round2']['effective_bandwidth']
                                                              + self.comm_bits_stats['round3']['effective_bandwidth']) / 3.0
        self.comm_bits_stats['total_message_similarity'] = (self.comm_bits_stats['round1']['message_similarity']
                                                             + self.comm_bits_stats['round2']['message_similarity']
                                                             + self.comm_bits_stats['round3']['message_similarity']) / 3.0
    
    def _reset_comm_stats(self):
        """重置所有通信统计指标为0"""
        for round_key in ['round1', 'round2', 'round3']:
            for metric_key in self.comm_bits_stats[round_key]:
                self.comm_bits_stats[round_key][metric_key] = 0.0
        
        for key in ['total_elements', 'total_comm_num', 'active_channels', 'total_entropy', 'total_max_entropy', 'total_effective_bandwidth', 'total_message_similarity']:
            self.comm_bits_stats[key] = 0.0
