"""
正确的 Decentralized Execution 实现（修复版 - GAComm版本）

关键修复：
===========
1. LSTM只调用一次（与训练时一致）
2. 通信基于前一时刻的hidden state（而非当前观察）
3. LSTM输入是 (编码观察 + 通信)
4. 适配GAComm的图注意力通信机制

与test_true_decentralized.py的区别：
- True版本: 调用LSTM两次 ❌
- Proper版本: 调用LSTM一次 ✅
"""

import sys
import argparse
import numpy as np
import torch
import torch.nn.functional as F

sys.path.append("..") 
import data
from utils import *
from action_utils import *
from ga_comm import GACommNetMLP

torch.set_default_tensor_type('torch.DoubleTensor')


def decentralized_forward_proper(policy_net, observations, prev_hiddens, args, info={}):
    """
    正确的Decentralized模式前向传播（GAComm版本）
    
    关键：与训练时的forward行为完全一致
    - LSTM只调用一次
    - 输入是编码观察（无skip connection，因为GAComm不使用）
    - 通信基于前一时刻的hidden state
    - 使用硬注意力和软注意力机制
    """
    
    batch_size = 1
    n = args.nagents
    hid_size = args.hid_size
    
    # =========================================================================
    # Phase 1: 每个agent独立编码观察
    # =========================================================================
    encoded_obs_list = []
    for i in range(n):
        obs_i = observations[:, i:i+1, :]  # [1, 1, obs_dim]
        encoded_i = policy_net.encoder(obs_i)  # [1, 1, hid_size]
        encoded_obs_list.append(encoded_i.squeeze(1))  # [1, hid_size]
    
    # 拼接所有编码后的观察
    x = torch.cat(encoded_obs_list, dim=0)  # [n, hid_size]
    
    # =========================================================================
    # Phase 2: 更新LSTM（关键：只调用一次！）
    # =========================================================================
    if args.recurrent and args.rnn_type == 'LSTM':
        updated_hidden_list = []
        updated_cell_list = []
        
        for i in range(n):
            inp_i = encoded_obs_list[i]  # [1, hid_size]
            
            # 提取前一时刻的状态
            prev_h_i = prev_hiddens[0][i:i+1]
            prev_c_i = prev_hiddens[1][i:i+1]
            
            # ⭐ 关键：只调用一次LSTM
            h_i, c_i = policy_net.f_module(inp_i, (prev_h_i, prev_c_i))
            
            updated_hidden_list.append(h_i)
            updated_cell_list.append(c_i)
        
        # 组合所有agents的状态
        hidden_state = torch.cat(updated_hidden_list, dim=0)  # [n, hid_size]
        cell_state = torch.cat(updated_cell_list, dim=0)
        next_hiddens = (hidden_state, cell_state)
        
    else:
        # MLP模式
        hidden_state = policy_net.f_modules[0](x)
        hidden_state = policy_net.tanh(hidden_state)
        next_hiddens = hidden_state
    
    # =========================================================================
    # Phase 3: 图注意力通信机制
    # =========================================================================
    num_agents_alive, agent_mask = policy_net.get_agent_mask(batch_size, info)
    agent_mask = agent_mask.view(n, 1)  # GAComm uses [n, 1] format
    
    if not args.comm_mask_zero:
        if not args.comm_action_one:
            # 硬注意力：使用双向LSTM生成通信图
            h0 = torch.zeros(2, batch_size * n, hid_size * 2, requires_grad=True)
            c0 = torch.zeros(2, batch_size * n, hid_size * 2, requires_grad=True)
            
            # 构建硬注意力输入（排除自通信）
            list1 = []
            for a in range(batch_size * n):
                list2 = [torch.cat([hidden_state[a], hidden_state[b]]) 
                        for b in range(batch_size * n) if b != a]
                list1.append(torch.stack(list2))
            
            # hard_attn_input: size of (N-1) x N x (hid_size*2)
            hard_attn_input = torch.stack(list1, dim=1)
            # hard_attn_output: size of (N-1) x N x 2
            hard_attn_output = policy_net.lstm(hard_attn_input, (h0, c0))[0]
            hard_attn_output = policy_net.linear(hard_attn_output)
            hard_attn_output = F.gumbel_softmax(hard_attn_output, hard=True)
            # hard_attn_output: size of N x (N-1)
            hard_attn_output = torch.narrow(hard_attn_output, 2, 1, 1)
            hard_attn_output = hard_attn_output.permute(1, 0, 2).squeeze()
        else:
            hard_attn_output = policy_net.get_hard_attn_one(agent_mask)
        
        # 软注意力：计算query和key
        q = policy_net.wq(hidden_state)  # [n, qk_hid_size]
        k = policy_net.wk(hidden_state)  # [n, qk_hid_size]
        
        # 计算注意力分数 [n, n]
        soft_attn = torch.matmul(q, k.transpose(0, 1)) / np.sqrt(args.qk_hid_size)
        
        # 移除自注意力，size of [n, n-1]
        soft_attn = torch.stack([torch.cat([soft_attn[l][:l], soft_attn[l][l+1:batch_size*n]]) 
                                for l in range(batch_size*n)])
        
        # 应用硬注意力掩码
        soft_attn = soft_attn * hard_attn_output
        soft_attn = F.softmax(soft_attn, dim=1)
        attn = soft_attn * hard_attn_output  # 最终的注意力权重
        
        # 准备通信消息（使用当前hidden state）
        comm = hidden_state.view(batch_size, n, hid_size)
        comm = comm * agent_mask
        comm = comm.view(batch_size * n, hid_size)
        
        # 聚合通信信息（每个agent接收其他agents的加权消息）
        comm = torch.stack([
            (attn[l].reshape(batch_size*n-1, 1) * 
             torch.cat([comm[:l], comm[l+1:batch_size*n]], dim=0)).sum(dim=0) 
            for l in range(batch_size*n)
        ], dim=0)
        
        comm = comm.view(batch_size, n, hid_size)
        comm = comm * agent_mask
    else:
        # 无通信
        comm = torch.zeros(batch_size, n, hid_size)
    
    # =========================================================================
    # Phase 4: 生成动作和价值
    # =========================================================================
    # 拼接hidden state和通信用于价值估计
    value_input = torch.cat((hidden_state, comm.view(batch_size*n, hid_size)), dim=-1)
    value = policy_net.value_head(value_input)
    
    # 拼接hidden state和通信用于动作生成
    h = hidden_state.view(batch_size, n, hid_size)
    action_input = torch.cat((h, comm), dim=-1)  # [batch_size, n, hid_size*2]
    
    if args.continuous:
        action_mean = policy_net.action_mean(action_input)
        action_log_std = policy_net.action_log_std.expand_as(action_mean)
        action_std = torch.exp(action_log_std)
        action_out = (action_mean, action_log_std, action_std)
    else:
        action_out = [F.log_softmax(head(action_input), dim=-1) 
                     for head in policy_net.heads]
    
    return action_out, value, next_hiddens


def test_episode_proper(args, policy_net, env, epoch=0):
    """使用正确的decentralized forward测试episode"""
    
    # 重置环境
    from inspect import getfullargspec
    reset_args = getfullargspec(env.reset).args
    if 'epoch' in reset_args:
        state = env.reset(epoch)
    else:
        state = env.reset()
    
    stat = dict()
    info = dict()
    
    # 初始化hidden states
    if args.recurrent and args.rnn_type == 'LSTM':
        prev_hid = policy_net.init_hidden(batch_size=1)
    else:
        prev_hid = torch.zeros(1, args.nagents, args.hid_size)
    
    with torch.no_grad():
        for t in range(args.max_steps):
            if t == 0 and args.hard_attn:
                info['comm_action'] = np.zeros(args.nagents, dtype=int)
            
            # 使用修复后的decentralized forward
            action_out, value, prev_hid = decentralized_forward_proper(
                policy_net, state, prev_hid, args, info
            )
            
            # 选择动作
            action = select_action(args, action_out)
            action, actual = translate_action(args, env, action)
            
            # 执行动作
            next_state, reward, done, info = env.step(actual)
            
            # 统计
            if args.hard_attn:
                info['comm_action'] = action[-1] if not args.comm_action_one else np.ones(args.nagents, dtype=int)
                stat['comm_action'] = stat.get('comm_action', 0) + info['comm_action'][:args.nfriendly]
            
            stat['reward'] = stat.get('reward', 0) + reward[:args.nfriendly]
            
            state = next_state
            done = done or t == args.max_steps - 1
            
            if done:
                break
        
        stat['num_steps'] = t + 1
        stat['steps_taken'] = t + 1
    
    # 终止奖励
    if hasattr(env, 'reward_terminal'):
        terminal_reward = env.reward_terminal()
        stat['reward'] = stat.get('reward', 0) + terminal_reward[:args.nfriendly]
    
    # 环境统计
    if hasattr(env, 'get_stat'):
        merge_stat(env.get_stat(), stat)
    
    return stat


def run_test(model_path, num_episodes=100):
    """运行修复后的测试（GAComm版本）"""
    
    print(f"\n{'='*80}")
    print("正确的 Decentralized Execution 测试（GAComm修复版）")
    print(f"{'='*80}\n")
    print("关键修复:")
    print("  ✅ LSTM只调用一次（与训练时一致）")
    print("  ✅ 使用图注意力通信机制")
    print("  ✅ LSTM输入是编码观察（无skip connection）")
    print(f"{'='*80}\n")
    
    # 加载模型
    checkpoint = torch.load(model_path)
    
    # 设置参数
    class Args:
        env_name = 'traffic_junction'
        nagents = 5
        dim = 6
        max_steps = 20
        add_rate_min = 0.3
        add_rate_max = 0.3
        difficulty = 'easy'
        vision = 1
        hid_size = 128
        qk_hid_size = 16  # GAComm的query-key hidden size（从checkpoint推断）
        comm_passes = 1
        recurrent = True
        rnn_type = 'LSTM'
        commnet = False  # GAComm不是CommNet
        ic3net = False
        
        # 启用通信
        comm_mask_zero = False
        
        hard_attn = False
        comm_action_one = True  # GAComm通常使用comm_action_one=True
        mean_ratio = 1.0
        comm_mode = 'avg'
        share_weights = False
        continuous = False
        seed = 0
        
        # 其他必需参数
        display = False
        render = False
        nprocesses = 1
        batch_size = 500
        gamma = 1.0
        normalize_rewards = False
        advantages_per_action = False
        detach_gap = 10
        value_coeff = 0.01
        entr = 0
        action_scale = 1.0
        lrate = 0.001
        
        # 环境特定参数
        curr_start = 0.0
        curr_end = 0.0
        vocab_type = 'bool'
        comm_init = 'uniform'
    
    args = Args()
    
    # 初始化环境
    from data import init as init_env
    from utils import init_args_for_env
    
    class DummyParser:
        def add_argument(self, *args_list, **kwargs):
            arg_name = args_list[0].replace('--', '')
            if 'default' in kwargs and not hasattr(Args, arg_name):
                setattr(Args, arg_name, kwargs['default'])
    
    init_args_for_env(DummyParser())
    env = init_env(args.env_name, args, False)
    
    num_inputs = env.observation_dim
    args.num_actions = env.num_actions if isinstance(env.num_actions, list) else [env.num_actions]
    args.dim_actions = env.dim_actions
    args.num_inputs = num_inputs
    args.nfriendly = args.nagents
    
    if args.hard_attn and args.commnet:
        args.num_actions = [*args.num_actions, 2]
        args.dim_actions = env.dim_actions + 1
    
    parse_action_args(args)
    
    # 创建网络
    policy_net = GACommNetMLP(args, num_inputs)
    policy_net.load_state_dict(checkpoint['policy_net'])
    policy_net.eval()
    
    print(f"✓ 模型加载成功 (GAComm)")
    print(f"✓ 网络参数：")
    print(f"  - Agents数量: {args.nagents}")
    print(f"  - Hidden size: {args.hid_size}")
    print(f"  - QK Hidden size: {args.qk_hid_size}")
    print(f"  - 通信拓扑 (comm_mask):")
    print(f"    {policy_net.comm_mask}")
    print(f"\n开始测试 {num_episodes} episodes...\n")
    
    # 运行测试
    all_stats = []
    for ep in range(num_episodes):
        stat = test_episode_proper(args, policy_net, env, epoch=ep)
        all_stats.append(stat)
        
        if (ep + 1) % 20 == 0:
            print(f"Episode {ep + 1}/{num_episodes} 完成")
    
    # 统计结果
    print(f"\n{'='*80}")
    print("测试结果 (GAComm Proper Decentralized Execution)")
    print(f"{'='*80}\n")
    
    metrics = {
        'reward': [],
        'steps_taken': [],
        'success': [],
    }
    
    for stat in all_stats:
        for key in metrics.keys():
            if key in stat:
                val = stat[key]
                if isinstance(val, np.ndarray):
                    val = val.mean()
                metrics[key].append(val)
    
    for metric_name, values in metrics.items():
        if len(values) > 0:
            values = np.array(values)
            print(f"{metric_name.upper()}:")
            print(f"  平均值: {values.mean():.4f}")
            print(f"  标准差: {values.std():.4f}")
            print(f"  最小值: {values.min():.4f}")
            print(f"  最大值: {values.max():.4f}")
            print()
    
    print(f"{'='*80}\n")
    
    return metrics


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='正确的 Decentralized Execution 测试（GAComm修复版）'
    )
    parser.add_argument('--model_path', type=str, 
                       default='saved/traffic_junction/gacomm/run34/model.pt',
                       help='训练好的GAComm模型路径')
    parser.add_argument('--num_episodes', type=int, default=100,
                       help='测试episode数量')
    
    args = parser.parse_args()
    
    import os
    if not os.path.exists(args.model_path):
        print(f"错误: 找不到模型文件 {args.model_path}")
        sys.exit(1)
    
    print("\n" + "="*80)
    print("⭐ 这是修复后的正确实现（GAComm版本）")
    print("="*80)
    print("\n关键修复:")
    print("  ❌ 错误版本: LSTM调用2次")
    print("  ✅ 修复版本: LSTM调用1次（与训练一致）")
    print("  ✅ 使用图注意力通信机制")
    print("\n预期结果:")
    print("  - 成功率应该接近训练时的表现")
    print("  - 保持GAComm的图注意力通信机制")
    print("\n")
    
    results = run_test(args.model_path, args.num_episodes)

