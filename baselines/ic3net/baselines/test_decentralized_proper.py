"""
正确的 Decentralized Execution 实现（修复版）

关键修复：
===========
1. LSTM只调用一次（与训练时一致）
2. 通信基于前一时刻的hidden state（而非当前观察）
3. LSTM输入是 (编码观察 + 通信)

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
from comm import CommNetMLP

torch.set_default_tensor_type('torch.DoubleTensor')


def decentralized_forward_proper(policy_net, observations, prev_hiddens, args, info={}):
    """
    正确的Decentralized模式前向传播
    
    关键：与训练时的forward行为完全一致
    - LSTM只调用一次
    - 输入是 (编码观察 + 通信)
    - 通信基于前一时刻的hidden state
    """
    
    batch_size = 1
    n = args.nagents
    hid_size = args.hid_size
    
    # =========================================================================
    # Phase 1: 每个agent独立编码观察（不调用LSTM）
    # =========================================================================
    encoded_obs_list = []
    for i in range(n):
        obs_i = observations[:, i:i+1, :]  # [1, 1, obs_dim]
        
        # 只编码，不更新hidden state
        encoded_i = policy_net.encoder(obs_i)  # [1, 1, hid_size]
        encoded_obs_list.append(encoded_i.squeeze(1))  # [1, hid_size]
    
    # =========================================================================
    # Phase 2: 生成通信矩阵（基于前一时刻的hidden state）
    # =========================================================================
    # 关键：使用前一时刻的hidden state，而非当前观察编码后的state
    
    if args.recurrent and args.rnn_type == 'LSTM':
        # LSTM模式：使用前一时刻的hidden state
        prev_hidden_state = prev_hiddens[0]  # [nagents, hid_size]
    else:
        # MLP模式：状态就是编码后的观察
        prev_hidden_state = torch.cat(encoded_obs_list, dim=0)
    
    # 构建通信矩阵
    comm = prev_hidden_state.unsqueeze(0).unsqueeze(0)  # [1, 1, n, hid_size]
    comm = comm.expand(batch_size, n, n, hid_size)  # [1, n, n, hid_size]
    
    # 应用通信拓扑（comm_mask）
    mask = policy_net.comm_mask.view(1, n, n).unsqueeze(-1)
    mask = mask.expand_as(comm)
    
    # 禁用自通信
    self_mask = (1 - torch.eye(n, n)).view(1, n, n, 1)
    self_mask = self_mask.expand_as(comm)
    
    comm = comm * mask * self_mask
    
    # 处理alive_mask（如果有agent死亡）
    num_agents_alive, agent_mask = policy_net.get_agent_mask(batch_size, info)
    if 'alive_mask' in info:
        # agent_mask已经是 [batch_size, n, n, 1]，直接expand
        agent_mask_expanded = agent_mask.expand_as(comm)
        comm = comm * agent_mask_expanded
        agent_mask_transpose = agent_mask.transpose(1, 2).expand_as(comm)
        comm = comm * agent_mask_transpose
    
    # 归一化（平均模式）
    if args.comm_mode == 'avg' and num_agents_alive > 1:
        # 计算每个agent实际接收的消息数
        active_comm = (comm.abs().sum(dim=-1) > 0).float()  # [1, n, n]
        num_received = active_comm.sum(dim=2, keepdim=True).unsqueeze(-1)  # [1, n, 1, 1]
        num_received = torch.clamp(num_received, min=1.0)
        comm = comm / num_received
    
    # =========================================================================
    # Phase 3: 为每个agent聚合通信
    # =========================================================================
    comm_agg_list = []
    for i in range(n):
        # Agent i接收的通信
        comm_received = comm[:, i, :, :]  # [1, n, hid_size]
        
        # 聚合（求和）
        comm_sum_i = comm_received.sum(dim=1)  # [1, hid_size]
        
        # 使用C_module处理
        c_i = policy_net.C_modules[0](comm_sum_i)
        comm_agg_list.append(c_i)
    
    # =========================================================================
    # Phase 4: 更新LSTM（关键：只调用一次！）
    # =========================================================================
    if args.recurrent and args.rnn_type == 'LSTM':
        updated_hidden_list = []
        updated_cell_list = []
        
        for i in range(n):
            # Skip connection: 编码观察 + 通信
            inp_i = encoded_obs_list[i] + comm_agg_list[i]
            
            # 提取前一时刻的状态
            prev_h_i = prev_hiddens[0][i:i+1]
            prev_c_i = prev_hiddens[1][i:i+1]
            
            # ⭐ 关键：只调用一次LSTM
            h_i, c_i = policy_net.f_module(inp_i, (prev_h_i, prev_c_i))
            
            updated_hidden_list.append(h_i)
            updated_cell_list.append(c_i)
        
        # 组合所有agents的状态
        final_hidden = torch.cat(updated_hidden_list, dim=0)  # [n, hid_size]
        final_cell = torch.cat(updated_cell_list, dim=0)
        next_hiddens = (final_hidden, final_cell)
        
    else:
        # MLP模式
        updated_hidden_list = []
        for i in range(n):
            h_i = sum([
                encoded_obs_list[i],
                policy_net.f_modules[0](prev_hidden_state[i:i+1]),
                comm_agg_list[i]
            ])
            h_i = policy_net.tanh(h_i)
            updated_hidden_list.append(h_i)
        
        final_hidden = torch.cat(updated_hidden_list, dim=0)
        next_hiddens = final_hidden
    
    # =========================================================================
    # Phase 5: 生成动作
    # =========================================================================
    final_hidden_reshaped = final_hidden.unsqueeze(0)  # [1, n, hid_size]
    
    # 价值估计
    value = policy_net.value_head(final_hidden_reshaped)
    
    # 动作生成
    if args.continuous:
        action_mean = policy_net.action_mean(final_hidden_reshaped)
        action_log_std = policy_net.action_log_std.expand_as(action_mean)
        action_std = torch.exp(action_log_std)
        action_out = (action_mean, action_log_std, action_std)
    else:
        action_out = [F.log_softmax(head(final_hidden_reshaped), dim=-1) 
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
    """运行修复后的测试"""
    
    print(f"\n{'='*80}")
    print("正确的 Decentralized Execution 测试（修复版）")
    print(f"{'='*80}\n")
    print("关键修复:")
    print("  ✅ LSTM只调用一次（与训练时一致）")
    print("  ✅ 通信基于前一时刻的hidden state")
    print("  ✅ LSTM输入是 (编码观察 + 通信)")
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
        comm_passes = 1
        recurrent = True
        rnn_type = 'LSTM'
        commnet = True
        ic3net = False
        
        # 启用通信
        comm_mask_zero = False
        
        hard_attn = False
        comm_action_one = False
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
    policy_net = CommNetMLP(args, num_inputs)
    
    # 处理LoRA模型加载问题
    # checkpoint可能包含LoRA参数，需要转换为标准参数
    state_dict = checkpoint['policy_net']
    
    # 创建新的state_dict，移除LoRA相关的键并转换参数
    new_state_dict = {}
    for key, value in state_dict.items():
        # 跳过LoRA相关的参数
        if 'lora' in key.lower():
            print(f"  跳过LoRA参数: {key}")
            continue
        
        # 如果是C_modules的lora包装层，需要特殊处理
        # 例如: C_modules.0.lora.weight 应该被忽略，因为我们需要的是 C_modules.0.weight
        new_state_dict[key] = value
    
    # 尝试加载，使用strict=False允许部分匹配
    missing_keys, unexpected_keys = policy_net.load_state_dict(new_state_dict, strict=False)
    
    if missing_keys:
        print(f"\n⚠️  缺失的参数键:")
        for key in missing_keys:
            print(f"    - {key}")
        print("\n这可能是因为训练时使用了LoRA，而测试时使用标准模型")
        print("尝试从LoRA参数重建标准参数...\n")
        
        # 如果C_modules的参数缺失，尝试从LoRA参数重建
        if any('C_modules' in k for k in missing_keys):
            for i in range(args.comm_passes):
                lora_a_key = f'C_modules.{i}.lora.lora_A.weight'
                lora_b_key = f'C_modules.{i}.lora.lora_B.weight'
                weight_key = f'C_modules.{i}.weight'
                bias_key = f'C_modules.{i}.bias'
                
                if lora_a_key in state_dict and lora_b_key in state_dict:
                    # LoRA: W = W0 + BA
                    # 我们需要找到原始权重W0
                    # 通常LoRA会保存原始层的权重，让我们查找
                    
                    # 尝试查找原始权重
                    original_weight_key = f'C_modules.{i}.original_module.weight'
                    original_bias_key = f'C_modules.{i}.original_module.bias'
                    
                    if original_weight_key in state_dict:
                        lora_a = state_dict[lora_a_key]
                        lora_b = state_dict[lora_b_key]
                        original_weight = state_dict[original_weight_key]
                        
                        # 计算完整权重: W = W0 + BA
                        full_weight = original_weight + torch.matmul(lora_b, lora_a)
                        new_state_dict[weight_key] = full_weight
                        print(f"  ✓ 从LoRA重建: {weight_key}")
                        
                        if original_bias_key in state_dict:
                            new_state_dict[bias_key] = state_dict[original_bias_key]
                            print(f"  ✓ 复制偏置: {bias_key}")
        
        # 重新加载
        policy_net.load_state_dict(new_state_dict, strict=False)
    
    if unexpected_keys:
        print(f"\n⚠️  未预期的参数键 (已忽略):")
        for key in unexpected_keys[:10]:  # 只显示前10个
            print(f"    - {key}")
        if len(unexpected_keys) > 10:
            print(f"    ... 还有 {len(unexpected_keys) - 10} 个")
    
    policy_net.eval()
    
    print(f"✓ 模型加载成功")
    print(f"✓ 网络参数：")
    print(f"  - Agents数量: {args.nagents}")
    print(f"  - Hidden size: {args.hid_size}")
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
    print("测试结果 (Proper Decentralized Execution)")
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
        description='正确的 Decentralized Execution 测试（修复版）'
    )
    parser.add_argument('--model_path', type=str, 
                       default='saved/traffic_junction/ic3net/run21/model.pt',
                       help='训练好的模型路径')
    parser.add_argument('--num_episodes', type=int, default=100,
                       help='测试episode数量')
    
    args = parser.parse_args()
    
    import os
    if not os.path.exists(args.model_path):
        print(f"错误: 找不到模型文件 {args.model_path}")
        sys.exit(1)
    
    print("\n" + "="*80)
    print("⭐ 这是修复后的正确实现")
    print("="*80)
    print("\n关键修复:")
    print("  ❌ 错误版本: LSTM调用2次")
    print("  ✅ 修复版本: LSTM调用1次（与训练一致）")
    print("\n预期结果:")
    print("  - 成功率应该接近Simple版本（~80%）")
    print("  - 同时保持通信机制的完整性")
    print("\n")
    
    results = run_test(args.model_path, args.num_episodes)

