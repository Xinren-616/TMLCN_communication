"""
正确的 Decentralized Execution 实现（修复版 - MAGIC）

关键修复：
===========
1. LSTM只调用一次（与训练时一致）
2. 通信基于前一时刻的hidden state（而非当前观察）
3. LSTM输入是 (编码观察 + 通信)

与test_true_decentralized.py的区别：
- True版本: 调用LSTM两次 ❌
- Proper版本: 调用LSTM一次 ✅

适配MAGIC:
- 使用MAGIC的Graph Attention机制
- 使用MAGIC的通信协议
"""

import sys
import argparse
import numpy as np
import torch
import torch.nn.functional as F

import data
from utils import *
from action_utils import *
from magic import MAGIC

torch.set_default_tensor_type('torch.DoubleTensor')


def decentralized_forward_proper(policy_net, observations, prev_hiddens, args, info={}):
    """
    正确的Decentralized模式前向传播（MAGIC版本）
    
    关键：与训练时的forward行为完全一致
    - LSTM只调用一次
    - 输入是 (编码观察 + 通信)
    - 通信基于前一时刻的hidden state，并使用Graph Attention
    """
    
    batch_size = 1
    n = args.nagents
    hid_size = args.hid_size
    
    # =========================================================================
    # Phase 1: 每个agent独立编码观察（不调用LSTM）
    # =========================================================================
    # encoded_obs: [1 * n * hid_size]
    encoded_obs = policy_net.obs_encoder(observations)
    hidden_state, cell_state = prev_hiddens
    
    # =========================================================================
    # Phase 2: 更新LSTM（关键：只调用一次！）
    # =========================================================================
    # 使用LSTM更新状态
    hidden_state, cell_state = policy_net.lstm_cell(encoded_obs.squeeze(0), (hidden_state, cell_state))
    
    # =========================================================================
    # Phase 3: 生成通信消息（基于更新后的hidden state）
    # =========================================================================
    # comm: [n * hid_size]
    comm = hidden_state
    
    # 获取agent mask
    num_agents_alive, agent_mask = policy_net.get_agent_mask(batch_size, info)
    
    # 如果禁用通信，则mask设为0
    if args.comm_mask_zero:
        agent_mask *= torch.zeros(n, 1)
    
    # 如果使用message encoder
    if args.message_encoder:
        comm = policy_net.message_encoder(comm)
    
    # 对已死亡的agent进行mask
    comm = comm * agent_mask
    
    # =========================================================================
    # Phase 4: 生成通信图（Scheduler）
    # =========================================================================
    if not args.first_graph_complete:
        # 使用scheduler学习通信图
        if args.use_gat_encoder:
            adj_complete = policy_net.get_complete_graph(agent_mask)
            encoded_state = policy_net.gat_encoder(comm, adj_complete)
            adj = policy_net.sub_scheduler(policy_net.sub_scheduler_mlp, encoded_state, agent_mask, args.directed)
        else:
            adj = policy_net.sub_scheduler(policy_net.sub_scheduler_mlp, comm, agent_mask, args.directed)
    else:
        # 使用完全连接图
        adj = policy_net.get_complete_graph(agent_mask)
    
    # =========================================================================
    # 统计通信信息（仿照trainer.py）
    # =========================================================================
    comm_stats = {
        'round1': {
            'total_elements': 0,
            'active_channels': 0,
            'entropy': 0.0,
            'max_entropy': 0.0,
            'effective_bandwidth': 0.0,
            'message_similarity': 0.0,
            'total_comm_num': 0,
        }
    }
    
    # 计算通信统计
    with torch.no_grad():
        # 统计活跃通信通道数（边数）
        # adj: [n, n], 统计非零边的数量
        active_channels = (adj > 0).sum().item()
        comm_stats['round1']['active_channels'] = active_channels
        
        # 统计传输的消息数（元素数）
        # comm: [n, hid_size], 统计非零消息的元素数
        total_elements = (comm.abs() > 1e-8).sum().item()
        comm_stats['round1']['total_elements'] = total_elements
        
        # 通信次数：每个时间步算一次通信
        comm_stats['round1']['total_comm_num'] = 1
        
        # 计算通信熵
        if active_channels > 0:
            adj_flat = adj.view(-1)
            adj_flat = adj_flat[adj_flat > 1e-8]  # 只考虑非零边
            if len(adj_flat) > 0:
                # 归一化为概率分布
                adj_prob = adj_flat / adj_flat.sum()
                entropy = -(adj_prob * torch.log(adj_prob + 1e-10)).sum().item()
                max_entropy = np.log(len(adj_flat))
                comm_stats['round1']['entropy'] = entropy
                comm_stats['round1']['max_entropy'] = max_entropy
        
        # 计算有效带宽利用率
        max_channels = n * n if args.directed else n * (n - 1) / 2
        if max_channels > 0:
            effective_bandwidth = active_channels / max_channels
            comm_stats['round1']['effective_bandwidth'] = effective_bandwidth
        
        # 计算消息相似度
        if n > 1:
            # 计算所有消息对之间的余弦相似度
            comm_norm = F.normalize(comm, p=2, dim=-1)
            similarity_matrix = torch.mm(comm_norm, comm_norm.t())
            # 去除对角线（自己和自己的相似度）
            mask = torch.eye(n, dtype=torch.bool)
            similarities = similarity_matrix[~mask]
            if len(similarities) > 0:
                message_similarity = similarities.mean().item()
                comm_stats['round1']['message_similarity'] = message_similarity
    
    # =========================================================================
    # Phase 5: 通过Graph Attention处理通信（Sub-processor）
    # =========================================================================
    comm = F.elu(policy_net.sub_processor(comm, adj))
    
    # 如果使用message decoder
    if args.message_decoder:
        comm = policy_net.message_decoder(comm)
    
    # =========================================================================
    # Phase 6: 生成动作和价值
    # =========================================================================
    # 将hidden state和通信concat
    value_head = policy_net.value_head(torch.cat((hidden_state, comm), dim=-1))
    h = hidden_state.view(batch_size, n, hid_size)
    c = comm.view(batch_size, n, hid_size)
    
    # 生成动作分布
    action_out = [F.log_softmax(action_head(torch.cat((h, c), dim=-1)), dim=-1) 
                  for action_head in policy_net.action_heads]
    
    return action_out, value_head, (hidden_state.clone(), cell_state.clone()), comm_stats


def test_episode_proper(args, policy_net, env, epoch=0):
    """使用正确的decentralized forward测试episode（MAGIC版本）"""
    
    # 重置环境
    from inspect import getfullargspec
    reset_args = getfullargspec(env.reset).args
    if 'epoch' in reset_args:
        state = env.reset(epoch)
    else:
        state = env.reset()
    
    stat = dict()
    info = dict()
    
    # 初始化hidden states - MAGIC使用init_hidden返回元组
    prev_hid = policy_net.init_hidden(batch_size=1)
    
    # 初始化episode级别的通信统计累积字典（仿照trainer.py）
    episode_comm_stats = {
        'round1': {
            'total_elements': 0,         # 传输的消息数(元素数)
            'active_channels': 0,        # 活跃通信通道数（边数）
            'entropy': 0.0,              # 通信信息熵
            'max_entropy': 0.0,          # 最大熵
            'effective_bandwidth': 0.0,  # 有效带宽利用率
            'message_similarity': 0.0,   # 消息相似度
            'total_comm_num': 0,         # 通信次数（累加型）
        },
        'step_count': 0                   # 记录有效步数
    }
    
    with torch.no_grad():
        for t in range(args.max_steps):
            # 使用修复后的decentralized forward
            action_out, value, prev_hid, comm_stats = decentralized_forward_proper(
                policy_net, state, prev_hid, args, info
            )
            
            # 累加当前步骤的通信统计（仿照trainer.py）
            if comm_stats is not None:
                episode_comm_stats['step_count'] += 1
                
                # 处理round1通信的统计数据
                if 'round1' in comm_stats:
                    # 累加型指标
                    for subkey in ['total_elements', 'total_comm_num']:
                        if subkey in comm_stats['round1']:
                            episode_comm_stats['round1'][subkey] += comm_stats['round1'][subkey]
                    
                    # 平均型指标 - 先累加后面再计算平均值
                    for subkey in ['active_channels', 'effective_bandwidth', 'message_similarity', 'entropy', 'max_entropy']:
                        if subkey in comm_stats['round1']:
                            episode_comm_stats['round1'][subkey] += comm_stats['round1'][subkey]
            
            # 选择动作
            action = select_action(args, action_out)
            action, actual = translate_action(args, env, action)
            
            # 执行动作
            next_state, reward, done, info = env.step(actual)
            
            # 统计
            stat['reward'] = stat.get('reward', 0) + reward[:args.nfriendly]
            
            state = next_state
            done = done or t == args.max_steps - 1
            
            if done:
                break
        
        stat['num_steps'] = t + 1
        stat['steps_taken'] = t + 1
    
    # 计算平均值（仿照trainer.py）
    if episode_comm_stats['step_count'] > 0:
        # 只对平均型指标计算平均值
        for subkey in ['active_channels', 'effective_bandwidth', 'message_similarity', 'entropy', 'max_entropy']:
            if subkey in episode_comm_stats['round1']:
                episode_comm_stats['round1'][subkey] /= episode_comm_stats['step_count']
    
    # 终止奖励
    if hasattr(env, 'reward_terminal'):
        terminal_reward = env.reward_terminal()
        stat['reward'] = stat.get('reward', 0) + terminal_reward[:args.nfriendly]
    
    # 环境统计
    if hasattr(env, 'get_stat'):
        merge_stat(env.get_stat(), stat)
    
    return stat, episode_comm_stats


def run_test(model_path, num_episodes=100):
    """运行修复后的测试（MAGIC版本）"""
    
    print(f"\n{'='*80}")
    print("正确的 Decentralized Execution 测试（修复版 - MAGIC）")
    print(f"{'='*80}\n")
    print("关键修复:")
    print("  ✅ LSTM只调用一次（与训练时一致）")
    print("  ✅ 通信基于前一时刻的hidden state")
    print("  ✅ 使用Graph Attention机制进行通信")
    print(f"{'='*80}\n")
    
    # 加载模型
    checkpoint = torch.load(model_path)
    
    # 设置参数 - MAGIC参数（根据实际checkpoint配置）
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
        
        # MAGIC特定参数 - 从checkpoint推断
        directed = False
        self_loop_type1 = 2
        self_loop_type2 = 2
        gat_num_heads = 4  # 从 a_i/a_j 形状 [4, 128, 1] 推断
        gat_num_heads_out = 1
        gat_hid_size = 128  # 从 W 形状 [128, 512] = [in, 4*128] 推断
        ge_num_heads = 4
        first_gat_normalize = False
        second_gat_normalize = False
        gat_encoder_normalize = False
        use_gat_encoder = False
        gat_encoder_out_size = 64
        first_graph_complete = True  # 模型没有 sub_scheduler_mlp
        second_graph_complete = False
        learn_second_graph = False
        message_encoder = False
        message_decoder = True  # 模型有 message_decoder
        
        # 启用/禁用通信
        comm_mask_zero = False
        
        mean_ratio = 1.0
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
    args.obs_size = num_inputs
    args.nfriendly = args.nagents
    args.naction_heads = args.num_actions
    
    parse_action_args(args)
    
    # 创建MAGIC网络
    policy_net = MAGIC(args)
    policy_net.load_state_dict(checkpoint['policy_net'])
    policy_net.eval()
    
    print(f"✓ 模型加载成功")
    print(f"✓ 网络参数：")
    print(f"  - Agents数量: {args.nagents}")
    print(f"  - Hidden size: {args.hid_size}")
    print(f"  - Graph Attention heads: {args.gat_num_heads}")
    print(f"  - First graph complete: {args.first_graph_complete}")
    print(f"\n开始测试 {num_episodes} episodes...\n")
    
    # 运行测试
    all_stats = []
    all_comm_stats = []
    
    # 初始化汇总通信统计（仿照trainer.py）
    total_comm_stats = {
        'round1': {
            'total_elements': 0,
            'active_channels': 0,
            'entropy': 0.0,
            'max_entropy': 0.0,
            'effective_bandwidth': 0.0,
            'message_similarity': 0.0,
            'total_comm_num': 0,
        },
        'episode_count': 0
    }
    
    for ep in range(num_episodes):
        stat, episode_comm_stats = test_episode_proper(args, policy_net, env, epoch=ep)
        all_stats.append(stat)
        all_comm_stats.append(episode_comm_stats)
        
        # 累加通信统计（仿照trainer.py）
        if episode_comm_stats['step_count'] > 0:
            total_comm_stats['episode_count'] += 1
            
            # 累加型指标
            for subkey in ['total_elements', 'total_comm_num']:
                if subkey in episode_comm_stats['round1']:
                    total_comm_stats['round1'][subkey] += episode_comm_stats['round1'][subkey]
            
            # 平均型指标 - 先累加后面再计算平均值
            for subkey in ['active_channels', 'effective_bandwidth', 'message_similarity', 'entropy', 'max_entropy']:
                if subkey in episode_comm_stats['round1']:
                    total_comm_stats['round1'][subkey] += episode_comm_stats['round1'][subkey]
        
        if (ep + 1) % 20 == 0:
            print(f"Episode {ep + 1}/{num_episodes} 完成")
    
    # 计算平均值（仿照trainer.py）
    if total_comm_stats['episode_count'] > 0:
        for subkey in ['active_channels', 'effective_bandwidth', 'message_similarity', 'entropy', 'max_entropy']:
            if subkey in total_comm_stats['round1']:
                total_comm_stats['round1'][subkey] /= total_comm_stats['episode_count']
    
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
    
    # 计算基本统计指标
    for metric_name, values in metrics.items():
        if len(values) > 0:
            values = np.array(values)
            print(f"{metric_name.upper()}:")
            print(f"  平均值: {values.mean():.4f}")
            print(f"  标准差: {values.std():.4f}")
            print(f"  最小值: {values.min():.4f}")
            print(f"  最大值: {values.max():.4f}")
            print()
    
    # 输出通信统计信息
    print(f"\n{'='*80}")
    print("通信统计 (Communication Statistics)")
    print(f"{'='*80}\n")
    
    if total_comm_stats['episode_count'] > 0:
        print(f"ROUND 1 通信统计:")
        print(f"  总通信次数 (Total Comm Num): {total_comm_stats['round1']['total_comm_num']}")
        print(f"  总传输元素数 (Total Elements): {total_comm_stats['round1']['total_elements']}")
        print(f"  平均活跃通道数 (Avg Active Channels): {total_comm_stats['round1']['active_channels']:.4f}")
        print(f"  平均有效带宽利用率 (Avg Effective Bandwidth): {total_comm_stats['round1']['effective_bandwidth']:.4f}")
        print(f"  平均通信熵 (Avg Entropy): {total_comm_stats['round1']['entropy']:.4f}")
        print(f"  平均最大熵 (Avg Max Entropy): {total_comm_stats['round1']['max_entropy']:.4f}")
        print(f"  平均消息相似度 (Avg Message Similarity): {total_comm_stats['round1']['message_similarity']:.4f}")
        print()
    
    # 计算TEI指标 (TEI = Total Communication Number / Total Success)
    print(f"{'='*80}")
    print("TEI 指标 (Task Efficiency Index)")
    print(f"{'='*80}\n")
    
    total_success = sum(metrics['success']) if 'success' in metrics and len(metrics['success']) > 0 else 0
    total_comm_num = total_comm_stats['round1']['total_comm_num']
    
    print(f"总成功次数 (Total Success): {total_success}")
    print(f"总通信次数 (Total Comm Num): {total_comm_num}")
    
    if total_success > 0:
        tei = total_comm_num / total_success
        print(f"\n✨ TEI = Total Comm Num / Total Success")
        print(f"✨ TEI = {total_comm_num} / {total_success}")
        print(f"✨ TEI = {tei:.4f}")
        print(f"\n(TEI越小表示完成任务所需的通信开销越低，效率越高)")
    else:
        print(f"\n⚠️  无法计算TEI (成功次数为0)")
        tei = None
    
    print(f"\n{'='*80}\n")
    
    # 将通信统计和TEI添加到返回的metrics中
    metrics['comm_stats'] = total_comm_stats
    metrics['tei'] = tei
    
    return metrics


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='正确的 Decentralized Execution 测试（修复版 - MAGIC）'
    )
    parser.add_argument('--model_path', type=str, 
                       default='saved/traffic_junction/run1/model.pt',
                       help='训练好的MAGIC模型路径')
    parser.add_argument('--num_episodes', type=int, default=100,
                       help='测试episode数量')
    
    args = parser.parse_args()
    
    import os
    if not os.path.exists(args.model_path):
        print(f"错误: 找不到模型文件 {args.model_path}")
        sys.exit(1)
    
    print("\n" + "="*80)
    print("⭐ 这是修复后的正确实现（MAGIC版本）")
    print("="*80)
    print("\n关键特性:")
    print("  ✅ LSTM调用1次（与训练一致）")
    print("  ✅ 使用Graph Attention进行通信")
    print("  ✅ 支持动态通信图学习")
    print("\n预期结果:")
    print("  - 成功率应该与训练时性能一致")
    print("  - 保持MAGIC通信机制的完整性")
    print("\n")
    
    results = run_test(args.model_path, args.num_episodes)

