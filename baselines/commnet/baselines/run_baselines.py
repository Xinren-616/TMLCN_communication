import sys
import time
import signal
import argparse
import os
from pathlib import Path

import numpy as np
import torch
import visdom
from comm import CommNetMLP
from ga_comm import GACommNetMLP
from tar_comm import TarCommNetMLP
from trainer import Trainer
from epoch_ci_calculator import EpochCICalculator

sys.path.append("..") 
import data
from utils import *
from action_utils import parse_action_args
from multi_processing import MultiProcessTrainer
# import gym

# gym.logger.set_level(40)

torch.utils.backcompat.broadcast_warning.enabled = True
torch.utils.backcompat.keepdim_warning.enabled = True

torch.set_default_tensor_type('torch.DoubleTensor')

def compute_ci_from_statistics(stats, hid_size):
    """
    从统计量计算 Epoch 级别的 KL 散度 CI
    
    Args:
        stats: 包含 count, mean_with, mean_without, M2_with, M2_without 的字典
        hid_size: 隐藏层大小
        
    Returns:
        CI_matrix: [n, n] 矩阵，每个元素是 agent i -> j 的 CI
        total_ci: 所有有效链路的平均 CI
        statistics: 统计信息字典
    """
    import numpy as np
    
    n = stats['count'].shape[0]
    CI_matrix = np.zeros((n, n))
    total_samples = 0
    valid_pairs = 0
    total_ci_sum = 0.0
    total_var_with = 0.0
    total_var_without = 0.0
    
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            
            count = stats['count'][i, j]
            if count < 2:  # 需要至少2个样本来估计方差
                continue
            
            # 计算方差
            var_with = stats['M2_with'][i, j] / (count - 1)  # 样本方差
            var_without = stats['M2_without'][i, j] / (count - 1)
            
            # 计算均值
            mean_with = stats['mean_with'][i, j]
            mean_without = stats['mean_without'][i, j]
            
            # 计算 KL 散度（假设高斯分布，方差为1，只考虑均值差异）
            # KL(p_with || p_without) = 0.5 * sum((mean_with - mean_without)^2)
            mean_diff_sq = np.sum((mean_with - mean_without) ** 2)
            ci = 0.5 * mean_diff_sq / hid_size  # 归一化
            
            CI_matrix[i, j] = ci
            total_ci_sum += ci
            valid_pairs += 1
            total_samples += count
            total_var_with += np.mean(var_with)
            total_var_without += np.mean(var_without)
    
    # 计算平均值
    if valid_pairs > 0:
        total_ci = total_ci_sum / valid_pairs
        mean_var_with = total_var_with / valid_pairs
        mean_var_without = total_var_without / valid_pairs
        mean_samples_per_pair = total_samples / valid_pairs
    else:
        total_ci = 0.0
        mean_var_with = 0.0
        mean_var_without = 0.0
        mean_samples_per_pair = 0.0
    
    statistics = {
        'total_samples': total_samples,
        'mean_samples_per_pair': mean_samples_per_pair,
        'mean_var_with': mean_var_with,
        'mean_var_without': mean_var_without,
        'valid_pairs': valid_pairs
    }
    
    return CI_matrix, total_ci, statistics

parser = argparse.ArgumentParser(description='PyTorch RL trainer')
# training
# note: number of steps per epoch = epoch_size X batch_size x nprocesses
parser.add_argument('--num_epochs', default=100, type=int,
                    help='number of training epochs')
parser.add_argument('--epoch_size', type=int, default=10,
                    help='number of update iterations in an epoch')
parser.add_argument('--batch_size', type=int, default=500,
                    help='number of steps before each update (per thread)')
parser.add_argument('--nprocesses', type=int, default=16,
                    help='How many processes to run')
# model
parser.add_argument('--hid_size', default=64, type=int,
                    help='hidden layer size')
parser.add_argument('--qk_hid_size', default=16, type=int,
                    help='key and query size for soft attention')
parser.add_argument('--value_hid_size', default=32, type=int,
                    help='value size for soft attention')
parser.add_argument('--recurrent', action='store_true', default=False,
                    help='make the model recurrent in time')

# optimization
parser.add_argument('--gamma', type=float, default=1.0,
                    help='discount factor')
parser.add_argument('--tau', type=float, default=1.0,
                    help='gae (remove?)')
parser.add_argument('--seed', type=int, default=-1,
                    help='random seed. Pass -1 for random seed') # TODO: works in thread?
parser.add_argument('--normalize_rewards', action='store_true', default=False,
                    help='normalize rewards in each batch')
parser.add_argument('--lrate', type=float, default=0.001,
                    help='learning rate')
parser.add_argument('--entr', type=float, default=0,
                    help='entropy regularization coeff')
parser.add_argument('--value_coeff', type=float, default=0.01,
                    help='coeff for value loss term')
# environment
parser.add_argument('--env_name', default="Cartpole",
                    help='name of the environment to run')
parser.add_argument('--max_steps', default=20, type=int,
                    help='force to end the game after this many steps')
parser.add_argument('--nactions', default='1', type=str,
                    help='the number of agent actions (0 for continuous). Use N:M:K for multiple actions')
parser.add_argument('--action_scale', default=1.0, type=float,
                    help='scale action output from model')
# other
parser.add_argument('--plot', action='store_true', default=False,
                    help='plot training progress')
parser.add_argument('--plot_env', default='main', type=str,
                    help='plot env name')
parser.add_argument('--plot_port', default='8097', type=str,
                    help='plot port')
parser.add_argument('--save', action="store_true", default=False,
                    help='save the model after training')
parser.add_argument('--save_every', default=0, type=int,
                    help='save the model after every n_th epoch')
parser.add_argument('--load', default='', type=str,
                    help='load the model')
parser.add_argument('--display', action="store_true", default=False,
                    help='Display environment state')
parser.add_argument('--random', action='store_true', default=False,
                    help="enable random model")

# CommNet specific args
parser.add_argument('--commnet', action='store_true', default=False,
                    help="enable commnet model")
parser.add_argument('--ic3net', action='store_true', default=False,
                    help="enable ic3net model")
parser.add_argument('--tarcomm', action='store_true', default=False,
                    help="enable tarmac model (with commnet or ic3net)")
parser.add_argument('--gacomm', action='store_true', default=False,
                    help="enable gacomm model")
parser.add_argument('--nagents', type=int, default=1,
                    help="Number of agents (used in multiagent)")
parser.add_argument('--comm_mode', type=str, default='avg',
                    help="Type of mode for communication tensor calculation [avg|sum]")
parser.add_argument('--comm_passes', type=int, default=1,
                    help="Number of comm passes per step over the model")
parser.add_argument('--comm_mask_zero', action='store_true', default=False,
                    help="Whether communication should be there")
parser.add_argument('--mean_ratio', default=1.0, type=float,
                    help='how much coooperative to do? 1.0 means fully cooperative')
parser.add_argument('--rnn_type', default='MLP', type=str,
                    help='type of rnn to use. [LSTM|MLP]')
parser.add_argument('--detach_gap', default=10000, type=int,
                    help='detach hidden state and cell state for rnns at this interval.'
                    + ' Default 10000 (very high)')
parser.add_argument('--comm_init', default='uniform', type=str,
                    help='how to initialise comm weights [uniform|zeros]')
parser.add_argument('--hard_attn', default=False, action='store_true',
                    help='Whether to use hard attention: action - talk|silent')
parser.add_argument('--comm_action_one', default=False, action='store_true',
                    help='Whether to always talk, sanity check for hard attention.')
parser.add_argument('--advantages_per_action', default=False, action='store_true',
                    help='Whether to multipy log porb for each chosen action with advantages')
parser.add_argument('--share_weights', default=False, action='store_true',
                    help='Share weights for hops')
parser.add_argument('--use_full_kl_divergence', default=False, action='store_true',
                    help='Use full KL divergence (estimate variance from batch) for causal influence. '
                         'If False, uses simplified version with fixed variance (equivalent to scaled L2 distance).')


init_args_for_env(parser)
args = parser.parse_args()

if args.ic3net:
    args.commnet = 1
    args.hard_attn = 1
    args.mean_ratio = 0
    
    # For TJ set comm action to 1 as specified in paper to showcase
    # importance of individual rewards even in cooperative games
    if args.env_name == "traffic_junction":
        args.comm_action_one = True
    
if args.gacomm:
    args.commnet = 1
    args.mean_ratio = 0
    if args.env_name == "traffic_junction":
        args.comm_action_one = True

# Enemy comm
args.nfriendly = args.nagents
if hasattr(args, 'enemy_comm') and args.enemy_comm:
    if hasattr(args, 'nenemies'):
        args.nagents += args.nenemies
    else:
        raise RuntimeError("Env. needs to pass argument 'nenemy'.")

if args.env_name == 'grf':
    render = args.render
    args.render = False
env = data.init(args.env_name, args, False)

num_inputs = env.observation_dim
args.num_actions = env.num_actions

# Multi-action
if not isinstance(args.num_actions, (list, tuple)): # single action case
    args.num_actions = [args.num_actions]
args.dim_actions = env.dim_actions
args.num_inputs = num_inputs

# Hard attention
if args.hard_attn and args.commnet:
    # add comm_action as last dim in actions
    args.num_actions = [*args.num_actions, 2]
    args.dim_actions = env.dim_actions + 1

# Recurrence
if args.commnet and (args.recurrent or args.rnn_type == 'LSTM'):
    args.recurrent = True
    args.rnn_type = 'LSTM'


parse_action_args(args)

if args.seed == -1:
    args.seed = np.random.randint(0,10000)
torch.manual_seed(args.seed)

print(args)

if args.gacomm:
    policy_net = GACommNetMLP(args, num_inputs)
elif args.commnet:
    if args.tarcomm:
        policy_net = TarCommNetMLP(args, num_inputs)
    else:
        policy_net = CommNetMLP(args, num_inputs)
elif args.random:
    policy_net = Random(args, num_inputs)
elif args.recurrent:
    policy_net = RNN(args, num_inputs)
else:
    policy_net = MLP(args, num_inputs)
        
if not args.display:
    display_models([policy_net])

# share parameters among threads, but not gradients
for p in policy_net.parameters():
    p.data.share_memory_()

if args.env_name == 'grf':
    args.render = render
if args.nprocesses > 1:
    trainer = MultiProcessTrainer(args, lambda: Trainer(args, policy_net, data.init(args.env_name, args)))
else:
    trainer = Trainer(args, policy_net, data.init(args.env_name, args))

disp_trainer = Trainer(args, policy_net, data.init(args.env_name, args, False))
disp_trainer.display = True
def disp():
    x = disp_trainer.get_episode()    
    
log = dict()
log['epoch'] = LogField(list(), False, None, None)
log['reward'] = LogField(list(), True, 'epoch', 'num_episodes')
log['enemy_reward'] = LogField(list(), True, 'epoch', 'num_episodes')
log['success'] = LogField(list(), True, 'epoch', 'num_episodes')
log['steps_taken'] = LogField(list(), True, 'epoch', 'num_episodes')
log['add_rate'] = LogField(list(), True, 'epoch', 'num_episodes')
log['comm_action'] = LogField(list(), True, 'epoch', 'num_steps')
log['enemy_comm'] = LogField(list(), True, 'epoch', 'num_steps')
log['value_loss'] = LogField(list(), True, 'epoch', 'num_steps')
log['action_loss'] = LogField(list(), True, 'epoch', 'num_steps')
log['entropy'] = LogField(list(), True, 'epoch', 'num_steps')
log['density1'] = LogField(list(), True, 'epoch', 'num_steps')
log['density2'] = LogField(list(), True, 'epoch', 'num_steps')

if args.plot:
    vis = visdom.Visdom(env=args.plot_env, port=args.plot_port)
if args.gacomm:
    model_dir = Path('./saved') / args.env_name / 'gacomm'
elif args.tarcomm:
    if args.ic3net:
        model_dir = Path('./saved') / args.env_name / 'tar_ic3net'
    elif args.commnet:
        model_dir = Path('./saved') / args.env_name / 'tar_commnet'
    else:
        model_dir = Path('./saved') / args.env_name / 'other'
elif args.ic3net:
    model_dir = Path('./saved') / args.env_name / 'ic3net'
elif args.commnet:
    model_dir = Path('./saved') / args.env_name / 'commnet'
else:
    model_dir = Path('./saved') / args.env_name / 'other'
if args.env_name == 'grf':
    model_dir = model_dir / args.scenario
if not model_dir.exists():
    curr_run = 'run1'
else:
    exst_run_nums = [int(str(folder.name).split('run')[1]) for folder in
                     model_dir.iterdir() if
                     str(folder.name).startswith('run')]
    if len(exst_run_nums) == 0:
        curr_run = 'run1'
    else:
        curr_run = 'run%i' % (max(exst_run_nums) + 1)
run_dir = model_dir / curr_run 
    
def process_epoch_comm_stats(batch_comm_stats_list, epoch_size):
    """处理一个epoch中所有batch的通信统计数据
    
    Args:
        batch_comm_stats_list: 一个epoch中所有batch的通信统计数据列表
        epoch_size: epoch的大小（batch数量）
    
    Returns:
        处理后的epoch级别统计数据
    """
    # 初始化累积数据字典
    epoch_stats = {
        'round1': {
            # 累加型指标
            'total_elements': 0,         # 传输的消息数(元素数)
            'total_comm_num': 0,         # 通信次数（累加型）
            # 平均型指标（先累加后面再平均）
            'active_channels': 0.0,      # 活跃通信通道数（边数）
            'entropy': 0.0,              # 通信信息熵
            'max_entropy': 0.0,          # 最大熵
            'effective_bandwidth': 0.0,  # 有效带宽利用率
            'message_similarity': 0.0,   # 消息相似度
            # 新增因果影响指标
            'causal_influence': 0.0,     # 因果影响
            'hidden_state_change': 0.0,  # 隐藏状态变化
            'message_impact_ratio': 0.0, # 消息影响效率比率
        },
        'valid_batch_count': 0           # 记录有效的batch数量
    }
    
    # 遍历所有batch的统计数据
    for batch_stats in batch_comm_stats_list:
        if batch_stats is None or 'episode_count' not in batch_stats or batch_stats['episode_count'] == 0:
            continue
            
        epoch_stats['valid_batch_count'] += 1
        
        # 处理第一轮通信的统计数据
        if 'round1' in batch_stats:
            # 累加型指标
            for subkey in ['total_elements', 'total_comm_num']:
                if subkey in batch_stats['round1']:
                    epoch_stats['round1'][subkey] += batch_stats['round1'][subkey]
            
            # 平均型指标（包括新增的因果影响指标）
            for subkey in ['active_channels', 'entropy', 'max_entropy', 'effective_bandwidth', 
                           'message_similarity', 'causal_influence', 'hidden_state_change', 
                           'message_impact_ratio']:
                if subkey in batch_stats['round1']:
                    epoch_stats['round1'][subkey] += batch_stats['round1'][subkey]
    
    # 如果有有效的batch数据，计算平均值
    if epoch_stats['valid_batch_count'] > 0:
        # 处理平均型指标（包括新增的因果影响指标）
        for subkey in ['active_channels', 'entropy', 'max_entropy', 'effective_bandwidth', 
                       'message_similarity', 'causal_influence', 'hidden_state_change', 
                       'message_impact_ratio']:
            epoch_stats['round1'][subkey] /= epoch_stats['valid_batch_count']
    
    return epoch_stats

def run(num_epochs): 
    num_episodes = 0
    if args.save:
        os.makedirs(run_dir)
    
    # 初始化 Epoch 级别 CI 计算器（仅在单进程模式下）
    # Epoch CI 现在在多进程模式下也支持（通过传递统计量而不是原始 tensor）
    print("✓ Epoch-level CI calculation enabled (uses statistical aggregation)")
    
    for ep in range(num_epochs):
        epoch_begin_time = time.time()
        stat = dict()
        batch_comm_stats_list = []
        
        # 初始化 epoch 级别的 CI 统计量汇总
        epoch_ci_stats = None
        
        for n in range(args.epoch_size):
            if n == args.epoch_size - 1 and args.display:
                trainer.display = True
                
            # 统一执行train_batch，获取通信统计
            s, batch_comm_stats = trainer.train_batch(ep)
            # 将batch的通信统计数据添加到列表中
            batch_comm_stats_list.append(batch_comm_stats)
            
            # 收集 CI 统计量（在多进程模式下也可以传递）
            if 'ci_statistics' in batch_comm_stats and batch_comm_stats['ci_statistics'] is not None:
                # 合并统计量
                if epoch_ci_stats is None:
                    epoch_ci_stats = batch_comm_stats['ci_statistics']
                else:
                    # 使用 trainer 的合并方法（根据是否多进程选择正确的对象）
                    if hasattr(trainer, '_merge_ci_statistics'):
                        # 单进程模式：trainer 是 Trainer 对象
                        epoch_ci_stats = trainer._merge_ci_statistics(epoch_ci_stats, batch_comm_stats['ci_statistics'])
                    else:
                        # 多进程模式：trainer 是 MultiProcessTrainer 对象，需要访问内部的 trainer
                        epoch_ci_stats = trainer.trainer._merge_ci_statistics(epoch_ci_stats, batch_comm_stats['ci_statistics'])
                
            print('batch: ', n)
            merge_stat(s, stat)
            trainer.display = False

        # 处理epoch级别的通信统计数据
        epoch_comm_stats = process_epoch_comm_stats(batch_comm_stats_list, args.epoch_size)
        
        # 计算 epoch 级别的 KL 散度 CI（使用汇总的统计量）
        if epoch_ci_stats is not None:
            CI_matrix, total_ci_kl, ci_statistics = compute_ci_from_statistics(epoch_ci_stats, args.hid_size)
        else:
            total_ci_kl = 0.0
            ci_statistics = {'total_samples': 0, 'mean_samples_per_pair': 0, 
                           'mean_var_with': 0, 'mean_var_without': 0}
        
        # 打印epoch级别的通信统计数据
        print("\nEpoch Communication Statistics:")
        for round_key in ['round1']:
            print(f"\n{round_key.upper()}:")
            # 打印累加型指标
            print(f"Total Elements: {epoch_comm_stats[round_key]['total_elements']}")
            print(f"Total Comm Num: {epoch_comm_stats[round_key]['total_comm_num']}")
            # 打印平均型指标
            print(f"Active Channels: {epoch_comm_stats[round_key]['active_channels']:.4f}")
            print(f"Entropy: {epoch_comm_stats[round_key]['entropy']:.4f}")
            print(f"Max Entropy: {epoch_comm_stats[round_key]['max_entropy']:.4f}")
            print(f"Effective Bandwidth: {epoch_comm_stats[round_key]['effective_bandwidth']:.4f}")
            print(f"Message Similarity: {epoch_comm_stats[round_key]['message_similarity']:.4f}")
            # 打印新增的因果影响指标（瞬时 L2 距离）
            print(f"Causal Influence (Instantaneous L2): {epoch_comm_stats[round_key]['causal_influence']:.6f}")
            print(f"Hidden State Change: {epoch_comm_stats[round_key]['hidden_state_change']:.6f}")
            print(f"Message Impact Ratio: {epoch_comm_stats[round_key]['message_impact_ratio']:.6f}")
        
        # 打印 Epoch 级别的 KL 散度 CI
        if epoch_ci_stats is not None:
            print(f"\nEpoch-Level Causal Influence (KL Divergence):")
            print(f"Total CI (KL): {total_ci_kl:.6f}")
            print(f"Samples Collected: {ci_statistics['total_samples']:.0f}")
            print(f"Mean Samples Per Pair: {ci_statistics['mean_samples_per_pair']:.1f}")
            print(f"Mean Variance (with): {ci_statistics['mean_var_with']:.6f}")
            print(f"Mean Variance (without): {ci_statistics['mean_var_without']:.6f}")
            
            # 打印 n×n CI 矩阵
            print(f"\nCI Matrix (Agent i → Agent j):")
            n_agents = CI_matrix.shape[0]
            
            # 打印表头
            header = "     "
            for j in range(n_agents):
                header += f"  j={j}   "
            print(header)
            print("-" * len(header))
            
            # 打印每一行
            for i in range(n_agents):
                row = f"i={i} |"
                for j in range(n_agents):
                    if i == j:
                        row += "   -    "  # 对角线不显示
                    else:
                        row += f"{CI_matrix[i, j]:7.5f} "
                print(row)
        
        # 打印epoch总体统计数据
        print("\nEpoch Overall Statistics:")
        # 累加型指标
        print(f"Total Elements: {epoch_comm_stats['round1']['total_elements']}")
        print(f"Total Comm Num: {epoch_comm_stats['round1']['total_comm_num']}")
        # 平均型指标
        print(f"Active Channels: {epoch_comm_stats['round1']['active_channels']:.4f}")
        print(f"Total Entropy: {epoch_comm_stats['round1']['entropy']:.4f}")
        print(f"Total Max Entropy: {epoch_comm_stats['round1']['max_entropy']:.4f}")
        print(f"Total Effective Bandwidth: {epoch_comm_stats['round1']['effective_bandwidth']:.4f}")
        print(f"Total Message Similarity: {epoch_comm_stats['round1']['message_similarity']:.4f}")
        # 打印新增的因果影响指标（瞬时 L2 距离）
        print(f"Total Causal Influence (Instantaneous L2): {epoch_comm_stats['round1']['causal_influence']:.6f}")
        print(f"Total Hidden State Change: {epoch_comm_stats['round1']['hidden_state_change']:.6f}")
        print(f"Total Message Impact Ratio: {epoch_comm_stats['round1']['message_impact_ratio']:.6f}")
        # 打印 Epoch 级别的 KL 散度 CI 和矩阵
        if epoch_ci_stats is not None:
            print(f"Epoch-Level CI (KL Divergence): {total_ci_kl:.6f}")
            
            # 打印简化版 CI 矩阵（只显示数值）
            print(f"\nCI Matrix (i→j):")
            n_agents = CI_matrix.shape[0]
            for i in range(n_agents):
                row_values = []
                for j in range(n_agents):
                    if i == j:
                        row_values.append("  -  ")
                    else:
                        row_values.append(f"{CI_matrix[i, j]:.4f}")
                print(f"  [{', '.join(row_values)}]")

        epoch_time = time.time() - epoch_begin_time
        epoch = len(log['epoch'].data) + 1
        num_episodes += stat['num_episodes']
        for k, v in log.items():
            if k == 'epoch':
                v.data.append(epoch)
            else:
                if k in stat and v.divide_by is not None and stat[v.divide_by] > 0:
                    stat[k] = stat[k] / stat[v.divide_by]
                v.data.append(stat.get(k, 0))

        np.set_printoptions(precision=2)
        
        print('Epoch {}'.format(epoch))
        print('Episode: {}'.format(num_episodes))
        print('Reward: {}'.format(stat['reward']))
        print('Time: {:.2f}s'.format(epoch_time))
        
        if 'enemy_reward' in stat.keys():
            print('Enemy-Reward: {}'.format(stat['enemy_reward']))
        if 'add_rate' in stat.keys():
            print('Add-Rate: {:.2f}'.format(stat['add_rate']))
        if 'success' in stat.keys():
            print('Success: {:.4f}'.format(stat['success']))
        if 'steps_taken' in stat.keys():
            print('Steps-Taken: {:.2f}'.format(stat['steps_taken']))
        if 'comm_action' in stat.keys():
            print('Comm-Action: {}'.format(stat['comm_action']))
        if 'enemy_comm' in stat.keys():
            print('Enemy-Comm: {}'.format(stat['enemy_comm']))
        if 'density1' in stat.keys():
            print('density1: {:.4f}'.format(stat['density1']))
        if 'density2' in stat.keys():
            print('density2: {:.4f}'.format(stat['density2']))        


        if args.plot:
            for k, v in log.items():
                if v.plot and len(v.data) > 0:
                    vis.line(np.asarray(v.data), np.asarray(log[v.x_axis].data[-len(v.data):]),
                    win=k, opts=dict(xlabel=v.x_axis, ylabel=k))
    
        # 每100个epoch保存checkpoint用于后续分析
        if args.save and (ep+1) % 100 == 0:
            save(final=False, epoch=ep+1)
        
        if args.save_every and ep and args.save and (ep+1) % args.save_every == 0:
            save(final=False, epoch=ep+1)

        if args.save:
            save(final=True)

def save(final, epoch=0): 
    d = dict()
    d['policy_net'] = policy_net.state_dict()
    d['log'] = log
    d['trainer'] = trainer.state_dict()
    if final:
        torch.save(d, run_dir / 'model.pt')
    else:
        torch.save(d, run_dir / ('model_ep%i.pt' %(epoch)))

def load(path):
    d = torch.load(path)
    # log.clear()
    policy_net.load_state_dict(d['policy_net'])
    log.update(d['log'])
    trainer.load_state_dict(d['trainer'])

def signal_handler(signal, frame):
        print('You pressed Ctrl+C! Exiting gracefully.')
        if args.display:
            env.end_display()
        sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)

if args.load != '':
    load(args.load)

run(args.num_epochs)
if args.display:
    env.end_display()

if args.save:
    save(final=True)

if sys.flags.interactive == 0 and args.nprocesses > 1:
    trainer.quit()
    import os
    os._exit(0)
