import sys
import time
import signal
import argparse
import os
from pathlib import Path

import numpy as np
import torch
import visdom
import data
from magic import MAGIC
from utils import *
from action_utils import parse_action_args
from trainer import Trainer
from multi_processing import MultiProcessTrainer
import gym
import random
gym.logger.set_level(40)

torch.utils.backcompat.broadcast_warning.enabled = True
torch.utils.backcompat.keepdim_warning.enabled = True

torch.set_default_tensor_type('torch.DoubleTensor')

parser = argparse.ArgumentParser(description='Multi-Agent Graph Attention Communication')

# training
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
parser.add_argument('--directed', action='store_true', default=False,
                    help='whether the communication graph is directed')
parser.add_argument('--self_loop_type1', default=2, type=int,
                    help='self loop type in the first gat layer (0: no self loop, 1: with self loop, 2: decided by hard attn mechanism)')
parser.add_argument('--self_loop_type2', default=2, type=int,
                    help='self loop type in the second gat layer (0: no self loop, 1: with self loop, 2: decided by hard attn mechanism)')
parser.add_argument('--gat_num_heads', default=1, type=int,
                    help='number of heads in gat layers except the last one')
parser.add_argument('--gat_num_heads_out', default=1, type=int,
                    help='number of heads in output gat layer')
parser.add_argument('--gat_hid_size', default=64, type=int,
                    help='hidden size of one head in gat')
parser.add_argument('--ge_num_heads', default=4, type=int,
                    help='number of heads in the gat encoder')
parser.add_argument('--first_gat_normalize', action='store_true', default=False,
                    help='whether normalize the coefficients in the first gat layer of the message processor')
parser.add_argument('--second_gat_normalize', action='store_true', default=False,
                    help='whether normilize the coefficients in the second gat layer of the message proccessor')
parser.add_argument('--gat_encoder_normalize', action='store_true', default=False,
                    help='whether normilize the coefficients in the gat encoder (they have been normalized if the input graph is complete)')
parser.add_argument('--use_gat_encoder', action='store_true', default=False,
                    help='whether use the gat encoder before learning the first graph')
parser.add_argument('--gat_encoder_out_size', default=64, type=int,
                    help='hidden size of output of the gat encoder')
parser.add_argument('--first_graph_complete', action='store_true', default=False,
                    help='whether the first communication graph is set to a complete graph')
parser.add_argument('--second_graph_complete', action='store_true', default=False,
                    help='whether the second communication graph is set to a complete graph')
parser.add_argument('--learn_second_graph', action='store_true', default=False,
                    help='whether learn a new communication graph at the second round of communication')
parser.add_argument('--message_encoder', action='store_true', default=False,
                    help='whether use the message encoder')
parser.add_argument('--message_decoder', action='store_true', default=False,
                    help='whether use the message decoder')
parser.add_argument('--nagents', type=int, default=1,
                    help="number of agents")
parser.add_argument('--mean_ratio', default=0, type=float,
                    help='how much coooperative to do? 1.0 means fully cooperative')
parser.add_argument('--detach_gap', default=10000, type=int,
                    help='detach hidden state and cell state for rnns at this interval')
parser.add_argument('--comm_init', default='uniform', type=str,
                    help='how to initialise comm weights [uniform|zeros]')
parser.add_argument('--advantages_per_action', default=False, action='store_true',
                    help='whether to multipy log porb for each chosen action with advantages')
parser.add_argument('--comm_mask_zero', action='store_true', default=False,
                    help="whether block the communication")

# optimization
parser.add_argument('--gamma', type=float, default=1.0,
                    help='discount factor')
parser.add_argument('--seed', type=int, default=-1,
                    help='random seed') 
parser.add_argument('--normalize_rewards', action='store_true', default=False,
                    help='normalize rewards in each batch')
parser.add_argument('--lrate', type=float, default=0.001,
                    help='learning rate')
parser.add_argument('--entr', type=float, default=0,
                    help='entropy regularization coeff')
parser.add_argument('--value_coeff', type=float, default=0.01,
                    help='coefficient for value loss term')

# environment
parser.add_argument('--env_name', default="grf",
                    help='name of the environment to run')
parser.add_argument('--max_steps', default=20, type=int,
                    help='force to end the game after this many steps')
parser.add_argument('--nactions', default='1', type=str,
                    help='the number of agent actions')
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
                    help='display environment state')
parser.add_argument('--random', action='store_true', default=False,
                    help="enable random model")


init_args_for_env(parser)
args = parser.parse_args()

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

args.obs_size = env.observation_dim
args.num_actions = env.num_actions

# Multi-action
if not isinstance(args.num_actions, (list, tuple)): # single action case
    args.num_actions = [args.num_actions]
args.dim_actions = env.dim_actions

parse_action_args(args)

if args.seed == -1:
    args.seed = np.random.randint(0,10000)
torch.manual_seed(args.seed)
# # new add
# np.random.seed(args.seed)
# random.seed(args.seed)

print(args)

policy_net = MAGIC(args)


if not args.display:
    display_models([policy_net])

# share parameters among threads, but not gradients
for p in policy_net.parameters():
    p.data.share_memory_()

disp_trainer = Trainer(args, policy_net, data.init(args.env_name, args, False))
disp_trainer.display = True
def disp():
    x = disp_trainer.get_episode()

if args.env_name == 'grf':
    args.render = render
if args.nprocesses > 1:
    trainer = MultiProcessTrainer(args, lambda: Trainer(args, policy_net, data.init(args.env_name, args)))
else:
    trainer = Trainer(args, policy_net, data.init(args.env_name, args))

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


if args.plot:
    vis = visdom.Visdom(env=args.plot_env, port=args.plot_port)

model_dir = Path('./saved') / args.env_name 
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

def compute_ci_from_statistics(ci_stats, n):
    """
    从聚合的统计量计算 KL 散度形式的 Epoch 级别因果影响
    
    Args:
        ci_stats: 包含 count, mean_with, mean_without, M2_with, M2_without 的字典
        n: agent 数量
        
    Returns:
        total_ci: 总体因果影响（所有 agent 对的平均）
        ci_matrix: n×n 的因果影响矩阵
    """
    import numpy as np
    
    if ci_stats is None:
        return 0.0, np.zeros((n, n))
    
    epsilon = 1e-6
    ci_matrix = np.zeros((n, n))
    
    for i in range(n):
        for j in range(n):
            if i != j and ci_stats['count'][i, j] > 1:  # 需要至少2个样本才能计算方差
                count = ci_stats['count'][i, j]
                mean_with = ci_stats['mean_with'][i, j]
                mean_without = ci_stats['mean_without'][i, j]
                var_with = ci_stats['M2_with'][i, j] / count + epsilon
                var_without = ci_stats['M2_without'][i, j] / count + epsilon
                
                # 计算 KL 散度: D_KL(P||Q) = 0.5 * sum[(mu_p - mu_q)^2/var_q + var_p/var_q - 1 - log(var_p/var_q)]
                mean_diff_term = ((mean_with - mean_without) ** 2) / var_without
                var_ratio_term = var_with / var_without
                log_var_term = np.log(var_with / var_without)
                
                kl_div = 0.5 * (mean_diff_term + var_ratio_term - 1 - log_var_term).sum()
                ci_matrix[i, j] = kl_div
    
    # 总体 CI = 所有非对角元素的平均值
    non_diagonal_count = n * (n - 1)
    total_ci = ci_matrix.sum() / non_diagonal_count if non_diagonal_count > 0 else 0.0
    
    return total_ci, ci_matrix

def process_epoch_comm_stats(batch_comm_stats_list, epoch_size, trainer=None):
    """处理一个epoch中所有batch的通信统计数据
    
    Args:
        batch_comm_stats_list: 一个epoch中所有batch的通信统计数据列表
        epoch_size: epoch的大小（batch数量）
        trainer: Trainer 对象，用于访问 _merge_ci_statistics 方法
    
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
            'causal_influence': 0.0,     # 因果影响（新增）
            'hidden_state_change': 0.0,  # 隐藏状态变化（新增）
            'message_impact_ratio': 0.0, # 消息影响效率比率（新增）
        },
        'valid_batch_count': 0           # 记录有效的batch数量
    }
    
    # 初始化 epoch 级别的 CI 统计量
    epoch_ci_stats = None
    
    # 遍历所有batch的统计数据
    for batch_stats in batch_comm_stats_list:
        if batch_stats is None or 'episode_count' not in batch_stats or batch_stats['episode_count'] == 0:
            continue
            
        epoch_stats['valid_batch_count'] += 1
        
        # 处理每轮通信的统计数据
        if 'round1' in batch_stats:
            # 累加型指标
            for subkey in ['total_elements', 'total_comm_num']:
                if subkey in batch_stats['round1']:
                    epoch_stats['round1'][subkey] += batch_stats['round1'][subkey]
            
            # 平均型指标（包括新增的因果影响指标）
            for subkey in ['active_channels', 'entropy', 'max_entropy', 'effective_bandwidth', 'message_similarity',
                           'causal_influence', 'hidden_state_change', 'message_impact_ratio']:
                if subkey in batch_stats['round1']:
                    epoch_stats['round1'][subkey] += batch_stats['round1'][subkey]
        
        # 合并 CI 统计量
        if 'ci_statistics' in batch_stats and batch_stats['ci_statistics'] is not None:
            if trainer is not None:
                # 使用 trainer 的 _merge_ci_statistics 方法
                if hasattr(trainer, '_merge_ci_statistics'):
                    epoch_ci_stats = trainer._merge_ci_statistics(epoch_ci_stats, batch_stats['ci_statistics'])
                elif hasattr(trainer, 'trainer') and hasattr(trainer.trainer, '_merge_ci_statistics'):
                    # 多进程情况
                    epoch_ci_stats = trainer.trainer._merge_ci_statistics(epoch_ci_stats, batch_stats['ci_statistics'])
    
    # 如果有有效的batch数据，计算平均值
    if epoch_stats['valid_batch_count'] > 0:
        # 处理每轮通信的平均型指标（包括新增的因果影响指标）
        for subkey in ['active_channels', 'entropy', 'max_entropy', 'effective_bandwidth', 'message_similarity',
                       'causal_influence', 'hidden_state_change', 'message_impact_ratio']:
            epoch_stats['round1'][subkey] /= epoch_stats['valid_batch_count']
    
    # 将 epoch 级别的 CI 统计量添加到结果中
    epoch_stats['ci_statistics'] = epoch_ci_stats
    
    return epoch_stats
def run(num_epochs): 
    num_episodes = 0
    if args.save:
        os.makedirs(run_dir)
    for ep in range(num_epochs):
        epoch_begin_time = time.time()
        stat = dict()
        batch_comm_stats_list = []

        for n in range(args.epoch_size):
            trainer.display = args.display
            trainer.last_step = n == args.epoch_size - 1
            
            result, batch_comm_stats = trainer.train_batch(ep)
            print('batch: ', n)
            #print('batch_comm_stats', batch_comm_stats)
            merge_stat(result, stat)
            # 将batch的通信统计数据添加到列表中
            batch_comm_stats_list.append(batch_comm_stats)
            trainer.display = False
        
        # 处理epoch级别的通信统计数据
        epoch_comm_stats = process_epoch_comm_stats(batch_comm_stats_list, args.epoch_size, trainer)
        
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
        # 新增因果影响指标
        print(f"Total Causal Influence (Instantaneous L2): {epoch_comm_stats['round1']['causal_influence']:.6f}")
        print(f"Total Hidden State Change: {epoch_comm_stats['round1']['hidden_state_change']:.6f}")
        print(f"Total Message Impact Ratio: {epoch_comm_stats['round1']['message_impact_ratio']:.6f}")
        
        # 计算并打印 Epoch 级别的 KL 散度因果影响
        if 'ci_statistics' in epoch_comm_stats and epoch_comm_stats['ci_statistics'] is not None:
            total_ci, ci_matrix = compute_ci_from_statistics(epoch_comm_stats['ci_statistics'], args.nagents)
            
            print(f"Epoch-Level CI (KL Divergence): {total_ci:.6f}")
            
            # 打印 CI 矩阵的统计信息
            ci_stats = epoch_comm_stats['ci_statistics']
            total_samples = ci_stats['count'].sum()
            mean_samples_per_pair = ci_stats['count'].mean()
            
            # 计算平均方差
            valid_pairs = ci_stats['count'] > 1
            if valid_pairs.sum() > 0:
                mean_var_with = np.mean([ci_stats['M2_with'][i, j].mean() / ci_stats['count'][i, j] 
                                        for i in range(args.nagents) for j in range(args.nagents) 
                                        if i != j and ci_stats['count'][i, j] > 1])
                mean_var_without = np.mean([ci_stats['M2_without'][i, j].mean() / ci_stats['count'][i, j] 
                                           for i in range(args.nagents) for j in range(args.nagents) 
                                           if i != j and ci_stats['count'][i, j] > 1])
            else:
                mean_var_with = 0.0
                mean_var_without = 0.0
            
            print(f"\nSamples Collected: {int(total_samples)}")
            print(f"Mean Samples Per Pair: {mean_samples_per_pair:.1f}")
            print(f"Mean Variance (with): {mean_var_with:.6f}")
            print(f"Mean Variance (without): {mean_var_without:.6f}")
            
            # 打印 CI 矩阵
            print("\nCI Matrix (i→j):")
            for i in range(args.nagents):
                row_str = "  ["
                for j in range(args.nagents):
                    if i == j:
                        row_str += "  -  , "
                    else:
                        row_str += f"{ci_matrix[i, j]:.4f}, "
                row_str = row_str.rstrip(", ") + "]"
                print(row_str)

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



