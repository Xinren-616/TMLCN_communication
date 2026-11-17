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
from bvme import BVMECommNetMLP
from gacg import GACGNetMLP
from trainer import Trainer
from models import RNN

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
parser.add_argument('--bvme', action='store_true', default=False,
                    help="enable BVME (variational comm) baseline on CommNet/IC3Net")
parser.add_argument('--bvme_kl_weight', type=float, default=0.0,
                    help="weight for BVME KL term in total loss")
parser.add_argument('--bvme_sigma0', type=float, default=0.1,
                    help="BVME prior std for KL")
parser.add_argument('--bvme_logvar_min', type=float, default=-5.0,
                    help="BVME clamp min for log-variance")
parser.add_argument('--bvme_logvar_max', type=float, default=3.0,
                    help="BVME clamp max for log-variance")
parser.add_argument('--gacg', action='store_true', default=False,
                    help="enable GACG baseline on CommNet/IC3Net")
parser.add_argument('--gacg_group_loss_weight', type=float, default=0.0,
                    help="weight for GACG group distance regularizer")
parser.add_argument('--gacg_num_groups', type=int, default=2,
                    help="GACG number of latent groups")
parser.add_argument('--gacg_history_k', type=int, default=4,
                    help="GACG observation history length for group context")
parser.add_argument('--gacg_group_var_scale', type=float, default=0.5,
                    help="GACG edge variance scaling for same-group pairs")
parser.add_argument('--gacg_edge_logvar_min', type=float, default=-5.0,
                    help="GACG edge log-variance clamp min")
parser.add_argument('--gacg_edge_logvar_max', type=float, default=1.0,
                    help="GACG edge log-variance clamp max")
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

if args.bvme or args.gacg:
    args.commnet = True
    args.hard_attn = True
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
    elif args.bvme:
        policy_net = BVMECommNetMLP(args, num_inputs)
    elif args.gacg:
        policy_net = GACGNetMLP(args, num_inputs)
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
elif args.bvme:
    model_dir = Path('./saved') / args.env_name / 'bvme'
elif args.gacg:
    model_dir = Path('./saved') / args.env_name / 'gacg'
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
        },
        'round2': {
            # 累加型指标
            'total_elements': 0,         # 传输的消息数(元素数)
            'total_comm_num': 0,         # 通信次数（累加型）
            # 平均型指标（先累加后面再平均）
            'active_channels': 0.0,      # 活跃通信通道数（边数）
            'entropy': 0.0,              # 通信信息熵
            'max_entropy': 0.0,          # 最大熵
            'effective_bandwidth': 0.0,  # 有效带宽利用率
            'message_similarity': 0.0,   # 消息相似度
        },
        'round3': {
            # 累加型指标
            'total_elements': 0,         # 传输的消息数(元素数)
            'total_comm_num': 0,         # 通信次数（累加型）
            # 平均型指标（先累加后面再平均）
            'active_channels': 0.0,      # 活跃通信通道数（边数）
            'entropy': 0.0,              # 通信信息熵
            'max_entropy': 0.0,          # 最大熵
            'effective_bandwidth': 0.0,  # 有效带宽利用率
            'message_similarity': 0.0,   # 消息相似度
        },
        # 累加型指标
        'total_elements': 0,             # 总传输消息数
        'total_comm_num': 0,             # 总通信次数（累加型）
        # 平均型指标
        'active_channels': 0.0,          # 总活跃通信通道数
        'total_entropy': 0.0,            # 总信息熵
        'total_max_entropy': 0.0,        # 总最大熵
        'total_effective_bandwidth': 0.0, # 总有效带宽利用率
        'total_message_similarity': 0.0,  # 总消息相似度
        'valid_batch_count': 0           # 记录有效的batch数量
    }
    
    # 遍历所有batch的统计数据
    for batch_stats in batch_comm_stats_list:
        if batch_stats is None or 'episode_count' not in batch_stats or batch_stats['episode_count'] == 0:
            continue
            
        epoch_stats['valid_batch_count'] += 1
        
        # 处理每轮通信的统计数据
        for key in ['round1', 'round2', 'round3']:
            if key in batch_stats:
                # 累加型指标
                for subkey in ['total_elements', 'total_comm_num']:
                    if subkey in batch_stats[key]:
                        epoch_stats[key][subkey] += batch_stats[key][subkey]
                
                # 平均型指标
                for subkey in ['active_channels', 'entropy', 'max_entropy', 'effective_bandwidth', 'message_similarity']:
                    if subkey in batch_stats[key]:
                        epoch_stats[key][subkey] += batch_stats[key][subkey]
        
        # 处理总体累加型指标
        if 'total_elements' in batch_stats:
            epoch_stats['total_elements'] += batch_stats['total_elements']
        if 'total_comm_num' in batch_stats:
            epoch_stats['total_comm_num'] += batch_stats['total_comm_num']
            
        # 处理总体平均型指标
        for subkey in ['active_channels', 'total_entropy', 'total_max_entropy', 'total_effective_bandwidth', 'total_message_similarity']:
            if subkey in batch_stats:
                epoch_stats[subkey] += batch_stats[subkey]
    
    # 如果有有效的batch数据，计算平均值
    if epoch_stats['valid_batch_count'] > 0:
        # 处理每轮通信的平均型指标
        for key in ['round1', 'round2', 'round3']:
            for subkey in ['active_channels', 'entropy', 'max_entropy', 'effective_bandwidth', 'message_similarity']:
                epoch_stats[key][subkey] /= epoch_stats['valid_batch_count']
        
        # 处理总体平均型指标
        for subkey in ['active_channels', 'total_entropy', 'total_max_entropy', 'total_effective_bandwidth', 'total_message_similarity']:
            epoch_stats[subkey] /= epoch_stats['valid_batch_count']
    
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
            if n == args.epoch_size - 1 and args.display:
                trainer.display = True
                
            # 统一执行train_batch，获取通信统计
            s, batch_comm_stats = trainer.train_batch(ep)
            # 将batch的通信统计数据添加到列表中
            batch_comm_stats_list.append(batch_comm_stats)
                
            print('batch: ', n)
            merge_stat(s, stat)
            trainer.display = False

        # 处理epoch级别的通信统计数据
        epoch_comm_stats = process_epoch_comm_stats(batch_comm_stats_list, args.epoch_size)
        
        # 打印epoch级别的通信统计数据
        print("\nEpoch Communication Statistics:")
        for round_key in ['round1', 'round2', 'round3']:
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
        print(f"Total Elements: {epoch_comm_stats['total_elements']}")
        print(f"Total Comm Num: {epoch_comm_stats['total_comm_num']}")
        # 平均型指标
        print(f"Active Channels: {epoch_comm_stats['active_channels']:.4f}")
        print(f"Total Entropy: {epoch_comm_stats['total_entropy']:.4f}")
        print(f"Total Max Entropy: {epoch_comm_stats['total_max_entropy']:.4f}")
        print(f"Total Effective Bandwidth: {epoch_comm_stats['total_effective_bandwidth']:.4f}")
        print(f"Total Message Similarity: {epoch_comm_stats['total_message_similarity']:.4f}")

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
    
        if args.save_every and ep and args.save and ep % args.save_every == 0:
            save(final=False, episode=ep)

        if args.save:
            save(final=True)

def save(final, episode=0): 
    d = dict()
    d['policy_net'] = policy_net.state_dict()
    d['log'] = log
    d['trainer'] = trainer.state_dict()
    if final:
        torch.save(d, run_dir / 'model.pt')
    else:
        torch.save(d, run_dir / ('model_ep%i.pt' %(episode)))

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
