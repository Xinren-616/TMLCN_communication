"""
训练脚本 - 仅记录 Shapley 值（不做 CTAP 剪枝）

每隔 --shapley_interval 个 epoch 计算一次边级别 Shapley 值并保存到
saved/<run_name>/shapley_results/，训练过程本身不做任何通信拓扑剪枝。

使用方法：
python train_with_shapley_only.py \\
    --env_name mpe --mpe_scenario simple_spread \\
    --nagents 3 --model_type commnet \\
    --num_epochs 2000 --shapley_interval 100 --shapley_episodes 10 \\
    --save_dir ./saved/mpe_commnet_shapley_only --cuda
"""

import sys
import time
import argparse
from pathlib import Path
import numpy as np
import torch
import os

torch.set_default_tensor_type('torch.DoubleTensor')

current_dir = Path(__file__).resolve().parent
parent_dir = current_dir.parent
sys.path.insert(0, str(parent_dir))

from comm import CommNetMLP
from ga_comm import GACommNetMLP
from tar_comm import TarCommNetMLP
from mat_comm import MATCommNet
from bvme_comm import BVMECommNet
from mappo_comm import MAPPOCommNet
from expocomm_comm import ExpoCommNet
sys.path.insert(0, str(parent_dir))
from magic import MAGIC

import data
from utils import *
from action_utils import parse_action_args, select_action, translate_action
from trainer import Trainer
from multi_processing import MultiProcessTrainer
from online_edge_shapley import OnlineEdgeShapleyAnalyzer, integrate_with_training


def parse_args():
    parser = argparse.ArgumentParser(
        description='Baseline Training with Shapley Value Recording (no pruning)'
    )

    # ── 基本训练参数 ──
    parser.add_argument('--env_name', type=str, required=True)
    parser.add_argument('--nagents', type=int, required=True)

    # ── 环境专属参数 ──
    parser.add_argument('--smac_map', type=str, default='10gen_terran')
    parser.add_argument('--smac_difficulty', type=str, default='7')
    parser.add_argument('--mpe_scenario', type=str, default='simple_spread')
    parser.add_argument('--mamujoco_scenario', type=str, default='Ant-v2')
    parser.add_argument('--mamujoco_agent_conf', type=str, default='2x4')

    parser.add_argument('--num_epochs', type=int, default=1000)
    parser.add_argument('--epoch_size', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=500)
    parser.add_argument('--nprocesses', type=int, default=1)

    # ── 模型参数 ──
    parser.add_argument('--model_type', type=str, required=True,
                        choices=['commnet', 'ic3net', 'tarmac', 'tarcomm', 'gacomm', 'magic', 'mat', 'bvme', 'mappo', 'expocomm'])
    parser.add_argument('--hid_size', type=int, default=128)
    parser.add_argument('--recurrent', action='store_true', default=False)
    parser.add_argument('--rnn_type', default='MLP', type=str)

    # ── Shapley 参数 ──
    parser.add_argument('--shapley_interval', type=int, default=100,
                        help='每隔多少 epoch 计算一次 Shapley 值')
    parser.add_argument('--shapley_episodes', type=int, default=10,
                        help='每次 Shapley 评估使用的 episode 数')
    parser.add_argument('--topology_threshold', type=float, default=0.1,
                        help='拓扑阈值：仅对使用频率 > 此值的边计算 Shapley')
    parser.add_argument('--disable_shapley', action='store_true', default=False,
                        help='禁用 Shapley 值计算（fullcomm baseline 使用）')

    # ── 保存 ──
    parser.add_argument('--save_dir', type=str, default='./saved/shapley_only_run')
    parser.add_argument('--save_every', type=int, default=100)
    parser.add_argument('--resume_from', type=str, default=None)

    # ── 环境 ──
    parser.add_argument('--max_steps', type=int, default=40)
    parser.add_argument('--nactions', default='1', type=str)

    # ── CommNet / IC3Net / TarComm / GAComm ──
    parser.add_argument('--commnet', action='store_true', default=False)
    parser.add_argument('--ic3net', action='store_true', default=False)
    parser.add_argument('--tarcomm', action='store_true', default=False)
    parser.add_argument('--gacomm', action='store_true', default=False)
    parser.add_argument('--magic', action='store_true', default=False)
    parser.add_argument('--hard_attn', default=False, action='store_true')
    parser.add_argument('--comm_passes', type=int, default=1)
    parser.add_argument('--comm_mask_zero', action='store_true', default=False)
    parser.add_argument('--comm_mode', type=str, default='avg')
    parser.add_argument('--comm_init', default='uniform', type=str)
    parser.add_argument('--share_weights', default=False, action='store_true')
    parser.add_argument('--advantages_per_action', default=False, action='store_true')
    parser.add_argument('--comm_action_one', default=False, action='store_true')
    parser.add_argument('--qk_hid_size', default=16, type=int)
    parser.add_argument('--value_hid_size', default=32, type=int)

    # ── MAGIC 专属 ──
    parser.add_argument('--directed', action='store_true', default=False)
    parser.add_argument('--self_loop_type1', default=2, type=int)
    parser.add_argument('--self_loop_type2', default=2, type=int)
    parser.add_argument('--gat_num_heads', default=1, type=int)
    parser.add_argument('--gat_num_heads_out', default=1, type=int)
    parser.add_argument('--ge_num_heads', default=4, type=int)
    parser.add_argument('--first_gat_normalize', action='store_true', default=False)
    parser.add_argument('--second_gat_normalize', action='store_true', default=False)
    parser.add_argument('--gat_encoder_normalize', action='store_true', default=False)
    parser.add_argument('--use_gat_encoder', action='store_true', default=False)
    parser.add_argument('--gat_encoder_out_size', default=64, type=int)
    parser.add_argument('--first_graph_complete', action='store_true', default=False)
    parser.add_argument('--second_graph_complete', action='store_true', default=False)
    parser.add_argument('--message_encoder', action='store_true', default=False)
    parser.add_argument('--message_decoder', action='store_true', default=False)

    # ── 优化 ──
    parser.add_argument('--lrate', type=float, default=0.001)
    parser.add_argument('--gamma', type=float, default=1.0)
    parser.add_argument('--value_coeff', type=float, default=0.01)
    parser.add_argument('--entr', type=float, default=0)
    parser.add_argument('--normalize_rewards', action='store_true', default=False)

    # ── 其他 ──
    parser.add_argument('--seed', type=int, default=-1)
    parser.add_argument('--mean_ratio', default=1.0, type=float)
    parser.add_argument('--detach_gap', default=10000, type=int)
    parser.add_argument('--display', action='store_true', default=False)
    parser.add_argument('--plot', action='store_true', default=False)
    parser.add_argument('--save', action='store_true', default=True)
    parser.add_argument('--continuous', action='store_true', default=False)
    parser.add_argument('--random', action='store_true', default=False)
    parser.add_argument('--cuda', action='store_true', default=False,
                        help='使用 GPU 训练')

    from utils import init_args_for_env
    init_args_for_env(parser)

    args = parser.parse_args()

    # model_type → flag
    if args.model_type == 'ic3net':
        args.ic3net = True
        args.commnet = True
        args.hard_attn = True
    elif args.model_type == 'commnet':
        args.commnet = True
    elif args.model_type in ('tarmac', 'tarcomm'):
        args.tarcomm = True
        args.commnet = True
    elif args.model_type == 'gacomm':
        args.gacomm = True
        args.commnet = True
    elif args.model_type == 'magic':
        args.magic = True
    elif args.model_type in ('mat', 'bvme', 'mappo', 'expocomm'):
        args.commnet = True

    if args.commnet and (args.recurrent or args.rnn_type == 'LSTM'):
        args.recurrent = True
        args.rnn_type = 'LSTM'
    else:
        args.recurrent = False
        if not hasattr(args, 'rnn_type') or args.rnn_type is None:
            args.rnn_type = 'MLP'

    return args


def create_model(args, num_inputs):
    if args.model_type == 'magic':
        return MAGIC(args)
    elif args.model_type == 'gacomm':
        return GACommNetMLP(args, num_inputs)
    elif args.model_type in ('tarmac', 'tarcomm'):
        return TarCommNetMLP(args, num_inputs)
    elif args.model_type == 'mat':
        return MATCommNet(args, num_inputs)
    elif args.model_type == 'bvme':
        return BVMECommNet(args, num_inputs)
    elif args.model_type == 'mappo':
        return MAPPOCommNet(args, num_inputs)
    elif args.model_type == 'expocomm':
        return ExpoCommNet(args, num_inputs)
    else:
        return CommNetMLP(args, num_inputs)


def train():
    args = parse_args()

    if args.seed != -1:
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)

    # device
    if args.cuda and torch.cuda.is_available():
        args.device = torch.device('cuda')
        torch.cuda.manual_seed(args.seed if args.seed != -1 else 0)
        torch.set_default_tensor_type('torch.cuda.DoubleTensor')
        print(f"使用 GPU: {torch.cuda.get_device_name(0)}")
    else:
        args.device = torch.device('cpu')
        torch.set_default_tensor_type('torch.DoubleTensor')
        if args.cuda:
            print("警告: 指定了 --cuda 但 CUDA 不可用，回退到 CPU")

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # 环境（需先初始化，make_smacv2_env 会自动修正 args.max_steps）
    env = data.init(args.env_name, args, False)
    args.obs_size = env.observation_dim
    args.num_actions = env.num_actions
    if not isinstance(args.num_actions, (list, tuple)):
        args.num_actions = [args.num_actions]
    args.dim_actions = env.dim_actions
    args.nfriendly = args.nagents
    args.num_inputs = env.observation_dim

    if args.hard_attn and args.commnet:
        args.num_actions = [*args.num_actions, 2]
        args.dim_actions = env.dim_actions + 1

    print("=" * 80)
    print("训练配置 - Shapley 值记录（无剪枝）")
    print("=" * 80)
    print(f"环境: {args.env_name}")
    if args.env_name == 'smac_v2':
        print(f"地图: {getattr(args, 'smac_map', 'N/A')}")
    print(f"模型: {args.model_type}")
    print(f"智能体数: {args.nagents}")
    print(f"总 Epoch: {args.num_epochs}")
    print(f"每 Epoch steps 上限: {args.max_steps}")
    print(f"Shapley 计算间隔: {args.shapley_interval} epochs")
    print(f"Shapley 评估 episodes: {args.shapley_episodes}")
    print(f"保存目录: {save_dir}")
    print("=" * 80 + "\n")

    parse_action_args(args)

    # 模型
    policy_net = create_model(args, env.observation_dim)
    policy_net = policy_net.double().to(args.device)

    # Trainer
    if args.nprocesses > 1:
        for p in policy_net.parameters():
            p.data.share_memory_()
        trainer = MultiProcessTrainer(args, lambda: Trainer(args, policy_net, data.init(args.env_name, args)))
    else:
        trainer = Trainer(args, policy_net, env)

    # Shapley 分析器
    shapley_analyzer = OnlineEdgeShapleyAnalyzer(
        args=args,
        env=env,
        n_eval_episodes=args.shapley_episodes,
        topology_threshold=args.topology_threshold,
    )

    def to_scalar(v):
        """将 numpy 数组或数值统一转为 Python float。"""
        if isinstance(v, np.ndarray):
            return float(np.mean(v))
        return float(v)

    log = {'epoch': [], 'reward': [], 'success': [], 'steps_taken': []}
    start_epoch = 1
    num_episodes = 0

    # 恢复 checkpoint
    if args.resume_from is not None:
        print(f"从 checkpoint 恢复: {args.resume_from}")
        checkpoint = torch.load(args.resume_from, weights_only=False)
        policy_net.load_state_dict(checkpoint['policy_net'])
        if hasattr(trainer, 'trainer'):
            trainer.trainer.optimizer.load_state_dict(checkpoint['optimizer'])
        else:
            trainer.optimizer.load_state_dict(checkpoint['optimizer'])
        if 'log' in checkpoint:
            log = checkpoint['log']
        start_epoch = checkpoint['epoch'] + 1
        num_episodes = checkpoint['epoch'] * args.epoch_size * args.nprocesses
        print(f"从 Epoch {start_epoch} 继续训练\n")

    print("开始训练...\n")

    for epoch in range(start_epoch, args.num_epochs + 1):
        epoch_begin_time = time.time()

        # 确保模型处于训练模式
        policy_net.train()
        for module in policy_net.modules():
            module.train()

        epoch_stats = {'reward': 0, 'success': 0, 'steps_taken': 0, 'num_episodes': 0}
        epoch_total_comm_num = 0.0

        for _ in range(args.epoch_size):
            # 不传 pruning_mask，保持全连接通信
            stat, batch_comm_stats = trainer.train_batch(epoch)

            epoch_stats['reward'] += stat.get('reward', 0)
            epoch_stats['success'] += stat.get('success', 0)
            epoch_stats['steps_taken'] += stat.get('steps_taken', 0)
            epoch_stats['num_episodes'] += stat.get('num_episodes', 0)

            # 累积 total_comm_num
            if isinstance(batch_comm_stats, dict) and 'round1' in batch_comm_stats:
                epoch_total_comm_num += batch_comm_stats['round1'].get('total_comm_num', 0.0)

        if epoch_stats['num_episodes'] > 0:
            epoch_stats['reward'] /= epoch_stats['num_episodes']
            epoch_stats['success'] /= epoch_stats['num_episodes']
            epoch_stats['steps_taken'] /= epoch_stats['num_episodes']

        num_episodes += epoch_stats['num_episodes']
        epoch_time = time.time() - epoch_begin_time

        log['epoch'].append(epoch)
        log['reward'].append(epoch_stats['reward'])
        log['success'].append(epoch_stats['success'])
        log['steps_taken'].append(epoch_stats['steps_taken'])

        reward_val  = to_scalar(epoch_stats['reward'])
        success_val = to_scalar(epoch_stats['success'])
        steps_val   = to_scalar(epoch_stats['steps_taken'])

        print(f"Epoch {epoch}/{args.num_epochs}")
        print(f"  Episodes: {num_episodes}")
        print(f"  Reward: {reward_val:.4f}")
        print(f"  Success: {success_val:.4f}")
        print(f"  Steps-Taken: {steps_val:.2f}")
        print(f"  Total Comm Num: {epoch_total_comm_num:.1f}")
        print(f"  Time: {epoch_time:.2f}s")

        # ── Shapley 值计算（仅记录，不剪枝）──
        if not getattr(args, 'disable_shapley', False):
            shapley_results = integrate_with_training(
                trainer=trainer,
                shapley_analyzer=shapley_analyzer,
                epoch=epoch,
                shapley_interval=args.shapley_interval,
            )

            if shapley_results is not None:
                shapley_analyzer.save_results(save_dir / 'shapley_results')
                shapley_analyzer.print_shapley_matrix(epoch)

        # 保存 checkpoint
        if epoch % args.save_every == 0:
            checkpoint = {
                'epoch': epoch,
                'policy_net': policy_net.state_dict(),
                'optimizer': trainer.state_dict(),
                'log': log,
            }
            save_dir.mkdir(parents=True, exist_ok=True)
            checkpoint_path = save_dir / f'model_ep{epoch}.pt'
            try:
                torch.save(checkpoint, str(checkpoint_path))
                print(f"  Checkpoint 已保存: model_ep{epoch}.pt")
            except Exception as e:
                print(f"  警告: 保存 checkpoint 失败: {e}")
                torch.save(checkpoint, str(Path(f'model_ep{epoch}.pt')))

        print()

    # 训练结束
    print("=" * 80)
    print("训练完成！")
    print("=" * 80)

    final_checkpoint = {
        'epoch': args.num_epochs,
        'policy_net': policy_net.state_dict(),
        'optimizer': trainer.state_dict(),
        'log': log,
    }
    save_dir.mkdir(parents=True, exist_ok=True)
    final_path = save_dir / 'model_final.pt'
    try:
        torch.save(final_checkpoint, str(final_path))
        print(f"最终模型已保存: {final_path}")
    except Exception as e:
        print(f"警告: 保存最终模型失败: {e}")
        torch.save(final_checkpoint, 'model_final.pt')

    # 保存最终 Shapley 值
    if not getattr(args, 'disable_shapley', False):
        shapley_analyzer.save_results(save_dir / 'shapley_results')
        print(f"Shapley 值结果已保存: {save_dir / 'shapley_results'}")
        if shapley_analyzer.computation_times:
            print(f"\nShapley 计算统计:")
            print(f"  总次数: {len(shapley_analyzer.computation_times)}")
            print(f"  平均耗时: {np.mean(shapley_analyzer.computation_times):.2f}s")
            print(f"  总耗时: {np.sum(shapley_analyzer.computation_times):.2f}s")


if __name__ == '__main__':
    train()
