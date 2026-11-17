"""
训练脚本 - 集成在线Shapley值计算

这个脚本展示如何将在线Shapley值计算集成到baseline模型的训练过程中。
每隔100个epoch，会自动计算一次边级别的Shapley值。

使用方法：
python train_with_online_shapley.py --env_name traffic_junction --nagents 5 \\
    --model_type commnet --num_epochs 1000 --shapley_interval 100 \\
    --shapley_episodes 10 --save_dir ./saved/tj_with_shapley
"""

import sys
import time
import argparse
from pathlib import Path
import numpy as np
import torch
import os

# 设置默认tensor类型为Double（MAGIC模型需要）
torch.set_default_tensor_type('torch.DoubleTensor')

# 添加父目录到path（支持从任何目录运行）
current_dir = Path(__file__).resolve().parent
parent_dir = current_dir.parent
sys.path.insert(0, str(parent_dir))

# 导入baseline模型
from comm import CommNetMLP
from ga_comm import GACommNetMLP
from tar_comm import TarCommNetMLP
from mat_comm import MATCommNet
from bvme_comm import BVMECommNet
from mappo_comm import MAPPOCommNet
from expocomm_comm import ExpoCommNet

# 添加父目录以导入MAGIC模型
import sys
sys.path.insert(0, str(parent_dir))
from magic import MAGIC

# 导入训练相关模块
import data
from utils import *
from action_utils import parse_action_args, select_action, translate_action
from trainer import Trainer
from multi_processing import MultiProcessTrainer

# 导入在线Shapley值分析器
from online_edge_shapley import OnlineEdgeShapleyAnalyzer, integrate_with_training

# 导入拓扑剪枝和通信统计模块（新增）
from topology_pruner import TopologyPruner
from communication_stats import CommunicationStatsCollector, CommunicationLogger


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='Baseline Training with Online Shapley Value Analysis')
    
    # 基本训练参数
    parser.add_argument('--env_name', type=str, required=True,
                       help='环境名称 (traffic_junction, predator_prey, smac_v2, mpe, mamujoco)')
    parser.add_argument('--nagents', type=int, required=True,
                       help='智能体数量')

    # ── SMAC v2 专属参数 ──
    parser.add_argument('--smac_map', type=str, default='10gen_terran',
                       help='[smac_v2] 地图名称，如 10gen_terran / 2s3z / 3m')
    parser.add_argument('--smac_difficulty', type=str, default='7',
                       help='[smac_v2] 难度 (1=VeryEasy … 7=VeryHard / A=CheatInsane)')

    # ── MPE 专属参数 ──
    parser.add_argument('--mpe_scenario', type=str, default='simple_spread',
                       help='[mpe] 场景名称：simple_spread / simple_tag / simple_adversary / simple_speaker_listener')

    # ── MAMuJoCo 专属参数 ──
    parser.add_argument('--mamujoco_scenario', type=str, default='Ant-v2',
                       help='[mamujoco] MuJoCo 场景：Ant-v2 / HalfCheetah-v2 / Hopper-v2 等')
    parser.add_argument('--mamujoco_agent_conf', type=str, default='2x4',
                       help='[mamujoco] Agent 分区配置：2x4 / 4x2 / 2x3 等')
    parser.add_argument('--num_epochs', type=int, default=1000,
                       help='训练epoch数')
    parser.add_argument('--epoch_size', type=int, default=10,
                       help='每个epoch的更新次数')
    parser.add_argument('--batch_size', type=int, default=500,
                       help='每次更新前的步数')
    parser.add_argument('--nprocesses', type=int, default=1,
                       help='进程数（建议设为1以便调试）')
    
    # 模型参数
    parser.add_argument('--model_type', type=str, required=True,
                       choices=['commnet', 'ic3net', 'tarmac', 'tarcomm', 'gacomm', 'magic', 'mat', 'bvme', 'mappo', 'expocomm'],
                       help='baseline模型类型')
    parser.add_argument('--hid_size', type=int, default=128,
                       help='隐藏层大小')
    parser.add_argument('--recurrent', action='store_true', default=False,
                       help='使用recurrent模型')
    parser.add_argument('--rnn_type', default='MLP', type=str,
                       help='RNN类型 (MLP/LSTM/GRU)')
    
    # 在线Shapley值参数
    parser.add_argument('--shapley_interval', type=int, default=100,
                       help='计算Shapley值的间隔（每N个epoch计算一次）')
    parser.add_argument('--shapley_episodes', type=int, default=10,
                       help='计算Shapley值时使用的episode数量')
    parser.add_argument('--topology_threshold', type=float, default=0.1,
                       help='拓扑阈值（只分析使用频率>此值的边）')
    parser.add_argument('--enable_shapley', action='store_true', default=True,
                       help='是否启用在线Shapley值计算')
    
    # 拓扑剪枝参数（新增）
    parser.add_argument('--enable_pruning', action='store_true', default=False,
                       help='是否启用基于Shapley值的拓扑剪枝')
    parser.add_argument('--pruning_threshold', type=float, default=-0.3,
                       help='Shapley值剪枝阈值（低于此值的边将被剪枝）')
    parser.add_argument('--pruning_start_epoch', type=int, default=500,
                       help='开始剪枝的epoch（warm-up期）')
    parser.add_argument('--pruning_interval', type=int, default=100,
                       help='剪枝mask更新间隔（与shapley_interval保持一致）')
    
    # 通信统计参数（新增）
    parser.add_argument('--enable_comm_stats', action='store_true', default=False,
                       help='是否启用通信统计（计算TEI等指标）')
    
    # CTAP参数（置信度引导的TEI最大化自适应剪枝）
    parser.add_argument('--enable_ctap', action='store_true', default=False,
                       help='启用CTAP（Confidence-guided TEI-maximizing Adaptive Pruning）')
    parser.add_argument('--ctap_mode', type=str, default='full',
                       choices=['full', 'conf_only', 'tei_only'],
                       help='CTAP模式：full=完整CTAP, conf_only=仅置信度, tei_only=仅TEI优化')
    parser.add_argument('--confidence_window', type=int, default=5,
                       help='置信度分析的观测窗口大小')
    parser.add_argument('--confidence_prob', type=float, default=0.8,
                       help='置信度阈值（需要多少比例的观测一致才认为是高置信度）')
    parser.add_argument('--initial_threshold', type=float, default=-0.3,
                       help='初始Shapley阈值')
    parser.add_argument('--min_threshold', type=float, default=-0.5,
                       help='最小Shapley阈值（最激进剪枝）')
    parser.add_argument('--max_threshold', type=float, default=0.2,
                       help='最大Shapley阈值（最保守剪枝）')
    parser.add_argument('--tei_learning_rate', type=float, default=0.05,
                       help='TEI优化器的学习率（threshold调整步长）')
    parser.add_argument('--tei_patience', type=int, default=3,
                       help='TEI未改善的容忍次数（之后将探索新的threshold）')
    
    # 保存和日志
    parser.add_argument('--save_dir', type=str, default='./saved/online_shapley_run',
                       help='保存目录')
    parser.add_argument('--save_every', type=int, default=100,
                       help='保存checkpoint的间隔')
    parser.add_argument('--resume_from', type=str, default=None,
                       help='从checkpoint恢复训练的路径')
    
    # 环境相关
    parser.add_argument('--max_steps', type=int, default=40,
                       help='每个episode的最大步数')
    parser.add_argument('--nactions', default='1', type=str,
                       help='动作数量')
    
    # CommNet相关参数
    parser.add_argument('--commnet', action='store_true', default=False,
                       help="启用commnet模型")
    parser.add_argument('--ic3net', action='store_true', default=False,
                       help="启用ic3net模型")
    parser.add_argument('--tarcomm', action='store_true', default=False,
                       help="启用tarmac模型")
    parser.add_argument('--gacomm', action='store_true', default=False,
                       help="启用gacomm模型")
    parser.add_argument('--magic', action='store_true', default=False,
                       help="启用magic模型")
    parser.add_argument('--hard_attn', default=False, action='store_true',
                       help='是否使用hard attention')
    parser.add_argument('--comm_passes', type=int, default=1,
                       help="通信轮数")
    parser.add_argument('--comm_mask_zero', action='store_true', default=False,
                       help="是否阻断通信")
    parser.add_argument('--comm_mode', type=str, default='avg',
                       help="通信模式 [avg|sum]")
    parser.add_argument('--comm_init', default='uniform', type=str,
                       help='通信权重初始化 [uniform|zeros]')
    parser.add_argument('--share_weights', default=False, action='store_true',
                       help='是否共享权重')
    parser.add_argument('--advantages_per_action', default=False, action='store_true',
                       help='是否为每个动作计算优势')
    parser.add_argument('--comm_action_one', default=False, action='store_true',
                       help='是否总是通信（hard attention的sanity check）')
    parser.add_argument('--qk_hid_size', default=16, type=int,
                       help='key和query的大小（用于soft attention）')
    parser.add_argument('--value_hid_size', default=32, type=int,
                       help='value的大小（用于soft attention）')
    
    # MAGIC特定参数
    parser.add_argument('--directed', action='store_true', default=False,
                       help='是否使用有向通信图（MAGIC专用）')
    parser.add_argument('--self_loop_type1', default=2, type=int,
                       help='第一层GAT的自环类型（MAGIC专用）')
    parser.add_argument('--self_loop_type2', default=2, type=int,
                       help='第二层GAT的自环类型（MAGIC专用）')
    parser.add_argument('--gat_num_heads', default=1, type=int,
                       help='GAT层的注意力头数（MAGIC专用）')
    parser.add_argument('--gat_num_heads_out', default=1, type=int,
                       help='输出GAT层的注意力头数（MAGIC专用）')
    parser.add_argument('--ge_num_heads', default=4, type=int,
                       help='GAT编码器的注意力头数（MAGIC专用）')
    parser.add_argument('--first_gat_normalize', action='store_true', default=False,
                       help='是否归一化第一层GAT的系数（MAGIC专用）')
    parser.add_argument('--second_gat_normalize', action='store_true', default=False,
                       help='是否归一化第二层GAT的系数（MAGIC专用）')
    parser.add_argument('--gat_encoder_normalize', action='store_true', default=False,
                       help='是否归一化GAT编码器的系数（MAGIC专用）')
    parser.add_argument('--use_gat_encoder', action='store_true', default=False,
                       help='是否使用GAT编码器（MAGIC专用）')
    parser.add_argument('--gat_encoder_out_size', default=64, type=int,
                       help='GAT编码器输出大小（MAGIC专用）')
    parser.add_argument('--first_graph_complete', action='store_true', default=False,
                       help='第一个通信图是否为完全图（MAGIC专用）')
    parser.add_argument('--second_graph_complete', action='store_true', default=False,
                       help='第二个通信图是否为完全图（MAGIC专用）')
    parser.add_argument('--message_encoder', action='store_true', default=False,
                       help='是否使用消息编码器（MAGIC专用）')
    parser.add_argument('--message_decoder', action='store_true', default=False,
                       help='是否使用消息解码器（MAGIC专用）')
    
    # 优化参数
    parser.add_argument('--lrate', type=float, default=0.001,
                       help='学习率')
    parser.add_argument('--gamma', type=float, default=1.0,
                       help='折扣因子')
    parser.add_argument('--value_coeff', type=float, default=0.01,
                       help='value loss系数')
    parser.add_argument('--entr', type=float, default=0,
                       help='熵正则化系数')
    parser.add_argument('--normalize_rewards', action='store_true', default=False,
                       help='是否归一化奖励')
    
    # 其他
    parser.add_argument('--seed', type=int, default=-1,
                       help='随机种子')
    parser.add_argument('--mean_ratio', default=1.0, type=float,
                       help='合作程度')
    parser.add_argument('--detach_gap', default=10000, type=int,
                       help='detach间隔')
    parser.add_argument('--display', action='store_true', default=False,
                       help='是否显示环境')
    parser.add_argument('--plot', action='store_true', default=False,
                       help='是否绘制训练曲线')
    parser.add_argument('--save', action='store_true', default=True,
                       help='是否保存模型')
    parser.add_argument('--continuous', action='store_true', default=False,
                       help='是否使用连续动作空间')
    parser.add_argument('--random', action='store_true', default=False,
                       help='是否使用随机策略')
    parser.add_argument('--cuda', action='store_true', default=False,
                       help='是否使用 GPU 训练（需要 CUDA 可用）')

    # 从utils导入环境特定参数
    from utils import init_args_for_env
    init_args_for_env(parser)
    
    args = parser.parse_args()
    
    # 根据model_type设置相应的flag
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
    
    # Recurrence设置（和run_baselines.py保持一致）
    # 注意：baseline的trainer.py总是以recurrent方式调用模型
    # 但是CommNet在args.recurrent=False时，forward_state_encoder会正确处理
    # 只有当明确指定--recurrent或--rnn_type LSTM时才启用真正的LSTM
    if args.commnet and (args.recurrent or args.rnn_type == 'LSTM'):
        args.recurrent = True
        args.rnn_type = 'LSTM'
    else:
        # 确保默认是非recurrent
        args.recurrent = False
        if not hasattr(args, 'rnn_type') or args.rnn_type is None:
            args.rnn_type = 'MLP'
    
    return args


def create_model(args, num_inputs):
    """创建模型"""
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
    else:  # commnet 或 ic3net
        return CommNetMLP(args, num_inputs)


def train():
    """主训练函数"""
    args = parse_args()
    
    # 设置随机种子
    if args.seed != -1:
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)

    # 设置 device
    if args.cuda and torch.cuda.is_available():
        args.device = torch.device('cuda')
        torch.cuda.manual_seed(args.seed if args.seed != -1 else 0)
        # GPU 上使用 cuda.DoubleTensor 替换全局默认值
        torch.set_default_tensor_type('torch.cuda.DoubleTensor')
        print(f"使用 GPU: {torch.cuda.get_device_name(0)}")
    else:
        args.device = torch.device('cpu')
        torch.set_default_tensor_type('torch.DoubleTensor')
        if args.cuda:
            print("警告: 指定了 --cuda 但 CUDA 不可用，回退到 CPU")
        else:
            print("使用 CPU 训练")
    
    # 创建保存目录
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*80)
    print(f"训练配置 - 集成在线Shapley值分析")
    print("="*80)
    print(f"环境: {args.env_name}")
    print(f"模型类型: {args.model_type}")
    print(f"智能体数量: {args.nagents}")
    print(f"训练epochs: {args.num_epochs}")
    print(f"Shapley值计算间隔: {args.shapley_interval} epochs")
    print(f"Shapley值评估episodes: {args.shapley_episodes}")
    print(f"保存目录: {save_dir}")
    print("="*80 + "\n")
    
    # 初始化环境
    print(f"初始化环境: {args.env_name}")
    env = data.init(args.env_name, args, False)
    
    # 设置环境参数
    args.obs_size = env.observation_dim
    args.num_actions = env.num_actions
    if not isinstance(args.num_actions, (list, tuple)):
        args.num_actions = [args.num_actions]
    args.dim_actions = env.dim_actions
    args.nfriendly = args.nagents
    args.num_inputs = env.observation_dim
    
    # Hard attention
    if args.hard_attn and args.commnet:
        args.num_actions = [*args.num_actions, 2]
        args.dim_actions = env.dim_actions + 1
    
    parse_action_args(args)
    
    # 创建模型
    print(f"创建模型: {args.model_type}")
    policy_net = create_model(args, env.observation_dim)
    policy_net = policy_net.double().to(args.device)

    # 创建trainer（根据进程数选择）
    print(f"创建trainer (nprocesses={args.nprocesses})...")
    if args.nprocesses > 1:
        # 共享内存（多进程需要）
        for p in policy_net.parameters():
            p.data.share_memory_()
        trainer = MultiProcessTrainer(args, lambda: Trainer(args, policy_net, data.init(args.env_name, args)))
    else:
        trainer = Trainer(args, policy_net, env)
    
    # 创建在线Shapley值分析器
    shapley_analyzer = None
    if args.enable_shapley:
        print("创建在线Shapley值分析器...")
        shapley_analyzer = OnlineEdgeShapleyAnalyzer(
            args=args,
            env=env,
            n_eval_episodes=args.shapley_episodes,
            topology_threshold=args.topology_threshold
        )
    
    # 创建拓扑剪枝器（新增）
    topology_pruner = None
    if args.enable_pruning:
        print("创建拓扑剪枝器...")
        topology_pruner = TopologyPruner(args, args.nagents)
        
        # 如果启用剪枝但未启用Shapley值计算，给出警告
        if not args.enable_shapley:
            print("警告: 启用了剪枝但未启用Shapley值计算，剪枝功能将无法工作！")
    
    # 创建通信统计收集器（新增）
    comm_stats_collector = None
    comm_logger = None
    if args.enable_comm_stats or args.enable_pruning or args.enable_ctap:
        print("创建通信统计收集器...")
        comm_stats_collector = CommunicationStatsCollector(
            n_agents=args.nagents,
            save_dir=save_dir / 'communication_stats'
        )
        comm_logger = CommunicationLogger()
    
    # 创建CTAP剪枝器（新增）
    ctap_pruner = None
    if args.enable_ctap:
        print(f"创建CTAP剪枝器（模式: {args.ctap_mode}）...")
        from confidence_tei_pruner import CTAPruner
        ctap_pruner = CTAPruner(args, args.nagents)
        
        # 如果启用CTAP但未启用Shapley值计算，给出警告
        if not args.enable_shapley:
            print("警告: 启用了CTAP但未启用Shapley值计算，CTAP功能将无法工作！")
        
        # 如果同时启用了基本剪枝和CTAP，给出警告
        if args.enable_pruning:
            print("警告: 同时启用了基本剪枝和CTAP，将使用CTAP（基本剪枝将被忽略）")
            topology_pruner = None  # 禁用基本剪枝器
    
    # 训练日志
    log = {
        'epoch': [],
        'reward': [],
        'success': [],
        'steps_taken': [],
    }
    
    # 从checkpoint恢复（如果指定）
    start_epoch = 1
    num_episodes = 0
    
    if args.resume_from is not None:
        print("\n" + "="*80)
        print(f"从checkpoint恢复训练: {args.resume_from}")
        print("="*80)
        
        # PyTorch 2.6+需要设置weights_only=False来加载包含numpy对象的checkpoint
        checkpoint = torch.load(args.resume_from, weights_only=False)
        policy_net.load_state_dict(checkpoint['policy_net'])
        
        # 处理MultiProcessTrainer的情况
        if hasattr(trainer, 'trainer'):
            # MultiProcessTrainer: 优化器在trainer.trainer中
            trainer.trainer.optimizer.load_state_dict(checkpoint['optimizer'])
        else:
            # 单进程Trainer: 优化器直接在trainer中
            trainer.optimizer.load_state_dict(checkpoint['optimizer'])
        
        if 'log' in checkpoint:
            log = checkpoint['log']
        
        start_epoch = checkpoint['epoch'] + 1
        
        # 计算已完成的episodes数量
        num_episodes = checkpoint['epoch'] * args.epoch_size * args.nprocesses
        
        print(f"  恢复到Epoch {checkpoint['epoch']}")
        print(f"  从Epoch {start_epoch}继续训练")
        print(f"  已完成episodes: {num_episodes}")
        print("="*80 + "\n")
    
    # 开始训练
    print("\n" + "="*80)
    print("开始训练...")
    print("="*80 + "\n")
    
    for epoch in range(start_epoch, args.num_epochs + 1):
        epoch_begin_time = time.time()
        
        # **关键修复**: 每个epoch开始时强制确保模型处于训练模式
        # 这样即使Shapley计算有问题，也能在下一个epoch恢复
        policy_net.train()
        for module in policy_net.modules():
            module.train()
        
        # 每100个epoch验证一次模型状态
        if epoch % 100 == 0:
            is_training = policy_net.training
            if not is_training:
                print(f"警告: Epoch {epoch} 模型不在训练模式！正在修复...")
                policy_net.train()
                for module in policy_net.modules():
                    module.train()
        
        # 训练一个epoch
        epoch_stats = {'reward': 0, 'success': 0, 'steps_taken': 0, 'num_episodes': 0}
        
        # 获取当前的剪枝mask（优先使用CTAP，其次使用基本剪枝器）
        current_pruning_mask = None
        if ctap_pruner is not None:
            current_pruning_mask = ctap_pruner.get_mask()
        elif topology_pruner is not None:
            current_pruning_mask = topology_pruner.get_mask()
        
        epoch_total_comm_num = 0.0
        for _ in range(args.epoch_size):
            stat, batch_comm_stats = trainer.train_batch(epoch, pruning_mask=current_pruning_mask)

            # 累积统计
            epoch_stats['reward'] += stat.get('reward', 0)
            epoch_stats['success'] += stat.get('success', 0)
            epoch_stats['steps_taken'] += stat.get('steps_taken', 0)
            epoch_stats['num_episodes'] += stat.get('num_episodes', 0)

            # 累积 total_comm_num
            if isinstance(batch_comm_stats, dict) and 'round1' in batch_comm_stats:
                epoch_total_comm_num += batch_comm_stats['round1'].get('total_comm_num', 0.0)
        
        # 计算平均值
        if epoch_stats['num_episodes'] > 0:
            epoch_stats['reward'] /= epoch_stats['num_episodes']
            epoch_stats['success'] /= epoch_stats['num_episodes']
            epoch_stats['steps_taken'] /= epoch_stats['num_episodes']
        
        num_episodes += epoch_stats['num_episodes']
        epoch_time = time.time() - epoch_begin_time
        
        # 记录日志
        log['epoch'].append(epoch)
        log['reward'].append(epoch_stats['reward'])
        log['success'].append(epoch_stats['success'])
        log['steps_taken'].append(epoch_stats['steps_taken'])
        
        # 打印训练信息
        print(f"Epoch {epoch}/{args.num_epochs}")
        print(f"  Episodes: {num_episodes}")
        
        # 每100个epoch打印学习率
        if epoch % 100 == 0:
            if hasattr(trainer, 'trainer'):
                # MultiProcessTrainer
                current_lr = trainer.trainer.optimizer.param_groups[0]['lr']
            else:
                # 单进程Trainer
                current_lr = trainer.optimizer.param_groups[0]['lr']
            print(f"  学习率: {current_lr:.6f} (固定)")
        
        # 处理可能是数组的情况
        reward_val = np.mean(epoch_stats['reward']) if isinstance(epoch_stats['reward'], np.ndarray) else epoch_stats['reward']
        success_val = np.mean(epoch_stats['success']) if isinstance(epoch_stats['success'], np.ndarray) else epoch_stats['success']
        steps_val = np.mean(epoch_stats['steps_taken']) if isinstance(epoch_stats['steps_taken'], np.ndarray) else epoch_stats['steps_taken']
        print(f"  Reward: {float(reward_val):.4f}")
        print(f"  Success: {float(success_val):.4f}")
        print(f"  Steps-Taken: {float(steps_val):.2f}")
        print(f"  Total Comm Num: {epoch_total_comm_num:.1f}")
        print(f"  Time: {epoch_time:.2f}s")
        
        # 在线Shapley值计算
        if shapley_analyzer is not None:
            shapley_results = integrate_with_training(
                trainer=trainer,
                shapley_analyzer=shapley_analyzer,
                epoch=epoch,
                shapley_interval=args.shapley_interval
            )
            
            if shapley_results is not None:
                # 保存当前的Shapley值结果
                shapley_analyzer.save_results(save_dir / 'shapley_results')
                
                # 可选：打印Shapley值矩阵
                shapley_analyzer.print_shapley_matrix(epoch)
                
                # 【新增】基于Shapley值更新剪枝mask
                if topology_pruner is not None:
                    # 获取Shapley值矩阵
                    shapley_matrix = shapley_analyzer.get_shapley_matrix(epoch)
                    
                    # 更新剪枝mask
                    if topology_pruner.should_update_mask(epoch):
                        pruning_mask = topology_pruner.update_mask(shapley_matrix, epoch)
                        
                        # 保存剪枝历史
                        topology_pruner.save_history(save_dir / 'pruning_results')
                
                # 【新增】CTAP剪枝更新
                if ctap_pruner is not None:
                    # 获取Shapley值矩阵
                    shapley_matrix = shapley_analyzer.get_shapley_matrix(epoch)
                    
                    # 判断是否需要更新
                    if ctap_pruner.should_update(epoch):
                        # 计算当前的通信密度
                        # 优先使用通信日志；若日志为空（未集成到trainer），则用当前mask密度估计
                        current_comm_density = 0.0
                        if comm_stats_collector is not None:
                            comm_history = comm_logger.get_and_reset() if comm_logger else []
                            if len(comm_history) > 0:
                                temp_stats = comm_stats_collector.compute_epoch_stats(
                                    comm_history=comm_history,
                                    success_rate=float(success_val),
                                    epoch=epoch
                                )
                                current_comm_density = temp_stats.get('comm_density', 0.0)
                        
                        # 若comm_density仍为0（通信日志未接入），使用mask密度作为代理
                        # mask密度 = 当前保留边数 / 最大可能边数，反映实际通信负载
                        if current_comm_density == 0.0:
                            current_mask = ctap_pruner.get_mask()
                            n = args.nagents
                            max_edges = n * (n - 1)
                            current_comm_density = float(current_mask.sum()) / max_edges if max_edges > 0 else 1.0
                        
                        # 更新CTAP剪枝mask
                        update_info = ctap_pruner.update_mask(
                            shapley_matrix=shapley_matrix,
                            epoch=epoch,
                            success_rate=float(success_val),
                            comm_density=current_comm_density
                        )
                        
                        # 打印详细报告
                        print("\n" + "="*80)
                        print(ctap_pruner.get_detailed_report(epoch))
                        print("="*80 + "\n")
                        
                        # 定期保存分析结果（每500 epoch）
                        if epoch % 500 == 0 or epoch == args.pruning_start_epoch:
                            ctap_pruner.save_analysis(save_dir / 'ctap_analysis', epoch)
        
        # 【新增】通信统计（如果启用）
        if comm_stats_collector is not None and comm_logger is not None:
            # 获取当前epoch的通信历史
            comm_history = comm_logger.get_and_reset()
            
            # 计算统计
            if len(comm_history) > 0:
                comm_stats = comm_stats_collector.compute_epoch_stats(
                    comm_history=comm_history,
                    success_rate=float(success_val),
                    epoch=epoch
                )
                
                # 每100 epoch打印统计
                if epoch % 100 == 0:
                    comm_stats_collector.print_stats(comm_stats)
                
                # 保存统计
                comm_stats_collector.save_stats()
        
        # 保存checkpoint
        if epoch % args.save_every == 0:
            checkpoint = {
                'epoch': epoch,
                'policy_net': policy_net.state_dict(),
                'optimizer': trainer.state_dict(),
                'log': log,
            }
            # 确保保存目录存在
            save_dir.mkdir(parents=True, exist_ok=True)
            checkpoint_path = save_dir / f'model_ep{epoch}.pt'
            try:
                # 将Path对象转换为字符串
                torch.save(checkpoint, str(checkpoint_path))
                print(f"  Checkpoint已保存: model_ep{epoch}.pt")
            except Exception as e:
                print(f"  警告: 保存checkpoint失败: {e}")
                print(f"  尝试保存到备用位置...")
                # 尝试保存到当前目录
                backup_path = Path(f'model_ep{epoch}.pt')
                torch.save(checkpoint, str(backup_path))
                print(f"  Checkpoint已保存到备用位置: {backup_path}")
        
        # **学习率调度器已禁用 - 保持固定学习率**
        # 不再使用学习率调度器，学习率固定为初始值
        # if hasattr(trainer, 'scheduler') and trainer.scheduler is not None:
        #     trainer.scheduler.step()
        #     current_lr = trainer.optimizer.param_groups[0]['lr']
        #     if epoch % 500 == 0:
        #         print(f"  当前学习率: {current_lr:.6f}")
        
        print()
    
    # 训练结束
    print("="*80)
    print("训练完成！")
    print("="*80)
    
    # 保存最终模型
    final_checkpoint = {
        'epoch': args.num_epochs,
        'policy_net': policy_net.state_dict(),
        'optimizer': trainer.state_dict(),
        'log': log,
    }
    # 确保保存目录存在
    save_dir.mkdir(parents=True, exist_ok=True)
    final_path = save_dir / 'model_final.pt'
    try:
        # 将Path对象转换为字符串
        torch.save(final_checkpoint, str(final_path))
        print(f"最终模型已保存: {final_path}")
    except Exception as e:
        print(f"警告: 保存最终模型失败: {e}")
        backup_path = Path('model_final.pt')
        torch.save(final_checkpoint, str(backup_path))
        print(f"最终模型已保存到备用位置: {backup_path}")
    
    # 保存最终的Shapley值结果
    if shapley_analyzer is not None:
        shapley_analyzer.save_results(save_dir / 'shapley_results')
        print(f"Shapley值结果已保存: {save_dir / 'shapley_results'}")
        
        # 打印统计信息
        if shapley_analyzer.computation_times:
            print(f"\nShapley值计算统计:")
            print(f"  - 总计算次数: {len(shapley_analyzer.computation_times)}")
            print(f"  - 平均计算时间: {np.mean(shapley_analyzer.computation_times):.2f}秒")
            print(f"  - 总计算时间: {np.sum(shapley_analyzer.computation_times):.2f}秒")
            print(f"  - 占总训练时间比例: 估计~{np.sum(shapley_analyzer.computation_times) / (time.time() - epoch_begin_time) * 100:.1f}%")


if __name__ == '__main__':
    train()

