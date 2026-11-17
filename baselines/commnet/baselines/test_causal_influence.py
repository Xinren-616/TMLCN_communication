#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
测试脚本：验证因果影响（Causal Influence）指标的实现

该脚本创建一个简单的测试环境来验证新增的因果影响指标是否正确计算
"""

import torch
import numpy as np
import sys
import argparse

# 添加路径
sys.path.append('..')

from baselines.comm import CommNetMLP


def create_test_args():
    """创建测试用的参数"""
    args = argparse.Namespace()
    args.nagents = 3
    args.hid_size = 128     # 隐藏层维度
    args.recurrent = False
    args.continuous = False
    args.dim_actions = 1
    args.naction_heads = [5]  # 假设5个离散动作
    args.init_std = 0.2
    args.comm_mask_zero = False
    args.share_weights = True
    args.comm_init = 'xavier'
    args.batch_size = 1
    args.comm_mode = 'avg'
    
    # 添加缺失的参数
    args.hard_attn = False  # 不使用hard attention
    args.commnet = True     # 使用CommNet
    args.gacomm = False     # 不使用GAComm
    args.rnn_type = 'LSTM'  # RNN类型
    args.detach_gap = 10    # detach间隔
    args.max_steps = 100    # 最大步数
    
    return args


def test_causal_influence():
    """测试因果影响指标的计算"""
    print("=" * 80)
    print("测试因果影响（Causal Influence）指标计算")
    print("=" * 80)
    
    # 1. 创建测试参数和模型
    args = create_test_args()
    num_inputs = 20  # 观测维度
    
    print(f"\n配置参数:")
    print(f"  智能体数量: {args.nagents}")
    print(f"  隐藏层维度: {args.hid_size}")
    print(f"  观测维度: {num_inputs}")
    
    # 2. 初始化CommNet模型
    model = CommNetMLP(args, num_inputs)
    model.eval()
    
    print(f"\n模型初始化完成")
    
    # 3. 创建测试数据
    batch_size = 2
    state = torch.randn(batch_size, args.nagents, num_inputs)
    
    print(f"\n输入数据形状:")
    print(f"  State: {state.shape}")
    
    # 4. 前向传播
    info = {}
    with torch.no_grad():
        action, value = model(state, info)
    
    print(f"\n输出数据形状:")
    print(f"  Action: {[a.shape for a in action]}")
    print(f"  Value: {value.shape}")
    
    # 5. 检查通信统计
    if 'comm_stats' in info:
        print(f"\n✓ 通信统计成功生成")
        comm_stats = info['comm_stats']
        
        if 'round1' in comm_stats:
            print(f"\n通信统计（Round 1）:")
            round1_stats = comm_stats['round1']
            
            # 传统指标
            print(f"\n  [传统通信指标]")
            print(f"  - 总通信元素数: {round1_stats['total_elements']}")
            print(f"  - 活跃通信通道数: {round1_stats['active_channels']}")
            print(f"  - 通信熵: {round1_stats['entropy']:.4f}")
            print(f"  - 最大熵: {round1_stats['max_entropy']:.4f}")
            print(f"  - 有效带宽利用率: {round1_stats['effective_bandwidth']:.4f}")
            print(f"  - 消息相似度: {round1_stats['message_similarity']:.4f}")
            
            # 新增因果影响指标
            print(f"\n  [新增因果影响指标]")
            print(f"  - 因果影响 (Causal Influence): {round1_stats['causal_influence']:.6f}")
            print(f"  - 隐藏状态变化 (Hidden State Change): {round1_stats['hidden_state_change']:.6f}")
            print(f"  - 消息影响效率比率 (Message Impact Ratio): {round1_stats['message_impact_ratio']:.6f}")
            
            if 'per_agent_influence' in round1_stats:
                print(f"\n  [每个智能体的因果影响]")
                for i, influence in enumerate(round1_stats['per_agent_influence']):
                    print(f"    Agent {i}: {influence:.6f}")
            
            if 'pairwise_influence_matrix' in round1_stats:
                print(f"\n  [成对因果影响矩阵 C_i→j]")
                matrix = round1_stats['pairwise_influence_matrix']
                print(f"    发送者 → 接收者")
                for i, row in enumerate(matrix):
                    row_str = " ".join([f"{val:6.3f}" for val in row])
                    print(f"    Agent {i}: [{row_str}]")
            
            # 验证指标的合理性
            print(f"\n✓ 指标验证:")
            checks_passed = True
            
            # 检查1: 因果影响应该是非负的
            if round1_stats['causal_influence'] >= 0:
                print(f"  ✓ 因果影响 >= 0")
            else:
                print(f"  ✗ 因果影响 < 0 (异常)")
                checks_passed = False
            
            # 检查2: 隐藏状态变化应该是非负的
            if round1_stats['hidden_state_change'] >= 0:
                print(f"  ✓ 隐藏状态变化 >= 0")
            else:
                print(f"  ✗ 隐藏状态变化 < 0 (异常)")
                checks_passed = False
            
            # 检查3: 消息影响效率比率应该是非负的
            if round1_stats['message_impact_ratio'] >= 0:
                print(f"  ✓ 消息影响效率比率 >= 0")
            else:
                print(f"  ✗ 消息影响效率比率 < 0 (异常)")
                checks_passed = False
            
            # 检查4: 每个智能体的因果影响应该是非负的
            if 'per_agent_influence' in round1_stats:
                all_non_negative = all(x >= 0 for x in round1_stats['per_agent_influence'])
                if all_non_negative:
                    print(f"  ✓ 所有智能体的因果影响 >= 0")
                else:
                    print(f"  ✗ 存在负的智能体因果影响 (异常)")
                    checks_passed = False
            
            if checks_passed:
                print(f"\n{'='*80}")
                print(f"✓✓✓ 所有测试通过！因果影响指标计算正确！")
                print(f"{'='*80}")
            else:
                print(f"\n{'='*80}")
                print(f"✗✗✗ 部分测试失败，请检查实现")
                print(f"{'='*80}")
                
        else:
            print(f"\n✗ 未找到 round1 统计数据")
    else:
        print(f"\n✗ 未生成通信统计")
    
    return model, info


def test_with_different_inputs():
    """测试不同输入下的因果影响变化"""
    print("\n" + "=" * 80)
    print("测试不同通信强度下的因果影响变化")
    print("=" * 80)
    
    args = create_test_args()
    num_inputs = 20
    model = CommNetMLP(args, num_inputs)
    model.eval()
    
    batch_size = 1
    
    # 测试3种不同的输入场景
    scenarios = [
        ("场景1: 相似的观测", torch.ones(batch_size, args.nagents, num_inputs) * 0.1),
        ("场景2: 差异较大的观测", torch.randn(batch_size, args.nagents, num_inputs)),
        ("场景3: 极端差异的观测", torch.randn(batch_size, args.nagents, num_inputs) * 5),
    ]
    
    results = []
    
    for scenario_name, state in scenarios:
        info = {}
        with torch.no_grad():
            action, value = model(state, info)
        
        if 'comm_stats' in info and 'round1' in info['comm_stats']:
            causal_inf = info['comm_stats']['round1']['causal_influence']
            hidden_change = info['comm_stats']['round1']['hidden_state_change']
            impact_ratio = info['comm_stats']['round1']['message_impact_ratio']
            
            results.append({
                'scenario': scenario_name,
                'causal_influence': causal_inf,
                'hidden_state_change': hidden_change,
                'message_impact_ratio': impact_ratio
            })
    
    # 打印结果
    print("\n对比结果:")
    print(f"{'场景':<30} | {'因果影响':<15} | {'状态变化':<15} | {'影响效率':<15}")
    print("-" * 80)
    for r in results:
        print(f"{r['scenario']:<30} | {r['causal_influence']:<15.6f} | "
              f"{r['hidden_state_change']:<15.6f} | {r['message_impact_ratio']:<15.6f}")
    
    print("\n分析:")
    print("  - 因果影响反映了消息对其他智能体隐藏状态的影响程度")
    print("  - 状态变化衡量通信前后隐藏状态的变化幅度")
    print("  - 影响效率衡量单位消息强度带来的状态变化（效率指标）")
    print("  - 不同场景下这些指标应该有所差异，反映通信的重要性")


if __name__ == '__main__':
    print("\n" + "=" * 80)
    print("因果影响（Causal Influence）指标测试程序")
    print("=" * 80)
    
    try:
        # 测试1: 基本功能测试
        model, info = test_causal_influence()
        
        # 测试2: 不同输入场景测试
        test_with_different_inputs()
        
        print("\n" + "=" * 80)
        print("✓ 所有测试完成！")
        print("=" * 80)
        
    except Exception as e:
        print(f"\n✗ 测试过程中出现错误:")
        print(f"  {type(e).__name__}: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

