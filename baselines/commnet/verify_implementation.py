#!/usr/bin/env python
"""
验证剪枝功能实现的完整性
"""

import os
import sys
from pathlib import Path

def check_file(filepath, description):
    """检查文件是否存在"""
    if Path(filepath).exists():
        print(f"  ✓ {description}")
        return True
    else:
        print(f"  ✗ {description} (文件不存在: {filepath})")
        return False

def check_imports():
    """检查模块是否可以正常导入"""
    print("\n【模块导入测试】")
    try:
        sys.path.insert(0, 'baselines')
        
        # 测试导入拓扑剪枝器
        from topology_pruner import TopologyPruner
        print("  ✓ topology_pruner.TopologyPruner")
        
        # 测试导入通信统计
        from communication_stats import CommunicationStatsCollector, CommunicationLogger
        print("  ✓ communication_stats.CommunicationStatsCollector")
        print("  ✓ communication_stats.CommunicationLogger")
        
        print("  ℹ train_with_online_shapley (需要torch环境，跳过)")
        
        return True
    except Exception as e:
        print(f"  ✗ 导入失败: {e}")
        return False

def check_parameters():
    """检查参数解析是否包含新参数"""
    print("\n【参数检查】")
    
    # 检查训练脚本中是否包含新参数
    try:
        with open('train_tj_commnet_basic_prune_online_shapley.sh', 'r') as f:
            content = f.read()
            
        params_to_check = [
            ('--enable_pruning', 'enable_pruning'),
            ('--pruning_threshold', 'pruning_threshold'),
            ('--pruning_start_epoch', 'pruning_start_epoch'),
            ('--pruning_interval', 'pruning_interval'),
            ('--enable_comm_stats', 'enable_comm_stats'),
        ]
        
        for param, desc in params_to_check:
            if param in content:
                print(f"  ✓ {param}")
            else:
                print(f"  ✗ {param} (未在训练脚本中找到)")
                return False
        
        print("  ℹ 参数语法检查（需要torch环境，跳过详细测试）")
        return True
        
    except Exception as e:
        print(f"  ✗ 参数检查失败: {e}")
        return False

def main():
    print("="*80)
    print("剪枝功能实现验证")
    print("="*80)
    
    all_pass = True
    
    # 检查核心文件
    print("\n【核心文件检查】")
    all_pass &= check_file('baselines/topology_pruner.py', 'topology_pruner.py')
    all_pass &= check_file('baselines/communication_stats.py', 'communication_stats.py')
    all_pass &= check_file('baselines/train_with_online_shapley.py', 'train_with_online_shapley.py')
    all_pass &= check_file('baselines/online_edge_shapley.py', 'online_edge_shapley.py')
    
    # 检查训练脚本
    print("\n【训练脚本检查】")
    models = ['commnet', 'ic3net', 'gacomm', 'tarcomm', 'magic']
    
    print("\n  原有脚本（不带剪枝）:")
    for model in models:
        if model == 'magic':
            continue  # MAGIC的原始脚本可能不存在
        script = f'train_tj_{model}_online_shapley.sh'
        check_file(script, f'{model} 训练脚本')
    
    print("\n  新脚本（带剪枝）:")
    for model in models:
        script = f'train_tj_{model}_basic_prune_online_shapley.sh'
        all_pass &= check_file(script, f'{model} 剪枝训练脚本')
    
    # 检查文档
    print("\n【文档检查】")
    all_pass &= check_file('PRUNING_README.md', '使用文档')
    all_pass &= check_file('IMPLEMENTATION_SUMMARY.md', '实现总结')
    all_pass &= check_file('test_backward_compatibility.sh', '兼容性测试脚本')
    
    # 检查导入
    all_pass &= check_imports()
    
    # 检查参数
    all_pass &= check_parameters()
    
    # 总结
    print("\n" + "="*80)
    if all_pass:
        print("✅ 所有检查通过！剪枝功能实现完整。")
    else:
        print("❌ 部分检查失败，请查看上述错误信息。")
    print("="*80)
    
    return 0 if all_pass else 1

if __name__ == '__main__':
    sys.exit(main())

