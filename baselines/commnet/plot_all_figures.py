#!/usr/bin/env python3
"""
一键生成所有可视化图表
用于Basic Prune实验结果分析
"""

import os
import sys
import argparse
import subprocess
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description='生成所有可视化图表')
    parser.add_argument('--date', type=str, default='20260122',
                       help='日志文件日期 (默认: 20260122)')
    parser.add_argument('--format', type=str, default='pdf',
                       choices=['pdf', 'png', 'both'],
                       help='输出格式 (默认: pdf)')
    parser.add_argument('--dpi', type=int, default=300,
                       help='图片分辨率 (默认: 300)')
    parser.add_argument('--style', type=str, default='seaborn-v0_8-paper',
                       help='绘图风格 (默认: seaborn-v0_8-paper)')
    args = parser.parse_args()
    
    # 创建输出目录
    output_dir = Path('figures')
    output_dir.mkdir(exist_ok=True)
    
    print("="*80)
    print("开始生成所有可视化图表")
    print("="*80)
    print(f"日志日期: {args.date}")
    print(f"输出格式: {args.format}")
    print(f"分辨率: {args.dpi} DPI")
    print(f"绘图风格: {args.style}")
    print("="*80)
    print()
    
    # 绘图脚本列表
    scripts = [
        ('plot_training_comparison.py', '训练性能对比'),
        ('plot_communication_analysis.py', '通信效率分析'),
        ('plot_shapley_analysis.py', 'Shapley值分析'),
        ('plot_pruning_effects.py', '剪枝效果分析'),
        ('plot_tei_metrics.py', 'TEI指标分析'),
        ('plot_comprehensive_radar.py', '综合对比雷达图'),
    ]
    
    success_count = 0
    failed_scripts = []
    
    for script, description in scripts:
        print(f"[{success_count+1}/{len(scripts)}] 运行: {description} ({script})")
        print("-"*80)
        
        try:
            cmd = [
                'python3', script,
                '--date', args.date,
                '--format', args.format,
                '--dpi', str(args.dpi),
                '--style', args.style,
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
            
            if result.returncode == 0:
                print(f"✓ {description} 完成")
                success_count += 1
            else:
                print(f"✗ {description} 失败")
                print(f"错误信息: {result.stderr}")
                failed_scripts.append((script, description))
        
        except FileNotFoundError:
            print(f"⚠ {script} 不存在，跳过")
            failed_scripts.append((script, description))
        
        except subprocess.TimeoutExpired:
            print(f"✗ {description} 超时")
            failed_scripts.append((script, description))
        
        except Exception as e:
            print(f"✗ {description} 异常: {e}")
            failed_scripts.append((script, description))
        
        print()
    
    # 输出总结
    print("="*80)
    print("生成完成")
    print("="*80)
    print(f"成功: {success_count}/{len(scripts)}")
    print(f"失败: {len(failed_scripts)}/{len(scripts)}")
    
    if failed_scripts:
        print()
        print("失败的脚本:")
        for script, description in failed_scripts:
            print(f"  - {script} ({description})")
    
    print()
    print(f"图表保存位置: {output_dir.absolute()}/")
    print("="*80)
    
    return 0 if len(failed_scripts) == 0 else 1

if __name__ == '__main__':
    sys.exit(main())






