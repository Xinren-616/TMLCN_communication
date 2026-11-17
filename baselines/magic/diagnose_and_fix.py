#!/usr/bin/env python
"""
诊断并修复 GRF 环境配置问题
"""

import sys
import os
import subprocess
import site

print("="*60)
print("诊断 GRF 环境配置问题")
print("="*60)

# 1. 检查当前 grf_envs 的安装位置
print("\n步骤 1: 检查 grf_envs 模块位置...")
try:
    import grf_envs
    print(f"✓ grf_envs 已安装")
    print(f"  路径: {grf_envs.__file__}")
    
    # 检查是否是正确的路径
    current_dir = os.getcwd()
    expected_path = os.path.join(current_dir, "envs", "grf-envs", "grf_envs")
    
    if expected_path.lower() in grf_envs.__file__.lower().replace('\\', '/'):
        print(f"✓ 使用的是正确的模块路径")
    else:
        print(f"✗ 使用的是错误的模块路径！")
        print(f"  期望路径: {expected_path}")
        print(f"  实际路径: {os.path.dirname(grf_envs.__file__)}")
        print(f"\n需要重新安装...")
        
except ImportError:
    print("✗ grf_envs 未安装")

# 2. 检查 grf_wrapper_env.py 的内容
print("\n步骤 2: 检查 grf_wrapper_env.py...")
wrapper_path = os.path.join("envs", "grf-envs", "grf_envs", "grf_wrapper_env.py")
if os.path.exists(wrapper_path):
    with open(wrapper_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    if "import gfootball.env as grf_env" in content.split('\n')[0:5]:
        print("✗ 文件仍在模块级别导入 gfootball（第1-5行）")
    else:
        print("✓ 文件已正确修改（延迟导入 gfootball）")
else:
    print(f"✗ 找不到文件: {wrapper_path}")

# 3. 提供修复建议
print("\n" + "="*60)
print("修复建议")
print("="*60)
print("\n请按以下步骤操作：")
print("\n1. 卸载旧的 grf_envs 包：")
print("   pip uninstall -y grf_envs")
print("\n2. 清理缓存：")
print("   find . -type d -name '__pycache__' -exec rm -rf {} + 2>/dev/null")
print("   find . -type f -name '*.pyc' -delete 2>/dev/null")
print("\n3. 重新安装（开发模式）：")
print("   cd envs/grf-envs")
print("   python setup.py develop")
print("   cd ../..")
print("\n4. 验证安装：")
print("   python -c \"import grf_envs; print(grf_envs.__file__)\"")
print("\n5. 运行训练：")
print("   sh train_grf.sh")
print("\n" + "="*60)

