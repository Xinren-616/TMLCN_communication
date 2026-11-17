#!/usr/bin/env python
"""
检查 GRF 训练所需的所有依赖
"""

import sys

print("="*60)
print("检查 MAGIC-GRF 训练依赖")
print("="*60)

errors = []
warnings = []

# 1. 检查 Python 版本
print("\n[1] 检查 Python 版本...")
python_version = sys.version_info
print(f"    Python {python_version.major}.{python_version.minor}.{python_version.micro}")
if python_version.major != 3 or python_version.minor < 6:
    errors.append("需要 Python 3.6 或更高版本")
else:
    print("    ✓ 版本符合要求")

# 2. 检查 PyTorch
print("\n[2] 检查 PyTorch...")
try:
    import torch
    print(f"    PyTorch 版本: {torch.__version__}")
    print("    ✓ PyTorch 已安装")
except ImportError:
    errors.append("PyTorch 未安装")

# 3. 检查 Gym
print("\n[3] 检查 OpenAI Gym...")
try:
    import gym
    print(f"    Gym 版本: {gym.__version__}")
    print("    ✓ Gym 已安装")
except ImportError:
    errors.append("Gym 未安装")

# 4. 检查 NumPy
print("\n[4] 检查 NumPy...")
try:
    import numpy
    print(f"    NumPy 版本: {numpy.__version__}")
    print("    ✓ NumPy 已安装")
except ImportError:
    errors.append("NumPy 未安装")

# 5. 检查 gfootball（关键）
print("\n[5] 检查 Google Research Football...")
try:
    import gfootball
    print(f"    gfootball 路径: {gfootball.__file__}")
    
    # 检查是否支持 multiagent representation
    from gfootball.env import create_environment
    import inspect
    sig = inspect.signature(create_environment)
    
    # 尝试创建一个测试环境（不实际运行）
    print("    ✓ gfootball 已安装")
    
    # 检查是否是作者修改版
    print("\n    检查是否支持 multiagent representation...")
    print("    提示：如果下面的测试失败，说明安装的是官方版本")
    print("    需要安装作者修改版：https://github.com/chrisyrniu/football")
    
except ImportError:
    errors.append("gfootball 未安装 - 运行 bash install_gfootball.sh")
except Exception as e:
    warnings.append(f"gfootball 检查时出现警告: {e}")

# 6. 检查 ic3net_envs
print("\n[6] 检查 ic3net_envs...")
try:
    import ic3net_envs
    print(f"    ic3net_envs 路径: {ic3net_envs.__file__}")
    print("    ✓ ic3net_envs 已安装")
except ImportError:
    warnings.append("ic3net_envs 未安装（如果只运行 GRF 任务，可以忽略）")
    print("    ⚠ ic3net_envs 未安装")
    print("    如需运行 Predator-Prey 或 Traffic Junction:")
    print("      cd envs/ic3net-envs && python setup.py develop")

# 7. 检查 grf_envs（关键）
print("\n[7] 检查 grf_envs...")
try:
    import grf_envs
    print(f"    grf_envs 路径: {grf_envs.__file__}")
    
    # 检查路径是否正确
    import os
    current_dir = os.getcwd()
    expected_path = os.path.join(current_dir, "envs", "grf-envs", "grf_envs")
    
    if expected_path.lower().replace('\\', '/') in grf_envs.__file__.lower().replace('\\', '/'):
        print("    ✓ grf_envs 路径正确")
    else:
        warnings.append(f"grf_envs 路径可能不正确")
        print(f"    ⚠ 当前路径: {grf_envs.__file__}")
        print(f"    ⚠ 期望路径: {expected_path}")
        print("    建议运行: bash fix_grf_env.sh")
    
    # 检查能否注册环境
    try:
        env = gym.make('GRFWrapper-v0')
        print("    ✓ GRFWrapper-v0 环境注册成功")
    except Exception as e:
        errors.append(f"GRFWrapper-v0 环境注册失败: {e}")
        
except ImportError:
    errors.append("grf_envs 未安装 - 运行 bash fix_grf_env.sh")

# 8. 总结
print("\n" + "="*60)
print("检查结果总结")
print("="*60)

if not errors and not warnings:
    print("\n✓ 所有依赖检查通过！可以运行训练了：")
    print("  sh train_grf.sh")
else:
    if errors:
        print(f"\n✗ 发现 {len(errors)} 个错误：")
        for i, err in enumerate(errors, 1):
            print(f"  {i}. {err}")
    
    if warnings:
        print(f"\n⚠ 发现 {len(warnings)} 个警告：")
        for i, warn in enumerate(warnings, 1):
            print(f"  {i}. {warn}")
    
    print("\n建议的修复步骤：")
    print("  1. 运行: bash fix_grf_env.sh")
    print("  2. 运行: bash install_gfootball.sh")
    print("  3. 重新运行此检查: python check_dependencies.py")

print("\n" + "="*60)

