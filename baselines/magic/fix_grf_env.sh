#!/bin/bash

echo "================================================"
echo "修复 GRF 环境配置问题"
echo "================================================"

# 保存当前目录
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

# 1. 卸载旧的 grf_envs 包
echo ""
echo "步骤 1: 卸载旧的 grf_envs 包..."
pip uninstall -y grf_envs 2>/dev/null || echo "  (包未安装，跳过)"

# 2. 清理 Python 缓存
echo ""
echo "步骤 2: 清理 Python 缓存..."
find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
find . -type f -name "*.pyc" -delete 2>/dev/null || true
find . -type f -name "*.pyo" -delete 2>/dev/null || true
echo "  ✓ 缓存已清理"

# 3. 重新安装 grf_envs（使用当前目录）
echo ""
echo "步骤 3: 重新安装 grf_envs 包..."
cd "envs/grf-envs" || exit 1
python setup.py develop
if [ $? -eq 0 ]; then
    echo "  ✓ 安装成功"
else
    echo "  ✗ 安装失败"
    exit 1
fi
cd "$SCRIPT_DIR"

# 4. 验证安装
echo ""
echo "步骤 4: 验证安装..."
python -c "import grf_envs; print('  ✓ grf_envs 模块路径:', grf_envs.__file__)"
if [ $? -ne 0 ]; then
    echo "  ✗ 验证失败"
    exit 1
fi

echo ""
echo "================================================"
echo "✓ 修复完成！现在可以运行 sh train_grf.sh"
echo "================================================"

