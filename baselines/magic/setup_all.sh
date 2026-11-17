#!/bin/bash

echo "================================================================"
echo "MAGIC-GRF 完整环境配置脚本"
echo "================================================================"

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# 错误计数
ERRORS=0

echo ""
echo "此脚本将执行以下操作："
echo "  1. 修复 grf_envs 模块"
echo "  2. 安装作者修改版的 Google Research Football"
echo "  3. 验证所有依赖"
echo ""
read -p "是否继续？(y/n) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "取消安装"
    exit 0
fi

# ===== 步骤 1: 修复 grf_envs =====
echo ""
echo "================================================================"
echo "步骤 1/3: 修复 grf_envs 模块"
echo "================================================================"

echo "卸载旧的 grf_envs 包..."
pip uninstall -y grf_envs 2>/dev/null || true

echo "清理 Python 缓存..."
find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
find . -type f -name "*.pyc" -delete 2>/dev/null || true

echo "重新安装 grf_envs..."
cd "envs/grf-envs" || exit 1
python setup.py develop
if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ grf_envs 安装成功${NC}"
else
    echo -e "${RED}✗ grf_envs 安装失败${NC}"
    ERRORS=$((ERRORS+1))
fi
cd "$SCRIPT_DIR"

# ===== 步骤 2: 安装作者修改版 gfootball =====
echo ""
echo "================================================================"
echo "步骤 2/3: 安装作者修改版 Google Research Football"
echo "================================================================"

echo "检查系统依赖..."
echo "需要以下系统包："
echo "  git, cmake, build-essential, libgl1-mesa-dev, libsdl2-dev,"
echo "  libsdl2-image-dev, libsdl2-ttf-dev, libsdl2-gfx-dev,"
echo "  libboost-all-dev, libdirectfb-dev, libst-dev,"
echo "  mesa-utils, xvfb, x11vnc, libsdl-sge-dev"
echo ""
read -p "是否需要安装系统依赖？(y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "安装系统依赖..."
    sudo apt-get update
    sudo apt-get install -y git cmake build-essential libgl1-mesa-dev libsdl2-dev \
        libsdl2-image-dev libsdl2-ttf-dev libsdl2-gfx-dev libboost-all-dev \
        libdirectfb-dev libst-dev mesa-utils xvfb x11vnc libsdl-sge-dev python3-pip
    if [ $? -ne 0 ]; then
        echo -e "${RED}✗ 系统依赖安装失败${NC}"
        ERRORS=$((ERRORS+1))
    fi
fi

echo ""
echo "卸载官方版本的 gfootball..."
pip uninstall -y gfootball 2>/dev/null || true

echo ""
echo "克隆作者修改版的 football 仓库..."
cd /tmp
if [ -d "football" ]; then
    rm -rf football
fi

git clone https://github.com/chrisyrniu/football.git
if [ $? -ne 0 ]; then
    echo -e "${RED}✗ 克隆失败！请检查网络连接${NC}"
    echo "可以尝试手动下载："
    echo "  wget https://github.com/chrisyrniu/football/archive/refs/heads/master.zip"
    echo "  unzip master.zip"
    echo "  cd football-master"
    echo "  pip install ."
    ERRORS=$((ERRORS+1))
else
    cd football
    echo "安装 gfootball..."
    pip install .
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}✓ gfootball 安装成功${NC}"
    else
        echo -e "${RED}✗ gfootball 安装失败${NC}"
        ERRORS=$((ERRORS+1))
    fi
fi

cd "$SCRIPT_DIR"

# ===== 步骤 3: 验证安装 =====
echo ""
echo "================================================================"
echo "步骤 3/3: 验证安装"
echo "================================================================"

python check_dependencies.py

# ===== 总结 =====
echo ""
echo "================================================================"
echo "安装完成"
echo "================================================================"

if [ $ERRORS -eq 0 ]; then
    echo -e "${GREEN}✓ 所有步骤完成！${NC}"
    echo ""
    echo "现在可以运行训练："
    echo "  sh train_grf.sh"
else
    echo -e "${RED}✗ 安装过程中出现 $ERRORS 个错误${NC}"
    echo ""
    echo "请查看上面的错误信息并手动修复"
    echo "或查看 GRF_环境修复指南.md 获取详细帮助"
fi

echo ""

