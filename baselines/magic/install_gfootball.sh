#!/bin/bash

echo "================================================"
echo "安装作者修改版的 Google Research Football"
echo "================================================"

# 1. 检查并安装系统依赖
echo ""
echo "步骤 1: 检查系统依赖..."
echo "需要的系统包："
echo "  - git, cmake, build-essential"
echo "  - libgl1-mesa-dev, libsdl2-dev, libsdl2-image-dev"
echo "  - libsdl2-ttf-dev, libsdl2-gfx-dev"
echo "  - libboost-all-dev, libdirectfb-dev, libst-dev"
echo "  - mesa-utils, xvfb, x11vnc, libsdl-sge-dev"
echo ""
read -p "是否需要安装系统依赖？(y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "安装系统依赖..."
    sudo apt-get update
    sudo apt-get install -y git cmake build-essential libgl1-mesa-dev libsdl2-dev \
        libsdl2-image-dev libsdl2-ttf-dev libsdl2-gfx-dev libboost-all-dev \
        libdirectfb-dev libst-dev mesa-utils xvfb x11vnc libsdl-sge-dev python3-pip
fi

# 2. 卸载官方版本的 gfootball
echo ""
echo "步骤 2: 卸载官方版本的 gfootball..."
pip uninstall -y gfootball

# 3. 克隆作者修改版的仓库
echo ""
echo "步骤 3: 克隆作者修改版的 Google Research Football..."
cd /tmp
if [ -d "football" ]; then
    echo "删除已存在的 football 目录..."
    rm -rf football
fi

git clone https://github.com/chrisyrniu/football.git
if [ $? -ne 0 ]; then
    echo "✗ 克隆失败！请检查网络连接"
    exit 1
fi

# 4. 安装作者修改版
echo ""
echo "步骤 4: 安装作者修改版的 gfootball..."
cd football
pip install .
if [ $? -eq 0 ]; then
    echo "  ✓ 安装成功"
else
    echo "  ✗ 安装失败"
    exit 1
fi

# 5. 验证安装
echo ""
echo "步骤 5: 验证安装..."
python -c "import gfootball; print('  ✓ gfootball 版本:', gfootball.__version__ if hasattr(gfootball, '__version__') else '自定义版本')"
python -c "from gfootball.env import create_environment; print('  ✓ create_environment 函数可用')"

# 6. 清理
echo ""
echo "步骤 6: 清理临时文件..."
cd /tmp
rm -rf football
echo "  ✓ 清理完成"

echo ""
echo "================================================"
echo "✓ 安装完成！"
echo "================================================"
echo ""
echo "现在请重新运行训练脚本："
echo "  cd \"/home/Causal Communication/MAGIC-main_one_round/MAGIC-main_download_baseline\""
echo "  sh train_grf.sh"
echo ""

