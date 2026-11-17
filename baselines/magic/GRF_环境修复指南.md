# GRF 环境配置修复指南

## 问题描述

### 问题 1：重复注册环境（已解决）
```
gym.error.Error: Cannot re-register id: GFootball-11_vs_11_competition-SMM-v0
```

**根本原因**：
1. Python 导入的是**旧路径**的 `grf_envs` 模块（`/mnt/e/OneDrive...`）
2. 旧版本的 `grf_wrapper_env.py` 在模块级别导入 `gfootball`，导致重复注册环境

### 问题 2：不支持 multiagent representation（当前问题）
```
ValueError: Unsupported representation: multiagent
```

**根本原因**：
- 您安装的是官方版本的 gfootball (2.10.2)
- 官方版本**不支持** `multiagent` representation
- 需要安装作者修改版：https://github.com/chrisyrniu/football

## 已完成的修复

我已经修改了以下文件：

### 1. `envs/grf-envs/grf_envs/grf_wrapper_env.py`
- ✅ 将 `import gfootball.env as grf_env` 从第 2 行移除
- ✅ 将导入移到 `multi_agent_init()` 方法内部（延迟导入）
- ✅ 避免模块导入时触发 gfootball 环境注册

### 2. `envs/grf-envs/setup.py`
- ✅ 添加 `find_packages()` 确保正确打包

## 需要您执行的步骤

### 第一部分：修复 grf_envs 模块（✓ 已完成）

如果遇到重复注册问题，使用自动修复脚本：

```bash
cd "/home/Causal Communication/MAGIC-main_one_round/MAGIC-main_download_baseline"
sh fix_grf_env.sh
```

### 第二部分：安装作者修改版的 gfootball（当前步骤）

⚠️ **重要**：必须使用作者修改版，官方版本不支持 multiagent！

#### 方法一：使用自动安装脚本（推荐）

```bash
cd "/home/Causal Communication/MAGIC-main_one_round/MAGIC-main_download_baseline"
bash install_gfootball.sh
```

#### 方法二：手动安装

```bash
# 1. 安装系统依赖（如果还没安装）
sudo apt-get update
sudo apt-get install -y git cmake build-essential libgl1-mesa-dev libsdl2-dev \
    libsdl2-image-dev libsdl2-ttf-dev libsdl2-gfx-dev libboost-all-dev \
    libdirectfb-dev libst-dev mesa-utils xvfb x11vnc libsdl-sge-dev python3-pip

# 2. 卸载官方版本
pip uninstall -y gfootball

# 3. 克隆作者修改版
cd /tmp
git clone https://github.com/chrisyrniu/football.git
cd football

# 4. 安装
pip install .

# 5. 验证
python -c "import gfootball; print('安装成功')"

# 6. 返回项目目录
cd "/home/Causal Communication/MAGIC-main_one_round/MAGIC-main_download_baseline"
```

### 第三部分：运行训练

所有依赖安装完成后，运行训练脚本：

```bash
cd "/home/Causal Communication/MAGIC-main_one_round/MAGIC-main_download_baseline"
sh train_grf.sh
```

## 诊断工具

如果问题仍然存在，可以运行诊断脚本：

```bash
python diagnose_and_fix.py
```

## 常见问题

### Q1: 为什么必须使用作者修改版的 gfootball？
**A**: 作者在官方 gfootball 基础上添加了：
- `multiagent` representation 支持
- 多智能体观察功能
- 修复了一些 bug

官方版本（2.10.2）不支持这些功能。

### Q2: 如何确认安装的是作者修改版？
**A**: 运行以下命令：
```bash
python -c "from gfootball.env import create_environment; help(create_environment)" | grep representation
```
如果看到 `multiagent` 作为选项之一，说明是作者修改版。

### Q3: 系统依赖安装失败怎么办？
**A**: 确保：
1. 有 sudo 权限
2. 网络连接正常
3. 运行在 Ubuntu/Debian 系统上

在 WSL 中，可以正常使用 `apt-get`。

### Q4: git clone 很慢或失败怎么办？
**A**: 可以尝试：
1. 使用代理
2. 使用 Gitee 镜像（如果有）
3. 直接下载 zip：https://github.com/chrisyrniu/football/archive/refs/heads/master.zip

### Q5: 为什么会导入错误的模块路径？
**A**: Python 之前安装的 `grf_envs` 包指向了另一个目录。使用 `python setup.py develop` 会创建一个链接指向当前目录的源代码。

### Q6: 关于 PyTorch 警告
警告信息：
```
torch.set_default_tensor_type() is deprecated
```
这只是一个警告，不影响运行。如果想消除，需要修改代码中使用 `torch.set_default_tensor_type()` 的地方。

## 技术细节

### 修改前后对比

**修改前**（会导致错误）：
```python
import gym
import gfootball.env as grf_env  # ← 在模块导入时就执行
import numpy as np

class GRFWrapperEnv(gym.Env):
    def multi_agent_init(self, args):
        self.env = grf_env.create_environment(...)
```

**修改后**（正确）：
```python
import gym
import numpy as np  # ← 移除了模块级别的 gfootball 导入

class GRFWrapperEnv(gym.Env):
    def multi_agent_init(self, args):
        import gfootball.env as grf_env  # ← 延迟到实际使用时才导入
        self.env = grf_env.create_environment(...)
```

**为什么这样修改？**
- `gfootball` 包在导入时会自动注册很多 Gym 环境
- 如果在模块级别导入，每次 `import grf_envs` 都会触发注册
- 如果系统中有多个 gfootball 副本，会导致重复注册错误
- 延迟导入可以避免这个问题

## 联系

如果按照以上步骤操作后仍有问题，请提供以下信息：
1. `python diagnose_and_fix.py` 的完整输出
2. `pip list | grep grf` 的输出
3. `which python` 的输出
4. 完整的错误堆栈信息

