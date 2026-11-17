# MAGIC-GRF 快速开始指南

## 🚀 一键安装（推荐）

```bash
cd "/home/Causal Communication/MAGIC-main_one_round/MAGIC-main_download_baseline"
bash setup_all.sh
```

这个脚本会自动完成所有配置。

---

## 📋 问题诊断

如果遇到问题，运行诊断脚本：

```bash
python check_dependencies.py
```

---

## 🔧 分步安装

### 步骤 1：修复 grf_envs 模块

```bash
bash fix_grf_env.sh
```

**解决问题**：
- ✓ 环境重复注册错误
- ✓ 模块路径错误

### 步骤 2：安装作者修改版 gfootball

```bash
bash install_gfootball.sh
```

**解决问题**：
- ✓ `Unsupported representation: multiagent` 错误
- ✓ 添加多智能体观察功能

### 步骤 3：运行训练

```bash
sh train_grf.sh
```

---

## 🐛 常见错误速查

### 错误 1: `Cannot re-register id: GFootball-...`

**原因**：grf_envs 模块路径错误

**解决**：
```bash
bash fix_grf_env.sh
```

### 错误 2: `Unsupported representation: multiagent`

**原因**：安装了官方版 gfootball，而非作者修改版

**解决**：
```bash
bash install_gfootball.sh
```

### 错误 3: 系统依赖缺失

**解决**：
```bash
sudo apt-get update
sudo apt-get install -y git cmake build-essential libgl1-mesa-dev libsdl2-dev \
    libsdl2-image-dev libsdl2-ttf-dev libsdl2-gfx-dev libboost-all-dev \
    libdirectfb-dev libst-dev mesa-utils xvfb x11vnc libsdl-sge-dev python3-pip
```

---

## ✅ 验证安装

```bash
# 检查 grf_envs
python -c "import grf_envs; print('✓ grf_envs:', grf_envs.__file__)"

# 检查 gfootball
python -c "import gfootball; print('✓ gfootball 已安装')"

# 检查环境注册
python -c "import gym; import grf_envs; env = gym.make('GRFWrapper-v0'); print('✓ 环境注册成功')"

# 完整检查
python check_dependencies.py
```

---

## 📚 其他资源

- **详细文档**：`GRF_环境修复指南.md`
- **项目 README**：`README.md`
- **作者修改版 GRF**：https://github.com/chrisyrniu/football
- **论文**：Multi-Agent Graph-Attention Communication and Teaming (AAMAS 2021)

---

## 💡 提示

1. **警告可以忽略**：PyTorch 的 `set_default_tensor_type()` 警告不影响运行
2. **使用 WSL**：在 Windows 上推荐使用 WSL Ubuntu 环境
3. **Python 环境**：确保在正确的 conda 环境中（Katerina）
4. **网络问题**：如果 git clone 失败，可以手动下载 zip 文件

---

## 🆘 获取帮助

如果仍有问题：

1. 运行 `python diagnose_and_fix.py` 查看详细诊断
2. 查看 `GRF_环境修复指南.md` 中的常见问题部分
3. 提供以下信息寻求帮助：
   - `python check_dependencies.py` 的输出
   - 完整的错误堆栈信息
   - Python 版本和环境信息

