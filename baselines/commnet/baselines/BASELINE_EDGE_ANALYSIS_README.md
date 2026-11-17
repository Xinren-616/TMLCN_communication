# Baseline模型边级别通信Shapley值分析工具

## 📋 概述

这个工具为baseline模型（CommNet, IC3Net, TarMAC, GAComm）提供**边级别（链路级别）通信Shapley值分析**，帮助您理解：

- **哪条具体的通信链路** (agent i → agent j) 对团队成功最重要
- **通信拓扑**在训练过程中如何演化
- **每条链路的边际贡献**（移除该链路后性能下降多少）

## 🆕 与MAGIC版本的区别

| 特性 | MAGIC版本 | Baseline版本 |
|------|----------|-------------|
| **支持模型** | MAGIC | CommNet, IC3Net, TarMAC, GAComm |
| **拓扑类型** | 动态图（学习的） | 静态图（CommNet）或动态图（TarMAC） |
| **Hard Attention** | ✓ | ✓ (IC3Net) |
| **软注意力** | ✓ | ✓ (TarMAC, GAComm) |
| **参数兼容性** | main.py | run_baselines.py |

## 📁 文件结构

```
baselines/
├── run_baselines.py                    # [已修改] 训练脚本，添加checkpoint保存
├── edge_level_cshav_baselines.py       # [新增] 边级别分析工具
├── example_edge_analysis_baseline.sh   # [新增] 使用示例脚本
└── BASELINE_EDGE_ANALYSIS_README.md    # [新增] 本文档
```

## 🔧 修改内容

### 1. `run_baselines.py` 修改

**修改内容**：自动保存checkpoint用于后续分析

```python
# 在run()函数中，添加每100个epoch保存checkpoint
if args.save and (ep+1) % 100 == 0:
    save(final=False, epoch=ep+1)

# 修改save函数参数名（episode → epoch）
def save(final, epoch=0):
    d = dict()
    d['policy_net'] = policy_net.state_dict()
    d['log'] = log
    d['trainer'] = trainer.state_dict()
    if final:
        torch.save(d, run_dir / 'model.pt')
    else:
        torch.save(d, run_dir / ('model_ep%i.pt' %(epoch)))
```

**效果**：
- Epoch 100 → `model_ep100.pt`
- Epoch 200 → `model_ep200.pt`
- ...
- 最终模型 → `model.pt`

### 2. `edge_level_cshav_baselines.py` 新增

**核心功能**：

1. **支持多种Baseline模型**
   ```python
   if self.model_type == 'gacomm':
       policy_net = GACommNetMLP(args, num_inputs)
   elif self.model_type == 'tarmac':
       policy_net = TarCommNetMLP(args, num_inputs)
   else:  # commnet 或 ic3net
       policy_net = CommNetMLP(args, num_inputs)
   ```

2. **拓扑提取适配不同模型**
   - **CommNet**：假设完全图（全连接）
   - **IC3Net**：根据hard attention (comm_action) 统计实际通信
   - **TarMAC**：提取软注意力权重作为拓扑
   - **GAComm**：提取图注意力权重

3. **边级别Mask机制**
   ```python
   info = {
       'comm_mask': edge_mask,  # [n, n] 矩阵
   }
   ```

## 🚀 使用方法

### 步骤1: 训练并保存Checkpoint

使用修改后的 `run_baselines.py` 训练模型：

```bash
cd baselines

# 训练CommNet
python run_baselines.py \
  --env_name traffic_junction \
  --nagents 5 \
  --difficulty easy \
  --commnet \
  --hid_size 128 \
  --num_epochs 1000 \
  --save \
  --save_every 100  # 每100个epoch保存

# 训练IC3Net
python run_baselines.py \
  --env_name traffic_junction \
  --nagents 5 \
  --difficulty easy \
  --ic3net \
  --hid_size 128 \
  --num_epochs 1000 \
  --save \
  --save_every 100
```

### 步骤2: 运行边级别分析

#### 基本用法

```bash
python edge_level_cshav_baselines.py \
  --checkpoint_dir ./saved/traffic_junction/commnet/run1 \
  --output_dir ./edge_results_commnet \
  --model_type commnet \
  --env_name traffic_junction \
  --nagents 5 \
  --difficulty easy \
  --vision 0 \
  --add_rate_min 0.3 \
  --add_rate_max 0.3 \
  --hid_size 128 \
  --max_steps 20 \
  --n_episodes 20
```

#### 参数说明

| 参数 | 说明 | 示例 |
|------|------|------|
| `--checkpoint_dir` | checkpoint目录路径 | `./saved/traffic_junction/commnet/run1` |
| `--output_dir` | 输出结果目录 | `./edge_results_commnet` |
| `--model_type` | 模型类型 | `commnet`, `ic3net`, `tarmac`, `gacomm` |
| `--env_name` | 环境名称 | `traffic_junction`, `predator_prey` |
| `--nagents` | 智能体数量 | `5` |
| `--n_episodes` | 每次评估的episode数 | `20` (默认) |
| `--topology_threshold` | 拓扑阈值 | `0.1` (默认) |
| `--checkpoint_indices` | 要分析的checkpoint索引 | `0 4 9` (可选) |

#### 不同模型的特殊参数

**IC3Net**:
```bash
--model_type ic3net \
--hard_attn  # 必须添加
```

**TarMAC**:
```bash
--model_type tarmac \
--qk_hid_size 16 \
--value_hid_size 32
```

**GAComm**:
```bash
--model_type gacomm
```

### 步骤3: 查看结果

生成的文件：

```
edge_results_commnet/
├── edge_level_cshav_results.json          # 边级别贡献数据
├── communication_topologies.json          # 通信拓扑数据
├── edge_contribution_matrix_epoch100.png  # 各epoch的贡献矩阵图
├── edge_contribution_matrix_epoch500.png
├── edge_contribution_evolution.png        # 贡献演化图
├── network_graph_epoch100.png             # 网络图（频率+贡献）
└── network_graph_epoch500.png
```

## 📊 结果解读

### 1. 边贡献矩阵图

<img src="example_output/edge_contribution_matrix_epoch100.png" width="800">

**三个子图**：

1. **Communication Topology Frequency**（通信拓扑频率）
   - 数值范围：0-1
   - 含义：链路 i→j 的使用频率
   - 示例：0.85 表示该链路在85%的时间步被使用

2. **Edge-Level Communication Contribution**（边级别贡献）
   - 数值范围：-0.1 到 +0.1（或更大）
   - 含义：链路 i→j 的边际贡献（移除后成功率下降）
   - 颜色：
     - 🟢 绿色（正值）：该链路有贡献
     - 🔴 红色（负值）：移除该链路反而提升性能（罕见）

3. **Weighted Edge Importance**（加权重要性）
   - 计算：贡献 × 频率
   - 含义：综合考虑贡献和使用频率的重要性
   - 用途：找出真正重要的链路（既常用又有效）

### 2. 贡献演化图

<img src="example_output/edge_contribution_evolution.png" width="800">

**两个子图**：

1. **Top 10 Important Edges**
   - 显示最重要的10条链路在训练过程中的贡献变化
   - 观察：
     - 📈 上升趋势：该链路越来越重要
     - 📉 下降趋势：依赖逐渐减弱
     - 🔄 波动：不稳定的通信模式

2. **All Edges Heatmap**
   - 显示所有链路在所有epoch的贡献
   - 横轴：训练epoch
   - 纵轴：通信链路
   - 颜色：贡献大小

### 3. 网络图

<img src="example_output/network_graph_epoch100.png" width="800">

**两个子图**：

1. **Frequency-based Network**
   - 边的颜色：使用频率（黄→橙→红 = 低→高）
   - 边的粗细：固定
   - 用途：看哪些链路经常被使用

2. **Contribution-based Network**
   - 边的颜色：边际贡献（红→白→绿 = 负→零→正）
   - 边的粗细：固定
   - 用途：看哪些链路真正有用

## 💡 应用场景

### 场景1: 模型对比

```bash
# 分析CommNet
python edge_level_cshav_baselines.py \
  --checkpoint_dir ./saved/traffic_junction/commnet/run1 \
  --output_dir ./results/tj_commnet \
  --model_type commnet \
  ...

# 分析IC3Net
python edge_level_cshav_baselines.py \
  --checkpoint_dir ./saved/traffic_junction/ic3net/run1 \
  --output_dir ./results/tj_ic3net \
  --model_type ic3net \
  --hard_attn \
  ...

# 对比：
# - CommNet: 所有链路贡献相似（完全图）
# - IC3Net: 部分链路贡献高（选择性通信）
```

### 场景2: 拓扑演化分析

```bash
# 分析TarMAC的软注意力演化
python edge_level_cshav_baselines.py \
  --checkpoint_dir ./saved/traffic_junction/tar_commnet/run1 \
  --output_dir ./results/tj_tarmac_evolution \
  --model_type tarmac \
  --checkpoint_indices 0 2 4 6 8  # 分析多个epoch
  ...

# 观察：
# - 早期：注意力分散（探索）
# - 中期：注意力集中（学习关键链路）
# - 后期：注意力稳定（策略固化）
```

### 场景3: 关键链路识别

```bash
# 高精度分析以识别关键链路
python edge_level_cshav_baselines.py \
  --checkpoint_dir ./saved/traffic_junction/ic3net/run1 \
  --output_dir ./results/critical_links \
  --model_type ic3net \
  --n_episodes 50 \
  --topology_threshold 0.05 \
  --hard_attn \
  ...

# 从结果中找出：
# 1. 贡献最大的Top 5链路
# 2. 频率最高但贡献低的链路（冗余通信）
# 3. 频率低但贡献高的链路（关键时刻通信）
```

## 🔬 不同模型的特点

### CommNet

**特点**：
- ✅ 完全图（所有agent通信）
- ✅ 简单有效
- ❌ 无法选择性通信

**预期结果**：
```
Topology: 全1矩阵（对角线为0）
Contribution: 所有边贡献相似
```

### IC3Net

**特点**：
- ✅ Hard attention（学习何时通信）
- ✅ 稀疏通信（节省带宽）
- ✅ 可解释性强

**预期结果**：
```
Topology: 稀疏矩阵（部分边为0）
Contribution: 选择性高的边贡献大
```

### TarMAC

**特点**：
- ✅ 软注意力（学习通信权重）
- ✅ 目标驱动的通信
- ✅ 表达能力强

**预期结果**：
```
Topology: 连续值矩阵（软权重）
Contribution: 权重大的边贡献大
```

### GAComm

**特点**：
- ✅ 图注意力机制
- ✅ 动态通信图
- ✅ 适合复杂场景

**预期结果**：
```
Topology: 动态变化的图结构
Contribution: 图结构与任务相关
```

## ⚙️ Episode数量配置

与MAGIC版本相同，参考 `EPISODE_SAMPLING_GUIDE_CN.md`：

| 场景 | `--n_episodes` | 单边总成本 | 精度 | 时间 |
|------|---------------|-----------|------|------|
| 快速测试 | 10 | 25 episodes | 低 | 50% |
| **标准分析** | **20** | **50 episodes** | **中** | **100%** |
| 高精度 | 30 | 75 episodes | 高 | 150% |
| 论文发表 | 50 | 125 episodes | 很高 | 250% |

## 📈 性能估算

### Traffic Junction (5 agents)

```
潜在链路数：5 × 4 = 20条
活跃链路（频率>0.1）：约12-15条

单个checkpoint分析时间：
- CommNet (完全图):  15条 × 50 episodes × 10秒 = 125分钟
- IC3Net (稀疏):     8条 × 50 episodes × 10秒 = 67分钟
- TarMAC (动态):    12条 × 50 episodes × 10秒 = 100分钟
```

### Predator-Prey (5 agents)

```
潜在链路数：5 × 4 = 20条
活跃链路（频率>0.1）：约8-12条

单个checkpoint分析时间：
- GAComm (动态):    10条 × 50 episodes × 15秒 = 125分钟
```

## 🐛 常见问题

### Q1: 提示找不到模块

```bash
ModuleNotFoundError: No module named 'comm'
```

**解决**：确保在 `baselines/` 目录下运行：
```bash
cd baselines
python edge_level_cshav_baselines.py ...
```

### Q2: 加载checkpoint失败

```bash
RuntimeError: Error(s) in loading state_dict
```

**解决**：检查 `--model_type` 是否与checkpoint匹配：
- CommNet checkpoint → `--model_type commnet`
- IC3Net checkpoint → `--model_type ic3net`
- TarMAC checkpoint → `--model_type tarmac`

### Q3: 拓扑全为1（无差异）

**原因**：CommNet默认使用完全图

**解决**：
1. 使用IC3Net或TarMAC获得动态拓扑
2. 或者关注"贡献"而非"频率"

### Q4: 内存不足

**解决**：
```bash
# 减少n_episodes
--n_episodes 10

# 只分析部分checkpoint
--checkpoint_indices 0 9  # 只分析第1个和最后1个

# 提高topology_threshold
--topology_threshold 0.2  # 只分析更重要的边
```

## 📚 相关文档

- `EPISODE_SAMPLING_GUIDE_CN.md` - Episode数量选择指南
- `EDGE_MASK_MECHANISM_CN.md` - Mask机制详解
- `MODEL_CHECKPOINT_EXPLANATION_CN.md` - Checkpoint内容说明

## 🎓 论文引用

如果使用此工具进行研究，请引用相关论文：

**CommNet**:
```
Sukhbaatar et al. "Learning Multiagent Communication with Backpropagation." NIPS 2016.
```

**IC3Net**:
```
Singh et al. "Learning when to Communicate at Scale in Multiagent Cooperative and Competitive Environments." AAMAS 2019.
```

**TarMAC**:
```
Das et al. "TarMAC: Targeted Multi-Agent Communication." ICML 2019.
```

## 🤝 反馈与改进

如有问题或建议，请：
1. 检查本文档的"常见问题"部分
2. 查看代码注释
3. 联系开发者

---

**祝分析顺利！** 🎉

