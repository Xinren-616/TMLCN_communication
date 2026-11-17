# Baseline模型Checkpoint命名规则说明

## 🎯 快速回答

不同baseline模型的checkpoint保存在**不同的目录**中，而不是通过文件名区分。

## 📁 目录结构和命名规则

### 基本结构

```
saved/
└── {env_name}/              # 环境名称
    ├── commnet/             # CommNet模型
    │   ├── run1/            # 第1次运行
    │   │   ├── model_ep100.pt
    │   │   ├── model_ep200.pt
    │   │   └── model.pt
    │   └── run2/            # 第2次运行
    │       └── ...
    ├── ic3net/              # IC3Net模型
    │   └── run1/
    │       └── ...
    ├── tar_commnet/         # TarMAC + CommNet
    │   └── run1/
    │       └── ...
    ├── tar_ic3net/          # TarMAC + IC3Net
    │   └── run1/
    │       └── ...
    └── gacomm/              # GAComm模型
        └── run1/
            └── ...
```

### 代码实现（run_baselines.py 第318-345行）

```python
if args.gacomm:
    model_dir = Path('./saved') / args.env_name / 'gacomm'
elif args.tarcomm:
    if args.ic3net:
        model_dir = Path('./saved') / args.env_name / 'tar_ic3net'
    elif args.commnet:
        model_dir = Path('./saved') / args.env_name / 'tar_commnet'
    else:
        model_dir = Path('./saved') / args.env_name / 'other'
elif args.ic3net:
    model_dir = Path('./saved') / args.env_name / 'ic3net'
elif args.commnet:
    model_dir = Path('./saved') / args.env_name / 'commnet'
else:
    model_dir = Path('./saved') / args.env_name / 'other'

# 自动生成run编号
if not model_dir.exists():
    curr_run = 'run1'
else:
    exst_run_nums = [int(str(folder.name).split('run')[1]) for folder in
                     model_dir.iterdir() if
                     str(folder.name).startswith('run')]
    if len(exst_run_nums) == 0:
        curr_run = 'run1'
    else:
        curr_run = 'run%i' % (max(exst_run_nums) + 1)

run_dir = model_dir / curr_run
```

## 📊 不同模型的完整路径示例

### Traffic Junction环境

#### CommNet
```bash
python run_baselines.py \
  --env_name traffic_junction \
  --commnet \
  --save

# 保存路径：
./saved/traffic_junction/commnet/run1/
├── model_ep100.pt    # Epoch 100的checkpoint
├── model_ep200.pt    # Epoch 200的checkpoint
├── model_ep300.pt    # Epoch 300的checkpoint
└── model.pt          # 最终模型
```

#### IC3Net
```bash
python run_baselines.py \
  --env_name traffic_junction \
  --ic3net \
  --save

# 保存路径：
./saved/traffic_junction/ic3net/run1/
├── model_ep100.pt
├── model_ep200.pt
└── model.pt
```

#### TarMAC (基于CommNet)
```bash
python run_baselines.py \
  --env_name traffic_junction \
  --commnet \
  --tarcomm \
  --save

# 保存路径：
./saved/traffic_junction/tar_commnet/run1/
├── model_ep100.pt
├── model_ep200.pt
└── model.pt
```

#### TarMAC (基于IC3Net)
```bash
python run_baselines.py \
  --env_name traffic_junction \
  --ic3net \
  --tarcomm \
  --save

# 保存路径：
./saved/traffic_junction/tar_ic3net/run1/
├── model_ep100.pt
├── model_ep200.pt
└── model.pt
```

#### GAComm
```bash
python run_baselines.py \
  --env_name traffic_junction \
  --gacomm \
  --save

# 保存路径：
./saved/traffic_junction/gacomm/run1/
├── model_ep100.pt
├── model_ep200.pt
└── model.pt
```

### Predator-Prey环境

同样的模型，但环境不同：

```bash
python run_baselines.py \
  --env_name predator_prey \
  --commnet \
  --save

# 保存路径：
./saved/predator_prey/commnet/run1/
├── model_ep100.pt
├── model_ep200.pt
└── model.pt
```

## 🔄 多次运行的run编号

### 自动递增机制

如果在**相同目录**下多次运行训练，会自动创建新的run目录：

```bash
# 第1次运行CommNet
python run_baselines.py --env_name traffic_junction --commnet --save
# 保存到: ./saved/traffic_junction/commnet/run1/

# 第2次运行CommNet
python run_baselines.py --env_name traffic_junction --commnet --save
# 保存到: ./saved/traffic_junction/commnet/run2/

# 第3次运行CommNet
python run_baselines.py --env_name traffic_junction --commnet --save
# 保存到: ./saved/traffic_junction/commnet/run3/
```

**效果**：
```
saved/traffic_junction/commnet/
├── run1/
│   ├── model_ep100.pt
│   ├── model_ep200.pt
│   └── model.pt
├── run2/
│   ├── model_ep100.pt
│   ├── model_ep200.pt
│   └── model.pt
└── run3/
    ├── model_ep100.pt
    ├── model_ep200.pt
    └── model.pt
```

## 📝 Checkpoint文件命名规则

### 在每个run目录内

| 文件名 | 说明 | 何时保存 |
|--------|------|---------|
| `model_ep100.pt` | Epoch 100的checkpoint | 每100个epoch自动保存 |
| `model_ep200.pt` | Epoch 200的checkpoint | 每100个epoch自动保存 |
| `model_ep300.pt` | Epoch 300的checkpoint | 每100个epoch自动保存 |
| ... | ... | ... |
| `model.pt` | 最终模型 | 每个epoch结束时更新 |

### 命名格式

```python
# 中间checkpoint
'model_ep%i.pt' % epoch
# 例如: model_ep100.pt, model_ep200.pt, model_ep300.pt

# 最终模型
'model.pt'
```

## 🎨 完整的目录树示例

假设您训练了多种模型：

```
saved/
├── traffic_junction/
│   ├── commnet/
│   │   ├── run1/
│   │   │   ├── model_ep100.pt
│   │   │   ├── model_ep200.pt
│   │   │   ├── model_ep300.pt
│   │   │   └── model.pt
│   │   └── run2/
│   │       └── ...
│   ├── ic3net/
│   │   └── run1/
│   │       ├── model_ep100.pt
│   │       ├── model_ep200.pt
│   │       └── model.pt
│   ├── tar_commnet/
│   │   └── run1/
│   │       ├── model_ep100.pt
│   │       └── model.pt
│   ├── tar_ic3net/
│   │   └── run1/
│   │       ├── model_ep100.pt
│   │       └── model.pt
│   └── gacomm/
│       └── run1/
│           ├── model_ep100.pt
│           └── model.pt
└── predator_prey/
    ├── commnet/
    │   └── run1/
    │       └── ...
    └── ic3net/
        └── run1/
            └── ...
```

## 🔍 如何使用edge_level_cshav_baselines.py分析

### 分析不同的baseline模型

根据您要分析的模型，指定对应的checkpoint目录：

#### 分析CommNet

```bash
python edge_level_cshav_baselines.py \
  --checkpoint_dir ./saved/traffic_junction/commnet/run1 \
  --output_dir ./results/tj_commnet \
  --model_type commnet \
  --env_name traffic_junction \
  --nagents 5 \
  --n_episodes 20
```

#### 分析IC3Net

```bash
python edge_level_cshav_baselines.py \
  --checkpoint_dir ./saved/traffic_junction/ic3net/run1 \
  --output_dir ./results/tj_ic3net \
  --model_type ic3net \
  --env_name traffic_junction \
  --nagents 5 \
  --n_episodes 20 \
  --hard_attn
```

#### 分析TarMAC (CommNet版)

```bash
python edge_level_cshav_baselines.py \
  --checkpoint_dir ./saved/traffic_junction/tar_commnet/run1 \
  --output_dir ./results/tj_tarmac_commnet \
  --model_type tarmac \
  --env_name traffic_junction \
  --nagents 5 \
  --n_episodes 20 \
  --commnet \
  --tarcomm
```

#### 分析TarMAC (IC3Net版)

```bash
python edge_level_cshav_baselines.py \
  --checkpoint_dir ./saved/traffic_junction/tar_ic3net/run1 \
  --output_dir ./results/tj_tarmac_ic3net \
  --model_type tarmac \
  --env_name traffic_junction \
  --nagents 5 \
  --n_episodes 20 \
  --ic3net \
  --tarcomm
```

#### 分析GAComm

```bash
python edge_level_cshav_baselines.py \
  --checkpoint_dir ./saved/traffic_junction/gacomm/run1 \
  --output_dir ./results/tj_gacomm \
  --model_type gacomm \
  --env_name traffic_junction \
  --nagents 5 \
  --n_episodes 20 \
  --gacomm
```

## 💡 关键要点

### ✅ 正确理解

1. **不同模型 = 不同目录**
   - CommNet → `saved/{env}/commnet/`
   - IC3Net → `saved/{env}/ic3net/`
   - TarMAC+CommNet → `saved/{env}/tar_commnet/`
   - TarMAC+IC3Net → `saved/{env}/tar_ic3net/`
   - GAComm → `saved/{env}/gacomm/`

2. **相同模型多次运行 = 不同run编号**
   - 第1次 → `run1/`
   - 第2次 → `run2/`
   - 第3次 → `run3/`

3. **checkpoint文件名是固定格式**
   - 中间checkpoint: `model_ep{epoch}.pt`
   - 最终模型: `model.pt`

### ❌ 常见误解

**误解1**: "不同模型的checkpoint文件名不同"
- ❌ 错误：文件名都是 `model_ep100.pt`, `model_ep200.pt`, ...
- ✅ 正确：通过**目录名**区分模型类型

**误解2**: "需要手动指定保存路径"
- ❌ 错误：不需要手动指定
- ✅ 正确：根据命令行参数自动确定路径

**误解3**: "重复训练会覆盖原有模型"
- ❌ 错误：不会覆盖
- ✅ 正确：自动创建新的run目录（run2, run3, ...）

## 🛠️ 实用技巧

### 技巧1: 查看所有已训练的模型

```bash
# 查看所有Traffic Junction的模型
ls -R saved/traffic_junction/

# 查看CommNet的所有run
ls saved/traffic_junction/commnet/

# 查看特定run的checkpoint
ls saved/traffic_junction/commnet/run1/
```

### 技巧2: 批量分析所有模型

```bash
# 创建批量分析脚本
cat > analyze_all_baselines.sh << 'EOF'
#!/bin/bash

ENV="traffic_junction"
BASE_DIR="./saved/${ENV}"

# 分析CommNet
if [ -d "${BASE_DIR}/commnet/run1" ]; then
  python edge_level_cshav_baselines.py \
    --checkpoint_dir ${BASE_DIR}/commnet/run1 \
    --output_dir ./results/${ENV}_commnet \
    --model_type commnet \
    --env_name ${ENV} \
    --nagents 5 \
    --n_episodes 20
fi

# 分析IC3Net
if [ -d "${BASE_DIR}/ic3net/run1" ]; then
  python edge_level_cshav_baselines.py \
    --checkpoint_dir ${BASE_DIR}/ic3net/run1 \
    --output_dir ./results/${ENV}_ic3net \
    --model_type ic3net \
    --env_name ${ENV} \
    --nagents 5 \
    --n_episodes 20 \
    --hard_attn
fi

# 分析TarMAC
if [ -d "${BASE_DIR}/tar_commnet/run1" ]; then
  python edge_level_cshav_baselines.py \
    --checkpoint_dir ${BASE_DIR}/tar_commnet/run1 \
    --output_dir ./results/${ENV}_tarmac \
    --model_type tarmac \
    --env_name ${ENV} \
    --nagents 5 \
    --n_episodes 20
fi

echo "所有模型分析完成！"
EOF

chmod +x analyze_all_baselines.sh
./analyze_all_baselines.sh
```

### 技巧3: 对比同一模型的不同run

```bash
# 对比CommNet的run1和run2
python edge_level_cshav_baselines.py \
  --checkpoint_dir ./saved/traffic_junction/commnet/run1 \
  --output_dir ./results/commnet_run1 \
  --model_type commnet \
  ...

python edge_level_cshav_baselines.py \
  --checkpoint_dir ./saved/traffic_junction/commnet/run2 \
  --output_dir ./results/commnet_run2 \
  --model_type commnet \
  ...
```

## 📊 目录命名规则总结表

| 训练命令参数 | 保存目录 | 示例 |
|-------------|---------|------|
| `--commnet` | `{env}/commnet/` | `saved/traffic_junction/commnet/run1/` |
| `--ic3net` | `{env}/ic3net/` | `saved/traffic_junction/ic3net/run1/` |
| `--commnet --tarcomm` | `{env}/tar_commnet/` | `saved/traffic_junction/tar_commnet/run1/` |
| `--ic3net --tarcomm` | `{env}/tar_ic3net/` | `saved/traffic_junction/tar_ic3net/run1/` |
| `--gacomm` | `{env}/gacomm/` | `saved/traffic_junction/gacomm/run1/` |
| 其他组合 | `{env}/other/` | `saved/traffic_junction/other/run1/` |

## 🎯 分析时的model_type对应关系

| checkpoint目录包含 | --model_type | 其他必需参数 |
|------------------|-------------|-------------|
| `commnet/` | `commnet` | - |
| `ic3net/` | `ic3net` | `--hard_attn` |
| `tar_commnet/` | `tarmac` | `--commnet --tarcomm` |
| `tar_ic3net/` | `tarmac` | `--ic3net --tarcomm` |
| `gacomm/` | `gacomm` | `--gacomm` |

## 📞 常见问题

### Q1: 如何知道某个checkpoint是哪种模型？

**A**: 看目录名！
```bash
# 例如这个路径：
./saved/traffic_junction/ic3net/run1/model_ep100.pt
#                        ^^^^^^
#                        这是IC3Net模型

# 那么分析时用：
--model_type ic3net
```

### Q2: 不同run的区别是什么？

**A**: 
- 相同模型、相同参数、不同随机种子的多次训练
- 或者相同模型但参数略有不同的实验
- 用于对比实验的可重复性

### Q3: 能否自定义保存目录？

**A**: 
代码中目录是自动确定的，如需自定义：
1. 训练后手动移动checkpoint到自定义目录
2. 或修改 `run_baselines.py` 的目录生成逻辑

### Q4: model.pt和model_ep*.pt有什么区别？

**A**:
- `model.pt`: 最终模型，每个epoch都会更新（覆盖）
- `model_ep100.pt`: 固定的Epoch 100快照，不会被覆盖
- 分析时通常用 `model_ep*.pt`（因为需要多个时间点）

## ✅ 检查清单

使用edge_level_cshav_baselines.py前，确认：

- [ ] 知道checkpoint在哪个目录（看目录名判断模型类型）
- [ ] `--checkpoint_dir` 指向正确的路径
- [ ] `--model_type` 与目录名对应
- [ ] 如果是IC3Net，添加 `--hard_attn`
- [ ] 如果是TarMAC，添加相应的 `--tarcomm` 等参数

---

**总结**: 不同baseline模型通过**目录名**区分，而不是文件名！看清楚checkpoint在哪个子目录下，就知道是哪种模型了。🎯

