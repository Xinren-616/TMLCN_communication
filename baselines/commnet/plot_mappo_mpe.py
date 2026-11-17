"""
Plot MAPPO MPE results:
  1. Success rate & reward curves (no_comm / fullcomm+shapley_only / ctap)
  2. Shapley value heatmaps at early vs late epochs (shapley_only and ctap)
"""
import re
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path

BASE = Path('/home/zxr/Shapley_Value_Online/MAGIC-main_one_round/MAGIC-main_download')
LOG_DIR = BASE / 'new_env_scripts/mappo/mpe'
SAVED_DIR = BASE / 'saved'
OUT_DIR = BASE / 'plots/mappo_mpe'
OUT_DIR.mkdir(parents=True, exist_ok=True)

SMOOTH = 50  # moving average window

# ── helpers ──────────────────────────────────────────────────────────────────

def parse_log(path):
    epochs, rewards, successes = [], [], []
    e = r = s = None
    for line in Path(path).read_text().splitlines():
        m = re.match(r'^Epoch (\d+)/', line)
        if m:
            e = int(m.group(1))
        m = re.match(r'\s+Reward:\s+([-\d.]+)', line)
        if m:
            r = float(m.group(1))
        m = re.match(r'\s+Success:\s+([-\d.]+)', line)
        if m:
            s = float(m.group(1))
            if e is not None and r is not None:
                epochs.append(e); rewards.append(r); successes.append(s)
                e = r = s = None
    return np.array(epochs), np.array(rewards), np.array(successes)

def smooth(x, w):
    if len(x) < w:
        return x
    return np.convolve(x, np.ones(w)/w, mode='valid')

def smooth_xy(epochs, values, w):
    sv = smooth(values, w)
    # align epochs to smoothed length
    se = epochs[w-1:][:len(sv)]
    return se, sv

def load_seeds(pattern_list):
    """Load and average multiple seed logs matching any pattern in list."""
    all_r, all_s = [], []
    ref_e = None
    for p in pattern_list:
        logs = sorted(LOG_DIR.glob(p))
        for log in logs:
            e, r, s = parse_log(log)
            if ref_e is None:
                ref_e = e
            min_len = min(len(ref_e), len(r))
            all_r.append(r[:min_len])
            all_s.append(s[:min_len])
    if not all_r:
        return None, None, None
    min_len = min(len(x) for x in all_r)
    all_r = np.array([x[:min_len] for x in all_r])
    all_s = np.array([x[:min_len] for x in all_s])
    return ref_e[:min_len], all_r, all_s

# ── load data ─────────────────────────────────────────────────────────────────

groups = {
    'No Comm':          ['train_mpe_mappo_no_comm_seed*.log'],
    'FullComm / Shapley-Only': ['train_mpe_mappo_fullcomm_seed*.log',
                                'train_mpe_mappo_shapley_only_seed*.log'],
    'CTAP':             ['train_mpe_mappo_ctap_seed*.log'],
}

colors = {
    'No Comm': '#888888',
    'FullComm / Shapley-Only': '#2196F3',
    'CTAP': '#E53935',
}

# ── Figure 1: learning curves ─────────────────────────────────────────────────

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle('MAPPO on MPE simple_spread (3 agents)', fontsize=14, fontweight='bold')

for label, patterns in groups.items():
    epochs, all_r, all_s = load_seeds(patterns)
    if epochs is None:
        continue
    c = colors[label]
    for ax, data in zip(axes, [all_r, all_s]):
        mean = data.mean(axis=0)
        std  = data.std(axis=0)
        se, sm = smooth_xy(epochs, mean, SMOOTH)
        _, ss  = smooth_xy(epochs, std,  SMOOTH)
        ax.plot(se, sm, label=label, color=c, linewidth=1.8)
        ax.fill_between(se, sm - ss, sm + ss, alpha=0.15, color=c)

axes[0].set_title('Reward')
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Mean Reward')
axes[0].legend(fontsize=9)
axes[0].grid(alpha=0.3)

axes[1].set_title('Success Rate')
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('Success Rate')
axes[1].set_ylim(-0.05, 1.05)
axes[1].legend(fontsize=9)
axes[1].grid(alpha=0.3)

plt.tight_layout()
out1 = OUT_DIR / 'mappo_mpe_curves.png'
plt.savefig(out1, dpi=150, bbox_inches='tight')
plt.close()
print(f'Saved: {out1}')

# ── Figure 2: Shapley heatmaps ────────────────────────────────────────────────

def load_shapley_matrix(saved_name, epoch_key):
    """Return n×n Shapley matrix for given epoch."""
    path = SAVED_DIR / saved_name / 'shapley_results/online_shapley_values.json'
    with open(path) as f:
        data = json.load(f)
    if epoch_key not in data:
        # find closest
        keys = sorted(data.keys(), key=int)
        epoch_key = min(keys, key=lambda k: abs(int(k) - int(epoch_key)))
    edges = data[epoch_key]
    # infer n
    agents = set()
    for e in edges:
        a, b = e.split('->')
        agents.update([int(a), int(b)])
    n = max(agents) + 1
    mat = np.zeros((n, n))
    for e, v in edges.items():
        src, dst = map(int, e.split('->'))
        mat[src, dst] = v['shapley_value']
    return mat

def plot_heatmap(ax, mat, title, vmin=None, vmax=None):
    n = mat.shape[0]
    im = ax.imshow(mat, cmap='RdYlGn', vmin=vmin, vmax=vmax, aspect='auto')
    ax.set_xticks(range(n)); ax.set_yticks(range(n))
    ax.set_xticklabels([f'Agent {i}' for i in range(n)], fontsize=8)
    ax.set_yticklabels([f'Agent {i}' for i in range(n)], fontsize=8)
    ax.set_xlabel('Receiver', fontsize=8)
    ax.set_ylabel('Sender', fontsize=8)
    ax.set_title(title, fontsize=9, fontweight='bold')
    # annotate values
    for i in range(n):
        for j in range(n):
            if i != j:
                ax.text(j, i, f'{mat[i,j]:.2f}', ha='center', va='center',
                        fontsize=7, color='black')
    return im

# pick early / late epochs
EARLY = '500'
LATE  = '5000'

cases = [
    ('mpe_mappo_shapley_only_seed1_20260507', 'Shapley-Only'),
    ('mpe_mappo_ctap_seed1_20260507',         'CTAP'),
]

fig2, axes2 = plt.subplots(2, 2, figsize=(11, 9))
fig2.suptitle('Shapley Value Heatmaps — MPE simple_spread (Seed 1)', fontsize=13, fontweight='bold')

for row, (saved_name, case_label) in enumerate(cases):
    mats = []
    for epoch_key in [EARLY, LATE]:
        try:
            m = load_shapley_matrix(saved_name, epoch_key)
            mats.append(m)
        except Exception as ex:
            print(f'Warning: {saved_name} epoch {epoch_key}: {ex}')
            mats.append(None)

    # shared color scale
    valid = [m for m in mats if m is not None]
    vmin = min(m.min() for m in valid) if valid else 0
    vmax = max(m.max() for m in valid) if valid else 1

    for col, (mat, epoch_key) in enumerate(zip(mats, [EARLY, LATE])):
        ax = axes2[row, col]
        if mat is None:
            ax.text(0.5, 0.5, 'No data', ha='center', va='center')
            ax.set_title(f'{case_label} — Epoch {epoch_key}', fontsize=9)
            continue
        im = plot_heatmap(ax, mat, f'{case_label} — Epoch {epoch_key}', vmin=vmin, vmax=vmax)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

plt.tight_layout()
out2 = OUT_DIR / 'mappo_mpe_shapley_heatmaps.png'
plt.savefig(out2, dpi=150, bbox_inches='tight')
plt.close()
print(f'Saved: {out2}')
