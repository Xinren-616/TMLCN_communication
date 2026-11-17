# TMLCN / MAGIC communication baselines

This repository contains cleaned research code for communication-based multi-agent reinforcement learning. Runtime artifacts (checkpoints, logs, generated plots, caches, and saved experiment data) are intentionally excluded.

## Repository layout

```text
baselines/                  One-round, no loss-adjust baselines
  commnet/
  magic/
  gacomm/
  ic3net/
  tarmac/
  bvme/
  gacg/
loss_adjust/                Loss-adjust experiment implementations
  one_round/
  two_round/
  three_round/
communication_rounds/       No-loss-adjust multi-round BVME/GACG variants
  two_round/bvme_gacg/
  three_round/bvme_gacg/
```

Each algorithm directory is self-contained because the original experiments evolved as independent snapshots. This preserves the exact trainer, environment wrappers, and command-line behavior used for that experiment instead of forcing incompatible snapshots into one shared package.

## Baseline entry points

Baseline training uses either `main.py` (MAGIC) or `baselines/run_baselines.py` (the other communication baselines). The task launchers are the retained `train_*.sh` files. In particular:

- BVME: `baselines/bvme/baselines/train_tj_easy.sh`
- GACG: `baselines/gacg/baselines/train_tj_easy.sh`
- IC3Net: `baselines/ic3net/baselines/train_tj_easy.sh`
- TarMAC: `baselines/tarmac/baselines/train_tj_easy.sh`
- GA-Comm: `baselines/gacomm/baselines/train_tj_easy.sh`

Run a launcher from its algorithm root so its relative paths resolve correctly.

## Outputs

Training recreates output folders such as `saved/`, `logs/`, and `plots/`. These paths, model checkpoints, Python caches, log files, and common experiment-result formats are ignored by Git. Do not commit generated artifacts; keep reproducible configuration in the launcher scripts instead.

## Historical directory mapping

The old download-oriented names were normalized as follows:

| Old directory | New directory |
| --- | --- |
| `MAGIC-main_download` | `baselines/commnet` |
| `MAGIC-main_download_baseline` | `baselines/magic` |
| `MAGIC-main_download_baseline_GAcomm` | `baselines/gacomm` |
| `MAGIC-main_download_baseline_IC3Net` | `baselines/ic3net` |
| `MAGIC-main_download_baseline_TarMAC` | `baselines/tarmac` |
| integrated one-round BVME implementation | `baselines/bvme` |
| integrated one-round GACG implementation | `baselines/gacg` |
| `TMLCN/Loss_Adjust_*` | `loss_adjust/{one,two,three}_round` |

