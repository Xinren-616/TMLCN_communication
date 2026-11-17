# TMLCN communication experiments

Research code for communication baselines and loss-adjust experiments based on the MAGIC / IC3Net training framework. This repository contains source code and launch scripts; training logs, checkpoints, generated figures, and caches are excluded.

## Layout

| Directory | Contents |
| --- | --- |
| `baselines/{commnet,magic,gacomm,ic3net,tarmac,bvme,gacg}` | One-round communication baselines without loss adjust |
| `loss_adjust/one_round/{ic3net,tarmac}` | Existing one-round loss-adjust experiments |
| `loss_adjust/two_round/tarmac` | Existing two-round loss-adjust experiment |
| `loss_adjust/three_round/{commnet,magic,gacomm,ic3net,tarmac}` | Existing three-round loss-adjust experiments |
| `communication_rounds/{two_round,three_round}/bvme_gacg` | Existing multi-round BVME/GACG experiments without loss adjust |

Each experiment is a self-contained snapshot of its model, trainer, environment wrappers, and scripts. The directory matrix reflects the implementations present in the source material; an absent combination has not been claimed or generated.

## Quick start

Use a Linux environment with Bash and Python. The original project documentation in each experiment directory describes its dependencies, including PyTorch, Gym, Visdom, and the included IC3Net environment package. Install the environment package for the chosen experiment from its root:

```bash
cd baselines/bvme
python -m pip install -e envs/ic3net-envs
```

Install the remaining Python dependencies appropriate to your Python/PyTorch environment before training. The historical documentation mentions PyTorch 1.5; compatibility with newer releases has not been verified here.

Run a baseline launcher **from the directory containing `run_baselines.py`**, because the scripts use relative paths:

```bash
cd baselines/bvme/baselines
bash train_tj_easy.sh
```

For GACG, substitute `baselines/gacg/baselines`. The same pattern applies to CommNet, GA-Comm, IC3Net, and TarMAC. MAGIC's launcher instead runs `main.py` from the experiment root:

```bash
cd baselines/magic
bash train_tj_easy.sh
```

Launchers for other environments and communication rounds are located alongside their respective training entry points. Review the command-line flags and seeds in a script before running it. BVME and GACG share a compatible runner, so their code snapshots contain both implementations; the dedicated launchers select the named algorithm with `--bvme` or `--gacg`.

## Source and validation

The original `MAGIC-main_download` snapshot supplied CommNet, while the other baseline snapshots supplied MAGIC, GA-Comm, IC3Net, and TarMAC. BVME and GACG were separated from `TMLCN/two_baseline_no_LA/one_round`; the multi-round versions were preserved under `communication_rounds/`. Source code and launchers were retained, and a pre-existing duplicate `cmap` keyword in CommNet's analysis script was corrected.

All 546 retained Python files passed a syntax compile check. A training run was not performed: the local environment lacked `visdom` and the project-specific runtime dependencies. Runtime output paths and common checkpoint/log formats are covered by `.gitignore`.

The included experiment-level `LICENSE` and `README.md` files retain the upstream licensing and citation information.
