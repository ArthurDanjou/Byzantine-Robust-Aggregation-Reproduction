# SignGuard Reproduction

This repository reproduces the experiments from [**Byzantine-robust Federated Learning through Collaborative Malicious Gradient Filtering**](https://arxiv.org/pdf/2109.05872v2) (Xu et al., 2022).

## Motivation

The goal is **not** to produce new results — it is to identify what is hard, redundant, and error-prone when implementing Byzantine-robust aggregation research from scratch. Every pain point logged here is a candidate for a first-class abstraction in [**krum**](https://github.com/calicarpa/krum), a research library for Byzantine-robust federated learning.

## What this repo implements

| Module | File | Contents |
|--------|------|----------|
| Models | `models.py` | CNN (MNIST), CifarCNN, ResNet-18/34/50/101/152 |
| Datasets | `datasets.py` | MNIST, CIFAR-10, split across workers with `Subset` |
| Attacks | `attacks.py` | A Little Is Enough (Baruch et al., 2019), Fall of Empires (Xie et al., 2019) |
| Aggregators | `aggregators.py` | Multi-Krum (Blanchard et al., 2017), SignGuard (Xu et al., 2022) |
| Training | `experiments.py` | Federated training loop with honest & byzantine workers, gradient flattening |
| Runner | `main.py` | Wires everything together for a full experiment run |

## Pain points → krum features

These are the problems discovered during this reproduction that motivate the design of krum:

### 1. Aggregators have no unified interface
`MultiKrum(n, m, f)` vs `SignGuard(f, use_dbscan=False)` — no shared constructor, no shared configuration schema. krum needs a **standardized aggregator protocol** (constructor + `aggregate()` signature).

### 2. Attacks have no unified signature
`a_little_is_enough_attack(honest_gradients, num_byzantine, z=...)` vs `fall_of_empires_attack(honest_gradients, defense_func, num_byzantine, epsilon=...)` — every attack takes different parameters. krum needs an **attack protocol** with a common callable interface and keyword-only configuration.

### 3. Gradient flattening is boilerplate
Every worker must flatten gradients into a 1D tensor (`get_gradient_values`) and the aggregator must un-flatten them back into model parameters (`set_gradient_values`). krum should **abstract gradient packing/unpacking** behind the aggregator.

### 4. Data partitioning is manual
`get_dataset` manually slices the training set with `Subset` and builds `DataLoader` per worker. krum should provide a **`partition()` utility** that returns per-worker loaders from a dataset.

### 5. The training loop is rigid
`train_step` hard-codes the honest/byzantine ordering (Byzantine at `[0:f)`, honest at `[f:n+f)`), the attack-before-aggregate flow, and the update step. krum must offer a **composable training loop** where aggregators, attacks, and workers are pluggable.

### 6. Byzantine worker logic is duplicated
`byzantine_worker` is identical to `honest_worker` — the only difference is that the attack transforms the gradient afterward. krum should treat **every worker as a gradient producer** and let attacks be post-processing steps.

### 7. Evaluation is decoupled from the training pipeline
`test_classification` is standalone and not hooked into the training loop (metrics must be collected manually in `main.py`). krum should provide a **callback/hook system** for evaluation, logging, and checkpointing.

### 8. Reproducibility primitives are missing
No seed management, no experiment config serialization, no run tracking. krum should provide **reproducibility helpers** (seeds, config snapshots, result export).

## Requirements

- Python ≥ 3.13
- PyTorch ≥ 2.11
- torchvision ≥ 0.26
- scikit-learn ≥ 1.8
- numpy ≥ 2.4

## Usage

```bash
uv sync
uv run python main.py
```

## Reference

Xu, J., Li, Z., Chen, Y., Lyu, L., & Zhou, X. (2022). *Byzantine-robust Federated Learning through Collaborative Malicious Gradient Filtering*. In Proceedings of the 42nd IEEE International Conference on Distributed Computing Systems (ICDCS). [arXiv:2109.05872v2](https://arxiv.org/pdf/2109.05872v2)
