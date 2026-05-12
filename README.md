# SignGuard Reproduction

This repository reproduces the experiments from [**Byzantine-robust Federated Learning through Collaborative Malicious Gradient Filtering**](https://arxiv.org/pdf/2109.05872v2) (Xu et al., 2022).

## Motivation

The goal is **not** to produce new results — it is to identify what is hard, redundant, and error-prone when implementing Byzantine-robust aggregation research from scratch. Every pain point logged here is a candidate for a first-class abstraction in [**krum**](https://github.com/calicarpa/krum), a research library for Byzantine-robust federated learning.

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

-- 

Ajouter un comportement différent pour les workers byzantins (ex: latence après calculs pour simuler un réseau lent)
Définir un rôle à chaque worker (coordonateur, worker normal, worker byzantin)

plot:
  - Attack impact (%) (for all attacks) vs % of Byz Clients for each aggregator
  - Test accuracy vs Training epochs with all aggregators