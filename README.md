# Byzantine-Robust Aggregation Reproduction

Reproduction of [**Byzantine-robust Federated Learning through Collaborative Malicious Gradient Filtering**](https://arxiv.org/pdf/2109.05872v2) (Xu et al., 2022) — implementing Multi-Krum, SignGuard, A Little Is Enough, and Fall of Empires in PyTorch.

Also includes [**Robust Collaborative Learning with Linear Gradient Overhead**](https://proceedings.mlr.press/v202/farhadkhani23a.html) (Farhadkhani et al., ICML 2023).

## Implemented aggregators

- **SignGuard** (Xu et al., 2022) — sign-based clustering and norm clipping.
- **Multi-Krum** (Blanchard et al., 2017) — distance-based outlier filtering.
- **MoNNA** (Farhadkhani et al., 2023) — Polyak momentum + Nearest-Neighbor Averaging.

## Implemented attacks

- **A Little Is Enough** (Baruch et al., 2019) — crafted malicious gradients within noise bounds.
- **Fall of Empires** (Xie et al., 2020) — model poisoning through parameter scaling.

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

## Tags

`deep-learning` `pytorch` `reproducibility` `adversarial-attacks` `federated-learning` `distributed-learning` `byzantine-robust` `signguard` `multi-krum` `gradient-aggregation`

## References

Xu, J., Li, Z., Chen, Y., Lyu, L., & Zhou, X. (2022). *Byzantine-robust Federated Learning through Collaborative Malicious Gradient Filtering*. In Proceedings of the 42nd IEEE International Conference on Distributed Computing Systems (ICDCS). [arXiv:2109.05872v2](https://arxiv.org/pdf/2109.05872v2)

Farhadkhani, S., Guerraoui, R., Gupta, N., Hoang, L.-N., Pinot, R., & Stephan, J. (2023). *Robust Collaborative Learning with Linear Gradient Overhead*. In Proceedings of the 40th International Conference on Machine Learning (ICML). [PMLR 202:9761–9813](https://proceedings.mlr.press/v202/farhadkhani23a.html)

## Planned experiments

- Attack impact (%) for all attacks vs. percentage of Byzantine clients, for each aggregator.
- Test accuracy vs. training epochs across all aggregators.
- Verify MoNNA's theoretical linear gradient overhead in practice.

## Open questions

- **Flatten puis relink** — Est-ce que le découpage (`parameters_to_vector`) suivi d'un `vector_to_parameters` avec un optimizer externe est vraiment nécessaire ? L'optimizer garde un état interne (momentum buffer, etc.) qui n'est pas recalculé après le relink, ce qui peut rendre le clipping via l'aggregateur incohérent. Une alternative serait de faire l'update directement sur les paramètres sans repasser par l'optimizer, ou de synchroniser les buffers de momentum entre le modèle et l'optimizer. Cette question est en cours d'investigation et peut guider la conception d'une abstraction plus propre dans krum.