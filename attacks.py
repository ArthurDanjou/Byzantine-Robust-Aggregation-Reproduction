import numpy as np
import torch

from aggregators import Aggregator


def a_little_is_enough_attack(
    gradients: list[torch.Tensor],
    num_byzantines: int,
    z: float = 1.0,
) -> list[torch.Tensor]:
    """Implements the a-little-is-enough attack (Baruch et al., 2019).

    Args:
    ----
        gradients: List of node gradients.
        num_byzantines: The number of byzantine workers in the network.
        z: The number of standard deviations to use for the attack.

    Returns:
    -------
        A list containing the identical malicious gradient for each byzantine worker.
    """
    if num_byzantines == 0:
        return list()

    stacked_gradients = torch.stack(gradients)

    mu = torch.mean(stacked_gradients, dim=0)
    sigma = torch.std(stacked_gradients, dim=0, unbiased=False)

    malicious_gradient = mu - z * sigma

    return [malicious_gradient] * num_byzantines


def fall_of_empires_attack(
    honest_gradients: list[torch.Tensor],
    defense_func: Aggregator,
    num_byzantine: int = 0,
    epsilon: float = -1.0,
) -> list[torch.Tensor]:
    if num_byzantine == 0:
        return list()

    grad_avg = torch.mean(torch.stack(honest_gradients), dim=0)

    if epsilon < 0:
        best_eps = 0
        max_damage = -1

        num_evals = int(np.ceil(-epsilon))
        test_epsilons = torch.linspace(0.1, 5.0, steps=num_evals)

        for test_eps in test_epsilons:
            test_attack = grad_avg * (-test_eps)
            simulated_grads = honest_gradients + [test_attack] * num_byzantine
            aggregated, _, _ = defense_func.aggregate(simulated_grads)

            damage = torch.sum((aggregated - grad_avg) ** 2).item()

            if damage > max_damage:
                max_damage = damage
                best_eps = test_eps

        epsilon = best_eps

    byz_grad = grad_avg * (-epsilon)
    return [byz_grad] * num_byzantine
