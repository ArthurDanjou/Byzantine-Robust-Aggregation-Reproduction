import torch


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
