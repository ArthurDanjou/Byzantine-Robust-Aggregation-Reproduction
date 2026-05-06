import numpy as np
import torch
from sklearn.cluster import DBSCAN, MeanShift, estimate_bandwidth


class Aggregator:
    """Base class for aggregators."""

    def aggregate(
        self, gradients: list[torch.Tensor]
    ) -> tuple[torch.Tensor, list[int], float]:
        """Aggregates the given node values using the aggregator's algorithm.

        Args:
        ----
            gradients (list[torch.Tensor]): The gradients from the nodes.

        Returns:
        -------
            tuple[torch.Tensor, list[int], float]:
                The aggregated gradient, selected indices, and Byzantine ratio.
        """
        raise NotImplementedError


class MultiKrum(Aggregator):
    """Implements the Multi-Krum aggregator ()."""

    def __init__(self, n: int, m: int, f: int):
        """Initializes the Multi-Krum aggregator with the given parameters.

        Args:
        ----
            n (int): The number of nodes.
            m (int): The number of good nodes.
            f (int): The number of Byzantine nodes.
        """
        self.n = n
        self.m = m
        self.f = f

    def aggregate(
        self, gradients: list[torch.Tensor]
    ) -> tuple[torch.Tensor, list[int], float]:
        """Aggregates the given node values using the Multi-Krum algorithm.

        Args:
        ----
            gradients (list[torch.Tensor]): The gradients from the nodes.

        Returns:
        -------
            tuple[torch.Tensor, list[int], float]:
                The aggregated gradient, selected indices, and Byzantine ratio.
        """
        grads = torch.stack(gradients)
        distances = torch.cdist(grads, grads, p=2.0)
        sorted_distances, _ = torch.sort(distances, dim=1)

        scores = torch.sum(sorted_distances[:, 1 : self.n - self.f], dim=1)
        _, best_indices = torch.topk(scores, k=self.m, largest=False)
        grad = torch.mean(grads[best_indices], dim=0)

        select_idx = best_indices.tolist()

        if self.f > 0:
            byz_num = sum(1 for i in select_idx if i < self.f)
            byz_ratio = byz_num / self.f
        else:
            byz_ratio = 0.0

        return grad, select_idx, byz_ratio


class SignGuard(Aggregator):
    """Implements the SignGuard aggregator (Xu et al. (2022))."""

    def __init__(self, f: int, use_dbscan: bool = False):
        """Initializes the SignGuard aggregator with the given parameters.

        Args:
        ----
            f (int): The number of Byzantine nodes.
            use_dbscan (bool): Whether to use DBSCAN for outlier detection.
        """
        self.f = f
        self.use_dbscan = use_dbscan

    def aggregate(
        self, gradients: list[torch.Tensor]
    ) -> tuple[torch.Tensor, list[int], float]:
        """Aggregates the gradients using the SignGuard algorithm.

        Args:
        ----
            gradients (list[torch.Tensor]): The gradients from the clients.

        Returns:
        -------
            tuple[torch.Tensor, list[int], float]:
                The aggregated gradient, the selected indices, and the Byzantine ratio.
        """
        n = len(gradients)

        grads = torch.stack(gradients, dim=0)
        torch.nan_to_num_(grads, nan=0.0)

        norms = torch.norm(grads, dim=1)
        norm_med = torch.median(norms)
        mask_norm = (norms > 0.1 * norm_med) & (norms < 3.0 * norm_med)
        num_param = grads.shape[1]
        num_spars = int(0.1 * num_param)
        idx = torch.randint(0, num_param - num_spars, size=(1,)).item()
        sign_grads = torch.sign(grads[:, idx : idx + num_spars])
        pos_ratio = (sign_grads == 1.0).float().mean(dim=1)
        zero_ratio = (sign_grads == 0.0).float().mean(dim=1)
        neg_ratio = (sign_grads == -1.0).float().mean(dim=1)
        pos_feat = pos_ratio / (pos_ratio.max() + 1e-8)
        zero_feat = zero_ratio / (zero_ratio.max() + 1e-8)
        neg_feat = neg_ratio / (neg_ratio.max() + 1e-8)
        sign_feat = torch.stack([pos_feat, zero_feat, neg_feat], dim=1).cpu().numpy()

        if self.use_dbscan:
            clustering = DBSCAN(eps=0.05, min_samples=2).fit(sign_feat)
        else:
            bandwidth = estimate_bandwidth(
                sign_feat, quantile=0.5, n_samples=min(50, n)
            )
            if bandwidth <= 0:
                bandwidth = 0.1
            clustering = MeanShift(
                bandwidth=bandwidth, bin_seeding=True, cluster_all=False
            ).fit(sign_feat)
        labels = clustering.labels_
        valid_labels = labels[labels != -1]
        if len(valid_labels) == 0:
            benign_class = 0
        else:
            values, counts = np.unique(valid_labels, return_counts=True)
            benign_class = values[np.argmax(counts)]
        mask_sign = torch.tensor(labels == benign_class, device=grads.device)

        final_mask = mask_norm & mask_sign
        benign_idx = torch.where(final_mask)[0].tolist()

        if len(benign_idx) == 0:
            benign_idx = list(range(n))

        norm_clip = norm_med.item()

        scales = norm_clip / torch.clamp_min(norms, norm_clip)
        grads_clipped = grads * scales.unsqueeze(1)

        global_grad = grads_clipped[benign_idx].mean(dim=0)

        byz_num = sum(1 for i in benign_idx if i < self.f)
        byz_ratio = (byz_num / self.f) if self.f > 0 else 0.0

        return global_grad, benign_idx, byz_ratio
