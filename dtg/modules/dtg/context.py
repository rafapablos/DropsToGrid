from typing import Tuple

import torch


class RandomMasker:
    """
    Return random subset mask, but:
      - Within each batch sample, the number of kept values is the same.
      - The mask is shared across T and C (same spatial mask across time and channels).


    Parameters
    ----------
    a : float or int, optional
        Minimum number of indices. If smaller than 1, represents a percentage of
        points.

    b : float or int, optional
        Maximum number of indices. If smaller than 1, represents a percentage of
        points.
    """

    def __init__(self, a=0.1, b=0.5):
        self.a = a
        self.b = b

    def __call__(self, x: torch.Tensor):
        b, t, c, h, w = x.shape

        x_flat = x.reshape(b, t * c, -1)
        valid_flat = ~torch.isnan(x_flat)
        valid_flat = torch.all(valid_flat, dim=1)  # [B, H*W] with all T,C non-nan

        # Count valid entries per sample
        num_valid = valid_flat.sum(dim=1)

        # Sample mask ratio ∈ [a, b] for each batch sample
        mask_ratio = torch.empty(b, device=x.device).uniform_(self.a, self.b)
        num_keep = (mask_ratio * num_valid.float()).floor().long()

        # Random values
        rand = torch.rand_like(x_flat[:, 0])
        rand.masked_fill_(~valid_flat, float("inf"))

        # Sort and threshold
        sorted_rand, _ = rand.sort(dim=1)
        threshold_indices = num_keep.clamp(max=rand.size(1) - 1).unsqueeze(1)
        threshold_vals = sorted_rand.gather(1, threshold_indices)

        # Mask
        keep_mask_flat = rand <= threshold_vals
        keep_mask = keep_mask_flat.view(b, 1, 1, h, w).expand(-1, t, c, -1, -1)
        return keep_mask


class NoMasker:
    """
    Return mask of ones.
    """

    def __call__(self, X: torch.Tensor):
        mask = torch.ones_like(X, dtype=torch.bool, device=X.device)
        return mask.masked_fill(torch.isnan(X), 0)


class GridCntxtTrgtGetter(torch.nn.Module):
    """
    Split grids of values (e.g. images) into context and target points.
    """

    def __init__(self, train_range: Tuple[float, float] = (0.0, 0.3)):
        super().__init__()
        self.train_masker = RandomMasker(a=train_range[0], b=train_range[1])
        self.val_masker = RandomMasker(a=0.5, b=0.5)
        self.no_masker = NoMasker()

    def __call__(self, x: torch.Tensor, context: bool, stage: str) -> torch.Tensor:
        if context and stage in ["train", "val"]:
            if stage == "train":
                context_idxs = self.train_masker(x)
            else:
                context_idxs = self.val_masker(x)
            return x.masked_fill(~context_idxs, torch.nan)
        else:
            target_idxs = self.no_masker(x)
            return x.masked_fill(~target_idxs, torch.nan)
