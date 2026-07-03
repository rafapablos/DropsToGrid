import torch
import torch.nn as nn


class CNPFLoss(nn.Module):
    """
    Compute the negative log likelihood loss for members of the conditional neural process (sub-)family.
    """

    def __init__(self, reduction="mean"):
        super().__init__()
        self.reduction = reduction

    def forward(self, pred_outputs, Y_trgt):
        # \sum_t log p(y^t|z)
        # \sum_t log p(y^t|z). size = [z_samples, batch_size]
        sum_log_p_yCz = sum_log_prob(pred_outputs, Y_trgt)

        # size = [batch_size]
        loss = -sum_log_p_yCz

        if self.reduction is None:
            # size = [batch_size]
            return loss
        elif self.reduction == "mean":
            # size = [1]
            return loss.mean(0)
        elif self.reduction == "sum":
            # size = [1]
            return loss.sum(0)
        else:
            raise ValueError(f"Unknown {self.reduction}")


def sum_from_nth_dim(t, dim, mask):
    """Sum all dims from `dim`. E.g. sum_after_nth_dim(torch.rand(2,3,4,5), 2).shape = [2,3]"""
    t = t.masked_fill(mask, torch.nan)
    return torch.nansum(t.view(*t.shape[:dim], -1), dim=-1)


def sum_log_prob(prob, sample):
    """Compute log probability then sum all but the batch."""
    # size = [batch_size, *]
    nan_mask = sample.isnan()
    sample = sample.masked_fill(nan_mask, 0)
    log_p = prob.compute_loss(sample)
    # size = [batch_size]
    sum_log_p = sum_from_nth_dim(log_p, 1, nan_mask.any(dim=2))
    return sum_log_p
