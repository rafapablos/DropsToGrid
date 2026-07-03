import math

import torch
import torch.distributions as td
from einops import rearrange
from torchmetrics.utilities.compute import _safe_divide

from dtg.loss.distributions import ZeroInflatedGamma
from dtg.metrics.plottable import PlotData, PlottableMetric


def beta_fn(a, b):
    a = torch.as_tensor(a, dtype=torch.float32)
    b = torch.as_tensor(b, dtype=torch.float32)
    return torch.exp(torch.lgamma(a) + torch.lgamma(b) - torch.lgamma(a + b))


def crps_gamma_full(alpha, beta, y):
    """
    Full CRPS for Gamma(alpha, beta) using Scheuerer & Hamill (2015) formula with Beta function.
    alpha, beta, y : tensors (broadcastable)
    """
    y = torch.clamp_min(y, 0.0)

    # Gamma CDF
    dist = td.Gamma(alpha, beta)
    F_y = dist.cdf(y)
    F_y_plus = td.Gamma(alpha + 1.0, beta).cdf(y)

    # Beta function term: B(alpha+1/2, 1/2)
    B_term = beta_fn(alpha + 0.5, 0.5)

    crps = (
        y * (2.0 * F_y - 1.0)
        - (alpha / beta) * (2.0 * F_y_plus - 1.0)
        - (alpha / (beta * math.pi)) * B_term
    )

    return torch.clamp(crps, min=0.0)

def _crps_zig(
    alpha: torch.Tensor, beta: torch.Tensor, logits_zero: torch.Tensor, y: torch.Tensor
) -> torch.Tensor:
    """CRPS for Zero-Inflated Gamma.
    Here sigmoid(logits_zero) = p_nonzero (per your class).
    """
    p_nonzero = torch.sigmoid(logits_zero)
    p_nonzero = (p_nonzero >= 0.5).float()
    p_zero = 1.0 - p_nonzero

    crps_g = crps_gamma_full(alpha, beta, y)
    is_zero = y <= 0.2

    # If y==0: CRPS = p_nonzero * CRPS_Gamma(y=0)
    # If y>0:  CRPS = p_zero * y + p_nonzero * CRPS_Gamma(y)
    return torch.where(is_zero, p_nonzero * crps_g, p_zero * y + p_nonzero * crps_g)


class ContinuousRankedProbabilityScore(PlottableMetric):
    """
    Continuous Ranked Probability Score for Zero-Inflated Gamma.
    Lower is better.
    """

    is_differentiable = False
    higher_is_better = False
    full_state_update = False

    def __init__(self, num_lead_times: int):
        super().__init__()
        if num_lead_times <= 0:
            raise ValueError("`num_lead_times` must be > 0")

        self.add_state(
            "crps_sum",
            default=torch.zeros(num_lead_times),
            dist_reduce_fx="sum",
        )
        self.add_state(
            "count",
            default=torch.zeros(num_lead_times),
            dist_reduce_fx="sum",
        )

    def update(
        self,
        distribution: ZeroInflatedGamma,
        target: torch.Tensor,
    ):
        assert distribution.mean.shape == target.shape, (
            "Distribution and target must have matching shape"
        )

        nan_mask = torch.isnan(target)
        target = target.masked_fill(nan_mask, 0.0)

        crps = _crps_zig(distribution.alphas, distribution.betas, distribution.logits_zero, target)

        crps = crps.masked_fill(nan_mask, 0.0)
        crps = rearrange(crps, "b t c h w -> (b c h w) t")

        self.crps_sum += torch.nan_to_num(crps, nan=0.0).sum(dim=0)
        self.count += (~nan_mask).sum(dim=(0, 2, 3, 4))

    def compute(self) -> torch.Tensor:
        return _safe_divide(self.crps_sum, self.count).mean()

    def _compute_per_leadtime(self) -> torch.Tensor:
        return _safe_divide(self.crps_sum, self.count)

    def get_plot_data(self) -> list[PlotData]:
        crps = self._compute_per_leadtime()
        num_lead_times = crps.shape[0]
        return [
            PlotData(
                title="CRPS for Zero-Inflated Gamma",
                x=list(range(1, num_lead_times + 1)),
                y=crps.cpu().tolist(),
                x_label="Lead Time",
                y_label="CRPS",
            )
        ]
