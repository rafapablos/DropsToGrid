import torch
import torch.nn.functional as F
from einops import rearrange, repeat
from torch import Tensor
from torchmetrics.utilities.compute import _safe_divide

from dtg.metrics.plottable import PlotData, PlottableMetric


class FractionSkillScore(PlottableMetric):
    """
    Computes the Fraction Skill Score (FSS) with grouped conv2d for speed:
    FSS = 1 - mean((P_f-P_o)^2) / mean((P_f)^2-(P_o)^2)
    P_f is activations in forecast above a threshold in a neighborhood
    P_o is activations in observations above a threshold in a neighborhood
    """

    is_differentiable = False
    higher_is_better = True
    full_state_update = False

    def __init__(
        self,
        num_lead_times: int,
        thresholds: int | float | list[float] | torch.Tensor = [
            0.2,
            1.0,
            2.0,
            5.0,
            10.0,
        ],
        neighborhood_sizes: int | list[int] = [3, 11, 21],
    ):
        super().__init__()
        if num_lead_times <= 0:
            raise ValueError("`num_lead_times` must be > 0")
        if not isinstance(thresholds, list):
            thresholds = [thresholds]

        if not isinstance(neighborhood_sizes, list):
            neighborhood_sizes = [neighborhood_sizes]
        for neighborhood_size in neighborhood_sizes:
            assert neighborhood_size % 2 == 1

        self.num_scales = len(neighborhood_sizes)
        self.neighborhood_sizes = neighborhood_sizes
        self.num_thresholds = len(thresholds)

        zeros_shape = (self.num_scales, self.num_thresholds, num_lead_times)

        thresholds = torch.tensor(thresholds)
        self.register_buffer("thresholds", thresholds, persistent=False)

        # Precompute all kernels for grouped convolution
        kernels = []
        max_scale = max(neighborhood_sizes)

        for scale in neighborhood_sizes:
            kernel = torch.ones((1, 1, scale, scale)) / (scale**2)

            pad_total = max_scale - scale
            pad_before = pad_total // 2
            pad_after = pad_total - pad_before

            padded = F.pad(kernel, (pad_before, pad_after, pad_before, pad_after))
            kernels.append(padded)

        kernels = torch.cat(kernels, dim=0)
        kernels = repeat(kernels, "s 1 h w -> (s th) 1 h w", th=self.num_thresholds)
        self.register_buffer("kernels", kernels, persistent=False)

        self.add_state(
            "sum_obs_sq",
            default=torch.zeros(zeros_shape),
            dist_reduce_fx="sum",
        )
        self.add_state(
            "sum_fct_obs",
            default=torch.zeros(zeros_shape),
            dist_reduce_fx="sum",
        )
        self.add_state(
            "sum_fct_sq",
            default=torch.zeros(zeros_shape),
            dist_reduce_fx="sum",
        )

    def update(self, preds: Tensor, target: Tensor):
        nan_mask = torch.isnan(target)
        preds = preds.masked_fill(nan_mask, torch.nan)

        sum_obs_sq, sum_fct_obs, sum_fct_sq = _fss_grouped_update(
            preds,
            target,
            self.thresholds,
            self.kernels,
            self.num_scales,
            self.num_thresholds,
        )
        self.sum_obs_sq += sum_obs_sq
        self.sum_fct_obs += sum_fct_obs
        self.sum_fct_sq += sum_fct_sq

    def compute(self) -> torch.Tensor:
        return self._compute(reduce_mean=True).mean()

    def _compute(self, reduce_mean) -> torch.Tensor:
        sum_obs_sq = self.sum_obs_sq
        sum_fct_obs = self.sum_fct_obs
        sum_fct_sq = self.sum_fct_sq

        if reduce_mean:
            sum_obs_sq = sum_obs_sq.sum(dim=2)
            sum_fct_obs = sum_fct_obs.sum(dim=2)
            sum_fct_sq = sum_fct_sq.sum(dim=2)

        numer = sum_fct_sq - 2.0 * sum_fct_obs + sum_obs_sq
        denom = sum_fct_sq + sum_obs_sq
        return 1.0 - _safe_divide(numer, denom)

    def get_plot_data(self) -> list[list[PlotData]]:
        data = self._compute(reduce_mean=False)
        plots = []
        for n, neighborhood_size in enumerate(self.neighborhood_sizes):
            plot_data = []
            for i, threshold in enumerate(self.thresholds):
                num_lead_times = data.shape[2]
                plot_data.append(
                    PlotData(
                        title=f"FSS {neighborhood_size}x{neighborhood_size} at {threshold:4.1f} mm/h",
                        x=list(range(1, num_lead_times + 1)),
                        y=data[n][i].cpu().tolist(),
                        x_label="Lead Time",
                        y_label=f"FSS {neighborhood_size}x{neighborhood_size}",
                    )
                )
            plots.append(plot_data)
        return plots


def _fss_grouped_update(
    preds: Tensor,
    target: Tensor,
    thresholds: Tensor,
    kernels: Tensor,
    num_scales: int,
    num_thresholds: int,
) -> tuple[Tensor, Tensor, Tensor]:
    b, t, _, _, _ = preds.shape

    n_combinations = num_scales * num_thresholds

    preds = rearrange(preds, "b t 1 h w -> (b t) 1 h w")
    target = rearrange(target, "b t 1 h w -> (b t) 1 h w")

    # Expand thresholds to (n_combinations, 1, 1)
    thresholds = repeat(thresholds, "th -> (s th)", s=num_scales).view(
        n_combinations, 1, 1
    )

    # Thresholding (bt, n_combinations, h, w)
    preds_bin = (preds >= thresholds).float()
    target_bin = (target >= thresholds).float()

    # Padding for grouped convolution
    max_kernel_size = kernels.shape[-1]
    pad = max_kernel_size // 2
    preds_bin = F.pad(preds_bin, (pad, pad, pad, pad))
    target_bin = F.pad(target_bin, (pad, pad, pad, pad))

    # Grouped convolution: each group handles one (scale, threshold) combination
    preds_conv = F.conv2d(preds_bin, kernels, groups=n_combinations)
    target_conv = F.conv2d(target_bin, kernels, groups=n_combinations)

    # Reshape
    preds_conv = rearrange(preds_conv, "(b t) s_th h w -> s_th t (b h w)", b=b, t=t)
    target_conv = rearrange(target_conv, "(b t) s_th h w -> s_th t (b h w)", b=b, t=t)

    # Compute sums
    sum_obs_sq = torch.nansum(target_conv**2, dim=2)
    sum_fct_obs = torch.nansum(preds_conv * target_conv, dim=2)
    sum_fct_sq = torch.nansum(preds_conv**2, dim=2)

    # Reshape back to [scales, thresholds, time]
    sum_obs_sq = rearrange(
        sum_obs_sq, "(s th) t -> s th t", s=num_scales, th=num_thresholds
    )
    sum_fct_obs = rearrange(
        sum_fct_obs, "(s th) t -> s th t", s=num_scales, th=num_thresholds
    )
    sum_fct_sq = rearrange(
        sum_fct_sq, "(s th) t -> s th t", s=num_scales, th=num_thresholds
    )

    return sum_obs_sq, sum_fct_obs, sum_fct_sq
