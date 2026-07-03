from typing import Literal

import torch
from einops import rearrange
from torchmetrics.utilities.compute import _safe_divide

from dtg.metrics.plottable import PlotData, PlottableMetric


class BinaryMetric(PlottableMetric):
    """Computes Binary Metrics: CSI, FBI, TP, FP, FN."""

    is_differentiable = False
    full_state_update = False

    true_positives: torch.Tensor
    false_positives: torch.Tensor
    false_negatives: torch.Tensor
    total: torch.Tensor

    def __init__(
        self,
        num_lead_times: int,
        variable: Literal["csi", "fbi", "tp", "fp", "fn"],
        thresholds: int | float | list[float] = [0.2, 1.0, 2.0, 5.0, 10.0],
    ):
        super().__init__()
        if num_lead_times <= 0:
            raise ValueError("`num_lead_times` must be > 0")

        self.variable = variable

        if not isinstance(thresholds, list):
            thresholds = [thresholds]
        self.thresholds = thresholds

        zeros_shape = (len(thresholds), num_lead_times)
        self.add_state(
            "true_positives",
            default=torch.zeros(zeros_shape),
            dist_reduce_fx="sum",
        )
        self.add_state(
            "false_positives",
            default=torch.zeros(zeros_shape),
            dist_reduce_fx="sum",
        )
        self.add_state(
            "false_negatives",
            default=torch.zeros(zeros_shape),
            dist_reduce_fx="sum",
        )
        self.add_state(
            "total",
            default=torch.zeros(num_lead_times),
            dist_reduce_fx="sum",
        )

    def update(self, preds: torch.Tensor, target: torch.Tensor):
        nan_mask = torch.isnan(target) | torch.isnan(preds)
        preds = preds.masked_fill(nan_mask, torch.nan)
        target = target.masked_fill(nan_mask, torch.nan)

        tp, fp, fn, count = _metric_update(preds, target, self.thresholds)
        self.true_positives += tp
        self.false_positives += fp
        self.false_negatives += fn
        self.total += count

    def compute(self) -> torch.Tensor:
        return self._compute(reduce_mean=True)

    def _compute(self, reduce_mean) -> torch.Tensor:
        true_positives = self.true_positives
        false_positives = self.false_positives
        false_negatives = self.false_negatives
        total = self.total

        if reduce_mean:
            true_positives = true_positives.sum(dim=1)
            false_positives = false_positives.sum(dim=1)
            false_negatives = false_negatives.sum(dim=1)
            total = total.sum(dim=0)

        match self.variable:
            case "csi":
                output = _safe_divide(
                    true_positives,
                    true_positives + false_positives + false_negatives,
                )
            case "fbi":
                output = _safe_divide(
                    true_positives + false_positives,
                    true_positives + false_negatives,
                )
            case "tp":
                output = true_positives / total
            case "fp":
                output = false_positives / total
            case "fn":
                output = false_negatives / total
            case _:
                raise NotImplementedError

        if reduce_mean:
            return output.mean()
        return output

    def get_plot_data(self) -> list[PlotData]:
        data = self._compute(reduce_mean=False)
        plot_data = []
        for i, threshold in enumerate(self.thresholds):
            num_lead_times = data.shape[1]
            plot_data.append(
                PlotData(
                    title=f"{self.variable.upper()} at {threshold:4.1f} mm/h",
                    x=list(range(1, num_lead_times + 1)),
                    y=data[i].cpu().tolist(),
                    x_label="Lead Time",
                    y_label=self.variable.upper(),
                )
            )
        return plot_data


def _metric_update(
    preds: torch.Tensor,
    target: torch.Tensor,
    thresholds: list[float],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    assert preds.shape == target.shape

    # Convert to binary fields with the given intensity threshold
    thresholds_tensor = torch.tensor(thresholds, device=preds.device).view(
        -1, 1, 1, 1, 1, 1
    )

    # [num_thresh, B, T, C, H, W]
    preds_binary = (preds.unsqueeze(0) >= thresholds_tensor).bool()
    target_binary = (target.unsqueeze(0) >= thresholds_tensor).bool()
    mask = torch.isnan(target)

    # Output [thresholds, time]
    preds_binary = rearrange(preds_binary, "th b t c h w -> (b c h w) th t")
    target_binary = rearrange(target_binary, "th b t c h w -> (b c h w) th t")
    # Output [time]
    mask = rearrange(mask, "b t c h w -> (b c h w) t")

    true_positives = torch.logical_and(preds_binary, target_binary).sum(dim=0)
    false_positives = torch.logical_and(preds_binary, ~target_binary).sum(dim=0)
    false_negatives = torch.logical_and(~preds_binary, target_binary).sum(dim=0)
    count = (~mask).sum(dim=0)

    return true_positives, false_positives, false_negatives, count
