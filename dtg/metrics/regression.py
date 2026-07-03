import torch
from einops import rearrange
from torchmetrics.utilities.compute import _safe_divide

from dtg.metrics.plottable import PlotData, PlottableMetric


class MeanAbsoluteError(PlottableMetric):
    """Computes Mean Absolute Error (MAE) per lead time."""

    is_differentiable = True
    higher_is_better = False
    full_state_update = False

    def __init__(self, num_lead_times: int):
        super().__init__()
        self.add_state(
            "abs_error_sum",
            default=torch.zeros(num_lead_times),
            dist_reduce_fx="sum",
        )
        self.add_state(
            "count",
            default=torch.zeros(num_lead_times),
            dist_reduce_fx="sum",
        )

    def update(self, preds: torch.Tensor, target: torch.Tensor):
        nan_mask = torch.isnan(target) | torch.isnan(preds)
        preds = preds.masked_fill(nan_mask, 0.0)
        target = target.masked_fill(nan_mask, 0.0)
        valid = (~nan_mask).float()

        preds = rearrange(preds, "b t c h w -> (b c h w) t")
        target = rearrange(target, "b t c h w -> (b c h w) t")
        valid = rearrange(valid, "b t c h w -> (b c h w) t")

        abs_error = torch.abs(preds - target)
        self.abs_error_sum += (abs_error * valid).sum(dim=0)
        self.count += valid.sum(dim=0)

    def compute(self) -> torch.Tensor:
        return _safe_divide(self.abs_error_sum, self.count).mean()

    def get_plot_data(self) -> list[PlotData]:
        mae = _safe_divide(self.abs_error_sum, self.count)
        return [
            PlotData(
                title="Mean Absolute Error",
                x=list(range(1, len(mae) + 1)),
                y=mae.cpu().tolist(),
                x_label="Lead Time",
                y_label="MAE",
            )
        ]


class MeanSquaredError(PlottableMetric):
    """Computes Mean Squared Error (MSE) per lead time."""

    is_differentiable = True
    higher_is_better = False
    full_state_update = False

    def __init__(self, num_lead_times: int):
        super().__init__()
        self.add_state(
            "squared_error_sum",
            default=torch.zeros(num_lead_times),
            dist_reduce_fx="sum",
        )
        self.add_state(
            "count",
            default=torch.zeros(num_lead_times),
            dist_reduce_fx="sum",
        )

    def update(self, preds: torch.Tensor, target: torch.Tensor):
        nan_mask = torch.isnan(target) | torch.isnan(preds)
        preds = preds.masked_fill(nan_mask, 0.0)
        target = target.masked_fill(nan_mask, 0.0)
        valid = (~nan_mask).float()

        preds = rearrange(preds, "b t c h w -> (b c h w) t")
        target = rearrange(target, "b t c h w -> (b c h w) t")
        valid = rearrange(valid, "b t c h w -> (b c h w) t")

        squared_error = (preds - target) ** 2
        self.squared_error_sum += (squared_error * valid).sum(dim=0)
        self.count += valid.sum(dim=0)

    def compute(self) -> torch.Tensor:
        return _safe_divide(self.squared_error_sum, self.count).mean()

    def get_plot_data(self) -> list[PlotData]:
        mse = _safe_divide(self.squared_error_sum, self.count)
        return [
            PlotData(
                title="Mean Squared Error",
                x=list(range(1, len(mse) + 1)),
                y=mse.cpu().tolist(),
                x_label="Lead Time",
                y_label="MSE",
            )
        ]
