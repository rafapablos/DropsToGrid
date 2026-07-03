from dataclasses import dataclass

import torchmetrics


@dataclass
class PlotData:
    title: str
    x_label: str
    y_label: str
    x: list[int]
    y: list[float]


class PlottableMetric(torchmetrics.Metric):
    """Base class for plottable metrics."""

    def get_plot_data(self) -> list[PlotData]:
        raise NotImplementedError()
