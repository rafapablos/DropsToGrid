import json
import os
from dataclasses import asdict

import lightning.pytorch as pl
from lightning.fabric.utilities.cloud_io import get_filesystem
from lightning.pytorch.utilities import rank_zero_only
from torchmetrics import MetricCollection
from typing_extensions import Literal

from dtg.logger.clearml import ClearMlLogger
from dtg.metrics.plottable import PlotData, PlottableMetric


class LogPlots(pl.Callback):
    def setup(self, trainer: pl.Trainer, pl_module: pl.LightningModule, stage: str):
        if trainer.log_dir is None:
            self.plot_dir = None
            return

        if "_vs_" not in trainer.log_dir:
            self.plot_dir = os.path.join(trainer.log_dir, "plots")
        else:
            parts = os.path.normpath(trainer.log_dir).split(os.sep)
            exp = parts[-1]

            base, targets = exp.split("_vs_")
            self.plot_dir = os.path.join(parts[0], base, "test", targets, "plots")

        self._fs = get_filesystem(self.plot_dir)
        if trainer.is_global_zero:
            self._fs.makedirs(self.plot_dir, exist_ok=True)

    @rank_zero_only
    def on_validation_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        self._log_plots("val", trainer, pl_module)

    @rank_zero_only
    def on_test_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        self._log_overall_metrics("test", pl_module)
        self._log_plots("test", trainer, pl_module)

    def _log_overall_metrics(
        self,
        split: Literal["val", "test"],
        pl_module: pl.LightningModule,
    ):
        metrics = getattr(pl_module, f"{split}_metrics", None)
        if (
            self.plot_dir is None
            or metrics is None
            or not isinstance(metrics, MetricCollection)
        ):
            return

        overall_metrics = {}
        for name, metric in metrics.items():
            overall_metrics[name] = metric.compute().item()

        out_path = os.path.join(self.plot_dir, f"{split}_overall.json")
        with self._fs.open(out_path, mode="w") as file:
            json.dump(overall_metrics, file, indent=2)

    def _log_plots(
        self,
        split: Literal["val", "test"],
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
    ):
        logger = trainer.logger
        if trainer.logger is None:
            return

        metrics = getattr(pl_module, f"{split}_metrics", None)
        if metrics is None or not isinstance(metrics, MetricCollection):
            return

        for name, metric in metrics.items():
            if not isinstance(metric, PlottableMetric):
                continue
            plot_data = metric.get_plot_data()

            def _plot(plot_data, name):
                if isinstance(logger, ClearMlLogger):
                    self._log_to_clearml(name, plot_data, logger)
                self._write_plots_to_file(name, plot_data)

            # FSS returns a list of list[PlotData]
            if isinstance(plot_data[0], PlotData):
                _plot(plot_data, name)
            else:
                for i, plot in enumerate(plot_data):
                    _plot(plot, f"{name}_{i}")

    def _log_to_clearml(
        self,
        name: str,
        plots: list[PlotData],
        logger: ClearMlLogger,
    ):
        for plot in plots:
            logger.experiment().logger.report_scatter2d(
                title=name,
                series=plot.title,
                scatter=[(x, y) for x, y in zip(plot.x, plot.y)],
                xaxis=plot.x_label,
                yaxis=plot.y_label,
            )

    def _write_plots_to_file(self, name: str, plots: list[PlotData]):
        if self.plot_dir is None:
            return

        out_path = os.path.join(self.plot_dir, f"{name.replace('/', '_')}.json")
        with self._fs.open(out_path, mode="w") as file:
            json.dump({"plots": [asdict(plot) for plot in plots]}, file, indent=2)
