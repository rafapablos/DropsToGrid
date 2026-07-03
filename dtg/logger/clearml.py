import os
from argparse import Namespace
from pathlib import Path
from typing import Any

from clearml import Task
from lightning.fabric.loggers.logger import rank_zero_experiment
from lightning.pytorch.loggers import Logger
from lightning.pytorch.utilities import rank_zero_only


class ClearMlLogger(Logger):
    def __init__(
        self,
        save_dir: str | Path,
        name: str,
        project: str,
        task_type: str,
        offline: bool = False,
    ) -> None:
        self._save_dir = os.fspath(save_dir)
        self._name = name
        self._project = project
        self._task_type = task_type
        self._init_rank_zero(offline)

    @rank_zero_only
    def _init_rank_zero(self, offline: bool) -> None:
        Task.set_offline(offline)
        self._task = Task.init(
            project_name=self._project,
            task_name=self.name,
            task_type=self._task_type,
            reuse_last_task_id=False,
            auto_connect_frameworks=False,
        )

    @rank_zero_experiment
    def experiment(self) -> Task:
        return self._task

    @property
    def name(self) -> str:
        return self._name

    @property
    def version(self) -> str | None:
        return None

    @property
    def save_dir(self) -> str | None:
        version = self.version
        name = self.name
        if version is None:
            return os.path.join(self._save_dir, name)
        else:
            return os.path.join(self._save_dir, name, version)

    @rank_zero_only
    def log_metrics(
        self,
        metrics: dict[str, float],
        step: int | None = None,
    ) -> None:
        if step is None:
            raise NotImplementedError()
        logger = self.experiment().get_logger()
        for metric, value in metrics.items():
            parts = metric.split("/")
            if len(parts) == 2:
                metric, series = parts[0], parts[1]
            else:
                series = ""
            logger.report_scalar(metric, series, value, iteration=step)

    @rank_zero_only
    def log_hyperparams(
        self,
        params: dict[str, Any] | Namespace,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        return None
