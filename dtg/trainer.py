from typing import Literal

import lightning.pytorch as pl
import torch

from dtg.callbacks.factory import create_callbacks
from dtg.logger.factory import create_logger

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


class DtGTrainer(pl.Trainer):
    def __init__(
        self,
        run_name: str,
        accelerator: str = "auto",
        devices: list[int] | str | int = "auto",
        precision: Literal["32", "bf16-mixed"] = "32",
        callbacks: list[pl.Callback] | pl.Callback | None = None,
        max_epochs: int = 50,
        limit_train_batches: int | float | None = 2000,
        limit_val_batches: int | float | None = None,
        limit_test_batches: int | float | None = None,
        log_every_n_steps: int | None = 50,
        accumulate_grad_batches: int = 1,
        gradient_clip_val: int | float | None = None,
        deterministic: bool | None = None,
        ema_decay: float | None = None,
        ema_delay: int | None = None,
        project: str = "DropsToGrid",
        logger: Literal["clearml"] | None = None,
        offline: bool = False,
        profiler: str | None = None,
    ):
        root_dir = "runs"

        _logger = create_logger(
            logger_type=logger,
            root_dir=root_dir,
            run_name=run_name,
            project=project,
            task_type="testing" if "_vs_" in run_name else "training",
            offline=offline,
        )

        if callbacks is None:
            callbacks = []
        elif not isinstance(callbacks, list):
            callbacks = [callbacks]
        callbacks.extend(create_callbacks(ema_decay=ema_decay, ema_delay=ema_delay))

        super().__init__(
            accelerator=accelerator,
            default_root_dir=root_dir,
            devices=devices,
            callbacks=callbacks,
            max_epochs=max_epochs,
            limit_train_batches=limit_train_batches,
            limit_val_batches=limit_val_batches,
            limit_test_batches=limit_test_batches,
            log_every_n_steps=log_every_n_steps,
            accumulate_grad_batches=accumulate_grad_batches,
            gradient_clip_val=gradient_clip_val,
            deterministic=deterministic,
            num_sanity_val_steps=0,
            precision=precision,
            logger=_logger,
            profiler=profiler,
        )
