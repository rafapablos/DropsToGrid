import itertools
from typing import Any

from lightning.pytorch.utilities.types import STEP_OUTPUT
from torch.optim.swa_utils import get_ema_avg_fn
from typing_extensions import override

from dtg.callbacks.weight_averaging import WeightAveraging


class EMA(WeightAveraging):
    def __init__(
        self,
        decay: float = 0.999,
        start_step: int | None = 0,
        device=None,
        use_buffers=True,
    ):
        """
        EMA with warmup delay for WeightAveraging callback.

        Args:
            decay (float): EMA decay factor (close to 1 = slower updates).
            start_step (int): Step after which EMA averaging starts.
            device (torch.device or None): Device for EMA weights.
            use_buffers (bool): Track buffers like BatchNorm running stats.
        """
        super().__init__(
            device=device, avg_fn=get_ema_avg_fn(decay), use_buffers=use_buffers
        )

        self.start_step = start_step if start_step else 0

    @override
    def should_update(self, step_idx=None, epoch_idx=None) -> bool:
        """
        Only start EMA updates after `start_step` training steps.
        """
        if step_idx is not None:
            return step_idx >= self.start_step
        return False

    def _copy_current_to_average(self, pl_module: "pl.LightningModule") -> None:
        """Copies the parameter values from the :class:`AveragedModel` to the current model.

        Args:
            pl_module: The current :class:`~lightning.pytorch.core.LightningModule` instance.

        """
        assert self._average_model is not None
        average_params = itertools.chain(
            self._average_model.module.parameters(),
            self._average_model.module.buffers(),
        )
        current_params = itertools.chain(pl_module.parameters(), pl_module.buffers())
        for average_param, current_param in zip(average_params, current_params):
            average_param.data.copy_(current_param.data)

    @override
    def on_train_batch_end(
        self,
        trainer: "pl.Trainer",
        pl_module: "pl.LightningModule",
        outputs: STEP_OUTPUT,
        batch: Any,
        batch_idx: int,
    ) -> None:
        """Called when a training batch ends.

        Updates the :class:`AveragedModel` parameters, if requested by ``self.should_update()``.

        Args:
            trainer: The current :class:`~lightning.pytorch.trainer.trainer.Trainer` instance.
            pl_module: The current :class:`~lightning.pytorch.core.LightningModule` instance.
            outputs: Outputs from the training batch.
            batch: The training batch.
            batch_idx: Index of the training batch.

        """
        # trainer.global_step is the number of optimizer steps taken so far, i.e. 1 after the first optimizer step. To
        # make step_idx consistent with epoch_idx, we'll pass a zero-based index.
        step_idx = trainer.global_step - 1
        if (trainer.global_step > self._latest_update_step) and self.should_update(
            step_idx=step_idx
        ):
            assert self._average_model is not None
            if self._latest_update_step == 0:
                self._copy_current_to_average(pl_module)
            else:
                self._average_model.update_parameters(pl_module)
            self._latest_update_step = trainer.global_step

    @override
    def on_train_epoch_end(
        self, trainer: "pl.Trainer", pl_module: "pl.LightningModule"
    ) -> None:
        """Called when a training epoch ends.

        Updates the :class:`AveragedModel` parameters, if requested by ``self.should_update()``.

        Args:
            trainer: The current :class:`~lightning.pytorch.trainer.trainer.Trainer` instance.
            pl_module: The current :class:`~lightning.pytorch.core.LightningModule` instance.

        """
        if (trainer.current_epoch > self._latest_update_epoch) and self.should_update(
            epoch_idx=trainer.current_epoch
        ):
            assert self._average_model is not None
            if self._latest_update_epoch == -1:
                self._copy_current_to_average(pl_module)
            else:
                self._average_model.update_parameters(pl_module)
            self._latest_update_epoch = trainer.current_epoch

    @override
    def on_validation_epoch_start(
        self, trainer: "pl.Trainer", pl_module: "pl.LightningModule"
    ) -> None:
        """Called when a validation epoch begins.

        Transfers parameter values from the :class:`AveragedModel` to the current model.

        Args:
            trainer: The current :class:`~lightning.pytorch.trainer.trainer.Trainer` instance.
            pl_module: The current :class:`~lightning.pytorch.core.LightningModule` instance.

        """
        if self._average_model is not None and (
            self._latest_update_epoch != -1 or self._latest_update_step != 0
        ):
            self._swap_models(pl_module)

    @override
    def on_validation_epoch_end(
        self, trainer: "pl.Trainer", pl_module: "pl.LightningModule"
    ) -> None:
        """Called when a validation epoch ends.

        Recovers the current model parameters from the :class:`AveragedModel`.

        Args:
            trainer: The current :class:`~lightning.pytorch.trainer.trainer.Trainer` instance.
            pl_module: The current :class:`~lightning.pytorch.core.LightningModule` instance.

        """
        if self._average_model is not None and (
            self._latest_update_epoch != -1 or self._latest_update_step != 0
        ):
            self._swap_models(pl_module)
