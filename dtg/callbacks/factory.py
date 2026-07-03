from lightning import Callback
from lightning.pytorch.callbacks import EarlyStopping

from dtg.callbacks.ema import EMA
from dtg.callbacks.log_animations import LogAnimations
from dtg.callbacks.log_avg_std import LogAverageStd
from dtg.callbacks.log_plots import LogPlots
from dtg.callbacks.model_checkpoint import ModelCheckpoint


def create_callbacks(
    ema_decay: float | None,
    ema_delay: int | None,
) -> list[Callback]:
    callbacks = [
        LogPlots(),
        LogAnimations(),
        LogAverageStd(),
        ModelCheckpoint(
            filename="best",
            monitor="val/loss",
            mode="min",
            save_top_k=1,
            save_on_train_epoch_end=False,
            save_last=True,
            enable_version_counter=False,
        ),
        EarlyStopping(monitor="val/loss", mode="min", patience=10),
    ]
    if ema_decay is not None:
        callbacks.append(EMA(ema_decay, ema_delay))
    return callbacks
