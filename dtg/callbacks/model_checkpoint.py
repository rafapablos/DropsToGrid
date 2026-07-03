import os

import lightning.pytorch as pl
from lightning.pytorch.callbacks import ModelCheckpoint as LightningModelCheckpoint


class ModelCheckpoint(LightningModelCheckpoint):
    def setup(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        stage: str,
    ) -> None:
        """
        Bugfix for ModelCheckpoint when using custom loggers.

        Default behavior is to save to
            <root>/<name>/<version>/<name>/<version>/checkpoints

        This changes it to
            <root>/<name>/<version>/checkpoints
        """
        if self.dirpath is None and trainer.log_dir is not None:
            self.dirpath = os.path.join(trainer.log_dir, "checkpoints")
        super().setup(trainer, pl_module, stage)
