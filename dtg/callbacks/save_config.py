from lightning.pytorch import LightningModule, Trainer
from lightning.pytorch.cli import SaveConfigCallback as LightningSaveConfigCallback


class SaveConfigCallback(LightningSaveConfigCallback):
    """
    Overrides the default SaveConfigCallback to only save the config file
    when the trainer is in the fit stage.
    This ensures the config file is not overwritten when running e.g. `test`.
    """

    def setup(self, trainer: Trainer, pl_module: LightningModule, stage: str) -> None:
        if stage != "fit":
            return

        super().setup(trainer, pl_module, stage)
