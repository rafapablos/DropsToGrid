from typing import Literal, Tuple, Type

import lightning as L
import torch
import torch.nn as nn
from einops import rearrange
from lightning.pytorch.utilities.types import OptimizerLRScheduler
from torch.optim.lr_scheduler import CosineAnnealingLR
from torchmetrics import MetricCollection

from dtg.data.sample import Sample
from dtg.data.source import Source
from dtg.loss.distributions import Distribution, ZeroInflatedGamma
from dtg.loss.loss import CNPFLoss
from dtg.metrics.binary import BinaryMetric
from dtg.metrics.crps import ContinuousRankedProbabilityScore
from dtg.metrics.fss import FractionSkillScore
from dtg.metrics.regression import MeanAbsoluteError, MeanSquaredError
from dtg.modules.dtg.context import GridCntxtTrgtGetter
from dtg.modules.output import EvalOutputs
from dtg.networks.dtg.model import DtGModelConfig
from dtg.networks.network import DensificationNetworkConfig
from dtg.transforms.factory import Transforms, create_transforms
from dtg.transforms.transforms import DenormalizeRainRates


class DtGModule(L.LightningModule):
    def __init__(
        self,
        input_source: Source,
        target_source: Source,
        num_lead_times: int,
        num_history_times: int,
        sources: list[Source] = [],
        network: DensificationNetworkConfig = DtGModelConfig(),
        distribution: Type[Distribution] = ZeroInflatedGamma,
        learning_rate: float = 3e-4,
        max_epochs: int = 30,
        weight_decay: float = 1e-4,
        context_masking: Tuple[float, float] = (0.0, 0.3),
        random_region_mask: bool = False,
    ):
        super().__init__()
        self.save_hyperparameters()

        assert input_source in sources and target_source in sources
        self.input_source: Source = input_source
        self.target_source: Source = (
            "targets_4km" if target_source == "wumap_4km" else target_source
        )

        assert (
            self.input_source == "wumap_4km"
        ), f"{self.input_source} input source not implemented"

        self.learning_rate = learning_rate
        self.max_epochs = max_epochs
        self.weight_decay = weight_decay
        self.num_lead_times = num_lead_times
        self.num_history_times = num_history_times

        self._transforms = create_transforms(
            for_training=True,
            random_region_mask=random_region_mask,
        )

        self.context_masker = GridCntxtTrgtGetter(train_range=context_masking)

        self.context_denormalizer = DenormalizeRainRates()
        self.target_denormalizer = nn.Identity()

        self.model = network.build(1, 1, distribution)
        self.criterion = CNPFLoss()

        self.val_metrics = _create_metrics(
            num_lead_times=num_lead_times + 1, stage="val"
        )
        self.test_metrics = _create_metrics(
            num_lead_times=num_lead_times + 1, stage="test"
        )

    @property
    def transforms(self) -> Transforms:
        return self._transforms

    def forward(self, batch: Sample, stage: Literal["train", "val", "test"]):
        input = batch[self.input_source]
        target = batch[self.target_source]

        # Mask all inputs according to RainViewer and Climate coverage
        rainviewer_mask = batch["rainviewer_mask_4km"][0][
            None, None, None, :, :
        ]  # [1, 1, 1, 128, 128]
        input = input.masked_fill(~rainviewer_mask, torch.nan)
        target = target.masked_fill(~rainviewer_mask, torch.nan)

        climate_mask = batch["climate_mask_4km"][0][
            None, None, None, :, :
        ]  # [1, 1, 1, 128, 128]
        input = input.masked_fill(~climate_mask, torch.nan)
        target = target.masked_fill(~climate_mask, torch.nan)

        additional_sources = {}
        if "rainviewer_4km" in batch:
            additional_sources["radar_4km"] = batch["rainviewer_4km"]
            additional_sources["radar_4km"] = additional_sources[
                "radar_4km"
            ].masked_fill(~rainviewer_mask, torch.nan)
            additional_sources["radar_4km"] = additional_sources[
                "radar_4km"
            ].masked_fill(~climate_mask, torch.nan)

        holdout_mask = rearrange(batch["wumap_mask_4km"], "b h w -> b 1 1 h w")
        if self.input_source == "wumap_4km":
            input = input.masked_fill(holdout_mask, torch.nan)
            input = self.context_masker(input, context=True, stage=stage)

        if self.target_source == "targets_4km":
            if stage == "train":
                target = target.masked_fill(~torch.isnan(input[:, -1:]), torch.nan)
            if stage in ["train", "val"]:
                target = target.masked_fill(holdout_mask, torch.nan)
            else:
                target = target.masked_fill(~holdout_mask, torch.nan)

        target = self.context_masker(target, context=False, stage=stage)

        outputs = self.model(input, additional_sources=additional_sources)

        return outputs, target, input, additional_sources

    def training_step(self, batch, batch_idx):
        outputs, targets, _, _ = self(batch, stage="train")
        loss = self.criterion(outputs, targets)
        self.log("train/loss", loss, on_epoch=True, sync_dist=True)

        optimizer = self.optimizers()
        if isinstance(optimizer, (list, tuple)):
            optimizer = optimizer[0]
        self.log(
            "train/lr",
            optimizer.param_groups[0]["lr"],
            on_step=True,
            on_epoch=False,
            sync_dist=True,
        )

        return loss

    def validation_step(self, batch, batch_idx):
        return self._shared_eval(batch, "val")

    def test_step(self, batch, batch_idx):
        return self._shared_eval(batch, "test")

    def predict_step(self, batch: Sample, batch_idx: int, return_targets: bool = True):
        if return_targets:
            return self._shared_eval(batch, "test", use_logger=False)

        print("Careful - no holdout stations here!!")
        input = batch[self.input_source]

        if self.input_source == "wumap_4km":
            input = self.context_masker(input, context=True, stage="predict")

        additional_sources = {}
        if "radar_4km" in batch.keys():
            additional_sources["radar_4km"] = batch["rainviewer_4km"]
        outputs = self.model(input, additional_sources=additional_sources)

        mean = self.target_denormalizer(outputs.mean)
        std = self.target_denormalizer(outputs.std)

        input = self.context_denormalizer(input)
        if "radar_4km" in additional_sources.keys():
            additional_sources["radar_4km"] = self.context_denormalizer(
                additional_sources["radar_4km"]
            )

        return EvalOutputs(
            mean,
            target=None,
            context=input,
            std=std,
            additional_sources=additional_sources,
        )

    def _shared_eval(
        self,
        batch: Sample,
        stage: Literal["val", "test"],
        use_logger: bool = True,
    ) -> EvalOutputs:
        outputs, targets, inputs, additional_sources = self(batch, stage=stage)

        loss = self.criterion(outputs, targets)
        if use_logger:
            self.log(f"{stage}/loss", loss, on_epoch=True, sync_dist=True)

        mean = self.target_denormalizer(outputs.mean)
        std = self.target_denormalizer(outputs.std)

        inputs = self.context_denormalizer(inputs)
        targets = self.target_denormalizer(targets)
        if "radar_4km" in additional_sources.keys():
            additional_sources["radar_4km"] = self.context_denormalizer(
                additional_sources["radar_4km"]
            )

        metrics = self.val_metrics if stage == "val" else self.test_metrics
        metrics.update(preds=mean, target=targets, distribution=outputs)
        if use_logger:
            self.log_dict(metrics)

        return EvalOutputs(
            mean,
            targets,
            context=inputs,
            std=std,
            additional_sources=additional_sources,
        )

    def configure_optimizers(self) -> OptimizerLRScheduler:
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )
        scheduler = CosineAnnealingLR(optimizer, T_max=self.max_epochs, eta_min=1e-6)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val/loss",
                "interval": "epoch",
            },
        }


def _create_metrics(num_lead_times: int, stage: str):
    metrics = {
        "csi": BinaryMetric(num_lead_times=num_lead_times, variable="csi"),
        "fbi": BinaryMetric(num_lead_times=num_lead_times, variable="fbi"),
        "mae": MeanAbsoluteError(num_lead_times=num_lead_times),
        "mse": MeanSquaredError(num_lead_times=num_lead_times),
        "crps": ContinuousRankedProbabilityScore(num_lead_times=num_lead_times),
    }
    if stage == "test":
        metrics["fss"] = FractionSkillScore(num_lead_times=num_lead_times)
    return MetricCollection(metrics, prefix=f"{stage}/")
