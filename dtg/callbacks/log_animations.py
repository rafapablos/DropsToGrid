import io
from tempfile import NamedTemporaryFile

import imageio.v3 as imageio
import lightning.pytorch as pl
import torch
from einops import rearrange
from lightning.pytorch.utilities import rank_zero_only
from torch import Tensor

from dtg.data.sample import Sample
from dtg.logger.clearml import ClearMlLogger
from dtg.modules.output import EvalOutputs


class LogAnimations(pl.Callback):
    def __init__(self, num_animations: int = 20):
        super().__init__()
        self.colorize_radar = ColorizeRainRates()
        self.num_animations = num_animations
        self.animations = []

    @rank_zero_only
    def on_validation_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs: EvalOutputs,
        batch: Sample,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        return self.on_stage_batch_end(eval_outputs=outputs)

    @rank_zero_only
    def on_test_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs: EvalOutputs,
        batch: Sample,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        return self.on_stage_batch_end(eval_outputs=outputs)

    def on_stage_batch_end(self, eval_outputs: EvalOutputs):
        if len(self.animations) >= self.num_animations:
            return

        outputs = eval_outputs.output.cpu()
        targets = eval_outputs.target.cpu()
        contexts = eval_outputs.context
        if contexts is not None:
            contexts = contexts.cpu()

        for i in range(len(outputs)):
            if len(self.animations) >= self.num_animations:
                break
            masked = targets[i][~torch.isnan(targets[i])]
            if masked.numel() == 0 or (masked > 1).sum() < 3:
                continue
            output = outputs[i]
            additional_sources = eval_outputs.additional_sources
            if (
                additional_sources is not None
                and "radar_4km" in additional_sources.keys()
            ):
                output = output.masked_fill(
                    torch.isnan(additional_sources["radar_4km"][i]).cpu(),
                    torch.nan,
                )
            output = self.colorize_radar(output)
            target = self.colorize_radar(targets[i])
            imgs = [output, target]
            if contexts is not None:
                context = self.colorize_radar(
                    contexts[i, :, -1:]
                )  # Plot last channel if many
                imgs.insert(0, context)
            # Plot additional sources (radar_4km)
            if (
                additional_sources is not None
                and "radar_4km" in additional_sources.keys()
            ):
                radar_4km = self.colorize_radar(
                    additional_sources["radar_4km"].cpu()[i]
                )
                imgs.insert(1, radar_4km)

            max_time = max(img.shape[0] for img in imgs)
            for i, img in enumerate(imgs):
                t = img.shape[0]
                if t < max_time:
                    repeat_factor = max_time // t
                    imgs[i] = img.repeat(repeat_factor, 1, 1, 1)
            animation = torch.concatenate(imgs, dim=2)

            # Upsample image for logging at higher resolution
            animation = torch.nn.functional.interpolate(
                animation.permute(0, 3, 1, 2), scale_factor=2, mode="nearest"
            ).permute(0, 2, 3, 1)

            self.animations.append(animation)

    @rank_zero_only
    def on_validation_end(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule
    ) -> None:
        return self.on_stage_end(trainer, pl_module, "val")

    @rank_zero_only
    def on_test_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        return self.on_stage_end(trainer, pl_module, "test")

    def on_stage_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        stage: str,
    ):
        if not self.animations:
            return

        logger = trainer.logger
        if isinstance(logger, ClearMlLogger):
            self.log_to_clearml(
                animations=self.animations,
                stage=stage,
                logger=logger,
                global_step=trainer.global_step,
            )
        self.animations = []

    def log_to_clearml(
        self,
        animations: list[Tensor],
        stage: str,
        logger: ClearMlLogger,
        global_step: int,
    ):
        for idx, animation in enumerate(animations):
            with NamedTemporaryFile(suffix=".mp4") as tmp_file:
                imageio.imwrite(tmp_file.name, animation, fps=8)
                tmp_file.seek(0)
                buffer = io.BytesIO(tmp_file.read())
                logger.experiment().logger.report_media(
                    title=f"{stage}/animation",
                    series=f"{idx}",
                    stream=buffer,
                    file_extension="mp4",
                    iteration=global_step,
                )

class ColorizeRainRates(torch.nn.Module):
    colors: torch.Tensor
    bounds: torch.Tensor

    def __init__(self):
        super().__init__()

        # RainViewer Dark Sky color scheme
        mmh_to_rgb = {
            0.000: (0, 0, 0),
            0.200: (191, 214, 236),
            0.237: (184, 210, 234),
            0.273: (177, 205, 232),
            0.316: (170, 201, 230),
            0.365: (157, 193, 226),
            0.421: (143, 184, 222),
            0.486: (130, 176, 219),
            0.562: (116, 167, 215),
            0.648: (102, 158, 211),
            0.749: (93, 148, 206),
            0.865: (84, 139, 201),
            0.999: (77, 130, 195),
            1.153: (70, 120, 190),
            1.332: (65, 110, 185),
            1.538: (78, 103, 180),
            1.776: (92, 96, 174),
            2.050: (107, 89, 168),
            2.368: (123, 81, 161),
            2.734: (142, 75, 155),
            3.158: (164, 76, 146),
            3.646: (186, 78, 137),
            4.211: (208, 79, 129),
            4.862: (230, 81, 120),
            5.615: (252, 83, 112),
            6.484: (252, 103, 111),
            7.488: (253, 123, 111),
            8.647: (253, 143, 110),
            9.985: (254, 163, 110),
            11.531: (255, 183, 110),
            13.315: (255, 197, 89),
            15.376: (255, 211, 68),
            17.756: (255, 225, 46),
            20.505: (255, 239, 25),
            23.679: (255, 253, 5),
            float("inf"): (128, 128, 128),  # nans,
        }

        bounds = torch.tensor(list(mmh_to_rgb.keys())[1:])
        colors = torch.tensor(list(mmh_to_rgb.values()), dtype=torch.uint8)
        self.register_buffer("bounds", bounds)
        self.register_buffer("colors", colors)

    def forward(self, rain_rates: Tensor) -> Tensor:
        assert rain_rates.dtype == torch.float32
        rain_rates = rearrange(rain_rates, "t 1 h w -> t h w")
        buckets = torch.bucketize(rain_rates, self.bounds, right=True)
        rain_rates = self.colors[buckets]
        return rain_rates  # (T, H, W, C)
