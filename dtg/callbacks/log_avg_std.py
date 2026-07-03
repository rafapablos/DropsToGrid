import io
from tempfile import NamedTemporaryFile

import imageio.v3 as imageio
import lightning.pytorch as pl
import matplotlib as mpl
import numpy as np
import torch
from einops import rearrange
from lightning.pytorch.utilities import rank_zero_only
from torch import Tensor

from dtg.data.sample import Sample
from dtg.logger.clearml import ClearMlLogger
from dtg.modules.output import EvalOutputs


class LogAverageStd(pl.Callback):
    def __init__(self):
        super().__init__()
        self.std_colorizer = CMapColorizer(mpl.colormaps["OrRd"])
        self.count = 0
        self.std_sums = None

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
        if outputs.std is None:
            return

        if self.std_sums is None:
            self.std_sums = outputs.std.sum(dim=0)
        else:
            self.std_sums += outputs.std.sum(dim=0)
        self.count += 1

    @rank_zero_only
    def on_test_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        if self.std_sums is None:
            return

        mean_std_map = self.std_sums / self.count
        mean_std_scalar = torch.nanmean(mean_std_map).item()

        # Log average std
        logger = trainer.logger
        if isinstance(logger, ClearMlLogger):
            logger.experiment().logger.report_scalar(
                title="test",
                series="mean_std",
                value=mean_std_scalar,
                iteration=trainer.global_step,
            )

        std = self.std_colorizer(self.std_sums / self.count)
        # Upsample image for logging at higher resolution
        std = torch.nn.functional.interpolate(
            std.permute(0, 3, 1, 2), scale_factor=5, mode="nearest"
        ).permute(0, 2, 3, 1)

        logger = trainer.logger
        if isinstance(logger, ClearMlLogger):
            self.log_to_clearml(
                std_map=std,
                stage="test",
                logger=logger,
                global_step=trainer.global_step,
            )

    def log_to_clearml(
        self,
        std_map: Tensor,
        stage: str,
        logger: ClearMlLogger,
        global_step: int,
    ):
        with NamedTemporaryFile(suffix=".mp4") as tmp_file:
            imageio.imwrite(tmp_file.name, std_map, fps=8)
            tmp_file.seek(0)
            buffer = io.BytesIO(tmp_file.read())
            logger.experiment().logger.report_media(
                title=f"{stage}/std",
                series="mean",
                stream=buffer,
                file_extension="mp4",
                iteration=global_step,
            )

class CMapColorizer(torch.nn.Module):
    def __init__(self, cmap, vmin=None, vmax=None):
        super().__init__()
        self.cmap = cmap
        self.vmin = vmin
        self.vmax = vmax

    def forward(self, x):
        x = rearrange(x, "t 1 h w -> t h w")
        img_np = x.detach().cpu().numpy()

        # Keep NaNs and normalize safely
        valid_mask = ~np.isnan(img_np)
        if np.any(valid_mask):
            min_val = self.vmin if self.vmin is not None else np.nanmin(img_np)
            max_val = self.vmax if self.vmax is not None else np.nanmax(img_np)
            img_np = np.clip(img_np, min_val, max_val)
            norm = (img_np - min_val) / (max_val - min_val + 1e-8)
            norm = np.where(valid_mask, norm, np.nan)
        else:
            norm = np.zeros_like(img_np.unsqueeze(0))

        # Apply colormap
        img_rgb = np.asarray(self.cmap(norm))[..., :3]
        img_rgb = torch.from_numpy(img_rgb)
        out = img_rgb * 255
        return out.to(torch.uint8)
