from typing import Callable

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from dtg.data.denmark_mask import LAND_MASK as _CLIMATE_MASK
from dtg.data.sample import Sample
from dtg.data.source import Source

_HEIGHT, _WIDTH = 128, 128
_RAINVIEWER_FRAMES = 6
_WUMAP_FRAMES = 3
_WETNESS_THRESHOLD = 0.55

_STATION_DENSITY = 0.1
_HOLDOUT_FRACTION = 0.2

_RADAR_COAST_BUFFER_PX = 4


def _dilate(mask: torch.Tensor, buffer_px: int) -> torch.Tensor:
    kernel_size = 2 * buffer_px + 1
    dilated = F.max_pool2d(
        mask.float()[None, None], kernel_size=kernel_size, stride=1, padding=buffer_px
    )
    return dilated[0, 0] > 0


def _smooth_noise(generator: torch.Generator, downscale: int = 12) -> torch.Tensor:
    low_res = torch.rand(
        1, 1, max(2, _HEIGHT // downscale), max(2, _WIDTH // downscale), generator=generator
    )
    field = F.interpolate(
        low_res, size=(_HEIGHT, _WIDTH), mode="bicubic", align_corners=False
    )[0, 0]
    return (field - field.min()) / (field.max() - field.min() + 1e-6)


def _fixed_bernoulli_mask(seed: int, density: float) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    return torch.rand(_HEIGHT, _WIDTH, generator=generator) < density


# Independent per-pixel draws (not smoothed) so the ~20% holdout share of the
# ~10% station pixels lands close to its expectation instead of depending on
# whether two smooth blobs happen to overlap.
_STATION_LOCATIONS = _fixed_bernoulli_mask(seed=100, density=_STATION_DENSITY)
_HOLDOUT_LOCATIONS = _STATION_LOCATIONS & _fixed_bernoulli_mask(
    seed=101, density=_HOLDOUT_FRACTION
)
_RAINVIEWER_MASK = _dilate(_CLIMATE_MASK, _RADAR_COAST_BUFFER_PX)


class FakeDtGDataset(Dataset):
    """Stand-in for `DtGDataset` that cannot be distributed."""

    def __init__(
        self,
        sources: list[Source],
        data_transforms: Callable[[Sample], Sample] | None = None,
        length: int = 512,
    ):
        self.sources = sources
        self.data_transforms = data_transforms
        self.length = length

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, idx: int) -> Sample:
        generator = torch.Generator().manual_seed(idx)
        storm = _storm_sequence(_RAINVIEWER_FRAMES, generator)

        sample: Sample = {}
        for source in self.sources:
            sample[source] = _fake_source(source, storm, generator)
        if self.data_transforms is not None:
            sample = self.data_transforms(sample)
        return sample


def _fake_source(
    source: Source, storm: torch.Tensor, generator: torch.Generator
) -> torch.Tensor:
    match source:
        case "rainviewer_4km":
            return _to_raw_radar(storm, generator)[:, None]
        case "wumap_4km":
            return _to_station_readings(storm[-_WUMAP_FRAMES:], generator)[:, None]
        case "wumap_mask_4km":
            return _HOLDOUT_LOCATIONS
        case "climate_mask_4km":
            return _CLIMATE_MASK
        case "rainviewer_mask_4km":
            return _RAINVIEWER_MASK
        case _:
            raise ValueError(f"FakeDtGDataset has no generator for source {source!r}")


def _storm_sequence(num_frames: int, generator: torch.Generator) -> torch.Tensor:
    """A smoothly advected [0, 1] intensity field per frame, mimicking a
    moving storm cell rather than i.i.d. noise."""
    field = _smooth_noise(generator)
    drift = tuple(torch.randint(-4, 5, (2,), generator=generator).tolist())

    frames = []
    for _ in range(num_frames):
        field = torch.roll(field, shifts=drift, dims=(0, 1))
        field = (0.85 * field + 0.15 * _smooth_noise(generator)).clamp(0, 1)
        frames.append(field)
    return torch.stack(frames)


def _to_raw_radar(storm: torch.Tensor, generator: torch.Generator) -> torch.Tensor:
    """Byte-encoded reflectivity, as `ToRainRate` expects (an integer index
    in [0, 255]). ~32 is a dbz-equivalent of 0 (no rain), ~96 is heavy rain."""
    wetness = (storm - _WETNESS_THRESHOLD).clamp(min=0) / (1 - _WETNESS_THRESHOLD)
    raw = 32 + wetness * 64 + torch.randn(storm.shape, generator=generator) * 1.5
    return raw.clamp(0, 255).round()


def _to_station_readings(storm: torch.Tensor, generator: torch.Generator) -> torch.Tensor:
    """Precipitation readings (mm/h) at the fixed station locations, NaN
    everywhere else, matching the real sparse `wumap_4km` coverage."""
    wetness = (storm - _WETNESS_THRESHOLD).clamp(min=0) / (1 - _WETNESS_THRESHOLD)
    rate_mmh = (wetness * 25 + torch.randn(storm.shape, generator=generator) * 0.5).clamp(min=0)

    readings = torch.full(storm.shape, float("nan"))
    readings[:, _STATION_LOCATIONS] = rate_mmh[:, _STATION_LOCATIONS]
    return readings
