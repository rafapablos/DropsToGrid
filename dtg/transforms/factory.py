import warnings
from copy import deepcopy
from dataclasses import dataclass
from typing import Callable

import torch.nn as nn

from dtg.data.sample import Sample
from dtg.data.source import ALL_SOURCES
from dtg.transforms.radar import ToRainRate
from dtg.transforms.transforms import (
    AverageTimeGroups,
    ChannelTransform,
    Clip,
    CloneSource,
    MaskRegion,
    NormalizeRainRates,
    SelectChannels,
    SelectTimes,
    SourceTransform,
)


@dataclass
class TransformPair:
    data: Callable[[Sample], Sample]


@dataclass
class Transforms:
    train: TransformPair
    eval: TransformPair


def create_transforms(
    random_region_mask: bool = False,
    for_training: bool = False,
) -> Transforms:
    transforms = []

    # wumap stations default transform for precipitation
    wumap_transform = [
        SelectChannels([0]),
        Clip(0, 30),
        SelectTimes([-1]),
    ]
    if for_training:
        transforms.append(CloneSource("wumap_4km", "targets_4km"))
        transforms.append(SourceTransform("targets_4km", *wumap_transform))
    else:
        transforms.append(SourceTransform("wumap_4km", *wumap_transform))

    for source in ALL_SOURCES:
        match source:
            case "wumap_4km":  # Input transformations on the inputs for model
                if for_training:
                    precipitation_transform = [Clip(0, 30), NormalizeRainRates()]

                    transform: list[nn.Module] | None = [
                        SelectChannels([0]),
                        ChannelTransform(0, *precipitation_transform),
                    ]
                else:
                    transform = None
            case "rainviewer_4km":
                transform = [
                    ToRainRate(),
                    AverageTimeGroups(frames_to_average=6),
                    NormalizeRainRates(),
                ]
            case (
                "wumap_mask_4km"
                | "climate_mask_4km"
                | "rainviewer_mask_4km"
                | "targets_4km"
            ):
                transform = None
            case _:
                transform = None
                warnings.warn(
                    f"Unknown transformation for source: {source}", UserWarning
                )

        if transform is not None:
            transforms.append(SourceTransform(source, *transform))

    train_transforms = deepcopy(transforms)
    eval_transforms = deepcopy(transforms)

    if random_region_mask:
        train_transforms.extend(
            [
                SourceTransform(
                    "wumap_4km",
                    MaskRegion(p=0.1, scale_factor=0.5),
                ),
                SourceTransform(
                    "rainviewer_4km",
                    MaskRegion(p=0.1, scale_factor=0.5),
                ),
            ]
        )

    train = TransformPair(data=nn.Sequential(*train_transforms))
    eval = TransformPair(data=nn.Sequential(*eval_transforms))
    return Transforms(train=train, eval=eval)
