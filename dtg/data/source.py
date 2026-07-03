from typing import Literal, get_args

Source = Literal[
    "targets_4km",
    "rainviewer_4km",
    "wumap_4km",
    "wumap_mask_4km",
    "climate_mask_4km",
    "rainviewer_mask_4km",
]

ALL_SOURCES: list[Source] = list(get_args(Source))
