from dataclasses import dataclass
from typing import Dict

import torch


@dataclass
class EvalOutputs:
    output: torch.Tensor
    target: torch.Tensor | None
    context: torch.Tensor | None
    std: torch.Tensor | None
    additional_sources: Dict[str, torch.Tensor] | None
