import torch

from dtg.data.source import Source

Sample = dict[Source, torch.Tensor]
