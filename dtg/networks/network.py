from abc import ABC, abstractmethod
from typing import Type

import torch.nn as nn

from dtg.loss.distributions import Distribution


class NetworkConfig(ABC):
    @abstractmethod
    def build(self, *args, **kwargs) -> nn.Module:
        pass


class DensificationNetworkConfig(NetworkConfig):
    @abstractmethod
    def build(
        self,
        input_channels: int,
        output_channels: int,
        distribution: Type[Distribution],
    ) -> nn.Module:
        pass
