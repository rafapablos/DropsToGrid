from typing import Callable, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
from check_shapes import check_shapes


def batch(x: torch.Tensor, other_dims: int) -> torch.Size:
    """Get the shape of the batch of a tensor."""
    return x.size()[:-other_dims]

def compress_batch_dimensions(
    x: torch.Tensor, other_dims: int
) -> Tuple[torch.Tensor, Callable]:
    """Compress multiple batch dimensions of a tensor into a single batch dimension."""
    b = batch(x, other_dims)
    if len(b) == 1:
        return x, lambda x: x

    def uncompress(x_after):
        return torch.reshape(x_after, (*b, *x_after.size()[1:]))

    return (
        torch.reshape(x, (int(np.prod(b)), *x.size()[len(b) :])),
        uncompress,
    )

class MLP(nn.Module):
    """MLP.

    Args:
        in_dim (int): Input dimensionality.
        out_dim (int): Output dimensionality.
        layers (tuple[int, ...], optional): Width of every hidden layer.
        num_layers (int, optional): Number of hidden layers.
        width (int, optional): Width of the hidden layers
        nonlinearity (function, optional): Nonlinearity.
        dtype (dtype, optional): Data type.

    Attributes:
        net (object): MLP, but which expects a different data format.
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        layers: Optional[Tuple[int, ...]] = None,
        num_layers: Optional[int] = None,
        width: Optional[int] = None,
        nonlinearity: Optional[nn.Module] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        super().__init__()

        if layers is None:
            # Check that one of the two specifications is given.
            assert (
                num_layers is not None and width is not None
            ), "Must specify either `layers` or `num_layers` and `width`."
            layers = (width,) * num_layers

        # Default to ReLUs.
        if nonlinearity is None:
            nonlinearity = nn.ReLU()

        # Build layers.
        if len(layers) == 0:
            self.net = nn.Linear(in_dim, out_dim, dtype=dtype)
        else:
            net = [nn.Linear(in_dim, layers[0], dtype=dtype)]
            for i in range(1, len(layers)):
                net.append(nonlinearity)
                net.append(nn.Linear(layers[i - 1], layers[i], dtype=dtype))
            net.append(nonlinearity)
            net.append(nn.Linear(layers[-1], out_dim, dtype=dtype))
            self.net = nn.Sequential(*net)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x, uncompress = compress_batch_dimensions(x, 1)
        x = self.net(x)
        x = uncompress(x)
        return x
    
class ConvBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        Conv: nn.Module,
        kernel_size: Union[int, Tuple[int, ...]] = 5,
        activation: nn.Module = nn.ReLU(),
        **kwargs,
    ):
        super().__init__()

        self.activation = activation

        padding: int | list[int]
        if isinstance(kernel_size, int):
            padding = kernel_size // 2
        else:
            kernel_size = tuple(kernel_size)
            padding = [k // 2 for k in kernel_size]

        # Conv = make_depth_sep_conv(Conv)
        self.conv = Conv(
            in_channels, out_channels, kernel_size, padding=padding, **kwargs
        )

    @check_shapes("x: [m, c, ...]")
    def forward(self, x: torch.Tensor):
        return self.conv(self.activation(x))