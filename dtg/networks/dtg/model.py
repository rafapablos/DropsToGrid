import copy
from abc import ABC
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Type, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from check_shapes import check_shapes
from einops import rearrange

from dtg.loss.distributions import Distribution, ZeroInflatedGamma
from dtg.networks.dtg.bottleneck import (
    TemporalPixelwiseTransformerBlock,
    TETransformer2D,
)
from dtg.networks.dtg.components import MLP, ConvBlock
from dtg.networks.network import DensificationNetworkConfig


class BaseDtGModel(nn.Module, ABC):
    """Base wrapper for DtG encoder, decoder, and distribution layer."""

    def __init__(
        self, encoder: nn.Module, decoder: nn.Module, distribution: type[Distribution]
    ):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.distribution = distribution

    @check_shapes("yc: [m, ..., dy]", "yc_grid: [m, ..., dy_grid]")
    def forward(self, yc: torch.Tensor, yc_grid: torch.Tensor) -> Distribution:
        encoded = self.encoder(yc, yc_grid)
        decoded = self.decoder(encoded)
        return self.distribution(decoded)


@dataclass
class DtGModelConfig(DensificationNetworkConfig):
    num_blocks: int = 8
    kernel_size: int = 11
    num_channels: int = 128
    p_dropout: float = 0.0
    num_heads: int = 8
    head_dim: int = 16
    depth: int = 1

    def build(
        self,
        input_channels: int,
        output_channels: int,
        distribution: Type[Distribution],
    ) -> "DtGModel":
        return DtGModel(
            x_dim=3,  # T H W
            y_dim=output_channels,
            distribution=distribution,
            num_blocks=self.num_blocks,
            kernel_size=self.kernel_size,
            num_channels=self.num_channels,
            p_dropout=self.p_dropout,
            num_heads=self.num_heads,
            head_dim=self.head_dim,
            depth=self.depth,
        )


class AbsConv(nn.Conv2d):
    def forward(self, input):
        return F.conv2d(
            input,
            self.weight.abs(),
            self.bias,
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
        )


class SetConv(nn.Module):
    def __init__(self, x_dim, kernel_size=11) -> None:
        super().__init__()
        self.conv = AbsConv(
            x_dim,
            x_dim,
            groups=x_dim,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            bias=False,
        )

    def forward(self, x: torch.Tensor, mask: torch.Tensor):
        # x: b t h w c
        # mask: b t h w c (1 / 0)
        b = x.size(0)
        x = rearrange(x, "b t h w c -> (b t) c h w")
        mask = rearrange(mask, "b t h w c -> (b t) c h w")

        X_cntxt = x * mask
        signal = self.conv(X_cntxt)
        mask_cntxt = mask.float()
        density = self.conv(mask_cntxt.expand_as(x))
        normalize_signal = signal / torch.clamp(density, min=1e-5)

        normalize_signal = rearrange(normalize_signal, "(b t) c h w -> b t h w c", b=b)
        density = rearrange(density, "(b t) c h w -> b t h w c", b=b)
        return normalize_signal, density


class DtGModelSetConv(nn.Module):

    def __init__(self) -> None:
        super().__init__()
        self.setconv_stations = SetConv(1)
        self.setconv_radar = SetConv(1)

    @check_shapes(
        "z: [m, ..., dz]",
        "z_grid: [m, ..., dz]",
        "return: [m, ..., d0]",
    )
    def forward(
        self,
        z: torch.Tensor,
        z_grid: torch.Tensor,
    ):
        # Filter gridded timesteps
        B, T, H, W, C = z.shape
        device = z.device

        # create tensor for radar data across timesteps
        radar_expanded = torch.zeros((B, T, H, W, 1), device=device)
        radar_flag = torch.zeros_like(radar_expanded)

        # insert radar only at timestep index 5 (the 6th timestep)
        radar_expanded[:, -1, ...] = z_grid[:, 0, ...]
        radar_flag[:, -1, ...] = 1.0  # mark as real data
        # Remove -1 from context data
        radar_flag[radar_expanded == -1] = 0
        radar_expanded[radar_expanded == -1] = 0

        # concatenate data + flag as 4 channels
        # shape: [B, T, H, W, 4]
        stations = torch.nan_to_num(z, nan=0)
        stations_flag = (~torch.isnan(z)).float()

        stations, stations_flag = self.setconv_stations(stations, stations_flag)
        radar_expanded, radar_flag = self.setconv_radar(radar_expanded, radar_flag)

        combined = torch.cat(
            [stations, radar_expanded, stations_flag, radar_flag], dim=-1
        )

        return combined


class Bottleneck(nn.Module):
    def __init__(
        self,
        num_channels,
        num_heads,
        head_dim,
        depth,
        p_dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.temporal_att = TemporalPixelwiseTransformerBlock(
            num_channels,
            num_heads=num_heads,
            head_dim=head_dim,
            p_dropout=p_dropout,
        )
        self.cross = TETransformer2D(
            dim=num_channels,
            depth=1,
            num_heads=num_heads,
            head_dim=head_dim,
            num_channels=num_channels,
            p_dropout=p_dropout,
        )
        self.combined = TETransformer2D(
            dim=num_channels,
            depth=depth,
            num_heads=num_heads,
            head_dim=head_dim,
            num_channels=num_channels,
            p_dropout=p_dropout,
        )
        self.radar_gate_conv = nn.Sequential(
            nn.Conv2d(
                num_channels, max(4, num_channels // 8), kernel_size=3, padding=1
            ),
            nn.ReLU(),
            nn.Conv2d(max(4, num_channels // 8), 1, kernel_size=1),
        )
        self.reduce = nn.Conv2d(num_channels * 3, num_channels, kernel_size=1)

    def forward(self, stations: torch.Tensor, radar: torch.Tensor):
        # radar: [b, 1, c, h, w] originally - in your code you rearrange earlier
        radar = rearrange(radar, "b 1 c h w -> b c h w")
        last_station = stations[:, -1]

        stations = self.temporal_att(stations)
        cross_out = self.cross(stations, radar)

        # spatial gate
        gate = torch.sigmoid(self.radar_gate_conv(radar))  # [b,1,h,w]
        gated_cross = cross_out * gate

        combined = torch.cat([stations, gated_cross, last_station], dim=1)
        x = self.reduce(combined)
        x = self.combined(x)
        return x.unsqueeze(1)


class UNet(nn.Module):
    def __init__(
        self,
        dim: int,
        num_channels: Union[int, List[int]],
        num_blocks: Optional[int] = None,
        max_num_channels: int = 256,
        pooling_size: Union[int, Tuple[int, ...]] = 2,
        factor_chan: int = 2,
        p_dropout: float = 0.0,
        num_heads: int = 8,
        head_dim: int = 16,
        depth: int = 1,
        **kwargs,
    ):
        super().__init__()
        self.max_num_channels = max_num_channels
        self.factor_chan = factor_chan

        if num_blocks is None:
            assert isinstance(num_channels, list)
            num_blocks = len(num_channels) - 1

        self.dim = dim
        self.num_blocks = num_blocks
        self.in_out_channels = self._get_in_out_channels(num_channels, num_blocks)

        if not isinstance(pooling_size, int):
            pooling_size = tuple(pooling_size)

        self.pooling_size = pooling_size
        self.pooling = nn.MaxPool2d(pooling_size)
        self.upsample_mode = "bilinear"
        self.bottleneck = Bottleneck(
            num_channels, num_heads, head_dim, depth, p_dropout=p_dropout
        )

        assert isinstance(num_channels, int)
        self.conv_blocks = nn.ModuleList(
            [
                ConvBlock(in_c, out_c, nn.Conv2d, **kwargs)
                for i, (in_c, out_c) in enumerate(self.in_out_channels)
            ]
        )

        self.radar_conv_blocks = nn.ModuleList(
            [
                ConvBlock(in_c, out_c, nn.Conv2d, **kwargs)
                for in_c, out_c in self.in_out_channels[: self.num_blocks // 2 + 1]
            ]
        )

    @check_shapes("stations: [m, ..., c]", "radar: [m, ..., c]", "return: [m, ..., c]")
    def forward(self, stations: torch.Tensor, radar: torch.Tensor) -> torch.Tensor:
        # Move channels to after batch dimension.
        B, T, C, H, W = stations.shape

        stations = rearrange(stations, "b t h w c -> (b t) c h w")
        radar = rearrange(radar, "b h w c -> b c h w")

        num_down_blocks = self.num_blocks // 2
        residuals_stations = []
        residuals_radar = []

        # Downwards convolutions.
        for i in range(num_down_blocks):
            stations = self.conv_blocks[i](stations)
            radar = self.radar_conv_blocks[i](radar)

            residuals_stations.append(
                rearrange(stations, "(b t) c h w -> b t c h w", t=T)[:, -1]
            )
            residuals_radar.append(radar)

            stations = self.pooling(stations)
            radar = self.pooling(radar)

        stations = self.conv_blocks[num_down_blocks](stations)
        radar = self.radar_conv_blocks[num_down_blocks](radar)

        # Bottleneck.
        stations = rearrange(stations, "(b t) c h w -> b t c h w", t=T)
        radar = radar.unsqueeze(1)
        fused = self.bottleneck(stations, radar)

        fused = rearrange(fused, "b 1 c h w -> b c h w")

        # Upwards convolutions.
        for i in range(num_down_blocks + 1, self.num_blocks):
            fused = nn.functional.interpolate(
                fused,
                size=residuals_stations[num_down_blocks - i].shape[-self.dim :],
                mode=self.upsample_mode,
                align_corners=True,
            )

            fused = torch.cat(
                (
                    fused,
                    residuals_stations[num_down_blocks - i],
                    residuals_radar[num_down_blocks - i],
                ),
                dim=1,
            )
            fused = self.conv_blocks[i](fused)

        fused = rearrange(fused, "b c h w -> b 1 h w c")
        return fused

    def _super_get_in_out_channels(
        self, num_channels: Union[List[int], int], num_blocks: int
    ) -> List[Tuple[int, int]]:
        """Return a list of tuple of input and output channels."""
        if isinstance(num_channels, int):
            channel_list = [num_channels] * (num_blocks + 1)
        else:
            channel_list = list(num_channels)

        assert len(channel_list) == (
            num_blocks + 1
        ), f"{len(channel_list)} != {num_blocks}."

        return list(zip(channel_list, channel_list[1:]))

    def _get_in_out_channels(
        self,
        num_channels: Union[List[int], int],
        num_blocks: int,
        T: int = 6,
    ) -> List[Tuple[int, int]]:
        # Doubles at every down layer, as in vanila UNet.
        factor_chan = self.factor_chan

        assert num_blocks % 2 == 1, f"n_blocks={num_blocks} not odd."

        if isinstance(num_channels, int):
            # e.g. if n_channels=16, n_blocks=5: [16, 32, 64].
            channel_list = [
                factor_chan**i * num_channels for i in range(num_blocks // 2 + 1)
            ]
        else:
            channel_list = list(num_channels)

        # e.g.: [16, 32, 64, 64, 32, 16].
        channel_list = channel_list + channel_list[::-1]

        # Bound max number of channels by self.max_nchannels (besides first and
        # last dim as this is input / output should not be changed).
        channel_list = (
            channel_list[:1]
            + [min(c, self.max_num_channels) for c in channel_list[1:-1]]
            + channel_list[-1:]
        )

        # e.g.: [(16, 32), (32, 64), (64, 64), (64, 32), (32, 16)].
        in_out_channels = self._super_get_in_out_channels(channel_list, num_blocks)
        # e.g.: [(16, 32), (32, 64), (64, 64), (128, 32), (64, 16)] due to concat.
        idcs = slice(len(in_out_channels) // 2 + 1, len(in_out_channels))
        in_out_channels[idcs] = [
            (in_chan * 3, out_chan) for in_chan, out_chan in in_out_channels[idcs]
        ]

        return in_out_channels


class DtGModelEncoder(nn.Module):
    def __init__(
        self,
        network: nn.Module,
        grid_encoder: DtGModelSetConv,
        z_encoder: nn.Module,
        num_channels: int,
    ):
        super().__init__()

        self.network = network
        self.grid_encoder = grid_encoder
        self.z_encoder = z_encoder
        self.z_radar_encoder = copy.deepcopy(z_encoder)

    @check_shapes(
        "yc: [m, ..., dy]",
        "yc_grid: [m, ..., dy_grid]",
        "return: [m, ..., dz]",
    )
    def forward(
        self,
        yc: torch.Tensor,
        yc_grid: torch.Tensor,
    ) -> torch.Tensor:
        z_grid = self.grid_encoder(yc, yc_grid)

        z_stations = self.z_encoder(z_grid[..., [0, 2]])
        z_radar = self.z_radar_encoder(z_grid[:, -1, ..., [1, 3]])

        z_grid = self.network(z_stations, z_radar)

        return z_grid


class DtGDecoder(nn.Module):
    def __init__(
        self,
        z_decoder: nn.Module,
    ):
        super().__init__()

        self.z_decoder = z_decoder

    @check_shapes("z: [m, ..., n, dz]", "return: [m, ..., nt, dy]")
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.z_decoder(z)


class DtGModel(nn.Module):
    def __init__(
        self,
        x_dim=3,
        y_dim=1,
        distribution: Type[Distribution] = ZeroInflatedGamma,
        num_channels=128,
        num_blocks=5,
        kernel_size=5,
        pooling_size=2,
        factor_chan=1,
        p_dropout: float = 0.0,
        num_heads: int = 8,
        head_dim: int = 16,
        depth: int = 1,
    ):
        super().__init__()

        self.input_dims = x_dim
        self.output_dims = y_dim * distribution.n_params

        network = UNet(
            dim=2,
            num_channels=num_channels,
            num_blocks=num_blocks,
            kernel_size=kernel_size,
            pooling_size=pooling_size,
            factor_chan=factor_chan,
            p_dropout=p_dropout,
            num_heads=num_heads,
            head_dim=head_dim,
            depth=depth,
        )

        z_encoder = MLP(
            in_dim=2,  # Data + Flag
            out_dim=num_channels,
            num_layers=2,
            width=num_channels,
        )
        setconv_encoder = DtGModelSetConv()
        convcnp_encoder = DtGModelEncoder(
            network=network,
            grid_encoder=setconv_encoder,
            z_encoder=z_encoder,
            num_channels=num_channels,
        )

        # -----------------------------
        # Decoder
        # -----------------------------
        z_decoder = MLP(
            in_dim=num_channels,
            out_dim=self.output_dims,
            num_layers=2,
            width=num_channels,
        )

        tnp_decoder = DtGDecoder(z_decoder=z_decoder)

        self.model = BaseDtGModel(convcnp_encoder, tnp_decoder, distribution)

    def forward(
        self,
        Y_cntxt,
        Y_trgt: torch.Tensor | None = None,
        additional_sources: Dict[str, torch.Tensor] | None = None,
    ) -> Distribution:
        B, T, C, H, W = Y_cntxt.shape

        # B T H W 1 for stations
        Y_cntxt = Y_cntxt.permute(0, 1, 3, 4, 2)

        # B H W for target
        if Y_trgt is not None:
            Y_trgt = Y_trgt.squeeze(dim=[1, 2])
            assert len(Y_trgt.shape) == 3

        # Radar: B T C H W -> B 1 H W 1
        assert additional_sources is not None
        yc_grid = rearrange(additional_sources["radar_4km"], "b 1 1 h w -> b 1 h w 1")
        yc_grid = torch.nan_to_num(yc_grid, nan=-1)

        logits = self.model.forward(Y_cntxt, yc_grid)

        return logits
