import torch

from dtg.data.sample import Sample
from dtg.data.source import Source


class SourceTransform(torch.nn.Module):
    def __init__(self, source: Source, *transforms: torch.nn.Module):
        super().__init__()
        self.source: Source = source
        self.transforms = torch.nn.Sequential(*transforms)

    def forward(self, sample: Sample) -> Sample:
        if self.source not in sample.keys():
            return sample
        sample[self.source] = self.transforms(sample[self.source])
        return sample

    def __repr__(self) -> str:
        return f"SourceTransform({self.source}, {self.transforms})"


class ChannelTransform(torch.nn.Module):
    def __init__(self, channel: int, *transforms: torch.nn.Module):
        super().__init__()
        self.channel = channel
        self.transforms = torch.nn.Sequential(*transforms)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x[:, self.channel : self.channel + 1] = self.transforms(
            x[:, self.channel : self.channel + 1]
        )
        return x

    def __repr__(self) -> str:
        return f"ChannelTransform({self.channel}, {self.transforms})"


class CloneSource(torch.nn.Module):
    def __init__(self, source: Source, new_source: Source):
        super().__init__()
        self.source: Source = source
        self.new_source: Source = new_source

    def forward(self, sample: Sample) -> Sample:
        sample[self.new_source] = sample[self.source].clone()
        return sample


class Identity(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, sample: Sample) -> Sample:
        return sample


class NormalizeRainRates(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.tanh(torch.log(x + 1) / 4)


class DenormalizeRainRates(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.exp(4 * torch.atanh(x)) - 1


class SelectTimes(torch.nn.Module):
    def __init__(self, times: list[int]):
        super().__init__()
        self.times = times

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x[self.times]
        return x


class Clip(torch.nn.Module):
    def __init__(self, min_value, max_value):
        super().__init__()
        self.min_value = min_value
        self.max_value = max_value

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.clip(x, self.min_value, self.max_value)


class Scale(torch.nn.Module):
    def __init__(self, factor: int):
        super().__init__()
        self.factor = factor

    def forward(self, x):
        return x * self.factor


class SelectChannels(torch.nn.Module):
    def __init__(self, channels: list[int]):
        super().__init__()
        self.channels = channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x[:, self.channels, :, :]


class AverageTimeGroups(torch.nn.Module):
    def __init__(self, frames_to_average: int):
        super().__init__()
        self.frames_to_average = frames_to_average

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        N, C, H, W = x.shape
        T = self.frames_to_average

        if N % T != 0:
            raise ValueError(f"Cannot evenly divide {N} timesteps into {T} groups.")

        groups = N // T
        x = x.view(groups, T, C, H, W)
        return torch.nanmean(x, dim=1)


class MaskRegion(torch.nn.Module):
    def __init__(self, p, scale_factor, value=float("nan")):
        super().__init__()
        self.p = p
        self.scale_factor = scale_factor
        self.mask_value = value

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if torch.rand(1).item() > self.p:
            return x

        H, W = x.shape[-2:]
        square_size = int(self.scale_factor * min(H, W))

        top = torch.randint(0, H - square_size + 1, (1,)).item()
        left = torch.randint(0, W - square_size + 1, (1,)).item()

        x[..., top : top + square_size, left : left + square_size] = self.mask_value

        return x
