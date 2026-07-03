import torch

MIN_DBZ = -1
MAX_DBZ = 64
GAIN = 1.0
OFFSET = -32


class ToDbz(torch.nn.Module):
    def __init__(self, no_data_value=255):
        super().__init__()
        self.no_data_value = no_data_value

    def forward(self, raw_radar: torch.Tensor) -> torch.Tensor:
        no_data = raw_radar == self.no_data_value
        dbz = raw_radar * GAIN + OFFSET
        dbz[no_data] = torch.nan
        return torch.clamp(dbz, MIN_DBZ, MAX_DBZ)


class ToRainRate(torch.nn.Module):
    raw_to_mmh: torch.Tensor

    def __init__(self):
        super().__init__()
        raw = torch.arange(256, dtype=torch.float32)
        to_dbz = ToDbz()
        dbz = to_dbz(raw)
        raw_to_mmh = self._dbz_to_mmh(dbz)
        raw_to_mmh[dbz == MIN_DBZ] = 0.0
        self.register_buffer("raw_to_mmh", raw_to_mmh)

    def forward(self, raw_radar: torch.Tensor) -> torch.Tensor:
        return self.raw_to_mmh[raw_radar.int()]

    @staticmethod
    def _dbz_to_mmh(x):
        return (10 ** (x / 10) / 200) ** (5 / 8)
