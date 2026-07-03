from typing import Optional, Tuple

import torch
import torch.nn as nn
from einops import rearrange


class PairwiseMLP(nn.Module):
    """
    Per-pair MLP applied to last dimension, supports arbitrary leading dimensions.
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        layers: Optional[Tuple[int, ...]] = None,
        num_layers: Optional[int] = None,
        width: Optional[int] = None,
        nonlinearity: Optional[nn.Module] = None,
    ):
        super().__init__()

        if layers is None:
            assert (
                num_layers is not None and width is not None
            ), "Must specify either `layers` or `num_layers` and `width`."
            layers = (width,) * num_layers

        if nonlinearity is None:
            nonlinearity = nn.ReLU()

        # build linear MLP as Sequential
        seq = []
        if len(layers) == 0:
            seq.append(nn.Linear(in_dim, out_dim))
        else:
            seq.append(nn.Linear(in_dim, layers[0]))
            for i in range(1, len(layers)):
                seq.append(nonlinearity)
                seq.append(nn.Linear(layers[i - 1], layers[i]))
            seq.append(nonlinearity)
            seq.append(nn.Linear(layers[-1], out_dim))
        self.net = nn.Sequential(*seq)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [..., in_dim] -> returns [..., out_dim]
        Flattens all leading dimensions automatically.
        """
        orig_shape = x.shape[:-1]
        x_flat = x.reshape(-1, x.shape[-1])  # flatten all leading dims
        out_flat = self.net(x_flat)
        out = out_flat.reshape(*orig_shape, out_flat.shape[-1])
        return out


class TEAttention2D(nn.Module):
    def __init__(
        self,
        dim,
        kernel: nn.Module,
        num_heads: int = 8,
        head_dim: int = 16,
        p_dropout: float = 0.0,
        window_size=16,
    ):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.scale = head_dim**-0.5
        self.window_size = window_size

        inner_dim = num_heads * head_dim

        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_k = nn.Linear(dim, inner_dim, bias=False)
        self.to_v = nn.Linear(dim, inner_dim, bias=False)
        self.to_out = nn.Linear(inner_dim, dim)

        self.dropout = nn.Dropout(p_dropout)
        self.kernel = kernel

    def _get_rel_pos(self, Ws: int, device: torch.device) -> torch.Tensor:
        """
        Return rel_pos tensor shaped [1, N, N, 2] (batched dimension later expanded as needed).
        Local relative positions: use coords within window [0..ws-1]
        """
        coords = (
            torch.stack(
                torch.meshgrid(
                    torch.arange(Ws, device=device),
                    torch.arange(Ws, device=device),
                    indexing="ij",
                ),
                dim=-1,
            )
            .view(Ws * Ws, 2)
            .float()
        )
        rel_pos = coords[:, None, :] - coords[None, :, :]
        rel_pos = rel_pos / float(Ws)
        rel_pos = rel_pos.unsqueeze(0).unsqueeze(0).unsqueeze(0)  # [1,1,1,N,N,2]

        return rel_pos

    def forward(
        self, x_q: torch.Tensor, x_kv: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        x_q: [B, H, W, C]  (queries)
        x_kv: [B, H, W, C] or None (keys/values). If None -> self-attention is used (kv = q).
        returns: [B, H, W, C]
        """
        if x_kv is None:
            x_kv = x_q

        B, H, W, C = x_q.shape

        # pad H and W to multiple of window_size
        pad_h = (self.window_size - H % self.window_size) % self.window_size
        pad_w = (self.window_size - W % self.window_size) % self.window_size
        x_q = nn.functional.pad(x_q, (0, 0, 0, pad_w, 0, pad_h))
        x_kv = nn.functional.pad(x_kv, (0, 0, 0, pad_w, 0, pad_h))

        # partition into windows
        x_q_win = rearrange(
            x_q,
            "b (nh ws) (nw ws2) c -> b nh nw ws ws2 c",
            ws=self.window_size,
            ws2=self.window_size,
        )
        x_kv_win = rearrange(
            x_kv,
            "b (nh ws) (nw ws2) c -> b nh nw ws ws2 c",
            ws=self.window_size,
            ws2=self.window_size,
        )

        # flatten window pixels
        B_, Nh, Nw, Ws, Ws2, C = x_q_win.shape
        x_q_flat = x_q_win.reshape(B_, Nh, Nw, Ws * Ws2, C)
        x_kv_flat = x_kv_win.reshape(B_, Nh, Nw, Ws * Ws2, C)

        # compute Q/K/V
        q = self.to_q(x_q_flat)
        k = self.to_k(x_kv_flat)
        v = self.to_v(x_kv_flat)
        q, k, v = [
            rearrange(t, "b nh nw n (h d) -> b nh nw h n d", h=self.num_heads)
            for t in (q, k, v)
        ]

        rel_pos = self._get_rel_pos(Ws, x_q.device)
        rel_pos = rel_pos.expand(B_, Nh, Nw, Ws * Ws, Ws * Ws, 2)

        # attention
        attn_logits = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attn_for_kernel = rearrange(attn_logits, "b nh nw h n m -> b nh nw n m h")

        kernel_input = torch.cat((rel_pos, attn_for_kernel), dim=-1)

        # kernel produces an attention map per head: [B, N, N, num_heads]
        attn = self.kernel(kernel_input)  # PairwiseMLP returns [B, N, N, num_heads]

        # back to [B, h, N, N]
        attn = rearrange(attn, "b nh nw n m h -> b nh nw h n m")

        # softmax over keys
        attn = attn.softmax(dim=-1)
        attn = self.dropout(attn)

        # attend to values: [B, h, N, d]
        out = torch.matmul(attn, v)
        out = rearrange(out, "b nh nw h n d -> b nh nw n (h d)")
        out = self.to_out(out)
        out = self.dropout(out)
        # reshape back to grid
        out = out.view(B_, Nh, Nw, Ws, Ws, C)
        out = rearrange(out, "b nh nw ws1 ws2 c -> b (nh ws1) (nw ws2) c")
        out = out[:, :H, :W, :]  # remove padding
        return out


class TETransformerBlock2D(nn.Module):
    def __init__(
        self,
        dim,
        num_heads: int = 8,
        head_dim: int = 16,
        num_channels: int = 48,
        mlp_dim: Optional[int] = None,
        p_dropout: float = 0.0,
    ):
        super().__init__()
        mlp_dim = mlp_dim or dim * 4

        # kernel takes (2 + num_heads) input channels and outputs num_heads logits per pair
        kernel = PairwiseMLP(
            in_dim=num_heads + 2,
            out_dim=num_heads,
            num_layers=2,
            width=num_channels,
        )

        self.attn = TEAttention2D(dim, kernel, num_heads, head_dim, p_dropout)

        self.norm1 = nn.LayerNorm(dim)
        self.ff = nn.Sequential(
            nn.Linear(dim, mlp_dim),
            nn.ReLU(),
            nn.Dropout(p_dropout),
            nn.Linear(mlp_dim, dim),
            nn.Dropout(p_dropout),
        )
        self.norm2 = nn.LayerNorm(dim)

    def forward(
        self, x_q: torch.Tensor, x_kv: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        x_q: [B, H, W, C]
        x_kv: optional [B, H, W, C] (if provided, cross-attend q->kv)
        returns: [B, H, W, C]
        """
        # Norms are applied to last dim (C)
        q_norm = self.norm1(x_q)
        kv_norm = self.norm1(x_kv) if x_kv is not None else None

        attn_out = self.attn(q_norm, kv_norm)
        x = x_q + attn_out

        x = x + self.ff(self.norm2(x))
        return x


class TETransformer2D(nn.Module):
    def __init__(
        self,
        dim: int,
        depth: int = 1,
        num_heads: int = 8,
        head_dim: int = 16,
        num_channels: int = 48,
        mlp_dim: Optional[int] = None,
        p_dropout: float = 0.0,
    ):
        super().__init__()
        self.layers = nn.ModuleList(
            [
                TETransformerBlock2D(
                    dim, num_heads, head_dim, num_channels, mlp_dim, p_dropout
                )
                for _ in range(depth)
            ]
        )

    def forward(
        self, x: torch.Tensor, kv: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        x: [B, C, H, W] -> [B, C, H, W]
        kv: optional [B, C, H, W] to be used as keys/values (cross-attention across all layers)
        """
        # move channels to last dimension
        x = x.permute(0, 2, 3, 1)  # [B, H, W, C]
        kv_t = None
        if kv is not None:
            kv_t = kv.permute(0, 2, 3, 1)  # [B, H, W, C]

        for layer in self.layers:
            x = layer(x, x_kv=kv_t)
        x = x.permute(0, 3, 1, 2)  # back to [B, C, H, W]
        return x


class TemporalPixelwiseAttention(nn.Module):
    """
    Temporal attention over the time axis (T) for each pixel independently.
    Input:  [B, T, C, H, W]
    Output: [B, C, H, W]
    """

    def __init__(
        self, dim, num_heads=8, head_dim=16, p_dropout=0.0, num_queries=1, T=3
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.scale = head_dim**-0.5
        inner_dim = num_heads * head_dim

        # Instead of learning Q from input, we define learnable query embeddings
        self.query = nn.Parameter(torch.randn(num_queries, inner_dim))
        nn.init.xavier_uniform_(self.query)

        self.to_k = nn.Linear(dim, inner_dim, bias=False)
        self.to_v = nn.Linear(dim, inner_dim, bias=False)
        self.to_out = nn.Linear(inner_dim, dim)
        self.dropout = nn.Dropout(p_dropout)

        self.pos_emb = nn.Parameter(torch.randn(1, T, dim, 1, 1))
        nn.init.normal_(self.pos_emb, std=0.02)

    def forward(self, x):
        """
        x: [B, T, C, H, W]
        returns: [B, C, H, W]
        """
        B, T, C, H, W = x.shape

        x = x + self.pos_emb  # [1, T, C, 1, 1]

        # flatten spatial dims to batch
        x_flat = rearrange(x, "b t c h w -> (b h w) t c")

        # KV projections
        k = self.to_k(x_flat)
        v = self.to_v(x_flat)

        # split heads
        k, v = [rearrange(t, "n t (h d) -> n h t d", h=self.num_heads) for t in (k, v)]

        # Prepare learnable query (shared across all pixels)
        q = self.query  # [num_queries, inner_dim]
        q = rearrange(q, "q (h d) -> h q d", h=self.num_heads)
        q = q.unsqueeze(0).expand(k.size(0), -1, -1, -1)  # [n, h, q, d]

        # attention weights
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.dropout(attn)

        # attention output: [n, h, q, d]
        out = attn @ v  # [n, h, q, d]
        out = rearrange(out, "n h q d -> n q (h d)")
        # project to output dim
        out = self.to_out(out)  # [n, q, dim]
        # if num_queries == 1, just squeeze
        out = out.squeeze(1)

        # reshape back to [B, C, H, W]
        out = rearrange(out, "(b h w) c -> b c h w", b=B, h=H, w=W)
        return out


class TemporalPixelwiseTransformerBlock(nn.Module):
    def __init__(self, dim, num_heads=8, head_dim=16, mlp_dim=None, p_dropout=0.0):
        super().__init__()
        mlp_dim = mlp_dim or dim * 4
        self.attn = TemporalPixelwiseAttention(dim, num_heads, head_dim, p_dropout)
        self.norm1 = nn.LayerNorm(dim)
        self.ff = nn.Sequential(
            nn.Linear(dim, mlp_dim),
            nn.ReLU(),
            nn.Dropout(p_dropout),
            nn.Linear(mlp_dim, dim),
            nn.Dropout(p_dropout),
        )
        self.norm2 = nn.LayerNorm(dim)

    def forward(self, x):
        """
        x: [B, T, C, H, W]
        returns: [B, C, H, W]
        """
        B, T, C, H, W = x.shape
        x_flat = rearrange(x, "b t c h w -> (b h w) t c")
        x_norm = self.norm1(x_flat)
        x_norm = rearrange(x_norm, "(b h w) t c -> b t c h w", b=B, h=H, w=W)

        out = self.attn(x_norm)
        # The FFN acts on the fused output (per-pixel)
        out_flat = rearrange(out, "b c h w -> (b h w) c")
        out_flat = out_flat + self.ff(self.norm2(out_flat))
        out = rearrange(out_flat, "(b h w) c -> b c h w", b=B, h=H, w=W)
        return out
