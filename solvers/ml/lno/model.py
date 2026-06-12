"""
Laplace Neural Operator (LNO) architecture for the gene regulatory network.

Maps initial conditions (G0, P0) → full trajectories (G(t), P(t)) by
operating in a latent Laplace/frequency domain.

Pipeline:
  IC [B, 2]
    → Lifting MLP          [B, d_model]
    → Temporal Embedding   [B, T, d_model]   (broadcast IC + sinusoidal t)
    → N × LNO Blocks       [B, T, d_model]   (frequency-domain mixing)
    → Projection MLP       [B, T, 2]
"""

import math

import torch
import torch.nn as nn


# ─────────────────────────────────────────────────────────────────────────────
# Sinusoidal Temporal Encoding
# ─────────────────────────────────────────────────────────────────────────────

class SinusoidalTimeEncoding(nn.Module):
    """Encodes a time grid t ∈ [0,1]^T into [T, d_enc] using sinusoidal bases."""

    def __init__(self, d_enc: int, max_period: float = 1e4):
        super().__init__()
        assert d_enc % 2 == 0, "d_enc must be even"
        self.d_enc = d_enc
        self.max_period = max_period

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        d = self.d_enc
        div = torch.exp(
            torch.arange(0, d, 2, dtype=torch.float32, device=t.device)
            * (-math.log(self.max_period) / d)
        )
        args = t.unsqueeze(1) * div.unsqueeze(0)
        return torch.cat([torch.sin(args), torch.cos(args)], dim=-1)


# ─────────────────────────────────────────────────────────────────────────────
# Laplace Mixing Layer (frequency-domain convolution)
# ─────────────────────────────────────────────────────────────────────────────

class LaplaceMixingLayer(nn.Module):
    """Learnable linear map in Fourier/Laplace frequency domain along time.

    1. rfft along T → complex spectrum
    2. Complex linear map (parameterised rational transfer function)
    3. irfft back to time domain
    """

    def __init__(self, d_model: int, n_modes: int, T: int):
        super().__init__()
        self.d_model = d_model
        self.n_modes = n_modes
        self.T = T

        scale = 1.0 / math.sqrt(d_model)
        self.W_re = nn.Parameter(scale * torch.randn(n_modes, d_model, d_model))
        self.W_im = nn.Parameter(scale * torch.randn(n_modes, d_model, d_model))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape
        x_ft = torch.fft.rfft(x, dim=1, norm="ortho")

        modes = min(self.n_modes, x_ft.shape[1])
        x_ft_trunc = x_ft[:, :modes, :]

        x_re = x_ft_trunc.real
        x_im = x_ft_trunc.imag

        out_re = (
            torch.einsum("bmd,mde->bme", x_re, self.W_re[:modes])
            - torch.einsum("bmd,mde->bme", x_im, self.W_im[:modes])
        )
        out_im = (
            torch.einsum("bmd,mde->bme", x_re, self.W_im[:modes])
            + torch.einsum("bmd,mde->bme", x_im, self.W_re[:modes])
        )
        out_ft_trunc = torch.complex(out_re, out_im)

        full_ft_size = T // 2 + 1
        out_ft = torch.zeros(B, full_ft_size, D, dtype=torch.cfloat, device=x.device)
        out_ft[:, :modes, :] = out_ft_trunc

        return torch.fft.irfft(out_ft, n=T, dim=1, norm="ortho")


# ─────────────────────────────────────────────────────────────────────────────
# Pointwise MLP Bypass
# ─────────────────────────────────────────────────────────────────────────────

class PointwiseMLP(nn.Module):
    """Small depth-2 MLP applied independently at each time step."""

    def __init__(self, d_in: int, d_hidden: int, d_out: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, d_hidden),
            nn.GELU(),
            nn.Linear(d_hidden, d_out),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ─────────────────────────────────────────────────────────────────────────────
# LNO Block
# ─────────────────────────────────────────────────────────────────────────────

class LNOBlock(nn.Module):
    """z_out = LayerNorm(z + σ(LapMix(z) + PointwiseMLP(z)))"""

    def __init__(self, d_model: int, n_modes: int, mlp_hidden: int, T: int):
        super().__init__()
        self.laplace_mix = LaplaceMixingLayer(d_model, n_modes, T)
        self.pointwise_mlp = PointwiseMLP(d_model, mlp_hidden, d_model)
        self.norm = nn.LayerNorm(d_model)
        self.activation = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        laplace_out = self.laplace_mix(x)
        pointwise_out = self.pointwise_mlp(x)
        combined = self.activation(laplace_out + pointwise_out)
        return self.norm(x + combined)


# ─────────────────────────────────────────────────────────────────────────────
# Helper MLP builders
# ─────────────────────────────────────────────────────────────────────────────

def _build_lifting_mlp(d_in: int, d_out: int, n_layers: int) -> nn.Sequential:
    layers = [nn.Linear(d_in, d_out), nn.GELU()]
    for _ in range(n_layers - 1):
        layers += [nn.Linear(d_out, d_out), nn.GELU()]
    return nn.Sequential(*layers)


def _build_proj_mlp(d_in: int, d_out: int, n_layers: int) -> nn.Sequential:
    layers = [nn.Linear(d_in, d_in), nn.GELU()]
    for _ in range(n_layers - 1):
        layers += [nn.Linear(d_in, d_in), nn.GELU()]
    layers.append(nn.Linear(d_in, d_out))
    return nn.Sequential(*layers)


# ─────────────────────────────────────────────────────────────────────────────
# Full Laplace Neural Operator
# ─────────────────────────────────────────────────────────────────────────────

class LaplaceNeuralOperator(nn.Module):
    """Maps initial conditions → full trajectory via frequency-domain blocks.

    Parameters
    ----------
    T              : number of output time steps
    d_model        : latent channel width
    n_lno_blocks   : number of LNO blocks
    n_modes        : Fourier modes retained per block
    mlp_hidden     : hidden width of pointwise bypass MLP
    lifting_layers : depth of IC lifting MLP
    proj_layers    : depth of output projection MLP
    n_species      : output dimension (2 for G, P)
    d_enc          : sinusoidal encoding dimension
    """

    def __init__(
        self, T, d_model, n_lno_blocks, n_modes, mlp_hidden,
        lifting_layers, proj_layers, n_species=2, d_enc=32,
    ):
        super().__init__()
        self.T = T
        self.d_model = d_model
        self.d_enc = d_enc

        self.lifting = _build_lifting_mlp(n_species, d_model, lifting_layers)
        self.time_enc = SinusoidalTimeEncoding(d_enc=d_enc)
        self.time_proj = nn.Linear(d_enc, d_model)
        self.blocks = nn.ModuleList([
            LNOBlock(d_model, n_modes, mlp_hidden, T)
            for _ in range(n_lno_blocks)
        ])
        self.projection = _build_proj_mlp(d_model, n_species, proj_layers)

    def forward(self, ic: torch.Tensor, t_grid: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        ic     : [B, 2]  — normalised initial conditions
        t_grid : [T]     — normalised time grid in [0, 1]

        Returns
        -------
        out : [B, T, 2]  — predicted trajectory
        """
        B = ic.shape[0]
        T = self.T

        h = self.lifting(ic)                                  # [B, d_model]
        h = h.unsqueeze(1).expand(B, T, self.d_model)         # [B, T, d_model]

        t_enc = self.time_enc(t_grid)                          # [T, d_enc]
        t_emb = self.time_proj(t_enc)                          # [T, d_model]
        h = h + t_emb.unsqueeze(0)                             # [B, T, d_model]

        for block in self.blocks:
            h = block(h)

        return self.projection(h)                              # [B, T, 2]
