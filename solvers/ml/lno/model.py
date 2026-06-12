"""Laplace Neural Operator (LNO) model architecture.

Maps initial conditions (x0, y0) -> full trajectories (x(t), y(t)) over time steps
by operating in a latent Laplace/frequency domain.
"""

import math
import torch
import torch.nn as nn

class SinusoidalTimeEncoding(nn.Module):
    """
    Encodes a scalar time grid t ∈ [0,1]^T into a [T, d_enc] matrix using
    sinusoidal basis functions.
    """
    def __init__(self, d_enc: int, max_period: float = 1e4):
        super().__init__()
        assert d_enc % 2 == 0, "d_enc must be even"
        self.d_enc     = d_enc
        self.max_period = max_period

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        T   = t.shape[0]
        d   = self.d_enc
        div = torch.exp(
            torch.arange(0, d, 2, dtype=torch.float32, device=t.device)
            * (-math.log(self.max_period) / d)
        )
        args = t.unsqueeze(1) * div.unsqueeze(0)
        enc  = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        return enc


class LaplaceMixingLayer(nn.Module):
    """
    Performs a learnable linear map in the real Fourier / Laplace frequency
    domain along the time axis.
    """
    def __init__(self, d_model: int, n_modes: int, T: int):
        super().__init__()
        self.d_model = d_model
        self.n_modes = n_modes
        self.T       = T

        scale = 1.0 / math.sqrt(d_model)
        self.W_re = nn.Parameter(scale * torch.randn(n_modes, d_model, d_model))
        self.W_im = nn.Parameter(scale * torch.randn(n_modes, d_model, d_model))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape

        # Step 1: real FFT along time dimension
        x_ft = torch.fft.rfft(x, dim=1, norm="ortho")

        # Keep only the first n_modes frequency components
        modes = min(self.n_modes, x_ft.shape[1])
        x_ft_trunc = x_ft[:, :modes, :]

        # Step 2: complex linear map in frequency domain
        x_re = x_ft_trunc.real
        x_im = x_ft_trunc.imag

        out_re = (
            torch.einsum("bmd,mde->bme", x_re, self.W_re)
            - torch.einsum("bmd,mde->bme", x_im, self.W_im)
        )
        out_im = (
            torch.einsum("bmd,mde->bme", x_re, self.W_im)
            + torch.einsum("bmd,mde->bme", x_im, self.W_re)
        )
        out_ft_trunc = torch.complex(out_re, out_im)

        # Pad back to full spectrum size with zeros
        full_ft_size = T // 2 + 1
        out_ft = torch.zeros(B, full_ft_size, D, dtype=torch.cfloat, device=x.device)
        out_ft[:, :modes, :] = out_ft_trunc

        # Step 3: inverse FFT back to time domain
        out = torch.fft.irfft(out_ft, n=T, dim=1, norm="ortho")
        return out


class PointwiseMLP(nn.Module):
    """
    A small depth-2 MLP applied independently at each time step.
    """
    def __init__(self, d_in: int, d_hidden: int, d_out: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in,     d_hidden),
            nn.GELU(),
            nn.Linear(d_hidden, d_out),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class LNOBlock(nn.Module):
    """
    One Laplace Neural Operator block.
    """
    def __init__(self, d_model: int, n_modes: int, mlp_hidden: int, T: int):
        super().__init__()
        self.laplace_mix   = LaplaceMixingLayer(d_model, n_modes, T)
        self.pointwise_mlp = PointwiseMLP(d_model, mlp_hidden, d_model)
        self.norm          = nn.LayerNorm(d_model)
        self.activation    = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        laplace_out  = self.laplace_mix(x)
        pointwise_out = self.pointwise_mlp(x)
        combined = self.activation(laplace_out + pointwise_out)
        return self.norm(x + combined)


def build_lifting_mlp(d_in: int, d_out: int, n_layers: int) -> nn.Sequential:
    """Lifts the initial condition into the latent space."""
    layers = [nn.Linear(d_in, d_out), nn.GELU()]
    for _ in range(n_layers - 1):
        layers += [nn.Linear(d_out, d_out), nn.GELU()]
    return nn.Sequential(*layers)


def build_proj_mlp(d_in: int, d_out: int, n_layers: int) -> nn.Sequential:
    """Maps latent representations back to physical outputs."""
    layers = [nn.Linear(d_in, d_in), nn.GELU()]
    for _ in range(n_layers - 1):
        layers += [nn.Linear(d_in, d_in), nn.GELU()]
    layers.append(nn.Linear(d_in, d_out))
    return nn.Sequential(*layers)


class LaplaceNeuralOperator(nn.Module):
    """
    Laplace Neural Operator (LNO) for trajectory mapping.
    """
    def __init__(
        self,
        T:              int,
        d_model:        int,
        n_lno_blocks:   int,
        n_modes:        int,
        mlp_hidden:     int,
        lifting_layers: int,
        proj_layers:    int,
        n_species:      int = 2,
        d_enc:          int = 32,
    ):
        super().__init__()
        self.T       = T
        self.d_model = d_model
        self.d_enc   = d_enc

        self.lifting = build_lifting_mlp(
            d_in    = n_species,
            d_out   = d_model,
            n_layers= lifting_layers,
        )

        self.time_enc = SinusoidalTimeEncoding(d_enc=d_enc)
        self.time_proj = nn.Linear(d_enc, d_model)

        self.blocks = nn.ModuleList([
            LNOBlock(d_model, n_modes, mlp_hidden, T)
            for _ in range(n_lno_blocks)
        ])

        self.projection = build_proj_mlp(
            d_in    = d_model,
            d_out   = n_species,
            n_layers= proj_layers,
        )

    def forward(self, ic: torch.Tensor, t_grid: torch.Tensor) -> torch.Tensor:
        B = ic.shape[0]
        T = self.T

        h = self.lifting(ic)
        h = h.unsqueeze(1).expand(B, T, self.d_model)

        t_enc = self.time_enc(t_grid)
        t_emb = self.time_proj(t_enc)
        h = h + t_emb.unsqueeze(0)

        for block in self.blocks:
            h = block(h)

        out = self.projection(h)
        return out


def build_model(hp: dict) -> LaplaceNeuralOperator:
    """Construct the LNO model from hyperparameter dictionary."""
    model = LaplaceNeuralOperator(
        T              = hp["n_timesteps"],
        d_model        = hp["d_model"],
        n_lno_blocks   = hp["n_lno_blocks"],
        n_modes        = hp["n_modes"],
        mlp_hidden     = hp["mlp_hidden"],
        lifting_layers = hp["lifting_layers"],
        proj_layers    = hp["proj_layers"],
        n_species      = hp["n_species"],
        d_enc          = max(32, hp["d_model"] // 2),
    )
    return model
