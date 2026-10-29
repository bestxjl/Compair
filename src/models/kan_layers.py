"""KAN Layer implementations: FastKAN (Gaussian RBF) and B-spline KAN (official pykan)."""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class FastKANLayer(nn.Module):
    """Gaussian RBF-based KAN layer (FastKAN).

    Replaces B-splines with Gaussian radial basis functions:
    φ(x) = Σ_b w_b * exp(-(x - c_b)² / (2σ²))

    This is ~3-5x faster than B-spline evaluation on consumer GPUs.
    Reference: Li et al., "FastKAN: Very Fast Kolmogorov-Arnold Network"
    """
    def __init__(self, in_dim, out_dim, grid_size=5, base_activation=nn.SiLU):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.grid_size = grid_size

        # Equally spaced grid in [-1, 1]
        grid = torch.linspace(-1, 1, grid_size)
        self.register_buffer('grid', grid)

        # RBF width: half the grid spacing
        self.sigma = 2.0 / (grid_size - 1) if grid_size > 1 else 1.0

        # RBF weights: (out_dim, in_dim, grid_size)
        self.rbf_weight = nn.Parameter(torch.randn(out_dim, in_dim, grid_size) * 0.1)

        # Base path: SiLU activation with learnable linear combination
        self.base_activation = base_activation()
        self.base_weight = nn.Parameter(torch.randn(out_dim, in_dim) * 0.1)

        # Optional LayerNorm for stability
        self.ln = nn.LayerNorm(out_dim)

        self._init_weights()

    def _init_weights(self):
        nn.init.kaiming_uniform_(self.rbf_weight, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.base_weight, a=math.sqrt(5))

    def forward(self, x):
        # x: (B, in_dim)
        B = x.shape[0]

        # RBF basis: (B, in_dim, grid_size)
        x_exp = x.unsqueeze(-1)  # (B, in_dim, 1)
        grid = self.grid.view(1, 1, -1)  # (1, 1, G)
        rbf = torch.exp(-((x_exp - grid) ** 2) / (2 * self.sigma ** 2))

        # Weighted combination: (B, out_dim)
        out_rbf = torch.einsum('big,oig->bo', rbf, self.rbf_weight)

        # Base path
        out_base = F.linear(self.base_activation(x), self.base_weight)

        out = out_rbf + out_base
        out = self.ln(out)
        return out


# ---------------------------------------------------------------------------
# Official pykan B-spline helpers (from Liu et al., KAN, 2024)
# ---------------------------------------------------------------------------

def _extend_grid(grid, k_extend):
    """Pad a uniform grid with k_extend extra points on each side.

    Mirrors pykan.spline.extend_grid, the extra margin allows B-spline
    basis functions to be evaluated near the domain boundaries without
    special-casing the edges.
    """
    h = (grid[:, [-1]] - grid[:, [0]]) / (grid.shape[1] - 1)
    for _ in range(k_extend):
        grid = torch.cat([grid[:, [0]] - h, grid], dim=1)
        grid = torch.cat([grid, grid[:, [-1]] + h], dim=1)
    return grid


def _B_batch(x, grid, k):
    """Evaluate x on B-spline bases of order k via Cox-de Boor recurrence.

    This is a direct PyTorch translation of pykan.spline.B_batch.

    Args:
        x:      (batch, in_dim) , input values.
        grid:   (in_dim, G+1+2k), extended knot grid, one row per input dim.
        k:      int             , spline order (0 = piecewise constant).

    Returns:
        (batch, in_dim, G+k), B-spline basis values.
    """
    x = x.unsqueeze(dim=2)           # (batch, in_dim, 1)
    grid = grid.unsqueeze(dim=0)     # (1, in_dim, G+1+2k)

    if k == 0:
        value = ((x >= grid[:, :, :-1]) & (x < grid[:, :, 1:])).to(x.dtype)
    else:
        B_km1 = _B_batch(x[:, :, 0], grid=grid[0], k=k - 1)

        denom1 = grid[:, :, k:-1] - grid[:, :, :-(k + 1)]
        denom2 = grid[:, :, k + 1:] - grid[:, :, 1:(-k)]

        term1 = (x - grid[:, :, :-(k + 1)]) / (denom1 + 1e-12) * B_km1[:, :, :-1]
        term2 = (grid[:, :, k + 1:] - x) / (denom2 + 1e-12) * B_km1[:, :, 1:]

        value = term1 + term2

    return torch.nan_to_num(value)  # guard against degenerate grid edges


class BSplineKANLayer(nn.Module):
    """Cubic B-spline KAN layer, faithful to the official pykan KANLayer.

    Uses true cubic (k=3) B-spline basis functions evaluated via the
    Cox-de Boor recurrence, with uniform grid + SiLU residual path +
    LayerNorm for stability.  This is the reference B-spline implementation
    from Liu et al., "KAN: Kolmogorov-Arnold Networks", ICLR 2025.
    """

    def __init__(self, in_dim, out_dim, grid_size=5, spline_order=3,
                 base_activation=nn.SiLU, grid_range=(-1.0, 1.0)):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.num = grid_size          # G intervals
        self.k = spline_order         # spline order (degree)

        # Uniform grid: G+1 points -> extend by k on each side -> G+1+2k points
        grid = torch.linspace(grid_range[0], grid_range[1], steps=self.num + 1)
        grid = grid.unsqueeze(0).expand(self.in_dim, self.num + 1)
        grid = _extend_grid(grid, k_extend=self.k)
        self.register_buffer('grid', grid)  # (in_dim, G+1+2k)

        # B-spline coefficients: (out_dim, in_dim, G+k)
        n_basis = self.num + self.k
        self.coef = nn.Parameter(
            torch.randn(out_dim, in_dim, n_basis) * (0.5 / self.num)
        )

        # Base path: SiLU + linear (residual connection, same as pykan)
        self.base_activation = base_activation()
        self.base_weight = nn.Parameter(torch.randn(out_dim, in_dim) * 0.1)

        # LayerNorm for stability (our addition, pykan uses scale parameters)
        self.ln = nn.LayerNorm(out_dim)

        self._init_weights()

    def _init_weights(self):
        nn.init.kaiming_uniform_(self.coef, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.base_weight, a=math.sqrt(5))

    def forward(self, x):
        # x: (B, in_dim)

        # Cubic B-spline basis: (B, in_dim, G+k)
        b_splines = _B_batch(x, self.grid, self.k)

        # Weighted combination: (B, out_dim)
        out_spline = torch.einsum('bik,oik->bo', b_splines, self.coef)

        # Base path: SiLU + linear
        out_base = F.linear(self.base_activation(x), self.base_weight)

        out = out_spline + out_base
        out = self.ln(out)
        return out


class KANClassifier(nn.Module):
    """Multi-layer KAN classifier.

    Supports both FastKAN (RBF) and BSpline KAN variants.
    """
    def __init__(self, layer_dims, grid_size=5, spline_order=3,
                 kan_type='fast', dropout=0.0):
        super().__init__()
        self.layer_dims = layer_dims
        self.kan_type = kan_type

        layers = []
        for i in range(len(layer_dims) - 1):
            if kan_type == 'fast':
                layers.append(FastKANLayer(
                    layer_dims[i], layer_dims[i+1], grid_size=grid_size
                ))
            else:
                layers.append(BSplineKANLayer(
                    layer_dims[i], layer_dims[i+1],
                    grid_size=grid_size, spline_order=spline_order
                ))
            if dropout > 0 and i < len(layer_dims) - 2:
                layers.append(nn.Dropout(dropout))

        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)

    def get_param_count(self):
        return sum(p.numel() for p in self.parameters())
