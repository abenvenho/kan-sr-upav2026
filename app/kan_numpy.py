# SPDX-License-Identifier: Apache-2.0
"""
Passagem direta (forward) da KAN treinada, em NumPy puro.

Reproduz exatamente o modelo pykan [12, 6, 1], grid = 10, k = 3, salvo em
models/kan_model_state.pt, sem exigir PyTorch nem pykan. Os pesos foram
exportados para models/kan_weights.npz por tools/exportar_pesos_kan.py.

Cada aresta (i -> j) de cada camada aplica
    phi_ij(x) = mask_ij * ( scale_base_ij * silu(x) + scale_sp_ij * spline_ij(x) )
e cada no soma as arestas que chegam a ele. O ramo simbolico, os vieses e as
escalas de no do modelo salvo sao nulos/unitarios e foram omitidos.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np


def _silu(x: np.ndarray) -> np.ndarray:
    return x / (1.0 + np.exp(-x))


def _b_batch(x: np.ndarray, grid: np.ndarray, k: int) -> np.ndarray:
    """Bases B-spline (Cox-de Boor), mesma convencao de kan.spline.B_batch.

    x: (batch, n_in)   grid: (n_in, n_pts)   ->   (batch, n_in, n_pts - 1 - k)
    """
    x = x[:, :, None]
    g = grid[None, :, :]
    if k == 0:
        return ((x >= g[:, :, :-1]) & (x < g[:, :, 1:])).astype(float)
    b = _b_batch(x[:, :, 0], grid, k - 1)
    with np.errstate(divide="ignore", invalid="ignore"):
        left = (x - g[:, :, : -(k + 1)]) / (g[:, :, k:-1] - g[:, :, : -(k + 1)]) * b[:, :, :-1]
        right = (g[:, :, k + 1 :] - x) / (g[:, :, k + 1 :] - g[:, :, 1:-k]) * b[:, :, 1:]
    return np.nan_to_num(left + right)


class KANNumpy:
    def __init__(self, weights_path: str | Path):
        w = np.load(weights_path)
        self.k = int(w["k"])
        self.layers = []
        n_layers = int(w["n_layers"])
        for l in range(n_layers):
            self.layers.append(
                {name: w[f"{name}_{l}"].astype(float) for name in ("grid", "coef", "mask", "scale_base", "scale_sp")}
            )
        self.y_min = float(w["y_min"])
        self.y_range = float(w["y_range"])

    def forward_norm(self, x_norm: np.ndarray) -> np.ndarray:
        """Entrada ja normalizada MinMax [0,1] -> saida normalizada."""
        x = np.asarray(x_norm, dtype=float)
        for p in self.layers:
            basis = _b_batch(x, p["grid"], self.k)                     # (b, in, G+k)
            spline = np.einsum("bik,iok->bio", basis, p["coef"])      # (b, in, out)
            y = p["scale_base"][None] * _silu(x)[:, :, None] + p["scale_sp"][None] * spline
            x = (p["mask"][None] * y).sum(axis=1)                      # (b, out)
        return x[:, 0]

    def predict_ln_valor_m2(self, x_norm: np.ndarray) -> np.ndarray:
        """Entrada normalizada -> ln(R$/m2)."""
        return self.forward_norm(x_norm) * self.y_range + self.y_min
