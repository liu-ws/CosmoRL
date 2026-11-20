"""Flat LambdaCDM background expansion and 1LPT initial conditions via CLASS."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite, log, pi, tau
from typing import Literal

import numpy as np
import torch
from classy import Class
from torch import Tensor

from ..config import CosmologyConfig
from .pm import spectral_geometry

_QUADRATURE_NODES, _QUADRATURE_WEIGHTS = np.polynomial.legendre.leggauss(8)


@dataclass(frozen=True)
class Background:
    """Flat Friedmann background with matter, radiation, and cosmological constant."""

    omega_m: float
    omega_lambda: float
    omega_r: float

    def expansion(self, a: float | np.ndarray) -> float | np.ndarray:
        """Evaluate dimensionless Hubble parameter E(a) = H(a) / H0."""
        return np.sqrt(self.omega_r / a**4 + self.omega_m / a**3 + self.omega_lambda)

    def time_factor(self, a0: float, a1: float, *, kind: Literal["drift", "kick"]) -> float:
        """Integrate the drift or kick factor over [a0, a1] via 8-node Gauss-Legendre in ln(a)."""
        if kind == "drift":
            power = 2
        elif kind == "kick":
            power = 1
        else:
            raise ValueError(f"Expected kind 'drift' or 'kick', got {kind!r}.")

        log_start, log_end = log(a0), log(a1)
        half_interval = 0.5 * (log_end - log_start)
        a = np.exp(0.5 * (log_start + log_end) + half_interval * _QUADRATURE_NODES)
        integrand = 1.0 / (a**power * self.expansion(a))
        return float(half_interval * np.dot(_QUADRATURE_WEIGHTS, integrand))


class InitialConditions:
    """Generate one-fluid 1LPT realizations from CLASS N-body gauge transfer functions."""

    def __init__(
        self,
        size: int,
        box_mpc: float,
        a_initial: float,
        cosmology: CosmologyConfig,
        device: torch.device,
    ) -> None:
        if size < 3 or not isfinite(box_mpc) or box_mpc <= 0 or not (0 < a_initial < 1):
            raise ValueError("Initial conditions require size >= 3, positive box_mpc, and 0 < a_initial < 1.")

        self.size = size
        self.shape = (size, size, size)
        self.device = device
        z_initial = 1.0 / a_initial - 1.0

        q_squared, self.inverse_gradient, nyquist = spectral_geometry(size, device)
        k = q_squared.sqrt() / box_mpc
        keep = (k > 0) & ~nyquist
        used_k = k[keep].cpu().numpy()
        pivot = 0.05  # Mpc^-1

        model = Class()
        try:
            model.set(
                {
                    "h": cosmology.h,
                    "omega_b": cosmology.omega_b,
                    "omega_cdm": cosmology.omega_cdm,
                    "A_s": cosmology.scalar_amplitude,
                    "n_s": cosmology.scalar_index,
                    "k_pivot": pivot,
                    "Omega_k": 0,
                    "N_ncdm": 0,
                    "output": "dTk,vTk",
                    "gauge": "newtonian",
                    "nbody_gauge_transfer_functions": "yes",
                    "P_k_max_1/Mpc": 1.05 * tau * (size // 2) * 3**0.5 / box_mpc,
                    "z_max_pk": z_initial + 0.1,
                }
            )
            table = model.get_transfer(z_initial)
            self.background = Background(
                omega_m=float(model.Omega_m()),
                omega_lambda=float(model.Omega_Lambda()),
                omega_r=float(model.Omega_r()),
            )
            hubble_0 = model.Hubble(0)  # H0/c in Mpc^-1
        finally:
            model.struct_cleanup()

        table_k = table["k (h/Mpc)"] * cosmology.h
        if used_k.min() < table_k[0] or used_k.max() > table_k[-1]:
            raise ValueError("The initial Fourier modes exceed the k range returned by CLASS.")

        baryon_fraction = cosmology.omega_b / (cosmology.omega_cdm + cosmology.omega_b)
        log_table_k = np.log(table_k)
        log_k = np.log(used_k)
        density_tf = baryon_fraction * table["d_b"] + (1.0 - baryon_fraction) * table["d_cdm"]
        divergence_tf = baryon_fraction * table["t_b"] + (1.0 - baryon_fraction) * table["t_cdm"]

        primordial_power = (
            2.0 * pi**2 * cosmology.scalar_amplitude / pivot**3 * (used_k / pivot) ** (cosmology.scalar_index - 4.0)
        )
        amplitude = np.sqrt((size / box_mpc) ** 3 * primordial_power)

        self.density_amplitude = torch.zeros_like(k)
        self.momentum_amplitude = torch.zeros_like(k)
        self.density_amplitude[keep] = torch.as_tensor(
            amplitude * np.interp(log_k, log_table_k, density_tf), device=device
        )
        # theta = div_x(v/c); dimensionless canonical momentum is u = a * (v/c) / [(H0/c) * L].
        self.momentum_amplitude[keep] = torch.as_tensor(
            -a_initial / hubble_0 * amplitude * np.interp(log_k, log_table_k, divergence_tf),
            device=device,
        )

        axis = torch.arange(size, dtype=torch.float64, device=device) / size
        self.lattice = torch.cartesian_prod(axis, axis, axis)

    @torch.no_grad()
    def sample(self, seed: int) -> tuple[Tensor, Tensor]:
        """Sample a seeded realization of initial particle positions and dimensionless momenta."""
        # Use a CPU generator so a given seed defines the exact same realization across backends.
        generator = torch.Generator().manual_seed(seed)
        white = torch.randn(self.shape, generator=generator, dtype=torch.float64).to(self.device)
        longitudinal = (1j * self.inverse_gradient) * torch.fft.rfftn(white)
        displacement = self._vector_field(longitudinal * self.density_amplitude)
        momentum = self._vector_field(longitudinal * self.momentum_amplitude)
        return (self.lattice + displacement).remainder(1.0), momentum

    def _vector_field(self, spectrum: Tensor) -> Tensor:
        field = torch.fft.irfftn(spectrum, s=self.shape, dim=(-3, -2, -1))
        return field.reshape(3, -1).T.contiguous()
