"""Generate a one-fluid 1LPT realization from CLASS transfer functions."""

from dataclasses import dataclass
from math import isfinite, pi, tau

import numpy as np
import torch
from classy import Class
from torch import Tensor

from .pm import Background, spectral_geometry


@dataclass(frozen=True)
class Cosmology:
    h: float = 0.674
    omega_b: float = 0.0224
    omega_cdm: float = 0.120
    scalar_amplitude: float = 2.1e-9
    scalar_index: float = 0.965


class InitialConditions:
    def __init__(
        self, size: int, box_mpc: float, a_initial: float, cosmology: Cosmology, device: torch.device
    ):
        if size < 3 or not isfinite(box_mpc) or box_mpc <= 0 or not 0 < a_initial < 1:
            raise ValueError("Initial conditions require size >= 3, a positive box and 0 < a_initial < 1.")
        self.shape = (size,) * 3
        self.device = device
        z_initial = 1 / a_initial - 1
        q_squared, self.inverse_gradient, nyquist = spectral_geometry(size, device)
        k = q_squared.sqrt() / box_mpc
        keep = (k > 0) & ~nyquist
        used_k = k[keep].cpu().numpy()
        pivot = 0.05  # Mpc^-1

        model = Class()
        try:
            model.set({
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
            })
            table = model.get_transfer(z_initial)
            self.background = Background(float(model.Omega_m()), float(model.Omega_Lambda()), float(model.Omega_r()))
            hubble_0 = model.Hubble(0)  # H0/c in Mpc^-1
        finally:
            model.struct_cleanup()

        table_k = table["k (h/Mpc)"] * cosmology.h
        if used_k.min() < table_k[0] or used_k.max() > table_k[-1]:
            raise ValueError("The initial Fourier modes exceed the k range returned by CLASS.")
        baryon_fraction = cosmology.omega_b / (cosmology.omega_cdm + cosmology.omega_b)
        log_table_k, log_k = np.log(table_k), np.log(used_k)
        density = baryon_fraction * table["d_b"] + (1 - baryon_fraction) * table["d_cdm"]
        divergence = baryon_fraction * table["t_b"] + (1 - baryon_fraction) * table["t_cdm"]
        primordial_power = 2 * pi**2 * cosmology.scalar_amplitude * (used_k / pivot) ** (cosmology.scalar_index - 1)
        primordial_power /= used_k**3
        amplitude = np.sqrt((size / box_mpc) ** 3 * primordial_power)
        self.density_amplitude = torch.zeros_like(k)
        self.momentum_amplitude = torch.zeros_like(k)
        self.density_amplitude[keep] = torch.as_tensor(
            amplitude * np.interp(log_k, log_table_k, density), device=device
        )
        # theta = div_x(v/c). The canonical momentum is a * (v/c) / [(H0/c) * L].
        self.momentum_amplitude[keep] = torch.as_tensor(
            -a_initial / hubble_0 * amplitude * np.interp(log_k, log_table_k, divergence), device=device
        )

        axis = torch.arange(size, dtype=torch.float64, device=device) / size
        self.lattice = torch.cartesian_prod(axis, axis, axis)

    @torch.no_grad()
    def sample(self, seed: int) -> tuple[Tensor, Tensor]:
        # Sharing a real white-noise field preserves both Hermitian symmetry and the IC phases.
        # Use a CPU generator so a seed defines the same realization on either backend.
        generator = torch.Generator().manual_seed(seed)
        white = torch.randn(self.shape, generator=generator, dtype=torch.float64).to(self.device)
        longitudinal = (1j * self.inverse_gradient) * torch.fft.rfftn(white)
        displacement = self.vector_field(longitudinal * self.density_amplitude)
        momentum = self.vector_field(longitudinal * self.momentum_amplitude)
        return (self.lattice + displacement).remainder(1), momentum

    def vector_field(self, spectrum: Tensor) -> Tensor:
        field = torch.fft.irfftn(spectrum, s=self.shape, dim=(-3, -2, -1))
        return field.reshape(3, -1).T.contiguous()
