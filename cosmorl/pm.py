"""Periodic Particle-Mesh gravity and kick-drift-kick integration."""

from dataclasses import dataclass
from math import exp, log, tau

import numpy as np
import torch
from torch import Tensor


def spectral_geometry(size: int, device: torch.device) -> tuple[Tensor, Tensor, Tensor]:
    """Return squared wave numbers, q/q², and a mask for all Nyquist planes."""
    args = {"n": size, "d": 1 / size, "device": device, "dtype": torch.float64}
    full = tau * torch.fft.fftfreq(**args)
    half = tau * torch.fft.rfftfreq(**args)
    wave = torch.stack(torch.meshgrid(full, full, half, indexing="ij"))
    q_squared = wave.square().sum(dim=0)
    nyquist = torch.zeros_like(wave, dtype=torch.bool)
    if size % 2 == 0:
        nyquist[0, size // 2, :, :] = True
        nyquist[1, :, size // 2, :] = True
        nyquist[2, :, :, size // 2] = True
    denominator = q_squared.clone()
    denominator[0, 0, 0] = 1
    return q_squared, wave.masked_fill(nyquist, 0) / denominator, nyquist.any(dim=0)


@dataclass(frozen=True)
class CICStencil:
    indices: Tensor
    weights: Tensor


class PeriodicMesh:
    def __init__(self, size: int, omega_m: float, device: torch.device):
        if size < 2:
            raise ValueError("The force grid needs at least two cells per axis.")
        self.size = size
        self.shape = (size,) * 3
        binary = torch.arange(2, device=device)
        self.offsets = torch.cartesian_prod(binary, binary, binary)
        self.upper_corner = self.offsets.bool()
        self.q_squared, inverse_gradient, _ = spectral_geometry(size, device)
        self.force_kernel = (1.5j * omega_m) * inverse_gradient

    def stencil(self, positions: Tensor) -> CICStencil:
        coordinate = positions.remainder(1) * self.size
        lower = coordinate.floor().long()
        fraction = coordinate - lower
        corners = (lower[:, None, :] + self.offsets).remainder(self.size)
        x, y, z = corners.unbind(dim=-1)
        indices = (x * self.size + y) * self.size + z
        weights = torch.where(self.upper_corner, fraction[:, None, :], 1 - fraction[:, None, :])
        return CICStencil(indices, weights.prod(dim=-1))

    def deposit(self, stencil: CICStencil, values: Tensor | None = None) -> Tensor:
        indices = stencil.indices.flatten()
        weights = stencil.weights
        if values is None:
            return weights.new_zeros(self.size**3).scatter_add_(0, indices, weights.flatten()).view(self.shape)
        channels = values.shape[1]
        contributions = (values.T.unsqueeze(-1) * weights).flatten(start_dim=1)
        grid = weights.new_zeros((channels, self.size**3))
        grid.scatter_add_(1, indices.expand(channels, -1), contributions)
        return grid.view(channels, *self.shape)

    def density_contrast(self, stencil: CICStencil) -> Tensor:
        return self.deposit(stencil) * (self.size**3 / stencil.indices.shape[0]) - 1

    def grid_force(self, delta: Tensor) -> Tensor:
        delta_hat = torch.fft.rfftn(delta, dim=(-3, -2, -1))
        return torch.fft.irfftn(self.force_kernel * delta_hat, s=self.shape, dim=(-3, -2, -1))

    def interpolate(self, grid: Tensor, stencil: CICStencil) -> Tensor:
        corners = grid.flatten(start_dim=1)[:, stencil.indices]
        return (corners * stencil.weights).sum(dim=-1).T.contiguous()


_QUADRATURE_NODES, _QUADRATURE_WEIGHTS = np.polynomial.legendre.leggauss(8)


@dataclass(frozen=True)
class Background:
    omega_m: float
    omega_lambda: float
    omega_r: float

    def time_factor(self, a0: float, a1: float, *, drift: bool) -> float:
        """Integrate the drift or kick factor using eight nodes in ln(a)."""
        log_start, log_end = log(a0), log(a1)
        half_interval = (log_end - log_start) / 2
        a = np.exp((log_start + log_end) / 2 + half_interval * _QUADRATURE_NODES)
        expansion = np.sqrt(self.omega_r / a**4 + self.omega_m / a**3 + self.omega_lambda)
        integrand = 1 / (a ** (2 if drift else 1) * expansion)
        return float(half_interval * np.dot(_QUADRATURE_WEIGHTS, integrand))


@dataclass(frozen=True)
class ParticleState:
    a: float
    positions: Tensor
    momentum: Tensor
    delta: Tensor
    force: Tensor
    stencil: CICStencil


class Integrator:
    def __init__(self, mesh: PeriodicMesh, background: Background):
        self.mesh = mesh
        self.background = background

    @torch.no_grad()
    def initialize(self, a: float, positions: Tensor, momentum: Tensor) -> ParticleState:
        if not 0 < a <= 1 or positions.ndim != 2 or positions.shape[1] != 3 or positions.shape[0] == 0:
            raise ValueError("Expected 0 < a <= 1 and a nonempty [particles, 3] position array.")
        if momentum.shape != positions.shape:
            raise ValueError("Momentum must have the same shape as positions.")
        if not (torch.isfinite(positions).all() & torch.isfinite(momentum).all()).item():
            raise FloatingPointError("Non-finite particle initial conditions.")
        positions = positions.remainder(1)
        stencil = self.mesh.stencil(positions)
        delta = self.mesh.density_contrast(stencil)
        force = self.mesh.interpolate(self.mesh.grid_force(delta), stencil)
        return ParticleState(a, positions, momentum, delta, force, stencil)

    @torch.no_grad()
    def step(self, state: ParticleState, a_next: float) -> ParticleState:
        if not state.a < a_next <= 1:
            raise ValueError("A KDK step must advance a towards a <= 1.")
        midpoint = (state.a + a_next) / 2
        kick_first = self.background.time_factor(state.a, midpoint, drift=False)
        drift = self.background.time_factor(state.a, a_next, drift=True)
        kick_last = self.background.time_factor(midpoint, a_next, drift=False)
        half_momentum = state.momentum + kick_first * state.force
        positions = (state.positions + drift * half_momentum).remainder(1)
        if not torch.isfinite(positions).all().item():
            raise FloatingPointError("Non-finite particle positions during KDK drift.")
        stencil = self.mesh.stencil(positions)
        delta = self.mesh.density_contrast(stencil)
        force = self.mesh.interpolate(self.mesh.grid_force(delta), stencil)
        momentum = half_momentum + kick_last * force
        if not torch.isfinite(momentum).all().item():
            raise FloatingPointError("Non-finite momentum during KDK kick.")
        return ParticleState(a_next, positions, momentum, delta, force, stencil)

    def advance_fixed(self, state: ParticleState, a_final: float, steps: int) -> ParticleState:
        if steps < 1 or not state.a < a_final <= 1:
            raise ValueError("Fixed integration requires positive steps and a_initial < a_final <= 1.")
        a_initial = state.a
        log_interval = log(a_final / a_initial)
        for index in range(1, steps):
            state = self.step(state, a_initial * exp(log_interval * index / steps))
        return self.step(state, a_final)
