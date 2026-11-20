"""Periodic Particle-Mesh gravity solver and kick-drift-kick integrator."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from math import exp, log, tau
from typing import TYPE_CHECKING

import torch
from torch import Tensor

if TYPE_CHECKING:
    from .cosmology import Background


def spectral_geometry(size: int, device: torch.device) -> tuple[Tensor, Tensor, Tensor]:
    """Return squared wave numbers q^2, q/q^2 with Nyquist derivatives zeroed, and Nyquist mask."""
    kwargs = {"n": size, "d": 1 / size, "device": device, "dtype": torch.float64}
    full = tau * torch.fft.fftfreq(**kwargs)
    half = tau * torch.fft.rfftfreq(**kwargs)
    wave = torch.stack(torch.meshgrid(full, full, half, indexing="ij"))
    q_squared = wave.square().sum(dim=0)

    nyquist = torch.zeros_like(wave, dtype=torch.bool)
    if size % 2 == 0:
        nyquist[0, size // 2, :, :] = True
        nyquist[1, :, size // 2, :] = True
        nyquist[2, :, :, size // 2] = True

    denominator = q_squared.clone()
    denominator[0, 0, 0] = 1.0
    inverse_gradient = wave.masked_fill(nyquist, 0.0) / denominator
    return q_squared, inverse_gradient, nyquist.any(dim=0)


@dataclass(frozen=True)
class CICStencil:
    """Precomputed Cloud-In-Cell (CIC) corner indices and trilinear weights."""

    indices: Tensor
    weights: Tensor


class PeriodicMesh:
    """Three-dimensional periodic Particle-Mesh grid with CIC and spectral Poisson force."""

    def __init__(self, size: int, omega_m: float, device: torch.device) -> None:
        if size < 2:
            raise ValueError("The force grid requires at least two cells per axis.")
        self.size = size
        self.shape = (size, size, size)
        binary = torch.arange(2, device=device)
        self.offsets = torch.cartesian_prod(binary, binary, binary)
        self.upper_corner = self.offsets.bool()
        self.q_squared, inverse_gradient, _ = spectral_geometry(size, device)
        self.force_kernel = (1.5j * omega_m) * inverse_gradient

    def stencil(self, positions: Tensor) -> CICStencil:
        """Compute periodic CIC neighbor indices and trilinear weights for particle positions."""
        coordinate = positions.remainder(1.0) * self.size
        lower = coordinate.floor().long()
        fraction = coordinate - lower
        corners = (lower[:, None, :] + self.offsets).remainder(self.size)
        x, y, z = corners.unbind(dim=-1)
        indices = (x * self.size + y) * self.size + z
        weights = torch.where(self.upper_corner, fraction[:, None, :], 1.0 - fraction[:, None, :])
        return CICStencil(indices=indices, weights=weights.prod(dim=-1))

    def deposit(self, stencil: CICStencil, values: Tensor | None = None) -> Tensor:
        """Deposit unit particle counts or multi-channel particle attributes onto the grid."""
        indices = stencil.indices.flatten()
        weights = stencil.weights
        if values is None:
            grid = weights.new_zeros(self.size**3)
            grid.scatter_add_(0, indices, weights.flatten())
            return grid.view(self.shape)

        channels = values.shape[1]
        contributions = (values.T.unsqueeze(-1) * weights).flatten(start_dim=1)
        grid = weights.new_zeros((channels, self.size**3))
        grid.scatter_add_(1, indices.expand(channels, -1), contributions)
        return grid.view(channels, *self.shape)

    def density_contrast(self, stencil: CICStencil) -> Tensor:
        """Compute dimensionless density contrast delta = (N^3 / N_p) * c - 1."""
        return self.deposit(stencil) * (self.size**3 / stencil.indices.shape[0]) - 1.0

    def grid_force(self, delta: Tensor) -> Tensor:
        """Solve the periodic Poisson equation in Fourier space for the dimensionless grid force."""
        delta_hat = torch.fft.rfftn(delta, dim=(-3, -2, -1))
        return torch.fft.irfftn(self.force_kernel * delta_hat, s=self.shape, dim=(-3, -2, -1))

    def interpolate(self, grid: Tensor, stencil: CICStencil) -> Tensor:
        """Interpolate a multi-channel grid field back to particle positions using the CIC stencil."""
        corners = grid.flatten(start_dim=1)[:, stencil.indices]
        return (corners * stencil.weights).sum(dim=-1).T.contiguous()


@dataclass(frozen=True)
class ParticleState:
    """Snapshot of particle positions, canonical momenta, and cached mesh quantities at scale factor a."""

    a: float
    positions: Tensor
    momentum: Tensor
    delta: Tensor
    force: Tensor
    stencil: CICStencil


def fixed_targets(a_initial: float, a_final: float, steps: int) -> Iterator[float]:
    """Yield scale-factor targets spaced uniformly in ln(a)."""
    if steps < 1 or not (a_initial < a_final <= 1.0):
        raise ValueError("Fixed integration requires steps >= 1 and a_initial < a_final <= 1.")
    log_interval = log(a_final / a_initial)
    for index in range(1, steps):
        yield a_initial * exp(log_interval * index / steps)
    yield a_final


class Integrator:
    """Symplectic Kick-Drift-Kick (KDK) integrator in scale factor a."""

    def __init__(self, mesh: PeriodicMesh, background: Background) -> None:
        self.mesh = mesh
        self.background = background

    @torch.no_grad()
    def initialize(self, a: float, positions: Tensor, momentum: Tensor) -> ParticleState:
        """Validate initial particle arrays and evaluate the initial density and force fields."""
        if not (0 < a <= 1.0) or positions.ndim != 2 or positions.shape[1] != 3 or positions.shape[0] == 0:
            raise ValueError("Expected 0 < a <= 1 and a non-empty [particles, 3] position tensor.")
        if momentum.shape != positions.shape:
            raise ValueError("Momentum must have the same shape as positions.")
        if not (torch.isfinite(positions).all() & torch.isfinite(momentum).all()).item():
            raise FloatingPointError("Non-finite particle initial conditions.")

        wrapped_positions = positions.remainder(1.0)
        stencil = self.mesh.stencil(wrapped_positions)
        delta = self.mesh.density_contrast(stencil)
        force = self.mesh.interpolate(self.mesh.grid_force(delta), stencil)
        return ParticleState(
            a=a,
            positions=wrapped_positions,
            momentum=momentum,
            delta=delta,
            force=force,
            stencil=stencil,
        )

    @torch.no_grad()
    def step(self, state: ParticleState, a_next: float) -> ParticleState:
        """Advance particle state from state.a to a_next using one KDK leapfrog step."""
        if not (state.a < a_next <= 1.0):
            raise ValueError("A KDK step must advance scale factor towards a <= 1.")
        midpoint = 0.5 * (state.a + a_next)
        kick_first = self.background.time_factor(state.a, midpoint, kind="kick")
        drift = self.background.time_factor(state.a, a_next, kind="drift")
        kick_last = self.background.time_factor(midpoint, a_next, kind="kick")

        half_momentum = state.momentum + kick_first * state.force
        positions = (state.positions + drift * half_momentum).remainder(1.0)
        if not torch.isfinite(positions).all().item():
            raise FloatingPointError("Non-finite particle positions during KDK drift.")

        stencil = self.mesh.stencil(positions)
        delta = self.mesh.density_contrast(stencil)
        force = self.mesh.interpolate(self.mesh.grid_force(delta), stencil)
        momentum = half_momentum + kick_last * force
        if not torch.isfinite(momentum).all().item():
            raise FloatingPointError("Non-finite momentum during KDK kick.")

        return ParticleState(
            a=a_next,
            positions=positions,
            momentum=momentum,
            delta=delta,
            force=force,
            stencil=stencil,
        )

    def advance_fixed(self, state: ParticleState, a_final: float, steps: int) -> ParticleState:
        """Advance particle state to a_final using fixed steps uniform in ln(a)."""
        for target in fixed_targets(state.a, a_final, steps):
            state = self.step(state, target)
        return state
